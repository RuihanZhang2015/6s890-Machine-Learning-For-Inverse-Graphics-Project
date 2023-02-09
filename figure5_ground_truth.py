import warnings
warnings.filterwarnings("ignore")
## environment: pypose
import argparse
import imageio
import lpips
import pickle
import os
import numpy as np
import pypose as pp
import torch
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm
import pickle
import pprint
import collections
import matplotlib as mpl


from src.data_loader import LineModDataset
from src.renderer import DiffRender
from src.ms_rgb import MS_RGB, MS_RGB_LAB

from src.rgb_to_lab import normalize_lab, rgb_to_lab
from src.rotation_continuity import ortho9DToTransform, transformToOrtho9D
from src.ssim import MS_SSIM
from src.utils.image_utils import batch_crop_resize, bboxToSquare, maskToBbox
from src.utils.pose_utils import poseError
from src.utils.visual_utils import torchImageToPlottable


class PGOTest(object):
    def __init__(self, epoch=100, lr=0.1, dtype=torch.float32) -> None:
        # Initial values for poses (quaternion order qxyzw)
        self.preds = []

        # Initial object-to-world pose
        # self.obj_pose_se3 = torch.tensor(
        #     [[0.1, 0.1, 0.1, 0, 0, 0]], dtype=dtype
        # )
        self.obj_pose_9d = transformToOrtho9D(pp.se3(
            torch.tensor([[0.1, 0.1, 0.1, 0, 0, 0]], dtype=dtype)
        ).matrix())
        self.obj_pose_9d.requires_grad = True
        # Pose predictions
        self.preds.append(pp.SE3(
            torch.tensor([[0.1, 0.0, 0.0, 0.6, 0, 0, 0.8]], dtype=dtype)
        ))
        print(self.preds[0])
        self.preds.append(pp.SE3(
            torch.tensor([[0.0, -0.1, 0.1, 0.6, 0, 0, 0.8]], dtype=dtype)
        ))
        self.preds.append(pp.SE3(
            torch.tensor([[-0.2, 0.1, -0.1, 0.6, 0, 0, 0.8]], dtype=dtype)
        ))

        optim = torch.optim.Adam(params=[self.obj_pose_9d], lr=lr)
        losses = []
        for ep in tqdm(range(epoch)):
            # self.obj_pose = th.SE3.exp_map(self.obj_pose_se3.clone())
            self.obj_pose = pp.from_matrix(
                ortho9DToTransform(self.obj_pose_9d.clone()), ltype=pp.SE3_type
            )

            loss = torch.tensor([0.0], dtype=dtype)
            # Define the residual pose error
            for pred in self.preds:
                error = (self.obj_pose.Inv() * pred).Log()
                loss += (error**2).sum()
            loss.backward()

            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                print(pp.from_matrix(
                    ortho9DToTransform(self.obj_pose_9d), ltype=pp.SE3_type
                ))
                losses.append(loss.item())

        plt.plot(losses)
        plt.show()

def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", "-lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", "-ep", type=int, default=500,
        help="Number of pose update iterations"
    )
    parser.add_argument(
        "--image_dir", "-i", type=str, help="Path to image folder",
        default="data/lm_images/000002/"
    )
    parser.add_argument(
        "--model_dir", "-m", type=str, help="Path to object CAD models",
        default="data/lm_models/"
    )
    parser.add_argument(
        "--out", "-o", type=str, default="/home/ziqi/Desktop/",
        help="The folder to save output files"
    )
    parser.add_argument(
        "--loss_configuration", "-loss", type=dict, default={},
        help="loss configuration"
    )
    args = parser.parse_args(args=[])
    return args

def main(args, perturb=pp.identity_SE3(1), show_image = False):
    """
    Align object pose with captured image(s) by self-supervision
    @param image_dir: [str] Path to image folder
    @param model_dir: [str] Path to object CAD model folder (.ply)
    @param out: [str] Path to output files
    @param epochs: [int] Number of update iterations
    @param lr: [float] learning rate
    @param perturb: [pp.LieTensor] Pose to perturb the camera pose
    """


    image_dir, model_dir, out, epochs, lr, loss_configuration= args.image_dir, args.model_dir, args.out, args.epochs, args.lr, args.loss_configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    linemod_data = LineModDataset(image_dir, model_dir)

    # Load image, GT obj poses, obj CAD models, K matrix
    rgb_gt, obj_pose_gt_all, obj_model_all, K_mat = linemod_data[0]
    rgb_gt = rgb_gt.to(device)
    obj_pose_gt = list(obj_pose_gt_all.values())[0].to(device)
    
    obj_pose = pp.mat2SE3(obj_pose_gt)
    translation = obj_pose.translation()
    translation = translation.cpu().detach().numpy()[0]
    print(translation)

    obj_model = list(obj_model_all.values())
    K_mat = K_mat.to(device)
    img_size = torch.tensor(rgb_gt.shape[-2:]).view(1, 2)

    # Perturb GT obj poses as initial value for optimization
    obj_pose_perturb = torch.einsum(
        "...ij,...jk->...ik", perturb.matrix().to(K_mat), obj_pose_gt.clone()
    )
    # print(f"Initial pose:\n{obj_pose_perturb.squeeze().cpu().numpy()}")

    # Initial object-to-world pose
    pose_0_torch = obj_pose_perturb
    pose_0_torch_9D = transformToOrtho9D(pose_0_torch)
    pose_init = pose_0_torch_9D.clone().to(K_mat)
    pose_init.requires_grad = True

    # Prepare the optimizer and loss function
    optim = torch.optim.AdamW(params=[pose_init], lr=lr)

    # Create a differentiable renderer
    diff_render = DiffRender(obj_model, device=device)

    losses,losses_dict, R_error, t_error, frames = [], {}, [], [], []
    for ii in tqdm(range(epochs)):

        # Convert 6D se3 reprs to rotations and translations
        obj_pose_opt = ortho9DToTransform(pose_init)


        # Render the RGB image and mask
        rgb_rendered, mask_rendered = diff_render.render(
            obj_pose_opt, K_mat, img_size, heuristic=False
        )
        # Extract the RoIs from the rendered RGB images
        bbox = maskToBbox(mask_rendered)
        bbox_sq = bboxToSquare(bbox, 1.0)
        # Crop and resize the rendered images
        rgb_rendered_roi = batch_crop_resize(
            rgb_rendered, bbox_sq, 256, 256
        )

        # Mask the ground truth image
        rgb_gt_masked = rgb_gt * mask_rendered
        # Crop and resize the ground truth image
        rgb_gt_masked_roi = batch_crop_resize(
            rgb_gt_masked, bbox_sq, 256, 256
        )

        x0 = int((translation[1]+0.5-0.2)*480)
        x1 = int((translation[1]+0.5+0.2)*480)
        y0 = int((translation[0]+0.5-0.2)*640)
        y1 = int((translation[0]+0.5+0.2)*640)

        plt.figure()
        # print(rgb_rendered.shape)
        # plt.imshow(rgb_rendered[0, ...].permute(1,2,0).cpu().detach().numpy())
        # plt.imshow(rgb_gt[0, ...].permute(1,2,0).cpu().detach().numpy(),alpha= 0.5)
        plt.imshow(rgb_rendered[0, :, x0:x1,y0:y1].permute(1,2,0).cpu().detach().numpy())
        plt.imshow(rgb_gt[0, :,x0:x1,y0:y1].permute(1,2,0).cpu().detach().numpy(),alpha= 0.5)
        plt.show()
        plt.savefig('plots/gt_object{}.jpg'.format(i))


        
args = retrieve_args()
args.epochs = 1

for i in [1,5,8,9]:
    args.image_dir = 'data/lm_images/{0:06d}/'.format(i)
    args.out = 'output{0:02d}'.format(i)
    folders = os.listdir(args.out)

    pose = [0, 0, 0, 0, 0, 0, 0]
    perturb = pp.SE3([pose])
    main(args, perturb,show_image = False)


      