# Test Theseus PGO example

import argparse
import os

import imageio
import numpy as np
import pypose as pp
# import theseus as th
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch3d.io import load_objs_as_meshes  # , load_obj, load_ply
# from pytorch3d.structures import Meshes
from pytorch3d.renderer import (AmbientLights, BlendParams, MeshRasterizer,
                                MeshRenderer, RasterizationSettings,
                                SoftPhongShader)
from pytorch3d.renderer.camera_conversions import \
    _cameras_from_opencv_projection
from tqdm import tqdm

from src.rgb_to_lab import normalize_lab, rgb_to_lab
from src.rotation_continuity import (ortho6DToRot, ortho9DToTransform, transformToOrtho9D)


def genSE3GaussNoise(
        std: float = 0.1, seed: int = 0, dtype=torch.float32) -> pp.LieTensor:
    '''
    Generate a Gaussian perturbation pose to add noise on a SE3 pose
    @param std: STD for the Gaussian random noise defined on se3
    @param seed: Random seed
    @return SE3noise: Perturbation SE3 pose
    '''
    generator = torch.Generator()
    generator.manual_seed(seed)
    SE3noise = pp.randn_SE3(1, sigma=std, generator=generator, dtype=dtype)
    return SE3noise


class LineModDataset(torch.utils.data.Dataset):
    def __init__(self, data_pth):
        """
        Load images and GT poses from the LineMod data folder
        @param data_pth: [str] LineMod data folder
        """
        assert os.path.isdir(data_pth), f"Error: {data_pth} not a folder!"

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class PGOTest(object):
    def __init__(self, epoch=100, lr=0.1, dtype=torch.float32) -> None:
        # Initial values for poses (quaternion order qwxyz)
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


class DiffRender(object):
    def __init__(self, models, dtype=torch.float32):
        """
        Load mesh model and camera parameters
        @param models: [str-list] Path to camera models
        @param type: Data type
        """
        # Define device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load mesh models
        self._meshes_ = load_objs_as_meshes(models, device=device)

    def camerasFromOpenCV(self, R, t, K, img_size):
        """
        Build cameras from OpenCV convention camera matrices
        OpenCV camera convention: x-right-y-down-z-forward
        @param R: [Bx3x3 tensor] object-to-camera rotations
        @param t: [Bx3 tensor] object-to-camera translations
        @param K: [Bx3x3 tensor] camera intrinsics
        @param img_size: [Bx2 tensor] image sizes [[width, height]]
        @return cameras: Pytorch cameras
        """
        # NOTE: Pytorch3D 0.7 has a bug in _cameras_from_opencv_projection see:
        # https://github.com/facebookresearch/pytorch3d/issues/1300
        # Basically they forget to flip the principal point of the cameras
        K[..., [0, 1], -1] = K[..., [1, 0], -1]  # Flip the principal points
        cameras = _cameras_from_opencv_projection(R, t, K, img_size)
        return cameras

    def render(self, rot, trans, intrinsics, img_size, dtype=torch.float32):
        """
        Render RGB images and masks
        @param rot: [Bx3x3 tensor] object-to-camera rotations
        @param trans: [Bx3 tensor] object-to-camera translations
        @param intrinsics: [Bx5 tensor] Camera intrinsics
        @param img_size: [Bx2] Image size [[width, height]]
        @return rgb: [BxHxWx3] Rendered RGB images
        @return mask: [BxHXWX1] Rendered binary masks
        """
        assert len(intrinsics.shape) == 2 and intrinsics.shape[-1] == 5, \
            "Error: camera intrinsics must have shape Bx5"
        assert len(rot.shape) == 3 and rot.shape[-1] == 3 and \
            rot.shape[-2] == 3, "Error: Rotation matrices must be Bx3x3"
        assert len(trans.shape) == 2 and trans.shape[-1] == 3,\
            "Error: Translation vectors must be Bx3"
        assert len(img_size.shape) == 2 and img_size.shape[-1] == 2, \
            "Error: Image size vectors must have shape Bx2"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = intrinsics.shape[0]
        # Read camera matrices
        img_size = torch.tensor(img_size, dtype=torch.int).to(device)
        # Populate the camera intrinsics matrix
        K = torch.zeros(batch_size, 3, 3, dtype=dtype).to(device)
        K[:, 0, 0], K[:, 1, 1] = intrinsics[:, 0], intrinsics[:, 1]
        K[:, 0, -1], K[:, 1, -1] = intrinsics[:, 2], intrinsics[:, 3]
        K[:, 0, 1] = intrinsics[:, -1]
        K[:, -1, -1] = torch.ones(batch_size)

        # Create cameras from the rotations
        self._cams_ = self.camerasFromOpenCV(rot, trans, K, img_size)
        # Rasterization setting
        raster_settings = RasterizationSettings(
            # NOTE: here takes [height width]
            image_size=(int(img_size[0, 1]), int(img_size[0, 0]))
        )
        # Create RGB renderer
        # NOTE: must apply lighting here otherwise rendered image too dark
        lights = AmbientLights(device=device)
        # Set background color to black
        blend_params = BlendParams(
            sigma=1e-4, gamma=1e-4, background_color=torch.zeros(3)
        )
        self._renderer_ = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self._cams_, raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, cameras=self._cams_, blend_params=blend_params,
                lights=lights  # TODO: how to set light?
            )
        )
        # Render RGBA images
        images = self._renderer_(self._meshes_)
        # Uncomment to visualize the 1st rendered image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(images[0, ..., :3].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(images[0, ..., -1].detach().cpu().numpy())
        # plt.show()
        # TODO: not sure why silhouette is not binary,
        # TODO: we need assert that within the mask we always have 0.5
        return images[..., :3], images[..., -1:]*2


def main(
        image, models, R, t, K, out, epochs=500, lr=1e-3, dtype=torch.float32
):
    """
    Align object pose with captured image(s) by self-supervision
    @param image: [str] Path to captured real image
    @param models: [str-list] Path to object textured CAD models
    @param R: [Bx3x3 tensor] object-to-camera rotations
    @param t: [Bx3 tensor] object-to-camera translations
    @param K: [Bx3x3 tensor] camera intrinsics
    @param out: [str] Path to output files
    @param epochs: [int] Number of update iterations
    @param lr: [float] learning rate
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare the captured image
    image = np.array(Image.open(image).convert("RGB"))
    # Uncomment to show image
    # plt.imshow(image)
    # plt.show()
    rgb_gt = torch.tensor(image).clone().to(device) / 255  # rescale to 0~1
    height, width = image.shape[:2]
    img_size = np.array([width, height]).reshape(1, 2)

    # Initial object-to-world pose
    # pose0 = th.SE3()
    # pose0.update_from_rot_and_trans(th.SO3(tensor=R), th.Point3(tensor=t))
    # pose_init = pose0.log_map()
    pose_0_torch = torch.cat((R.clone(), t.clone().unsqueeze(-1)), dim=-1)
    pose_0_torch_9D = transformToOrtho9D(pose_0_torch)
    pose_init = pose_0_torch_9D.clone().to(dtype).to(device)
    pose_init.requires_grad = True

    # Prepare the optimizer and loss function
    optim = torch.optim.AdamW(params=[pose_init], lr=lr)
    loss_fun = torch.nn.SmoothL1Loss()

    # Create a differentiable renderer
    diff_render = DiffRender(models)

    losses = []
    frames = []
    for ii in tqdm(range(epochs)):

        # Convert 6D se3 reprs to rotations and translations
        # rot = th.SE3.exp_map(pose_init.clone()).rotation().tensor
        # trans = th.SE3.exp_map(pose_init.clone()).translation().tensor
        rot = ortho6DToRot(pose_init[..., :6])
        trans = pose_init[..., 6:]

        # Render the RGB image and mask
        # TODO: it's still a mystery why the rendered images have nan grads
        # https://github.com/pytorch/pytorch/issues/40497#issuecomment-656312357
        rgb_rendered, mask_rendered = diff_render.render(
            rot, trans, K, img_size
        )
        # NOTE: We don't backprop through the rendered mask
        mask_rendered = mask_rendered.detach()
        # Uncomment to see gradient of rendered image
        # rgb_rendered.register_hook(lambda x:print(x))

        # Convert the rendered RGB to lab
        lab_rendered = normalize_lab(rgb_to_lab(
            rgb_rendered.permute(0, 3, 1, 2)
        ))

        # Mask the ground truth image
        rgb_gt_masked = rgb_gt * mask_rendered.squeeze(0)
        lab_gt_masked = normalize_lab(rgb_to_lab(
            rgb_gt_masked.permute(2, 0, 1).unsqueeze(0),
        ))
        # Lets take a look at the image alignment
        if ii % 10 == 0:
            with torch.no_grad():
                background = Image.fromarray((
                    rgb_rendered.squeeze(0).detach().cpu().numpy() * 255
                ).astype(np.uint8))
                overlay = Image.fromarray(image.astype(np.uint8))
                blend = Image.blend(background, overlay, 0.5)
                frames.append(np.array(blend))
                # Uncomment to vis the overlayed images
                # plt.imshow(rgb_rendered.squeeze(0).detach().cpu().numpy())
                # plt.imshow(image, alpha=0.5)
                # plt.show()

        # NOTE: must divide the photometric loss by the mask area
        # NOTE: otherwise rendered obj is encouraged to shrink to 1 pixel
        mask_area = mask_rendered.sum()
        # Avoid division by zero
        mask_area = torch.max(torch.ones_like(mask_area), mask_area).detach()
        # NOTE: Normalize to make the loss computation numerically stable
        mask_area = mask_area / width / height
        # TODO: Prafull said better disentangle the mask_area and photo loss
        # So still not sure how to add mask_area

        # NOTE: only use AB channel to compute the loss
        loss = loss_fun(
            lab_rendered[..., 1:, :, :],
            lab_gt_masked[..., 1:, :, :]
        )
        # Uncomment to vis the rendered and gt images
        # plt.imshow(img_gt_masked.detach().squeeze().cpu().numpy())
        # plt.show()
        # plt.imshow(rgb_rendered.detach().squeeze().cpu().numpy())
        # plt.show()

        loss.backward()

        # Gradient descent
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())

    plt.plot(losses)
    plt.show()
    plt.imshow(rgb_rendered.squeeze(0).detach().cpu().numpy())
    plt.imshow(image, alpha=0.5)
    plt.show()
    imageio.mimwrite(out + "video.mp4", frames, fps=10, quality=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", "-lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", "-ep", type=int, default=500,
        help="Number of pose update iterations"
    )
    parser.add_argument(
        "--image", "-i", type=str, help="Observed image",
        default="data/lm_images/ape/000000.png"
    )
    parser.add_argument(
        "--models", "-m", nargs="+", type=str, help="Object CAD models",
        default=[
            "data/lm_models/ape/textured.obj"
        ]
    )
    # TODO: Allow reading multiple camera intrinsics, image sizes and poses
    parser.add_argument(
        "--intrinsics", "-K", type=float, nargs=5, help="fx, fy, cx, cy, s",
        default=[572.4414, 573.5704, 325.2611, 242.0490, 0.0]
    )
    parser.add_argument(
        "--R", "-R", type=float, nargs=9,
        help="Initial object rotation estimate",
        default=[0.0963, 0.9940, 0.0510, 0.5733, -0.0135,
                 - 0.8192, -0.8137, 0.1081, -0.5712]
    )
    parser.add_argument(
        "--t", "-t", type=float, nargs=3,
        help="Initial object translation estimate",
        default=[-0.1054, -0.1475, 0.8149]
    )
    parser.add_argument(
        "--out", "-o", type=str, default="/home/ziqi/Desktop/",
        help="The folder to save output files"
    )
    args = parser.parse_args()
    # pgo_test = PGOTest(epoch=args.epochs, lr=args.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K = torch.tensor(args.intrinsics).reshape(1, 5).to(device)
    R = torch.tensor(args.R).reshape(1, 3, 3).to(device)
    t = torch.tensor(args.t).reshape(1, 3).to(device)

    # diff_render = DiffRender(args.models)
    # diff_render.render(R, t, K, np.array([[640, 480]]))

    main(
        args.image, args.models, R, t, K, args.out,
        args.epochs, args.lr
    )

    print("Done!")
