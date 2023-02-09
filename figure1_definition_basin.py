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
import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm

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
    # print(obj_pose_gt)
    obj_model = list(obj_model_all.values())
    K_mat = K_mat.to(device)
    img_size = torch.tensor(rgb_gt.shape[-2:]).view(1, 2)

    # Perturb GT obj poses as initial value for optimization
    obj_pose_perturb = torch.einsum(
        "...ij,...jk->...ik", perturb.matrix().to(K_mat), obj_pose_gt.clone()
    )


    # Initial object-to-world pose
    pose_0_torch = obj_pose_perturb
    pose_0_torch_9D = transformToOrtho9D(pose_0_torch)
    pose_init = pose_0_torch_9D.clone().to(K_mat)
    pose_init.requires_grad = True

    obj_pose = pp.mat2SE3(obj_pose_gt)
    translation = obj_pose.translation()
    translation = translation.cpu().detach().numpy()[0]
    print(translation)

    # Prepare the optimizer and loss function
    optim = torch.optim.AdamW(params=[pose_init], lr=lr)
   
    if 'rgb' in loss_configuration:
        color_loss_fun = torch.nn.SmoothL1Loss(beta=0, reduction="sum")

    # if 'rgb_ms' in loss_configuration:
    #     color_loss_fun = MS_RGB()

    # if 'rgb_ms_lab' in loss_configuration:
    #     color_loss_fun = MS_RGB_LAB()

    if 'ssim_ms' in loss_configuration:
        ms_ssim_loss_fun = MS_SSIM(data_range=1.0, normalize=True).to(device)
    
    if 'perceptual' in loss_configuration:
        percep_loss_fun = lpips.LPIPS(net="alex").to(device)

    # Create a differentiable renderer
    diff_render = DiffRender(obj_model, device=device)

    import collections
    losses_all, losses,losses_dict, R_error, t_error, frames = collections.defaultdict(list), [], {}, [], [], []
    for ii in tqdm(range(epochs)):
        
        # Convert 6D se3 reprs to rotations and translations
        obj_pose_opt = ortho9DToTransform(pose_init)

        # Compute pose to GT error
        with torch.no_grad():
            t_err, R_err = poseError(
                pp.mat2SE3(obj_pose_opt.detach()), pp.mat2SE3(obj_pose_gt)
            )
            # if ii>10 and t_err < 12:
            #     break
            R_error.append(R_err)
            t_error.append(t_err)
        
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

        # Lets take a look at the image alignment
        # if ii == 0:
        # with torch.no_grad():
            
            # plt.figure()
            # plt.imshow(rgb_rendered[0, :,144:336,192:448].permute(1,2,0).cpu().numpy())
            # plt.imshow(rgb_gt[0, :,144:336,192:448].permute(1,2,0).cpu().numpy(),alpha= 0.5)
            # plt.show()
            # print('hello')
            # plt.savefig('plots/ground_truth.jpg')

                # frames.append(np.array(blend))
                # if show_image:
                    # frames.append(np.array(blend))
                # Uncomment to vis the overlayed images
                # plt.imshow(torchImageToPlottable(rgb_rendered))
                # plt.imshow(torchImageToPlottable(rgb_gt), alpha=0.5)
                # plt.show()

        lab_rendered_roi = rgb_to_lab(rgb_rendered_roi)
        lab_gt_masked_roi = rgb_to_lab(rgb_gt_masked_roi)
        if 'rgb' in loss_configuration:
            # Normalize the photometric loss by the mask area
            mask_area = (
                rgb_rendered_roi[:, 0, ...] > 0.0
            ).sum(dim=(-2, -1), keepdims=True)
            # Avoid division by zero
            mask_area = torch.max(torch.ones_like(mask_area), mask_area)

            # Convert the rendered and real image from RGB to lab
            
            losses_dict['rgb'] = color_loss_fun(
                normalize_lab(lab_rendered_roi)[..., 1:, :, :]/mask_area,
                normalize_lab(lab_gt_masked_roi)[..., 1:, :, :]/mask_area,
            )

        # if 'rgb_ms' in loss_configuration:
        #     losses_dict['rgb_ms'] = color_loss_fun(
        #         rgb_rendered, 
        #         rgb_gt,
        #         mask_rendered.float()
        #     )

        # if 'rgb_ms_lab' in loss_configuration:
        #     losses_dict['rgb_ms_lab'] = color_loss_fun(
        #         rgb_rendered, 
        #         rgb_gt,
        #         mask_rendered.float()
            # )
        if 'perceptual' in loss_configuration:
            losses_dict['perceptual'] = percep_loss_fun(
                rgb_gt_masked_roi, rgb_rendered_roi, normalize=True
            )
        
        if 'ssim_ms' in loss_configuration:
            losses_dict['ssim_ms'] = (
                1.0 - ms_ssim_loss_fun(rgb_gt_masked_roi, rgb_rendered_roi)
            ).mean()

        loss_list = []
        for k in loss_configuration:
            # print(ii, k,losses_dict[k])
            loss_list.append(losses_dict[k] * loss_configuration[k])
            losses_all[k].append(losses_dict[k])
        loss = sum(loss_list)

        

        # Uncomment to vis the rendered and gt images
        # plt.imshow(torchImageToPlottable(rgb_gt_masked_roi[0, ...]))
        # plt.show()
        
        # plt.imshow(torchImageToPlottable(rgb_rendered_roi[0, ...]))
        # plt.show()

        loss.backward()

        # Gradient descent
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())

    # with open('output09/rgb_gt.pkl','wb') as f:
    #     pickle.dump(torchImageToPlottable(rgb_gt_masked[0, ...]),file=f)
        
    if show_image:
        plt.plot(losses)
        plt.show()
        plt.plot(t_error)
        plt.plot(R_error)
        plt.legend(["Translation error (cm)", "Rotation error (deg)"])
        plt.show()
        plt.imshow(torchImageToPlottable(rgb_rendered[0, ...]))
        plt.imshow(torchImageToPlottable(rgb_gt[0, ...]), alpha=0.5)
        plt.show()
        imageio.mimwrite(out + "video.mp4", frames, fps=10, quality=8)

    

    # with open('output/basin/dim_{0}_weight_{1:0.3f}_value_{2}.pkl'.format(dim,weight,value),'wb') as f:
    #     out = {
    #         'losses':losses,
    #         't_error':t_error,
    #         'R_error':R_error
    #     }

    # plt.close()
    # plt.figure()
    # plt.imshow(torchImageToPlottable(rgb_rendered[0, ...]))
    # plt.imshow(torchImageToPlottable(rgb_gt[0, ...]), alpha=0.5)
    # plt.savefig('output/image/dim_{0}_weight_{1:0.3f})_value_{2}.png'.format(dim,weight,value))
    x0 = int((translation[1]+0.5-0.2)*480)
    x1 = int((translation[1]+0.5+0.2)*480)
    y0 = int((translation[0]+0.5-0.2)*640)
    y1 = int((translation[0]+0.5+0.2)*640)
    return losses_all,t_error,R_error,torchImageToPlottable(rgb_rendered[0, :,x0:x1,y0:y1]),obj_pose_opt.detach()

def valid_3d(perturb,dirname):

    try:
        os.makedirs(dirname)
    except:
        pass

    filename = dirname + '/pose_' + '_'.join(['{0:0.2f}'.format(x) for x in perturb[:6]]) + '.pkl'

    if os.path.exists(filename):
        with open(filename,'rb') as f:
            out = pickle.load(f)
        return True
        # if out['t_error'][-1]<12:
        #     return True
        # else:
        #     return False

    perturb_pp = pp.SE3([perturb])
        
    losses_dict,t_error,R_error,img, pose =  main(args, perturb_pp,show_image = False)
    # print('losses_dict', losses_dict)

    out = {
            'perturb':perturb,
            'losses_dict':losses_dict,
            't_error':t_error,
            'R_error':R_error,
            'img':img
    }
    with open(filename,'wb') as f:
        pickle.dump(out,f)
    
    if t_error[-1]<12:
        return True
    else:
        return False

def enumerate_initial_losses(args):

    ################ Output #################
    dirname = args.out +  '/converge'
    for k in args.loss_configuration:
        dirname += '_{0}_{1:0.2f}'.format(k, args.loss_configuration[k])
    dirname += '/'
    print(dirname)

    try:
        os.makedirs(dirname)
    except:
        pass

    for dim1,dim2 in [[0,1]]:#,[0,2]]:#,[3,4],[4,5],[3,5]]:
        for i in np.linspace(-0.15,0.15,7):
            for j in np.linspace(-0.15,0.15,7):
                perturb = [0,0,0,0,0,0,1.0]
                perturb[dim1] = i
                perturb[dim2] = j
                print('perturb',perturb)
                valid_3d(perturb,dirname)
                # try:
                #     valid_3d(perturb,dirname)
                # except:
                #     pass

    
def plot_initial(args):

    dirname = args.out +  '/converge'
    for k in args.loss_configuration:
        dirname += '_{0}_{1:0.2f}'.format(k, args.loss_configuration[k])
    dirname += '/'
    # print(dirname)
    
    for dim1,dim2 in [[0,1]]:#[[1,2],[0,1],[0,2],[3,4],[4,5],[3,5]]:
        h = np.ones((9,9))
        for file in os.listdir(dirname):
            try:
                with open(dirname + file,'rb') as f:
                    out = pickle.load(f)
            except:
                continue
                
            # temp = [i for i in range(6) if abs(out['perturb' ][i]-0)>0.001]
            # if set(temp).issubset(set([dim1,dim2])):
            if True:
                i,j = out['perturb'][dim1],out['perturb'][dim2]
                ii = np.rint(i/0.05)*0.05
                jj = np.rint(j/0.05)*0.05
                # print(ii,i,jj,j)
                if abs(ii-i)<0.001 and abs(jj-j)<0.01:
                    # print(int(np.rint((i+0.2)*20)),int(np.rint((j+0.2)*20)) )
                    h[ int(np.rint((i+0.2)*20)),int(np.rint((j+0.2)*20))  ] = 0 if out['t_error'][-1]<12 else 1
                # plt.imshow(out['img'])
                # plt.savefig('plots/figure1/{}_{}.jpg'.format(int(np.rint((i+0.1)*50)),int(np.rint((j+0.1)*50))))

        fig = plt.figure()

        plt.imshow(h,cmap=mpl.colormaps['hot'])
        # import seaborn as sns; sns.set()
        # hmax = sns.heatmap(h_large,
        #         #cmap = al_winter, # this worked but I didn't like it
        #         # cmap = matplotlib.cm.winter,
        #         # alpha = 0.5, # whole heatmap is translucent
        #         # annot = True,
        #         # zorder = 2,
        #         )
        # plt.show()
        ticks = (np.linspace(-0.2,0.2,9)+0.2)*20
        ticks = [int(np.rint(x)) for x in ticks]
        labels = np.linspace(-0.2,0.2,9)
        labels = ['{0:0.2f}'.format(x) for x in labels]
        plt.xticks(ticks,labels)
        # mapping = ['z','x','y','angle1','angle2','angle3']
        # plt.ylabel(mapping[dim2])
        # plt.xlabel(mapping[dim1])
        plt.yticks(ticks,labels)
        # plt.colorbar()
        # print('plots/object_{}_converge_{}_dim{}_{}.jpg'.format(k,dim1,dim2))
        plt.title('dim {} {}'.format(dim1,dim2))
        plt.savefig('plots/object_{}_converge_{}_dim{}_{}.jpg'.format(obj,k,dim1,dim2),dpi = 200)


args = retrieve_args()
for obj in [1]:
    args.epochs = 500
    args.image_dir = 'data/lm_images/{0:06d}/'.format(obj)
    args.out = 'output{0:02d}'.format(obj)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    args.loss_configuration = {
                        'rgb':1.0,
                        'ssim_ms':0.2,
                        'perceptual':1.0
                    }
    # enumerate_initial_losses(args)
    plot_initial(args)