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

from src.data_loader import LineModDataset
from src.renderer import DiffRender
from src.ms_rgb import MS_RGB, MS_RGB_LAB

from src.rgb_to_lab import normalize_lab, rgb_to_lab
from src.rotation_continuity import ortho9DToTransform, transformToOrtho9D
from src.ssim import MS_SSIM
from src.utils.image_utils import batch_crop_resize, bboxToSquare, maskToBbox
from src.utils.pose_utils import poseError
from src.utils.visual_utils import torchImageToPlottable
    
# plt.figure()
fig,axs = plt.subplots(1,3,figsize = (12,4))  

for obj_ind,obj in enumerate([1,8,9]):
    outdir = 'output0{}/loss_weight'.format(obj)

    loss_configurations = []
    for a in np.linspace(0.1,0.9,9):
        for b in np.linspace(0.1,0.9-a,int(np.rint((0.9-a)*10))):
            # print(a,b)
            loss_configurations.append(
                {
                    'rgb':a,
                    'ssim_ms':b,
                    'perceptual':1-a-b
                }
            )

    # for dim in range(3):
        
    #     data = []
        
    #     for loss_configuration in loss_configurations:
    #         dirname = outdir
    #         for k in loss_configuration:
    #             dirname += '_{0}_{1:0.2f}'.format(k, loss_configuration[k])
    #         dirname += '/'

    #         filename = dirname + '/basin_dim_{}.pkl'.format(dim)
    #         try:
    #             # print(filename)
    #             with open(filename,'rb') as f:
    #                 out = pickle.load(f)
    #             if out['left']<0 or out['right']>0:
    #                 # print('he')
    #                 data.append([list(loss_configuration.values())[0],list(loss_configuration.values())[1],out['right']-out['left']])
    #         except:
    #             pass
    #     data = np.asarray(data)
    #     print(data[:,2])
    #     plt.scatter(data[:,0],data[:,1],s = 20, c = data[:,2],cmap = 'hot')
    #     plt.title('object{}_dim_{}'.format(obj,dim))
    #     plt.savefig('plots/figure3_object{}_dim{}.jpg'.format(obj,dim))
    
    
    data = []
    for loss_configuration in loss_configurations:
        dirname = outdir
        for k in loss_configuration:
            dirname += '_{0}_{1:0.2f}'.format(k, loss_configuration[k])
        dirname += '/'
        
        temp = 1
        for dim in range(3):
            filename = dirname + '/basin_dim_{}.pkl'.format(dim)
            try:
                with open(filename,'rb') as f:
                    out = pickle.load(f)
                if out['left']<0 or out['right']>0:
                    temp *= out['right']-out['left']
            except:
                pass
        data.append([list(loss_configuration.values())[0],list(loss_configuration.values())[1],temp])
    data = np.asarray(data)
    axs[obj_ind].set_xlabel('weight for color loss')
    axs[obj_ind].set_ylabel('weight for ms-ssim loss')
    im = axs[obj_ind].scatter(data[:,0],data[:,1],s = 20, c = data[:,2],cmap = 'hot')
    axs[obj_ind].set_title('object{}'.format(obj))
fig.colorbar(im, ax=axs[2])
plt.savefig('plots/figure3_object_final.jpg'.format(obj,dim),dpi = 200)
