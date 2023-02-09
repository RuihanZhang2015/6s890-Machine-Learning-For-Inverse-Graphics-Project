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

fig,axs = plt.subplots(2,3,figsize = (12,4))
x = [-0.2, -0.1, -0.075, -0.0625, -0.05625, -0.05, 0, 0.025, 0.05, 0.05625, 0.0625, 0.075,0.1, 0.2]
y = [1,     1,   1,      1,      1,          0,    0, 0,      0,   0,        1,     1,   1,   1]
axs[0,0].plot(x,y,'.-')
axs[0,0].set_ylabel('x basin')
# axs[0,0].set_ylabel('converge')
axs[0,0].set_yticks([0,1],['yes','no'])
axs[0,0].set_title('object1')
# axs[0].savefig('plots/figure2_object1.jpg', dpi = 200)

# plt.figure(figsize = (8,4))
x = [-0.2, -0.1, -0.05, -0.04375,-0.0375,  -0.025, 0, 0.0125, 0.01875, 0.025, 0.05, 0.1, 0.2]
y = [1,     1,   1,     0,        0,         0,    0,    0,     0,      1,      1,  1,   1]
axs[0,1].plot(x,y,'.-')
# axs[0,1].set_ylabel('x basin')
# axs[0,1].set_ylabel('converge')
axs[0,1].set_yticks([0,1],['yes','no'])
axs[0,1].set_title('object8')
# axs[1].savefig('plots/figure2_object8.jpg', dpi = 200)

# plt.figure(figsize = (8,4))
x = [-0.2, -0.1, -0.05, -0.0375, -0.03125, -0.025, 0, 0.0125, 0.01875, 0.025, 0.05, 0.1, 0.2]
y = [1,     1,   1,      1,        1,        0,     0,  0,      0,      1,    1,      1,   1]
axs[0,2].plot(x,y,'.-')
# axs[0,2].set_ylabel('x basin')
# axs[0,2].set_ylabel('converge')
axs[0,2].set_yticks([0,1],['yes','no'])
axs[0,2].set_title('object9')
# plt.savefig('plots/figure2_object9.jpg', dpi = 200)


# output01/loss_weight_rgb_1.00_ssim_ms_0.20_perceptual_1.00/
# output01/loss_weight_rgb_1.00_ssim_ms_0.20_perceptual_1.00/ {'left': -0.15000000000000002, 'right': 0.08125000000000002}
# output09/loss_weight_rgb_1.00_ssim_ms_0.20_perceptual_1.00/
# output09/loss_weight_rgb_1.00_ssim_ms_0.20_perceptual_1.00/ {'left': -0.06875, 'right': 0.037500000000000006}
# output08/loss_weight_rgb_1.00_ssim_ms_0.20_perceptual_1.00/
# output08/loss_weight_rgb_1.00_ssim_ms_0.20_perceptual_1.00/ {'left': -0.09375, 'right': 0.018750000000000003}

# plt.figure(figsize = (8,4))
x = [-0.2,-0.175,-0.1625, -0.15, -0.1, 0, 0.05,  0.075, 0.08125, 0.0875,0.1, 0.2]
y = [1,     1,   1,       0,       0,  0, 0,      0,      0,      1,     1,  1 ]
axs[1,0].plot(x,y,'.-')
axs[1,0].set_ylabel('y basin')
axs[1,0].set_xlabel('converge')
axs[1,0].set_yticks([0,1],['yes','no'])
# axs[1,0].set_title('object1')
# plt.savefig('plots/figure2_object1_dim1.jpg', dpi = 200)

# plt.figure(figsize = (8,4))
x = [-0.2, -0.1, -0.075, -0.06875, -0.0625, -0.05, 0, 0.025, 0.0375,0.04375,0.05, 0.1, 0.2]
y = [1,     1,   1,     0,        0,         0,    0,    0,     0,      1,      1,  1,   1]
axs[1,1].plot(x,y,'.-')
# axs[1,1].set_ylabel('y basin')
axs[1,1].set_xlabel('converge')
axs[1,1].set_yticks([0,1],['yes','no'])
# axs[1,1].set_title('object8')
# plt.savefig('plots/figure2_object8_dim1.jpg', dpi = 200)

# plt.figure(figsize = (8,4))
x = [-0.2, -0.1, -0.093750,-0.0875,-0.075,-0.05, 0, 0.0125, 0.01875, 0.025, 0.05, 0.1, 0.2]
y = [1,     1,   0,        0,        0,      0,   0,  0,      0,      1,    1,      1,   1]
axs[1,2].plot(x,y,'.-')
# axs[1,2].set_ylabel('y basin')
axs[1,2].set_xlabel('converge')
axs[1,2].set_yticks([0,1],['yes','no'])
# axs[1,2].set_title('object9')
# plt.savefig('plots/figure2_object9_dim1.jpg', dpi = 200)

plt.savefig('plots/figure2_final.jpg', dpi =300)