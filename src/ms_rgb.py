import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
# from utils.utils import bilinear_sampler, coords_grid

class MS_RGB:

    def __init__(self, num_levels=6, radius=4):
        
        self.num_levels = num_levels
        self.radius = radius
    
    def color_loss_fun(self,img1,img2,mask):

        loss_fun = torch.nn.SmoothL1Loss(beta=0, reduction="none")
        temp = loss_fun(img1,img2)
        return torch.einsum("bcij,bdij->b",temp,mask)/torch.sum(mask,axis = [1,2,3])

    def __call__(self,img1, img2, mask):
        
        color_loss = self.color_loss_fun(img1,img2,mask)
        for i in range(self.num_levels-1):
            padding = (img1.shape[2] % 2, img1.shape[3] % 2)
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=padding)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=padding)
            mask = F.avg_pool2d(mask, kernel_size=2, stride=2, padding=padding)
            mask[mask>0.5] = 1
            # fig, axs = plt.subplots(1,3,figsize = (10,8))
            # axs[0].imshow(img1.permute(0,2,3,1)[0,...].detach().cpu())
            # axs[1].imshow(img2.permute(0,2,3,1)[0,...].detach().cpu())
            # axs[2].imshow(mask[0,0,...].detach().cpu())
            # plt.savefig('{}.jpg'.format(time.time()))
            # print(self.color_loss_fun(img1,img2,mask))
            color_loss += self.color_loss_fun(img1,img2,mask)
        return color_loss/self.num_levels

class MS_RGB_LAB:

    def __init__(self, num_levels=6, radius=4):
        
        self.num_levels = num_levels
        self.radius = radius
    
    def color_loss_fun(self,img1,img2,mask):

        loss_fun = torch.nn.SmoothL1Loss(beta=0, reduction="none")
        temp = loss_fun(img1,img2)
        return torch.einsum("bcij,bdij->b",temp,mask)/torch.sum(mask,axis = [1,2,3])

    def __call__(self,img1, img2, mask):
        
        color_loss = self.color_loss_fun(img1,img2,mask)
        for i in range(self.num_levels-1):
            padding = (img1.shape[2] % 2, img1.shape[3] % 2)
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=padding)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=padding)
            mask = F.avg_pool2d(mask, kernel_size=2, stride=2, padding=padding)
            mask[mask>0.5] = 1
            color_loss += self.color_loss_fun(img1,img2,mask)
        return color_loss/self.num_levels