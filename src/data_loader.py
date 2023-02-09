# Data loader
import argparse
import glob
import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from .utils.pose_utils import tRToPose4x4
from .utils.visual_utils import torchImageToPlottable


class LineModDataset(torch.utils.data.Dataset):
    """
    Dataloader for the LineMod dataset in BOP challenge
    """

    def __init__(self, data_pth, model_pth):
        """
        Load images and GT poses from the LineMod data folder
        @param data_pth: [str] LineMod data folder
        """
        assert os.path.isdir(data_pth), f"Error: {data_pth} not a folder!"
        assert os.path.isdir(f"{data_pth}/rgb"), \
            f"Error: {data_pth}/rgb not a folder"
        assert os.path.isdir(model_pth), f"Error: {model_pth} not a folder!"
        assert os.path.isfile(f"{data_pth}/scene_gt.json"), \
            f"Error: {data_pth}/scene_gt.json not a file"
        assert os.path.isfile(f"{data_pth}/scene_camera.json"), \
            f"Error: {data_pth}/scene_camera.json not a file"

        # Get image names
        self._img_names_ = sorted(
            glob.glob(f"{data_pth}/rgb/*.png") +
            glob.glob(f"{data_pth}/rgb/*.jpg")
        )
        # Get image indices
        self._index_ = [
            int(os.path.basename(f)[:-4]) for f in self._img_names_
        ]
        # Get object CAD model names and ids
        model_names = sorted(glob.glob(f"{model_pth}/*.ply"))
        self._model_names_ = {
            int(os.path.basename(m)[:-4][4:]): m for m in model_names
        }
        # Read GT object poses
        self._obj_pose_all_ = dict()
        with open(f"{data_pth}/scene_gt.json") as obj_json:
            obj_pose_data = json.load(obj_json)
        for img_id, img_poses in obj_pose_data.items():
            obj_pose_img = dict()
            for pose in img_poses:
                R = torch.tensor(pose["cam_R_m2c"]).view(1, 3, 3)
                t = torch.tensor(pose["cam_t_m2c"]).view(1, 3) / 1e3
                obj_id = int(pose["obj_id"])
                obj_pose_img[obj_id] = tRToPose4x4(t, R)
            self._obj_pose_all_[int(img_id)] = obj_pose_img

        # Read camera intrinsic params
        self._K_, self._depth_scale_ = dict(), dict()
        with open(f"{data_pth}/scene_camera.json") as cam_json:
            cam_data = json.load(cam_json)
        for img_id, cam_param in cam_data.items():
            K_mat = torch.tensor(cam_param["cam_K"]).view(1, 3, 3)
            self._K_[int(img_id)] = K_mat
            ds = cam_param["depth_scale"]
            self._depth_scale_[int(img_id)] = ds

    def __len__(self):
        return len(self._img_names_)

    def __getitem__(self, ind):
        """
        Get image, object poses, camera params, etc.
        @param ind: [int] Image index
        @return image: [1xCxHxW] Image
        @return obj_pose: [{obj_id:1x4x4}] GT object pose
        @return model_path: [{obj_id:str}] Object CAD model paths
        @return K: [1x3x3] Camera intrinsic matrix
        """
        # Load image
        img_name = self._img_names_[ind]
        image = np.array(Image.open(img_name).convert("RGB"))
        image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2) / 255
        # GT Object poses
        obj_pose = self._obj_pose_all_[self._index_[ind]]
        # Intrinsic matrix
        K_mat = self._K_[self._index_[ind]]
        # Object CAD model paths
        model_path = {
            obj_id: self._model_names_[obj_id] for obj_id in obj_pose.keys()
        }
        return image, obj_pose, model_path, K_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", "-i", type=str, help="Path to image folder",
        default="data/lm_images/000001/"
    )
    parser.add_argument(
        "--model_dir", "-m", type=str, help="Path to object CAD models",
        default="data/lm_models/"
    )
    args = parser.parse_args()

    linemod_data = LineModDataset(args.image_dir, args.model_dir)
    rgb_gt, obj_pose_gt_all, obj_model_all, K_mat = linemod_data[0]
    plt.imshow(torchImageToPlottable(rgb_gt[0, ...]))
    plt.show()
    print(f"GT pbject poses: \n{obj_pose_gt_all}")
    print(f"Object model paths: \n{obj_model_all}")
    print(f"Intrinsic matrix: \n{K_mat.squeeze().detach().cpu().numpy()}")
