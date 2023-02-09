# Differentiable mesh renderer by Pytorch3D
import argparse

import torch
from matplotlib import pyplot as plt
from pytorch3d.io import IO  # , load_obj, load_ply
from pytorch3d.renderer import (AmbientLights, BlendParams, MeshRasterizer,
                                MeshRenderer, PerspectiveCameras,
                                RasterizationSettings, SoftPhongShader)
from pytorch3d.structures import join_meshes_as_batch

from .data_loader import LineModDataset
from .utils.visual_utils import torchImageToPlottable


class DiffRender(object):
    """
    Pytorch3D diff mesh renderer
    """

    def __init__(self, models, device):
        """
        Load mesh model and camera parameters
        NOTE: if same obj models across image batch, pass in 1 model
        NOTE: if different obj models across batch, pass in B models
        @param models: [str-list] Path to camera models
        @param device: [torch.device] Device
        """
        # Load mesh models
        meshes = []
        for model in models:
            mesh = IO().load_mesh(model)
            meshes.append(mesh.scale_verts(0.001).to(device))
        self._meshes_ = join_meshes_as_batch(meshes)

    def camerasFromOpenCV(self, pose, K, img_size):
        """
        Build cameras from OpenCV convention camera matrices
        OpenCV camera convention: x-right-y-down-z-forward
        @param pose: [Bx4x4 tensor] object-to-camera poses
        @param K: [Bx3x3 tensor] camera intrinsics
        @param img_size: [Bx2 tensor] image sizes [[height, width]]
        @return cameras: Pytorch cameras
        """
        # OpenCV pose to Pytorch3D pose
        cv2torch3d = torch.diag(pose.new_tensor([-1, -1, 1, 1]))
        RT = torch.einsum("ik,...kl->...il", cv2torch3d, pose)
        R_torch3d = RT[..., :3, :3].transpose(-2, -1)
        t_torch3d = RT[..., :3, 3]
        # Focal length
        focal_len = K[..., [0, 1], [0, 1]]
        # Principle points
        principle_point = K[..., :2, 2]
        # Build Pytorch3D cameras
        cameras = PerspectiveCameras(
            R=R_torch3d, T=t_torch3d, focal_length=focal_len,
            principal_point=principle_point, image_size=img_size,
            device=pose.device, in_ndc=False
        )
        return cameras

    def render(self, pose, intrinsics, img_size, heuristic=False):
        """
        Render RGB images and masks
        @param pose: [Bx4x4 tensor] object-to-camera poses
        @param intrinsics: [Bx3x3 tensor] Camera intrinsics
        @param img_size: [Bx2] Image size [[height, width]]
        @param heuristic: [bool] Use heuristic to set bin_size and #faces/bin
        NOTE: Use heuristic will have the risk of incomplete render but faster
        @return rgb: [Bx3xHxW] Rendered RGB images
        @return mask: [Bx1xHxW] Rendered binary masks
        """
        assert intrinsics.shape[-2:] == (3, 3), \
            "Error: Intrinsic matrix must be Bx3x3"
        assert len(pose.shape) == 3 and pose.shape[-2:] == (4, 4),\
            "Error: Rotation matrices must be Bx4x4"
        assert len(img_size.shape) == 2 and img_size.shape[-1] == 2, \
            "Error: Image size vectors must have shape Bx2"

        device = self._meshes_.device
        # Create cameras from the rotations
        self._cams_ = self.camerasFromOpenCV(pose, intrinsics, img_size)
        # Rasterization setting
        raster_settings = RasterizationSettings(
            # NOTE: assume all images have same size
            image_size=(int(img_size[0, 0]), int(img_size[0, 1])),
            max_faces_per_bin=None if heuristic else 100000,
            bin_size=None if heuristic else 31
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
        # Get rendered RGB and mask images
        rgb, mask = images[..., :3], images[..., -1:] > 0.0
        rgb = rgb.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        # Uncomment to visualize the 1st rendered image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(torchImageToPlottable(rgb[0, ...]))
        # plt.show()
        # plt.imshow(torchImageToPlottable(mask[0, ...]))
        # plt.show()
        return rgb, mask


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    linemod_data = LineModDataset(args.image_dir, args.model_dir)
    rgb_gt, obj_pose_gt_all, obj_model_all, K_mat = linemod_data[0]
    rgb_gt = rgb_gt.to(device)
    obj_pose_gt = list(obj_pose_gt_all.values())[0].to(device)
    obj_model = list(obj_model_all.values())
    K_mat = K_mat.to(device)
    img_size = torch.tensor(rgb_gt.shape[-2:]).view(1, 2)

    diff_render = DiffRender(obj_model, device)
    rgb, mask = diff_render.render(obj_pose_gt, K_mat, img_size)
    # Uncomment to vis the overlayed images
    plt.imshow(torchImageToPlottable(rgb[0, ...]))
    plt.show()
    plt.imshow(torchImageToPlottable(mask[0, ...]))
    plt.show()
