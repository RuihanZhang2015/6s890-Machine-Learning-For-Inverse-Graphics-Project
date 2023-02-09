# Image related utility functions
import torch
from torchvision.ops import RoIAlign, RoIPool


def maskToBbox(masks):
    """
    Convert a masked images to object bboxes
    @param masks: [BxHxW or Bx1xHxW] Masks for the target objects
    @return bbox: [Bx4] Object bboxes [[xmin, ymin, xmax, ymax]];
    if no mask in the image, return [-1]*4
    """
    assert len(masks.shape) == 3 or masks.shape[1] == 1, \
        "Error: mask must have shape (B, H, W) or (B, 1, H, W)"
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    # Initialize the bboxes as all -1
    n = masks.shape[0]
    bboxes = torch.ones((n, 4), device=masks.device, dtype=torch.float) * -1.0

    for index, mask in enumerate(masks):
        y, x = torch.where(mask.squeeze())
        if len(y) == 0:  # No mask in image
            continue
        bboxes[index, 0] = torch.min(x)
        bboxes[index, 1] = torch.min(y)
        bboxes[index, 2] = torch.max(x)
        bboxes[index, 3] = torch.max(y)

    return bboxes


def bboxToSquare(bboxes, scale=1.0):
    """
    Convert bboxes to square bboxes and apply a scale factor
    @param bbox: [Bx4] Object bboxes
    @param scale: [float] Enlarge the bbox
    @return bbox_sq: [Bx4] Rescaled square box
    """
    assert bboxes.shape[-1] == 4, "Error: Bboxes must be a Bx4 tensor"
    bbox_cx = (bboxes[..., 0:1] + bboxes[..., 2:3])/2.0
    bbox_cy = (bboxes[..., 1:2] + bboxes[..., 3:4])/2.0
    bbox_h = -bboxes[..., 1:2] + bboxes[..., 3:4]
    bbox_w = -bboxes[..., 0:1] + bboxes[..., 2:3]
    bbox_center = torch.cat((bbox_cx, bbox_cy), dim=-1)
    bbox_size = torch.maximum(bbox_w, bbox_h) * scale
    bbox_sq_xymin = bbox_center - bbox_size / 2.0
    bbox_sq_xymax = bbox_center + bbox_size / 2.0
    bbox_sq = torch.cat((bbox_sq_xymin, bbox_sq_xymax), dim=-1)
    return bbox_sq


def batch_crop_resize(
        img, rois, out_H, out_W, aligned=True, interpolation="bilinear"):
    """
    Crop and resize images
    @param img: [BxCxHxW] Batch of images
    @param rois: [Bx4 or 5] Region of Interest [[(idx), x1, y1, x2, y1], ...]
    @param out_H: [int] Output size height
    @param out_W: [int] Output size width
    @return cropped: [BxCxHxW] Cropped image batch
    """
    assert len(img.shape) >= 3 and img.shape[-3] == 3, \
        "Error: Image size must be (*, 3, H, W)"
    assert rois.shape[-1] in [4, 5], "Error: Bboxes should be Bx4 or Bx5"
    # NOTE: If img idx not assigned, it's going to be arange(#images)
    if rois.shape[-1] == 4:
        roi_idx = torch.arange(rois.size(0)).view(-1, 1).to(rois)
        rois = torch.cat((roi_idx, rois), dim=-1)
    # Crop and resize
    output_size = (out_H, out_W)
    if interpolation == "bilinear":
        op = RoIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")
    return op(img, rois)


def get_K_crop_resize(K, crop_xy, resize_ratio):
    """
    Update intrinsics matrix after crop and resize
    @param K: [Bx3x3] Original intrinsics
    @param crop_xy: [Bx2]  left top of crop boxes
    @param resize_ratio: [Bx2 or Bx1] ratio in H (and W)
    @return new_K: [Bx3x3] Updated intrinsic matrix
    """
    assert K.shape[1:] == (3, 3), "Error: Intrinsic matrix must be Bx3x3"
    assert crop_xy.shape[1:] == (2,), "Error: Crop box location must be Bx2"
    assert resize_ratio.shape[-1] in [1, 2], \
        "Error: Resize ratio must be Bx1 or Bx2"

    bs = K.shape[0]

    new_K = K.clone()
    new_K[:, [0, 1], 2] = K[:, [0, 1], 2] - crop_xy  # [b, 2]
    new_K[:, [0, 1]] = new_K[:, [0, 1]] * resize_ratio.view(bs, -1, 1)
    return new_K
