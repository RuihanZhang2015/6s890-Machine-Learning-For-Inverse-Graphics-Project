# Code borrowed from self6dpp (Wang et al. 2021)
# https://github.com/THU-DA-6D-Pose-Group/self6dpp
import torch


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    Args:
        image (torch.Tensor):
        RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a batch of RGB images to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image (torch.Tensor):
        RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    # Convert from Linear RGB to sRGB
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    rs: torch.Tensor = torch.where(r > 0.04045, torch.pow(
        ((r + 0.055) / 1.055), 2.4), r / 12.92)
    gs: torch.Tensor = torch.where(g > 0.04045, torch.pow(
        ((g + 0.055) / 1.055), 2.4), g / 12.92)
    bs: torch.Tensor = torch.where(b > 0.04045, torch.pow(
        ((b + 0.055) / 1.055), 2.4), b / 12.92)

    image_s = torch.stack([rs, gs, bs], dim=-3)

    xyz_im: torch.Tensor = rgb_to_xyz(image_s)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype
    )[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    power = torch.pow(xyz_normalized, 1 / 3)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > 0.008856, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


def normalize_lab(lab: torch.Tensor) -> torch.Tensor:
    """
    Args:
        lab: NCHW, normalize to [0,1]
    """
    assert len(lab.shape) >= 3 and lab.shape[-3] == 3, \
        "Error: Image size must be (*, 3, H, W)"
    # L_chan: black and white with input range [0, 100]
    # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
    # [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
    _min = torch.tensor([0.0, -110.0, -110.0]).view(3, 1, 1).to(lab)
    _max = torch.tensor([100.0, 110.0, 110.0]).view(3, 1, 1).to(lab)
    lab_normed = (lab - _min) / (_max - _min)
    return lab_normed
