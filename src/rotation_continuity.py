# Code adapted from RotationContinuity (Zhao et al. 2019)
# https://github.com/papagina/RotationContinuity
import torch


def ortho6DToRot(ortho_6d):
    """
    Convert 6D ortho rotation repr to rotation matrices
    """
    assert ortho_6d.shape[-1] == 6
    x_raw = ortho_6d[..., 0:3]
    y_raw = ortho_6d[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack((x, y, z), -1)
    return matrix


def ortho9DToTransform(ortho_9d):
    """
    Convert 9D ortho rotation repr to rotation matrices
    """
    assert ortho_9d.shape[-1] == 9
    R = ortho6DToRot(ortho_9d[..., :6])
    trans = ortho_9d[..., 6:]
    T = ortho_9d.new_zeros(*ortho_9d.shape[:-1], 4, 4)
    T[..., 0:3, 0:3] = R
    T[..., 0:3, 3] = trans
    T[..., 3, 3] = 1
    return T


def rotationToOrtho6D(rotation):
    """
    From "On The Continuity of Rotation Representations in Neural Networks"
    """
    assert rotation.shape[-1] == 3 and rotation.shape[-1] == 3,\
        "Error: rotations must have shape 3x3"
    x_raw = rotation[..., :3, 0]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    y_raw = rotation[..., :3, 1]
    y = y_raw / torch.norm(y_raw, p=2, dim=-1, keepdim=True)
    return torch.cat((x, y), dim=-1)


def transformToOrtho9D(transform):
    """
    From transformation (...x3x4 or ...x4x4) to 9D representation
    @return Ortho9D: [...x9] 9D pose repr
    """
    assert transform.shape[-1] == 4 and transform.shape[-2] in [3, 4],\
        "Error: poses must have shape 3x4 or 4x4"
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    ortho_6d = rotationToOrtho6D(R)
    return torch.cat((ortho_6d, t), dim=-1)
