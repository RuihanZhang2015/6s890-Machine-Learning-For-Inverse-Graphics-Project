# Pose related utility functions
import numpy as np
import pypose as pp
import torch


def tRToPose4x4(t, R):
    """
    Convert translation and rotation matrices to pose matrices
    @param t: [...x3] Translation vectors
    @param R: [...x3x3] Rotation matrix
    @return pose: [...x4x4] Pose matrices
    """
    assert R.shape[-2:] == (3, 3),  "Error: Rotation matrices must be ...x3x3"
    assert t.shape[-1] == 3, "Error: Translation vectors must be Bx3"
    assert R.shape[:-2] == t.shape[:-1], "Error: R & t batch size mismatch"
    batch_size = R.shape[:-2]
    Rt = torch.cat((R, t.view(*batch_size, 3, 1)), dim=-1)
    OI = Rt.new_zeros(*batch_size, 1, 4)
    pose = torch.cat((Rt, OI), dim=-2)
    pose[..., -1, -1] = 1
    return pose


def poseError(pose1: pp.LieTensor, pose2: pp.LieTensor):
    """
    Compute the translation and rotation erros btw two poses
    @param pose1: Pose 1
    @param pose2: Pose 2
    @return trans_err: [cm] translation error
    @return rot_err: [deg] rotation error
    """
    assert len(pose1) == 1 and len(pose2) == 1, \
        "Error: only support 1-1 pose compare"
    # NOTE: pypose has a weird bug if there's a 0 pose element
    # We fix the problem by multiplying a close to 0 pose
    pose_rel = pose1.Inv() * pose2 * pp.se3(pose1.new_ones(1, 6)*1e-20).Exp()
    trans_rel = pose_rel.translation()
    trans_err = trans_rel.norm().item() * 100
    rot_rel = pose_rel.rotation()
    rot_err = rot_rel.Log().tensor().norm().item() / np.pi * 180
    return trans_err, rot_err


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


def genSE3ManualNoise(noise: torch.Tensor = torch.ones(1, 6)) -> pp.LieTensor:
    """
    Manually generate a deterministic perturbation pose
    @param noise: [...x6 Tensor] noise list defined in the R^3 x so3 space
    @return SE3noise: Perturbation SE3 pose
    """
    assert noise.shape[-1] == 6, "Error: noise must have shape ...x6"
    r3xso3 = torch.cat(
        (noise[..., :3], pp.so3(noise[..., 3:]).Exp().tensor()), dim=-1
    )
    return pp.SE3(r3xso3)
