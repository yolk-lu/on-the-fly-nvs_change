from __future__ import annotations

import torch
import torch.nn.functional as F

from scene.adaptive_tsdf import AdaptiveTSDF
from scene.local_gaussian_model import quaternion_to_matrix


def tsdf_surface_loss(points_local: torch.Tensor, tsdf: AdaptiveTSDF, min_weight: float = 1.0) -> torch.Tensor:
    query = tsdf.query(points_local)
    mask = query.valid & (query.weight >= min_weight) & torch.isfinite(query.tsdf)
    if not mask.any():
        return points_local.sum() * 0.0
    penalty = 1.0 - torch.exp(-query.tsdf[mask].abs())
    return penalty.square().mean()


def tsdf_normal_alignment_loss(
    points_local: torch.Tensor,
    gaussian_rotation: torch.Tensor,
    tsdf_normals: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align Gaussian principal z axis with TSDF normals."""
    if points_local.shape[0] == 0:
        return points_local.sum() * 0.0
    R = quaternion_to_matrix(gaussian_rotation)
    gaussian_normals = F.normalize(R[..., :, 2], dim=-1)
    tsdf_normals = F.normalize(tsdf_normals, dim=-1)
    mask = torch.isfinite(gaussian_normals).all(dim=-1) & torch.isfinite(tsdf_normals).all(dim=-1)
    if valid_mask is not None:
        mask &= valid_mask
    if not mask.any():
        return points_local.sum() * 0.0
    alignment = (gaussian_normals[mask] * tsdf_normals[mask]).sum(dim=-1).abs()
    return (1.0 - alignment).square().mean()


def anisotropy_regularization(log_scaling: torch.Tensor, max_ratio: float = 8.0) -> torch.Tensor:
    if log_scaling.shape[0] == 0:
        return log_scaling.sum() * 0.0
    scales = torch.exp(log_scaling.clamp(-20.0, 20.0))
    ratio = scales.max(dim=-1).values / scales.min(dim=-1).values.clamp_min(1e-8)
    excess = (ratio / float(max_ratio)).clamp_min(1.0).log()
    return excess.square().mean()
