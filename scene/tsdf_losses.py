from __future__ import annotations

import torch

from scene.adaptive_tsdf import AdaptiveTSDF


def tsdf_surface_loss(points_local: torch.Tensor, tsdf: AdaptiveTSDF, min_weight: float = 1.0) -> torch.Tensor:
    return points_local.sum() * 0.0


def tsdf_normal_alignment_loss(
    points_local: torch.Tensor,
    gaussian_rotation: torch.Tensor,
    tsdf_normals: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return points_local.sum() * 0.0


def anisotropy_regularization(log_scaling: torch.Tensor, max_ratio: float = 8.0) -> torch.Tensor:
    if log_scaling.shape[0] == 0:
        return log_scaling.sum() * 0.0
    scales = torch.exp(log_scaling.clamp(-20.0, 20.0))
    ratio = scales.max(dim=-1).values / scales.min(dim=-1).values.clamp_min(1e-8)
    excess = (ratio / float(max_ratio)).clamp_min(1.0).log()
    return excess.square().mean()
