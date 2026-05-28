from __future__ import annotations

import torch

from scene.adaptive_tsdf import AdaptiveTSDF


class TSDFFusion:
    """No-op compatibility interface for the removed TSDF fusion backend."""

    def __init__(self, truncation: float = 0.5, stride: int = 8, min_conf: float = 0.35):
        self.truncation = float(truncation)
        self.stride = int(stride)
        self.min_conf = float(min_conf)

    @torch.no_grad()
    def integrate_depth(
        self,
        tsdf: AdaptiveTSDF,
        mono_idepth: torch.Tensor,
        confidence: torch.Tensor,
        R_world_to_anchor: torch.Tensor,
        t_world_to_anchor: torch.Tensor,
        R_cam_to_world: torch.Tensor,
        t_cam_to_world: torch.Tensor,
        f: torch.Tensor | float,
        centre: torch.Tensor,
    ) -> None:
        return None

    @torch.no_grad()
    def integrate_optimized_gaussians(
        self,
        tsdf: AdaptiveTSDF,
        gaussian_model,
        anchor=None,
        min_opacity: float = 0.05,
        max_samples: int = 250_000,
    ) -> dict:
        candidates = 0 if gaussian_model is None else int(getattr(gaussian_model, "n", 0))
        before = int(tsdf.keys.shape[0])
        stats = {
            "mode": "disabled",
            "reason": "tsdf_backend_removed",
            "candidates": candidates,
            "integrated": 0,
            "skipped_low_opacity": 0,
            "skipped_nonfinite": 0,
            "skipped_bad_scale": 0,
            "voxel_count_before": before,
            "voxel_count_after": before,
            "weight_mean": 0.0,
            "local_bbox_min": [],
            "local_bbox_max": [],
        }
        if anchor is not None:
            stats["anchor_id"] = int(getattr(anchor, "anchor_id", -1))
        return stats
