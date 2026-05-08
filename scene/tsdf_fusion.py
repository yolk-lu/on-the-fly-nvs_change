from __future__ import annotations

import torch

from scene.adaptive_tsdf import AdaptiveTSDF
from utils import depth2points


class TSDFFusion:
    """Fuse depth observations into an anchor-local AdaptiveTSDF."""

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
        h, w = mono_idepth.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(0, h, self.stride, device=mono_idepth.device),
            torch.arange(0, w, self.stride, device=mono_idepth.device),
            indexing="ij",
        )
        uv = torch.stack([x, y], dim=-1).reshape(-1, 2).float()
        idepth = mono_idepth.reshape(-1)[(y * w + x).reshape(-1)]
        conf = confidence.reshape(-1)[(y * w + x).reshape(-1)]
        depth = 1.0 / idepth.clamp(1e-6, 1e6)
        valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(conf) & (conf > self.min_conf)
        if not valid.any():
            return
        points_cam = depth2points(uv[valid], depth[valid, None], f, centre)
        points_world = (R_cam_to_world @ points_cam.T).T + t_cam_to_world[None]
        points_anchor = (R_world_to_anchor @ points_world.T).T + t_world_to_anchor[None]
        sdf = torch.zeros(points_anchor.shape[0], device=points_anchor.device, dtype=points_anchor.dtype)
        tsdf.integrate_samples(points_anchor, sdf.clamp(-self.truncation, self.truncation), conf[valid])
