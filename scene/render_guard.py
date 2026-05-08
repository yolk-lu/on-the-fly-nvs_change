from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RenderGuardResult:
    params: dict[str, torch.Tensor]
    mask: torch.Tensor
    reason_counts: dict[str, int]


class RenderGuard:
    """Pre-rasterizer validation for Gaussian tensors."""

    def __init__(self, f: torch.Tensor | float, max_screen_px: float = 200.0, max_depth: float = 1e4):
        self.f = f
        self.max_screen_px = float(max_screen_px)
        self.max_depth = float(max_depth)

    @torch.no_grad()
    def filter(self, params: dict[str, torch.Tensor], cam_centre: torch.Tensor) -> RenderGuardResult:
        xyz = params["xyz"]
        n = xyz.shape[0]
        if n == 0:
            return RenderGuardResult(params={k: v[:0] for k, v in params.items()}, mask=torch.zeros(0, dtype=torch.bool, device=xyz.device), reason_counts={})
        finite = torch.ones(n, dtype=torch.bool, device=xyz.device)
        for tensor in params.values():
            finite &= torch.isfinite(tensor.flatten(1)).all(dim=1)
        dist = torch.linalg.vector_norm(xyz - cam_centre[None], dim=-1)
        scaling = torch.exp(params["scaling"].clamp(-20.0, 20.0))
        screen = self.f * scaling.max(dim=-1).values / dist.clamp_min(1e-6)
        valid = finite & torch.isfinite(dist) & (dist > 1e-6) & (dist < self.max_depth)
        valid &= torch.isfinite(screen) & (screen > 0) & (screen < self.max_screen_px)
        reason_counts = {
            "nonfinite": int((~finite).sum().item()),
            "bad_distance": int((~(torch.isfinite(dist) & (dist > 1e-6) & (dist < self.max_depth))).sum().item()),
            "bad_screen": int((~(torch.isfinite(screen) & (screen > 0) & (screen < self.max_screen_px))).sum().item()),
        }
        return RenderGuardResult(params={k: v[valid].contiguous() for k, v in params.items()}, mask=valid, reason_counts=reason_counts)
