from __future__ import annotations

from dataclasses import dataclass

import torch

from utils import inverse_sigmoid


@dataclass
class OpacityResetResult:
    reset_mask: torch.Tensor
    num_reset: int


class ViewDiversityOpacityReset:
    """Localized opacity reset for visible, poorly constrained Gaussians."""

    def __init__(self, min_views: int = 2, grad_quantile: float = 0.95, reset_opacity: float = 0.01):
        self.min_views = int(min_views)
        self.grad_quantile = float(grad_quantile)
        self.reset_opacity = float(reset_opacity)

    @torch.no_grad()
    def apply(
        self,
        gaussian_params: dict[str, dict[str, torch.Tensor]],
        visible_mask: torch.Tensor,
        view_counts: torch.Tensor,
        grad_norm: torch.Tensor,
        tsdf_valid_mask: torch.Tensor | None = None,
    ) -> OpacityResetResult:
        opacity = gaussian_params["opacity"]["val"]
        if opacity.shape[0] == 0:
            empty = torch.empty(0, dtype=torch.bool, device=opacity.device)
            return OpacityResetResult(empty, 0)
        visible_mask = visible_mask.to(opacity.device).bool()
        view_counts = view_counts.to(opacity.device)
        grad_norm = grad_norm.to(opacity.device)
        stable_tsdf = torch.zeros_like(visible_mask) if tsdf_valid_mask is None else tsdf_valid_mask.to(opacity.device).bool()
        finite = torch.isfinite(opacity.flatten(1)).all(dim=1) & torch.isfinite(grad_norm)
        if finite.any():
            threshold = torch.quantile(grad_norm[finite], self.grad_quantile)
        else:
            threshold = torch.tensor(float("inf"), device=opacity.device)
        reset_mask = visible_mask & finite & (view_counts < self.min_views) & (grad_norm >= threshold) & (~stable_tsdf)
        if reset_mask.any():
            reset = torch.full_like(opacity[reset_mask], self.reset_opacity).clamp(1e-6, 1.0 - 1e-6)
            opacity[reset_mask] = inverse_sigmoid(reset)
        return OpacityResetResult(reset_mask=reset_mask, num_reset=int(reset_mask.sum().item()))
