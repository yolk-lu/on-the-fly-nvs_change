from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ScaleAlignmentResult:
    scale_map: torch.Tensor
    valid_mask: torch.Tensor
    global_scale: torch.Tensor
    num_valid_cells: int


class GridScaleAligner:
    """Grid-based alignment for relative monocular inverse-depth scale."""

    def __init__(self, grid_size: int = 16, min_conf: float = 0.35, min_samples_per_cell: int = 8):
        self.grid_size = int(grid_size)
        self.min_conf = float(min_conf)
        self.min_samples_per_cell = int(min_samples_per_cell)

    @torch.no_grad()
    def estimate(
        self,
        reference_idepth: torch.Tensor,
        new_idepth: torch.Tensor,
        reference_conf: torch.Tensor,
        new_conf: torch.Tensor,
    ) -> ScaleAlignmentResult:
        ref = self._as_bchw(reference_idepth).float()
        new = self._as_bchw(new_idepth).float().to(ref.device)
        ref_conf = self._as_bchw(reference_conf).float().to(ref.device)
        new_conf = self._as_bchw(new_conf).float().to(ref.device)
        if new.shape[-2:] != ref.shape[-2:]:
            new = F.interpolate(new, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            new_conf = F.interpolate(new_conf, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        valid = (
            torch.isfinite(ref)
            & torch.isfinite(new)
            & (ref > 1e-6)
            & (new > 1e-6)
            & torch.isfinite(ref_conf)
            & torch.isfinite(new_conf)
            & (ref_conf > self.min_conf)
            & (new_conf > self.min_conf)
        )
        ratio = (ref / new.clamp_min(1e-6)).clamp(1e-3, 1e3)
        weighted_ratio = torch.where(valid, ratio, torch.zeros_like(ratio))
        kernel = min(self.grid_size, int(ref.shape[-2]), int(ref.shape[-1]))
        ratio_sum = F.avg_pool2d(weighted_ratio, kernel, stride=kernel) * (kernel**2)
        count = F.avg_pool2d(valid.float(), kernel, stride=kernel) * (kernel**2)
        valid_cells = count >= self.min_samples_per_cell
        scale_map = torch.where(valid_cells, ratio_sum / count.clamp_min(1.0), torch.ones_like(ratio_sum))
        global_scale = scale_map[valid_cells].median() if valid_cells.any() else torch.ones((), dtype=ref.dtype, device=ref.device)
        return ScaleAlignmentResult(
            scale_map=scale_map[0, 0].contiguous(),
            valid_mask=valid_cells[0, 0].contiguous(),
            global_scale=global_scale,
            num_valid_cells=int(valid_cells.sum().item()),
        )

    @staticmethod
    def _as_bchw(tensor: torch.Tensor) -> torch.Tensor:
        while tensor.dim() > 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 2:
            return tensor[None, None]
        if tensor.dim() == 3:
            return tensor[None]
        if tensor.dim() == 4:
            return tensor
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {tuple(tensor.shape)}")
