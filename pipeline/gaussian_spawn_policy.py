from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pipeline.frame_state import FrameState
from utils import RGB2SH, depth2points, get_lapla_norm, make_torch_sampler


@dataclass
class GaussianSpawnResult:
    uv: torch.Tensor
    depth: torch.Tensor
    xyz_cam: torch.Tensor
    f_dc: torch.Tensor
    init_probability: torch.Tensor
    source: torch.Tensor


class GaussianSpawnPolicy:
    """LoG direct sampling with DepthAnything confidence and optional render penalty."""

    SOURCE_LOG_SAMPLE = 0
    SOURCE_MATCHED_POINT = 1

    def __init__(
        self,
        width: int,
        height: int,
        f: torch.Tensor | float,
        centre: torch.Tensor,
        init_proba_scaler: float = 2.0,
        min_depth_conf: float = 0.35,
    ):
        self.width = int(width)
        self.height = int(height)
        self.f = f
        self.centre = centre
        self.init_proba_scaler = float(init_proba_scaler)
        self.min_depth_conf = float(min_depth_conf)
        self.disc_kernel = torch.ones(1, 1, 5, 5, device=centre.device) / 25.0
        y, x = torch.meshgrid(
            torch.arange(self.height, device=centre.device),
            torch.arange(self.width, device=centre.device),
            indexing="ij",
        )
        self.uv_grid = torch.stack([x, y], dim=-1).float()

    @torch.no_grad()
    def sample(
        self,
        frame: FrameState,
        rendered_image: torch.Tensor | None = None,
        rendered_invdepth: torch.Tensor | None = None,
        random_map: torch.Tensor | None = None,
    ) -> GaussianSpawnResult:
        image = frame.image
        init_proba = get_lapla_norm(image, self.disc_kernel) * self.init_proba_scaler
        penalty = 0.0
        if rendered_image is not None:
            penalty = get_lapla_norm(rendered_image.detach(), self.disc_kernel) * self.init_proba_scaler
        sample_proba = (init_proba - penalty).clamp(0, 1)
        if frame.mask is not None:
            mask = frame.mask
            if mask.ndim == 3:
                mask = mask[0]
            sample_proba = sample_proba * mask.to(sample_proba.device).bool()

        rand = torch.rand_like(sample_proba) if random_map is None else random_map.to(sample_proba.device)
        sample_mask = rand < sample_proba
        uv = self.uv_grid[sample_mask]
        if uv.numel() == 0:
            return self._empty(image.device, image.dtype)

        sampler = make_torch_sampler(uv, self.width, self.height)
        mono_idepth = F.grid_sample(frame.mono_idepth, sampler[None, None], mode="bilinear", align_corners=True)[0, 0, 0]
        mono_conf = F.grid_sample(frame.mono_depth_conf, sampler[None, None], mode="bilinear", align_corners=True)[0, 0, 0]
        depth = 1.0 / mono_idepth.clamp(1e-6, 1e6)
        valid = torch.isfinite(depth) & (depth > 1e-6) & torch.isfinite(mono_conf) & (mono_conf > self.min_depth_conf)
        if rendered_invdepth is not None:
            rendered_depth = 1.0 / rendered_invdepth[0, sample_mask].clamp_min(1e-8)
            valid &= depth < rendered_depth

        uv = uv[valid]
        depth = depth[valid]
        if uv.numel() == 0:
            return self._empty(image.device, image.dtype)

        xyz_cam = depth2points(uv, depth[:, None], self.f, self.centre)
        colors = image[:, sample_mask][:, valid].T.contiguous()
        f_dc = RGB2SH(colors[:, None, :])
        source = torch.full((uv.shape[0],), self.SOURCE_LOG_SAMPLE, dtype=torch.long, device=uv.device)
        return GaussianSpawnResult(uv, depth, xyz_cam, f_dc, sample_proba[sample_mask][valid], source)

    def _empty(self, device: torch.device, dtype: torch.dtype) -> GaussianSpawnResult:
        return GaussianSpawnResult(
            uv=torch.empty(0, 2, device=device, dtype=torch.float32),
            depth=torch.empty(0, device=device, dtype=dtype),
            xyz_cam=torch.empty(0, 3, device=device, dtype=dtype),
            f_dc=torch.empty(0, 1, 3, device=device, dtype=dtype),
            init_probability=torch.empty(0, device=device, dtype=dtype),
            source=torch.empty(0, device=device, dtype=torch.long),
        )
