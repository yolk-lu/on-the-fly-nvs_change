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
    stats: dict


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
        target_samples: int = 8192,
        surface_sample_floor: float = 0.01,
        low_frequency_fraction: float = 0.15,
        edge_probability_threshold: float = 0.02,
        min_sample_probability: float = 1e-5,
    ):
        self.width = int(width)
        self.height = int(height)
        self.f = f
        self.centre = centre
        self.init_proba_scaler = float(init_proba_scaler)
        self.min_depth_conf = float(min_depth_conf)
        self.target_samples = int(target_samples)
        self.surface_sample_floor = float(surface_sample_floor)
        self.low_frequency_fraction = float(low_frequency_fraction)
        self.edge_probability_threshold = float(edge_probability_threshold)
        self.min_sample_probability = float(min_sample_probability)
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
        # penalty = 0.0

        if rendered_image is not None:
            penalty = get_lapla_norm(rendered_image.detach(), self.disc_kernel) * self.init_proba_scaler
            beta = 2.0
            edge_proba = (init_proba * torch.exp(-beta * penalty))
        else:
            edge_proba = init_proba


        edge_proba = edge_proba.clamp(0, 1)
        sample_proba = edge_proba

        conf_map = self._confidence_map(frame)
        if self.surface_sample_floor > 0:
            low_freq_floor = self.surface_sample_floor * conf_map.clamp(0, 1)
            sample_proba = torch.where(
                edge_proba >= self.edge_probability_threshold,
                edge_proba,
                torch.maximum(edge_proba, low_freq_floor),
            )
        if frame.mask is not None:
            mask = frame.mask
            if mask.ndim == 3:
                mask = mask[0]
            sample_proba = sample_proba * mask.to(sample_proba.device).bool()

        rand = torch.rand_like(sample_proba) if random_map is None else random_map.to(sample_proba.device)
        sample_mask = rand < sample_proba
        sample_mask = self._enforce_sample_quota(sample_mask, sample_proba, edge_proba)
        uv = self.uv_grid[sample_mask]
        stats = {
            "edge_probability_mean": float(edge_proba.mean().detach().item()) if edge_proba.numel() else 0.0,
            "edge_probability_max": float(edge_proba.max().detach().item()) if edge_proba.numel() else 0.0,
            "probability_mean": float(sample_proba.mean().detach().item()) if sample_proba.numel() else 0.0,
            "probability_max": float(sample_proba.max().detach().item()) if sample_proba.numel() else 0.0,
            "sampled_before_depth_gate": int(sample_mask.sum().item()),
            "target_samples": int(self.target_samples),
            "surface_sample_floor": float(self.surface_sample_floor),
            "low_frequency_fraction": float(self.low_frequency_fraction),
            "edge_probability_threshold": float(self.edge_probability_threshold),
        }
        if uv.numel() == 0:
            return self._empty(image.device, image.dtype, stats)

        sampler = make_torch_sampler(uv, self.width, self.height)
        mono_idepth = F.grid_sample(frame.mono_idepth, sampler[None, None], mode="bilinear", align_corners=True)[0, 0, 0]
        mono_conf = F.grid_sample(frame.mono_depth_conf, sampler[None, None], mode="bilinear", align_corners=True)[0, 0, 0]
        depth = 1.0 / mono_idepth.clamp(1e-6, 1e6)
        valid = torch.isfinite(depth) & (depth > 1e-6) & (depth < 150.0) & torch.isfinite(mono_conf) & (mono_conf > self.min_depth_conf)
        if rendered_invdepth is not None:
            rendered_depth = 1.0 / rendered_invdepth[0, sample_mask].clamp_min(1e-8)
            valid &= depth < rendered_depth
        stats["depth_gate_valid"] = int(valid.sum().item())
        stats["depth_gate_total"] = int(valid.numel())
        stats["depth_gate_ratio"] = float(valid.float().mean().item()) if valid.numel() else 0.0

        uv = uv[valid]
        depth = depth[valid]
        if uv.numel() == 0:
            return self._empty(image.device, image.dtype, stats)

        xyz_cam = depth2points(uv, depth[:, None], self.f, self.centre)
        colors = image[:, sample_mask][:, valid].T.contiguous()
        f_dc = RGB2SH(colors[:, None, :])
        source = torch.full((uv.shape[0],), self.SOURCE_LOG_SAMPLE, dtype=torch.long, device=uv.device)
        stats["spawned"] = int(uv.shape[0])
        stats["depth_min"] = float(depth.min().detach().item())
        stats["depth_median"] = float(depth.median().detach().item())
        stats["depth_max"] = float(depth.max().detach().item())
        return GaussianSpawnResult(uv, depth, xyz_cam, f_dc, sample_proba[sample_mask][valid], source, stats)

    def _confidence_map(self, frame: FrameState) -> torch.Tensor:
        conf = frame.mono_depth_conf
        if conf.ndim == 4:
            conf = conf[0, 0]
        elif conf.ndim == 3:
            conf = conf[0]
        if conf.shape[-2:] != (self.height, self.width):
            conf = F.interpolate(conf[None, None].float(), (self.height, self.width), mode="bilinear", align_corners=True)[0, 0]
        return conf.to(frame.image.device).float()

    def _enforce_sample_quota(
        self,
        sample_mask: torch.Tensor,
        sample_proba: torch.Tensor,
        edge_proba: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_samples <= 0 or sample_mask.sum().item() >= self.target_samples:
            return sample_mask
        flat_scores = sample_proba.flatten()
        flat_edge = edge_proba.flatten()
        eligible = flat_scores >= self.min_sample_probability
        high_freq = eligible & (flat_edge >= self.edge_probability_threshold)
        low_freq = eligible & ~high_freq
        if int(eligible.sum().item()) == 0:
            return sample_mask
        flat_mask = sample_mask.flatten().clone()
        needed = max(0, int(self.target_samples) - int(flat_mask.sum().item()))

        high_needed = needed
        high_available = int((high_freq & ~flat_mask).sum().item())
        if high_available > 0 and high_needed > 0:
            k_high = min(high_needed, high_available)
            high_scores = flat_scores.masked_fill(~(high_freq & ~flat_mask), -1.0)
            high_idx = torch.topk(high_scores, k=k_high, largest=True).indices
            flat_mask[high_idx] = True
            needed -= k_high

        low_cap = int(round(max(0.0, min(1.0, self.low_frequency_fraction)) * float(self.target_samples)))
        current_low = int((flat_mask & low_freq).sum().item())
        low_needed = min(max(0, low_cap - current_low), needed)
        low_available = int((low_freq & ~flat_mask).sum().item())
        if low_available > 0 and low_needed > 0:
            k_low = min(low_needed, low_available)
            low_scores = flat_scores.masked_fill(~(low_freq & ~flat_mask), -1.0)
            low_idx = torch.topk(low_scores, k=k_low, largest=True).indices
            flat_mask[low_idx] = True
        return flat_mask.view_as(sample_mask)

    def _empty(self, device: torch.device, dtype: torch.dtype, stats: dict | None = None) -> GaussianSpawnResult:
        return GaussianSpawnResult(
            uv=torch.empty(0, 2, device=device, dtype=torch.float32),
            depth=torch.empty(0, device=device, dtype=dtype),
            xyz_cam=torch.empty(0, 3, device=device, dtype=dtype),
            f_dc=torch.empty(0, 1, 3, device=device, dtype=dtype),
            init_probability=torch.empty(0, device=device, dtype=dtype),
            source=torch.empty(0, device=device, dtype=torch.long),
            stats={} if stats is None else stats,
        )
