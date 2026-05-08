from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TSDFQuery:
    tsdf: torch.Tensor
    weight: torch.Tensor
    valid: torch.Tensor
    level: torch.Tensor


class AdaptiveTSDF:
    """Sparse TSDF samples with variance-driven voxel levels."""

    def __init__(
        self,
        base_voxel_size: float = 0.25,
        num_levels: int = 4,
        fine_var_quantile: float = 0.75,
        coarse_var_quantile: float = 0.25,
        device: str | torch.device = "cuda",
    ):
        self.base_voxel_size = float(base_voxel_size)
        self.num_levels = int(num_levels)
        self.fine_var_quantile = float(fine_var_quantile)
        self.coarse_var_quantile = float(coarse_var_quantile)
        self.device = torch.device(device)
        self.keys = torch.empty(0, 3, dtype=torch.long, device=self.device)
        self.tsdf_mean = torch.empty(0, dtype=torch.float32, device=self.device)
        self.m2 = torch.empty(0, dtype=torch.float32, device=self.device)
        self.weight = torch.empty(0, dtype=torch.float32, device=self.device)
        self.level = torch.empty(0, dtype=torch.long, device=self.device)

    def to(self, device: str | torch.device) -> "AdaptiveTSDF":
        self.device = torch.device(device)
        for name in ("keys", "tsdf_mean", "m2", "weight", "level"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def voxel_size_for_level(self, level: torch.Tensor | int) -> torch.Tensor:
        lvl = torch.as_tensor(level, dtype=torch.float32, device=self.device)
        return self.base_voxel_size / torch.pow(torch.tensor(2.0, device=self.device), lvl)

    def _keys_for_points(self, points: torch.Tensor, level: int = 0) -> torch.Tensor:
        voxel_size = self.base_voxel_size / (2**level)
        return torch.floor(points / voxel_size).to(torch.long)

    @torch.no_grad()
    def integrate_samples(self, points_local: torch.Tensor, sdf: torch.Tensor, obs_weight: torch.Tensor | float = 1.0) -> None:
        if points_local.numel() == 0:
            return
        points_local = points_local.to(self.device)
        sdf = sdf.to(self.device).float().flatten()
        w_obs = torch.as_tensor(obs_weight, device=self.device, dtype=torch.float32).expand_as(sdf)
        valid = torch.isfinite(points_local).all(dim=-1) & torch.isfinite(sdf) & torch.isfinite(w_obs) & (w_obs > 0)
        points_local, sdf, w_obs = points_local[valid], sdf[valid], w_obs[valid]
        if points_local.numel() == 0:
            return

        new_keys = self._keys_for_points(points_local, level=0)
        all_keys = torch.cat([self.keys, new_keys], dim=0)
        unique_keys, inverse = torch.unique(all_keys, dim=0, return_inverse=True)
        old_n = self.keys.shape[0]
        new_inverse = inverse[old_n:]

        mean = torch.zeros(unique_keys.shape[0], device=self.device)
        m2 = torch.zeros_like(mean)
        weight = torch.zeros_like(mean)
        level = torch.zeros(unique_keys.shape[0], dtype=torch.long, device=self.device)
        if old_n > 0:
            old_inverse = inverse[:old_n]
            mean[old_inverse] = self.tsdf_mean
            m2[old_inverse] = self.m2
            weight[old_inverse] = self.weight
            level[old_inverse] = self.level

        for idx in range(sdf.shape[0]):
            k = new_inverse[idx]
            w_old = weight[k]
            w_new = w_old + w_obs[idx]
            delta = sdf[idx] - mean[k]
            mean[k] = mean[k] + w_obs[idx] * delta / w_new.clamp_min(1e-8)
            m2[k] = m2[k] + w_obs[idx] * delta * (sdf[idx] - mean[k])
            weight[k] = w_new

        self.keys = unique_keys
        self.tsdf_mean = mean
        self.m2 = m2
        self.weight = weight
        self.level = level
        self.refine_by_variance()

    @torch.no_grad()
    def refine_by_variance(self) -> None:
        if self.keys.shape[0] == 0:
            return
        var = self.variance()
        observed = self.weight > 1
        if not observed.any():
            return
        fine_thr = torch.quantile(var[observed], self.fine_var_quantile)
        coarse_thr = torch.quantile(var[observed], self.coarse_var_quantile)
        self.level = torch.where((var >= fine_thr) & observed, (self.level + 1).clamp_max(self.num_levels - 1), self.level)
        self.level = torch.where((var <= coarse_thr) & observed, (self.level - 1).clamp_min(0), self.level)

    def variance(self) -> torch.Tensor:
        denom = (self.weight - 1).clamp_min(1.0)
        return self.m2 / denom

    def query(self, points_local: torch.Tensor) -> TSDFQuery:
        if self.keys.shape[0] == 0 or points_local.numel() == 0:
            n = points_local.shape[0]
            return TSDFQuery(
                tsdf=torch.zeros(n, device=points_local.device),
                weight=torch.zeros(n, device=points_local.device),
                valid=torch.zeros(n, dtype=torch.bool, device=points_local.device),
                level=torch.zeros(n, dtype=torch.long, device=points_local.device),
            )
        points = points_local.to(self.device)
        keys = self._keys_for_points(points, level=0)
        match = (keys[:, None, :] == self.keys[None]).all(dim=-1)
        valid = match.any(dim=1)
        idx = match.float().argmax(dim=1).long()
        return TSDFQuery(
            tsdf=torch.where(valid, self.tsdf_mean[idx], torch.zeros_like(valid, dtype=torch.float32)),
            weight=torch.where(valid, self.weight[idx], torch.zeros_like(valid, dtype=torch.float32)),
            valid=valid,
            level=torch.where(valid, self.level[idx], torch.zeros_like(valid, dtype=torch.long)),
        )
