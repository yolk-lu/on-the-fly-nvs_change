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
    """Compatibility placeholder for the removed TSDF backend.

    Progressive code still owns an ``anchor.tsdf`` object and serializes the
    historical TSDF fields, but this class intentionally stores no samples.
    """

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
        self._reset_storage()

    def _reset_storage(self) -> None:
        self.keys = torch.empty(0, 3, dtype=torch.long, device=self.device)
        self.hashes = torch.empty(0, dtype=torch.long, device=self.device)
        self.tsdf_mean = torch.empty(0, dtype=torch.float32, device=self.device)
        self.m2 = torch.empty(0, dtype=torch.float32, device=self.device)
        self.weight = torch.empty(0, dtype=torch.float32, device=self.device)
        self.level = torch.empty(0, dtype=torch.long, device=self.device)

    def to(self, device: str | torch.device) -> "AdaptiveTSDF":
        self.device = torch.device(device)
        for name in ("keys", "hashes", "tsdf_mean", "m2", "weight", "level"):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    def voxel_size_for_level(self, level: torch.Tensor | int) -> torch.Tensor:
        lvl = torch.as_tensor(level, dtype=torch.float32, device=self.device)
        return self.base_voxel_size / torch.pow(torch.tensor(2.0, device=self.device), lvl)

    def _keys_for_points(self, points: torch.Tensor, level: int = 0) -> torch.Tensor:
        voxel_size = self.base_voxel_size / (2**level)
        return torch.floor(points / voxel_size).to(torch.long)

    @staticmethod
    def hash_keys(keys: torch.Tensor) -> torch.Tensor:
        keys = keys.to(torch.long)
        primes = torch.tensor([73856093, 19349663, 83492791], dtype=torch.long, device=keys.device)
        return (keys * primes).sum(dim=-1)

    @torch.no_grad()
    def integrate_samples(
        self,
        points_local: torch.Tensor,
        sdf: torch.Tensor,
        obs_weight: torch.Tensor | float = 1.0,
    ) -> None:
        return None

    @torch.no_grad()
    def refine_by_variance(self) -> None:
        return None

    def variance(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.float32, device=self.device)

    def query(self, points_local: torch.Tensor) -> TSDFQuery:
        n = points_local.shape[0]
        return TSDFQuery(
            tsdf=torch.zeros(n, device=points_local.device),
            weight=torch.zeros(n, device=points_local.device),
            valid=torch.zeros(n, dtype=torch.bool, device=points_local.device),
            level=torch.zeros(n, dtype=torch.long, device=points_local.device),
        )

    def lookup_keys(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(keys.shape[0], dtype=torch.long, device=keys.device),
            torch.zeros(keys.shape[0], dtype=torch.bool, device=keys.device),
        )

    def active_block_keys(self, block_size: int = 8) -> torch.Tensor:
        return torch.empty(0, 3, dtype=torch.long, device=self.device)

    def _sort_by_hash(self) -> None:
        return None
