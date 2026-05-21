from __future__ import annotations

from dataclasses import dataclass, field

import torch

from scene.adaptive_tsdf import AdaptiveTSDF
from scene.local_gaussian_model import (
    LocalGaussianModel,
    covariance_from_scaling_rotation,
    scaling_rotation_from_covariance,
)


@dataclass
class AnchorLocalMap:
    anchor_id: int
    gaussian_model: LocalGaussianModel
    tsdf: AdaptiveTSDF
    R_anchor_to_world: torch.Tensor
    t_anchor_to_world: torch.Tensor
    s_anchor_to_world: torch.Tensor
    keyframe_ids: list[int] = field(default_factory=list)

    @classmethod
    def create(cls, anchor_id: int, sh_degree: int = 3, device: str | torch.device = "cuda") -> "AnchorLocalMap":
        return cls(
            anchor_id=int(anchor_id),
            gaussian_model=LocalGaussianModel.empty(sh_degree=sh_degree, device=device),
            tsdf=AdaptiveTSDF(device=device),
            R_anchor_to_world=torch.eye(3, device=device),
            t_anchor_to_world=torch.zeros(3, device=device),
            s_anchor_to_world=torch.ones((), device=device),
        )

    @property
    def device(self) -> torch.device:
        return self.gaussian_model.device

    def to(self, device: str | torch.device) -> "AnchorLocalMap":
        self.gaussian_model.to(device)
        self.tsdf.to(device)
        self.R_anchor_to_world = self.R_anchor_to_world.to(device)
        self.t_anchor_to_world = self.t_anchor_to_world.to(device)
        self.s_anchor_to_world = self.s_anchor_to_world.to(device)
        return self

    @property
    def T_anchor_to_world(self) -> torch.Tensor:
        T = torch.eye(4, dtype=self.R_anchor_to_world.dtype, device=self.R_anchor_to_world.device)
        T[:3, :3] = self.s_anchor_to_world.clamp_min(1e-8) * self.R_anchor_to_world
        T[:3, 3] = self.t_anchor_to_world
        return T

    @property
    def R_world_to_anchor(self) -> torch.Tensor:
        return self.R_anchor_to_world.T

    @property
    def t_world_to_anchor(self) -> torch.Tensor:
        return -(self.R_anchor_to_world.T @ self.t_anchor_to_world) / self.s_anchor_to_world.clamp_min(1e-8)

    def world_to_local(self, points_world: torch.Tensor) -> torch.Tensor:
        scale = self.s_anchor_to_world.to(points_world.device, points_world.dtype).clamp_min(1e-8)
        R = self.R_world_to_anchor.to(points_world.device, points_world.dtype)
        t = self.t_anchor_to_world.to(points_world.device, points_world.dtype)
        return (R @ (points_world - t[None]).T).T / scale

    def local_to_world(self, points_local: torch.Tensor) -> torch.Tensor:
        scale = self.s_anchor_to_world.to(points_local.device, points_local.dtype).clamp_min(1e-8)
        R = self.R_anchor_to_world.to(points_local.device, points_local.dtype)
        t = self.t_anchor_to_world.to(points_local.device, points_local.dtype)
        return scale * (R @ points_local.T).T + t[None]

    @torch.no_grad()
    def update_pose_with_covariance_similarity(
        self,
        R_new: torch.Tensor,
        t_new: torch.Tensor,
        s_new: torch.Tensor | float = 1.0,
    ) -> None:
        """
        Update anchor Sim(3) pose and compensate local Gaussian covariance so the
        world-space ellipsoids remain continuous after pose-graph correction.
        """
        R_delta = self.R_anchor_to_world.T @ R_new.to(self.device)
        s_new_tensor = torch.as_tensor(s_new, dtype=self.s_anchor_to_world.dtype, device=self.device).clamp_min(1e-8)
        scale_ratio = (self.s_anchor_to_world.to(self.device).clamp_min(1e-8) / s_new_tensor).square()
        params = self.gaussian_model.params
        if params["xyz"]["val"].shape[0] > 0:
            cov_local = covariance_from_scaling_rotation(params["scaling"]["val"], params["rotation"]["val"])
            cov_rot = scale_ratio * (R_delta.T @ cov_local @ R_delta)
            scaling, rotation = scaling_rotation_from_covariance(cov_rot)
            params["scaling"]["val"] = scaling.contiguous()
            params["rotation"]["val"] = rotation.contiguous()
        self.R_anchor_to_world = R_new.to(self.device)
        self.t_anchor_to_world = t_new.to(self.device)
        self.s_anchor_to_world = s_new_tensor.to(self.device)

    @torch.no_grad()
    def update_pose_with_covariance_rotation(self, R_new: torch.Tensor, t_new: torch.Tensor) -> None:
        self.update_pose_with_covariance_similarity(R_new, t_new, self.s_anchor_to_world)

    def covariance_positive_mask(self) -> torch.Tensor:
        params = self.gaussian_model.params
        if params["xyz"]["val"].shape[0] == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        cov = covariance_from_scaling_rotation(params["scaling"]["val"], params["rotation"]["val"])
        eigvals = torch.linalg.eigvalsh(0.5 * (cov + cov.transpose(-1, -2)))
        return torch.isfinite(eigvals).all(dim=-1) & (eigvals > 0).all(dim=-1)
