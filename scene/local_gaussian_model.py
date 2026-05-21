from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


GAUSSIAN_KEYS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")


@dataclass
class LocalGaussianModel:
    """Anchor-local Gaussian parameter container."""

    params: dict[str, dict[str, torch.Tensor]]
    anchor_ids: torch.Tensor | None = None

    @classmethod
    def empty(cls, sh_degree: int = 3, device: str | torch.device = "cuda") -> "LocalGaussianModel":
        rest_dim = (sh_degree + 1) * (sh_degree + 1) - 1
        params = {
            "xyz": {"val": torch.empty(0, 3, device=device)},
            "f_dc": {"val": torch.empty(0, 1, 3, device=device)},
            "f_rest": {"val": torch.empty(0, rest_dim, 3, device=device)},
            "opacity": {"val": torch.empty(0, 1, device=device)},
            "scaling": {"val": torch.empty(0, 3, device=device)},
            "rotation": {"val": torch.empty(0, 4, device=device)},
        }
        return cls(params=params, anchor_ids=torch.empty(0, dtype=torch.long, device=device))

    @property
    def device(self) -> torch.device:
        return self.params["xyz"]["val"].device

    @property
    def n(self) -> int:
        return int(self.params["xyz"]["val"].shape[0])

    def to(self, device: str | torch.device) -> "LocalGaussianModel":
        for item in self.params.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(device)
        if self.anchor_ids is not None:
            self.anchor_ids = self.anchor_ids.to(device)
        return self

    def clone_params(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in item.items()} for name, item in self.params.items()}

    def append(self, extension: dict[str, torch.Tensor], anchor_id: int | None = None) -> None:
        n_new = int(extension["xyz"].shape[0])
        for key in GAUSSIAN_KEYS:
            self.params[key]["val"] = torch.cat([self.params[key]["val"], extension[key]], dim=0).contiguous()
        if self.anchor_ids is not None:
            fill = -1 if anchor_id is None else int(anchor_id)
            self.anchor_ids = torch.cat(
                [self.anchor_ids, torch.full((n_new,), fill, dtype=torch.long, device=self.device)],
                dim=0,
            )

    def prune(self, keep_mask: torch.Tensor) -> None:
        for key in GAUSSIAN_KEYS:
            self.params[key]["val"] = self.params[key]["val"][keep_mask].contiguous()
        if self.anchor_ids is not None:
            self.anchor_ids = self.anchor_ids[keep_mask].contiguous()

    def world_params(
        self,
        R_anchor_to_world: torch.Tensor,
        t_anchor_to_world: torch.Tensor,
        s_anchor_to_world: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        xyz_local = self.params["xyz"]["val"]
        scale = torch.as_tensor(s_anchor_to_world, dtype=xyz_local.dtype, device=xyz_local.device).clamp_min(1e-8)
        xyz_world = scale * (R_anchor_to_world @ xyz_local.T).T + t_anchor_to_world[None]
        out = {key: self.params[key]["val"] for key in GAUSSIAN_KEYS}
        out["xyz"] = xyz_world.contiguous()
        out["scaling"] = (self.params["scaling"]["val"] + torch.log(scale)).contiguous()
        out["rotation"] = self.rotate_quaternions_world(self.params["rotation"]["val"], R_anchor_to_world)
        return out

    @staticmethod
    def rotate_quaternions_world(q_local: torch.Tensor, R_anchor_to_world: torch.Tensor) -> torch.Tensor:
        # First version preserves stored quaternion if local covariance rotation
        # is not requested by the caller. AnchorLocalMap handles covariance-level
        # rotation explicitly for loop-closure pose updates.
        del R_anchor_to_world
        norm = torch.linalg.vector_norm(q_local, dim=-1, keepdim=True).clamp_min(1e-8)
        q = q_local / norm
        identity = torch.zeros_like(q)
        if identity.numel() > 0:
            identity[:, 0] = 1
        return torch.where(torch.isfinite(q).all(dim=-1, keepdim=True), q, identity)

    def finite_mask(self) -> torch.Tensor:
        if self.n == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        for key in GAUSSIAN_KEYS:
            mask &= torch.isfinite(self.params[key]["val"].flatten(1)).all(dim=1)
        return mask


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    # Stable enough for covariance rotation tests; assumes R is close to SO(3).
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    qw = torch.sqrt((1.0 + trace).clamp_min(1e-8)) * 0.5
    denom = (4.0 * qw).clamp_min(1e-8)
    qx = (R[..., 2, 1] - R[..., 1, 2]) / denom
    qy = (R[..., 0, 2] - R[..., 2, 0]) / denom
    qz = (R[..., 1, 0] - R[..., 0, 1]) / denom
    return F.normalize(torch.stack([qw, qx, qy, qz], dim=-1), dim=-1)


def covariance_from_scaling_rotation(log_scaling: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    R = quaternion_to_matrix(quaternion)
    scales2 = torch.exp(log_scaling).square().clamp_min(1e-12)
    return R @ torch.diag_embed(scales2) @ R.transpose(-1, -2)


def scaling_rotation_from_covariance(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(1e-12)
    det = torch.linalg.det(eigvecs)
    eigvecs = torch.where((det < 0)[..., None, None], eigvecs * torch.tensor([1, 1, -1], device=cov.device, dtype=cov.dtype), eigvecs)
    log_scaling = 0.5 * torch.log(eigvals)
    quat = matrix_to_quaternion(eigvecs)
    return log_scaling, quat
