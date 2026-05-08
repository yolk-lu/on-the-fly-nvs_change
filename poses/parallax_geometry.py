from __future__ import annotations

import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix v^ for vectors shaped (..., 3)."""
    if v.shape[-1] != 3:
        raise ValueError(f"hat expects vectors with last dim 3, got {tuple(v.shape)}")
    out = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    x, y, z = v.unbind(dim=-1)
    out[..., 0, 1] = -z
    out[..., 0, 2] = y
    out[..., 1, 0] = z
    out[..., 1, 2] = -x
    out[..., 2, 0] = -y
    out[..., 2, 1] = x
    return out


def vee(m: torch.Tensor) -> torch.Tensor:
    """Inverse of hat for skew-symmetric matrices shaped (..., 3, 3)."""
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"vee expects matrices (..., 3, 3), got {tuple(m.shape)}")
    return torch.stack(
        [
            0.5 * (m[..., 2, 1] - m[..., 1, 2]),
            0.5 * (m[..., 0, 2] - m[..., 2, 0]),
            0.5 * (m[..., 1, 0] - m[..., 0, 1]),
        ],
        dim=-1,
    )


def _eye_like(batch_shape: torch.Size | tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    eye = torch.eye(3, dtype=dtype, device=device)
    return eye.expand(*batch_shape, 3, 3).clone()


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential map for rotation vectors shaped (..., 3)."""
    if phi.shape[-1] != 3:
        raise ValueError(f"so3_exp expects vectors with last dim 3, got {tuple(phi.shape)}")
    theta2 = (phi * phi).sum(dim=-1, keepdim=True)
    theta = theta2.sqrt()
    k = hat(phi)
    k2 = k @ k
    small = theta2 < 1e-12
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-12),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    eye = _eye_like(phi.shape[:-1], phi.dtype, phi.device)
    return eye + a[..., None] * k + b[..., None] * k2


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithm map for rotation matrices shaped (..., 3, 3)."""
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"so3_log expects matrices (..., 3, 3), got {tuple(R.shape)}")
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    skew = R - R.transpose(-1, -2)
    raw = vee(skew)
    sin_theta = torch.sin(theta)
    scale = theta / (2.0 * sin_theta.clamp_min(1e-12))
    small = theta.abs() < 1e-6
    return torch.where(small[..., None], 0.5 * raw, scale[..., None] * raw)


def right_perturb_pose(
    R: torch.Tensor,
    t: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a local-frame right perturbation.

    delta layout is [..., 0:3] = delta_t and [..., 3:6] = delta_phi:
        R_new = R exp(delta_phi^)
        t_new = t + R delta_t
    """
    if delta.shape[-1] != 6:
        raise ValueError(f"delta must have last dim 6, got {tuple(delta.shape)}")
    delta_t = delta[..., :3]
    delta_phi = delta[..., 3:6]
    R_new = R @ so3_exp(delta_phi)
    t_new = t + (R @ delta_t.unsqueeze(-1)).squeeze(-1)
    return R_new, t_new


def relative_pose_lr(
    R_l: torch.Tensor,
    t_l: torch.Tensor,
    R_r: torch.Tensor,
    t_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return R_lr = R_l^T R_r and b = R_l^T (t_r - t_l)."""
    R_lr = R_l.transpose(-1, -2) @ R_r
    b = (R_l.transpose(-1, -2) @ (t_r - t_l).unsqueeze(-1)).squeeze(-1)
    return R_lr, b


def project_points(points_cam: torch.Tensor, f: torch.Tensor | float, centre: torch.Tensor) -> torch.Tensor:
    """Project camera-frame 3D points to pixel coordinates."""
    z = points_cam[..., 2:3]
    z_safe = torch.where(z.abs() > 1e-8, z, z.sign().clamp_min(0.0) + 1e-8)
    xy = points_cam[..., :2] / z_safe
    return xy * f + centre


def rotated_point_jacobian_right(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Jacobian of R exp(delta^) p wrt local right perturbation delta at zero."""
    return -(R @ hat(p))


def rotated_point_jacobian_left(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Jacobian of exp(delta^) R p wrt left perturbation delta at zero."""
    return -hat((R @ p.unsqueeze(-1)).squeeze(-1))
