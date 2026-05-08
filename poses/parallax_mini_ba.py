from __future__ import annotations

from dataclasses import dataclass

import torch

from poses.parallax_geometry import (
    hat,
    project_points,
    relative_pose_lr,
    right_perturb_pose,
)


@dataclass
class ParallaxBAResult:
    R_l: torch.Tensor
    t_l: torch.Tensor
    R_r: torch.Tensor
    t_r: torch.Tensor
    y: torch.Tensor
    residual: torch.Tensor
    initial_residual: torch.Tensor
    converged: torch.Tensor


def direction_from_angles(theta_phi: torch.Tensor) -> torch.Tensor:
    """Convert (..., theta, phi) to a unit ray in the left camera frame."""
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    cos_phi = torch.cos(phi)
    return torch.stack(
        [cos_phi * torch.sin(theta), torch.sin(phi), cos_phi * torch.cos(theta)],
        dim=-1,
    )


def parallax_point_from_state(y: torch.Tensor, baseline_l: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a left-frame point from y=(theta, phi, alpha).

    alpha is clamped to a valid parallax angle. The point depth follows the
    sine-law construction used by parallax-angle parameterizations.
    """
    ray_l = direction_from_angles(y[..., :2])
    alpha = y[..., 2].clamp(1e-5, torch.pi - 1e-4)
    b_norm = torch.linalg.vector_norm(baseline_l, dim=-1).clamp_min(1e-8)
    b_dir = baseline_l / b_norm[..., None]
    gamma = torch.acos((ray_l * b_dir).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    max_alpha = (torch.pi - gamma - 1e-4).clamp_min(1e-4)
    alpha = torch.minimum(alpha, max_alpha)
    depth = b_norm * torch.sin(gamma + alpha) / torch.sin(alpha).clamp_min(1e-6)
    depth = torch.nan_to_num(depth, nan=1.0, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
    return ray_l * depth[..., None]


def transform_left_point_to_right(
    point_l: torch.Tensor,
    R_lr: torch.Tensor,
    baseline_l: torch.Tensor,
) -> torch.Tensor:
    """Map a point from left camera coordinates to right camera coordinates."""
    return (R_lr.transpose(-1, -2) @ (point_l - baseline_l).unsqueeze(-1)).squeeze(-1)


def residual_two_view(
    R_l: torch.Tensor,
    t_l: torch.Tensor,
    R_r: torch.Tensor,
    t_r: torch.Tensor,
    y: torch.Tensor,
    uv_l: torch.Tensor,
    uv_r: torch.Tensor,
    f: torch.Tensor | float,
    centre: torch.Tensor,
) -> torch.Tensor:
    """Return stacked left/right reprojection residuals shaped (..., 4)."""
    R_lr, baseline_l = relative_pose_lr(R_l, t_l, R_r, t_r)
    point_l = parallax_point_from_state(y, baseline_l)
    point_r = transform_left_point_to_right(point_l, R_lr, baseline_l)
    res_l = project_points(point_l, f, centre) - uv_l
    res_r = project_points(point_r, f, centre) - uv_r
    return torch.cat([res_l, res_r], dim=-1)


def residual_from_relative(
    R_lr: torch.Tensor,
    baseline_l: torch.Tensor,
    y: torch.Tensor,
    uv_l: torch.Tensor,
    uv_r: torch.Tensor,
    f: torch.Tensor | float,
    centre: torch.Tensor,
) -> torch.Tensor:
    """Two-view residual parameterized only by relative pose and point state."""
    point_l = parallax_point_from_state(y, baseline_l)
    point_r = transform_left_point_to_right(point_l, R_lr, baseline_l)
    res_l = project_points(point_l, f, centre) - uv_l
    res_r = project_points(point_r, f, centre) - uv_r
    return torch.cat([res_l, res_r], dim=-1)


def baseline_translation_jacobians(R_lr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Jacobians of b = R_l^T(t_r - t_l) under local-frame right translation.

    db / d(delta_t_l) = -I
    db / d(delta_t_r) = R_lr
    """
    eye = torch.eye(3, dtype=R_lr.dtype, device=R_lr.device).expand_as(R_lr)
    return -eye, R_lr


def assemble_translation_chain_from_right_block(
    J_t_r: torch.Tensor,
    R_lr: torch.Tensor,
) -> torch.Tensor:
    """Strict relation required by the plan: J_t_l = -J_t_r R_lr^T."""
    return -(J_t_r @ R_lr.transpose(-1, -2))


def assemble_left_rotation_chain_jacobian(
    de_dphi_lr: torch.Tensor,
    de_dt_lr: torch.Tensor,
    R_lr: torch.Tensor,
    t_lr: torch.Tensor,
    *,
    include_translation_compensation: bool = True,
) -> torch.Tensor:
    """
    Assemble de / d phi_l under right perturbation.

    de/dphi_l = de/dphi_lr (-R_lr^T) + de/dt_lr t_lr^
    The compensation term is intentionally optional so tests can prove that
    omitting it is detectable.
    """
    J = de_dphi_lr @ (-R_lr.transpose(-1, -2))
    if include_translation_compensation:
        J = J + de_dt_lr @ hat(t_lr)
    return J


def baseline_left_rotation_jacobian(t_lr: torch.Tensor) -> torch.Tensor:
    """db / d(delta_phi_l) = t_lr^ for right perturbation of the left camera."""
    return hat(t_lr)


def _apply_lm_delta(
    R_l: torch.Tensor,
    t_l: torch.Tensor,
    R_r: torch.Tensor,
    t_r: torch.Tensor,
    y: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    R_l_new, t_l_new = right_perturb_pose(R_l, t_l, delta[..., 0:6])
    R_r_new, t_r_new = right_perturb_pose(R_r, t_r, delta[..., 6:12])
    y_new = y + delta[..., 12:15]
    y_new = torch.stack(
        [
            y_new[..., 0],
            y_new[..., 1].clamp(-1.55, 1.55),
            y_new[..., 2].clamp(1e-5, torch.pi - 1e-4),
        ],
        dim=-1,
    )
    return R_l_new, t_l_new, R_r_new, t_r_new, y_new


def _perturb_relative(
    R_lr: torch.Tensor,
    baseline_l: torch.Tensor,
    y: torch.Tensor,
    delta_b: torch.Tensor | None = None,
    delta_phi_lr: torch.Tensor | None = None,
    delta_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if delta_b is None:
        delta_b = torch.zeros_like(baseline_l)
    if delta_phi_lr is None:
        delta_phi_lr = torch.zeros_like(baseline_l)
    if delta_y is None:
        delta_y = torch.zeros_like(y)
    from poses.parallax_geometry import so3_exp

    return R_lr @ so3_exp(delta_phi_lr), baseline_l + delta_b, y + delta_y


class ParallaxMiniBA:
    """
    Correctness-first two-view ParallaxBA.

    The LM layout is fixed:
        [delta_t_l, delta_phi_l, delta_t_r, delta_phi_r, delta_y_point]
    Rotations use local-frame right perturbations.
    """

    def __init__(self, iters: int = 10, lm: float = 1e-3, eps: float = 1e-4):
        self.iters = int(iters)
        self.lm = float(lm)
        self.eps = float(eps)

    def finite_difference_jacobian(
        self,
        R_l: torch.Tensor,
        t_l: torch.Tensor,
        R_r: torch.Tensor,
        t_r: torch.Tensor,
        y: torch.Tensor,
        uv_l: torch.Tensor,
        uv_r: torch.Tensor,
        f: torch.Tensor | float,
        centre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r0 = residual_two_view(R_l, t_l, R_r, t_r, y, uv_l, uv_r, f, centre)
        jac_cols = []
        for idx in range(15):
            step = torch.zeros(*r0.shape[:-1], 15, dtype=r0.dtype, device=r0.device)
            step[..., idx] = self.eps
            perturbed = _apply_lm_delta(R_l, t_l, R_r, t_r, y, step)
            r_eps = residual_two_view(*perturbed, uv_l, uv_r, f, centre)
            jac_cols.append((r_eps - r0) / self.eps)
        return torch.stack(jac_cols, dim=-1), r0

    def strict_chain_jacobian(
        self,
        R_l: torch.Tensor,
        t_l: torch.Tensor,
        R_r: torch.Tensor,
        t_r: torch.Tensor,
        y: torch.Tensor,
        uv_l: torch.Tensor,
        uv_r: torch.Tensor,
        f: torch.Tensor | float,
        centre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble the camera-node Jacobian with the required chain rule.

        Layout:
            [delta_t_l, delta_phi_l, delta_t_r, delta_phi_r, delta_y_point]

        The left translation block is not independently approximated. It is
        forced to satisfy J_t_l = -J_t_r R_lr^T. The left rotation block includes
        the required de/db * b^ compensation term.
        """
        R_lr, baseline_l = relative_pose_lr(R_l, t_l, R_r, t_r)
        r0 = residual_from_relative(R_lr, baseline_l, y, uv_l, uv_r, f, centre)

        J_b_cols = []
        J_phi_lr_cols = []
        J_y_cols = []
        for idx in range(3):
            step_b = torch.zeros_like(baseline_l)
            step_b[..., idx] = self.eps
            r_b = residual_from_relative(
                *_perturb_relative(R_lr, baseline_l, y, delta_b=step_b),
                uv_l,
                uv_r,
                f,
                centre,
            )
            J_b_cols.append((r_b - r0) / self.eps)

            step_phi = torch.zeros_like(baseline_l)
            step_phi[..., idx] = self.eps
            r_phi = residual_from_relative(
                *_perturb_relative(R_lr, baseline_l, y, delta_phi_lr=step_phi),
                uv_l,
                uv_r,
                f,
                centre,
            )
            J_phi_lr_cols.append((r_phi - r0) / self.eps)

            step_y = torch.zeros_like(y)
            step_y[..., idx] = self.eps
            r_y = residual_from_relative(
                *_perturb_relative(R_lr, baseline_l, y, delta_y=step_y),
                uv_l,
                uv_r,
                f,
                centre,
            )
            J_y_cols.append((r_y - r0) / self.eps)

        J_b = torch.stack(J_b_cols, dim=-1)
        J_phi_lr = torch.stack(J_phi_lr_cols, dim=-1)
        J_y = torch.stack(J_y_cols, dim=-1)
        J_t_r = J_b @ R_lr
        J_t_l = assemble_translation_chain_from_right_block(J_t_r, R_lr)
        J_phi_l = assemble_left_rotation_chain_jacobian(J_phi_lr, J_b, R_lr, baseline_l)
        J_phi_r = J_phi_lr
        return torch.cat([J_t_l, J_phi_l, J_t_r, J_phi_r, J_y], dim=-1), r0

    def solve(
        self,
        R_l: torch.Tensor,
        t_l: torch.Tensor,
        R_r: torch.Tensor,
        t_r: torch.Tensor,
        y: torch.Tensor,
        uv_l: torch.Tensor,
        uv_r: torch.Tensor,
        f: torch.Tensor | float,
        centre: torch.Tensor,
    ) -> ParallaxBAResult:
        initial = residual_two_view(R_l, t_l, R_r, t_r, y, uv_l, uv_r, f, centre)
        initial_cost = initial.square().mean(dim=-1)
        lm = torch.full_like(initial_cost, self.lm)

        for _ in range(self.iters):
            J, r = self.strict_chain_jacobian(R_l, t_l, R_r, t_r, y, uv_l, uv_r, f, centre)
            H = J.transpose(-1, -2) @ J
            g = (J.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
            eye = torch.eye(15, dtype=H.dtype, device=H.device).expand_as(H)
            H_lm = H + lm[..., None, None] * eye
            delta = torch.linalg.solve_ex(H_lm, g.unsqueeze(-1))[0].squeeze(-1)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.25, 0.25)

            candidate = _apply_lm_delta(R_l, t_l, R_r, t_r, y, -delta)
            r_new = residual_two_view(*candidate, uv_l, uv_r, f, centre)
            old_cost = r.square().mean(dim=-1)
            new_cost = r_new.square().mean(dim=-1)
            accept = torch.isfinite(new_cost) & (new_cost <= old_cost)

            R_l = torch.where(accept[..., None, None], candidate[0], R_l)
            t_l = torch.where(accept[..., None], candidate[1], t_l)
            R_r = torch.where(accept[..., None, None], candidate[2], R_r)
            t_r = torch.where(accept[..., None], candidate[3], t_r)
            y = torch.where(accept[..., None], candidate[4], y)
            lm = torch.where(accept, (lm * 0.5).clamp_min(1e-7), (lm * 2.0).clamp_max(1e7))

        final = residual_two_view(R_l, t_l, R_r, t_r, y, uv_l, uv_r, f, centre)
        converged = torch.isfinite(final).all(dim=-1) & (final.square().mean(dim=-1) <= initial_cost)
        return ParallaxBAResult(R_l, t_l, R_r, t_r, y, final, initial, converged)
