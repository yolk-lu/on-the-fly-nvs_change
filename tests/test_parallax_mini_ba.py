import torch

from poses.parallax_geometry import relative_pose_lr, right_perturb_pose, so3_exp
from poses.parallax_mini_ba import (
    ParallaxMiniBA,
    assemble_left_rotation_chain_jacobian,
    assemble_translation_chain_from_right_block,
    baseline_left_rotation_jacobian,
    baseline_translation_jacobians,
    parallax_point_from_state,
    residual_two_view,
    transform_left_point_to_right,
)


def _camera_pair(dtype=torch.float64):
    R_l = so3_exp(torch.tensor([0.03, -0.02, 0.01], dtype=dtype))
    t_l = torch.tensor([0.2, -0.1, 1.0], dtype=dtype)
    R_r = so3_exp(torch.tensor([0.031, -0.018, 0.012], dtype=dtype))
    t_r = torch.tensor([0.55, -0.09, 1.02], dtype=dtype)
    return R_l, t_l, R_r, t_r


def test_translation_chain_relation_matches_required_formula():
    R_l, t_l, R_r, t_r = _camera_pair()
    R_lr, _ = relative_pose_lr(R_l, t_l, R_r, t_r)
    J_t_r = torch.tensor([[0.2, -0.5, 0.7], [1.1, 0.3, -0.4]], dtype=torch.float64)
    J_t_l = assemble_translation_chain_from_right_block(J_t_r, R_lr)
    assert torch.allclose(J_t_l, -J_t_r @ R_lr.T, atol=1e-12)


def test_baseline_translation_finite_difference_right_perturbation():
    R_l, t_l, R_r, t_r = _camera_pair()
    R_lr, b0 = relative_pose_lr(R_l, t_l, R_r, t_r)
    J_l, J_r = baseline_translation_jacobians(R_lr)

    eps = 1e-6
    fd_l, fd_r = [], []
    for i in range(3):
        d = torch.zeros(6, dtype=torch.float64)
        d[i] = eps
        _, t_l_eps = right_perturb_pose(R_l, t_l, d)
        _, b_eps_l = relative_pose_lr(R_l, t_l_eps, R_r, t_r)
        fd_l.append((b_eps_l - b0) / eps)

        _, t_r_eps = right_perturb_pose(R_r, t_r, d)
        _, b_eps_r = relative_pose_lr(R_l, t_l, R_r, t_r_eps)
        fd_r.append((b_eps_r - b0) / eps)

    assert torch.allclose(J_l, torch.stack(fd_l, dim=-1), atol=1e-8)
    assert torch.allclose(J_r, torch.stack(fd_r, dim=-1), atol=1e-8)


def test_left_rotation_baseline_compensation_matches_finite_difference():
    R_l, t_l, R_r, t_r = _camera_pair()
    _, b0 = relative_pose_lr(R_l, t_l, R_r, t_r)
    J = baseline_left_rotation_jacobian(b0)

    eps = 1e-6
    fd_cols = []
    for i in range(3):
        d = torch.zeros(6, dtype=torch.float64)
        d[3 + i] = eps
        R_l_eps, t_l_eps = right_perturb_pose(R_l, t_l, d)
        _, b_eps = relative_pose_lr(R_l_eps, t_l_eps, R_r, t_r)
        fd_cols.append((b_eps - b0) / eps)
    J_fd = torch.stack(fd_cols, dim=-1)
    assert torch.allclose(J, J_fd, atol=1e-6)


def test_left_rotation_chain_requires_translation_compensation():
    R_l, t_l, R_r, t_r = _camera_pair()
    R_lr, b = relative_pose_lr(R_l, t_l, R_r, t_r)
    de_dphi_lr = torch.tensor([[0.3, -0.2, 0.4], [0.1, 0.5, -0.6]], dtype=torch.float64)
    de_dt_lr = torch.tensor([[1.0, -0.4, 0.2], [-0.3, 0.8, 0.5]], dtype=torch.float64)
    with_comp = assemble_left_rotation_chain_jacobian(
        de_dphi_lr, de_dt_lr, R_lr, b, include_translation_compensation=True
    )
    without_comp = assemble_left_rotation_chain_jacobian(
        de_dphi_lr, de_dt_lr, R_lr, b, include_translation_compensation=False
    )
    expected_delta = de_dt_lr @ baseline_left_rotation_jacobian(b)
    assert torch.allclose(with_comp - without_comp, expected_delta, atol=1e-12)
    assert not torch.allclose(with_comp, without_comp, atol=1e-8)


def test_strict_chain_jacobian_matches_camera_finite_difference_blocks():
    R_l, t_l, R_r, t_r = _camera_pair()
    f = torch.tensor(700.0, dtype=torch.float64)
    centre = torch.tensor([320.0, 240.0], dtype=torch.float64)
    y = torch.tensor([0.02, -0.03, 0.2], dtype=torch.float64)
    R_lr, b = relative_pose_lr(R_l, t_l, R_r, t_r)
    point_l = parallax_point_from_state(y, b)
    point_r = transform_left_point_to_right(point_l, R_lr, b)
    uv_l = point_l[:2] / point_l[2] * f + centre
    uv_r = point_r[:2] / point_r[2] * f + centre

    solver = ParallaxMiniBA(iters=1, eps=1e-6)
    J_chain, _ = solver.strict_chain_jacobian(R_l, t_l, R_r, t_r, y, uv_l, uv_r, f, centre)
    J_fd, _ = solver.finite_difference_jacobian(R_l, t_l, R_r, t_r, y, uv_l, uv_r, f, centre)
    assert torch.allclose(J_chain[:, 0:12], J_fd[:, 0:12], atol=2e-3, rtol=2e-3)


def test_low_parallax_lm_residual_stays_finite_and_non_divergent():
    dtype = torch.float64
    R_l = torch.eye(3, dtype=dtype)[None]
    t_l = torch.zeros(1, 3, dtype=dtype)
    R_r = so3_exp(torch.tensor([[0.0, 0.001, 0.0]], dtype=dtype))
    t_r = torch.tensor([[0.05, 0.0, 0.0]], dtype=dtype)
    y_true = torch.tensor([[0.02, -0.01, 0.03]], dtype=dtype)
    f = torch.tensor(500.0, dtype=dtype)
    centre = torch.tensor([320.0, 240.0], dtype=dtype)

    R_lr, b = relative_pose_lr(R_l, t_l, R_r, t_r)
    point_l = parallax_point_from_state(y_true, b)
    point_r = transform_left_point_to_right(point_l, R_lr, b)
    uv_l = point_l[..., :2] / point_l[..., 2:3] * f + centre
    uv_r = point_r[..., :2] / point_r[..., 2:3] * f + centre

    y_init = y_true + torch.tensor([[0.002, -0.001, 0.003]], dtype=dtype)
    solver = ParallaxMiniBA(iters=5, lm=1e-3, eps=1e-5)
    initial = residual_two_view(R_l, t_l, R_r, t_r, y_init, uv_l, uv_r, f, centre)
    result = solver.solve(R_l, t_l, R_r, t_r, y_init, uv_l, uv_r, f, centre)
    assert torch.isfinite(result.residual).all()
    assert result.residual.square().mean() <= initial.square().mean() + 1e-9
    assert result.converged.all()
