import torch

from poses.parallax_geometry import (
    hat,
    right_perturb_pose,
    rotated_point_jacobian_right,
    so3_exp,
    so3_log,
)


def test_hat_cross_product_property():
    a = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
    b = torch.tensor([-0.4, 0.5, 0.1], dtype=torch.float64)
    assert torch.allclose(hat(a) @ b, torch.cross(a, b, dim=0), atol=1e-12)


def test_so3_exp_log_roundtrip_small_rotation():
    phi = torch.tensor([0.02, -0.03, 0.04], dtype=torch.float64)
    R = so3_exp(phi)
    phi_back = so3_log(R)
    assert torch.allclose(phi_back, phi, atol=1e-8)
    assert torch.allclose(R.T @ R, torch.eye(3, dtype=torch.float64), atol=1e-10)


def test_right_perturb_pose_uses_local_translation():
    R = so3_exp(torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64))
    t = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    delta = torch.tensor([0.3, -0.1, 0.2, 0.01, 0.02, -0.03], dtype=torch.float64)
    R_new, t_new = right_perturb_pose(R, t, delta)
    assert torch.allclose(t_new, t + R @ delta[:3], atol=1e-12)
    assert torch.allclose(R_new, R @ so3_exp(delta[3:]), atol=1e-12)


def test_right_rotation_point_jacobian_matches_finite_difference():
    R = so3_exp(torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64))
    p = torch.tensor([0.4, -0.7, 2.0], dtype=torch.float64)
    J = rotated_point_jacobian_right(R, p)
    eps = 1e-6
    cols = []
    for i in range(3):
        d = torch.zeros(3, dtype=torch.float64)
        d[i] = eps
        fd = (R @ so3_exp(d) @ p - R @ p) / eps
        cols.append(fd)
    J_fd = torch.stack(cols, dim=-1)
    assert torch.allclose(J, J_fd, atol=1e-5)
