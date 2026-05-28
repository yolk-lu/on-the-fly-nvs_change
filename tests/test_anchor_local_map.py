import torch

from poses.parallax_geometry import so3_exp
from scene.anchor_local_map import AnchorLocalMap
from scene.local_gaussian_model import covariance_from_scaling_rotation, matrix_to_quaternion, quaternion_to_matrix


def test_anchor_covariance_rotation_preserves_positive_definiteness():
    anchor = AnchorLocalMap.create(0, device="cpu")
    params = {
        "xyz": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "f_dc": torch.zeros(1, 1, 3),
        "f_rest": torch.zeros(1, 15, 3),
        "opacity": torch.zeros(1, 1),
        "scaling": torch.log(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    anchor.gaussian_model.append(params, anchor_id=0)
    R_new = so3_exp(torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32))
    anchor.update_pose_with_covariance_rotation(R_new, torch.tensor([1.0, 2.0, 3.0]))
    assert anchor.covariance_positive_mask().all()


def test_anchor_local_world_roundtrip():
    anchor = AnchorLocalMap.create(0, device="cpu")
    anchor.R_anchor_to_world = so3_exp(torch.tensor([0.1, 0.2, -0.1]))
    anchor.t_anchor_to_world = torch.tensor([1.0, -2.0, 0.5])
    anchor.s_anchor_to_world = torch.tensor(1.7)
    pts = torch.tensor([[0.2, 0.3, 0.4], [-1.0, 2.0, 3.0]])
    assert torch.allclose(anchor.world_to_local(anchor.local_to_world(pts)), pts, atol=1e-6)


def test_anchor_covariance_similarity_preserves_world_scale():
    anchor = AnchorLocalMap.create(0, device="cpu")
    anchor.s_anchor_to_world = torch.tensor(2.0)
    params = {
        "xyz": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "f_dc": torch.zeros(1, 1, 3),
        "f_rest": torch.zeros(1, 15, 3),
        "opacity": torch.zeros(1, 1),
        "scaling": torch.log(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    anchor.gaussian_model.append(params, anchor_id=0)
    cov_before = covariance_from_scaling_rotation(
        anchor.gaussian_model.params["scaling"]["val"],
        anchor.gaussian_model.params["rotation"]["val"],
    )
    anchor.update_pose_with_covariance_similarity(torch.eye(3), torch.zeros(3), torch.tensor(4.0))
    cov_after = covariance_from_scaling_rotation(
        anchor.gaussian_model.params["scaling"]["val"],
        anchor.gaussian_model.params["rotation"]["val"],
    )
    assert torch.allclose(cov_after, cov_before * 0.25, atol=1e-6)
    assert anchor.covariance_positive_mask().all()


def test_anchor_world_params_compose_gaussian_rotation():
    anchor = AnchorLocalMap.create(0, device="cpu")
    anchor.R_anchor_to_world = so3_exp(torch.tensor([0.2, -0.3, 0.1], dtype=torch.float32))
    R_local = so3_exp(torch.tensor([-0.1, 0.05, 0.4], dtype=torch.float32))
    params = {
        "xyz": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "f_dc": torch.zeros(1, 1, 3),
        "f_rest": torch.zeros(1, 15, 3),
        "opacity": torch.zeros(1, 1),
        "scaling": torch.log(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)),
        "rotation": matrix_to_quaternion(R_local[None]),
    }
    anchor.gaussian_model.append(params, anchor_id=0)

    world_params = anchor.gaussian_model.world_params(
        anchor.R_anchor_to_world,
        anchor.t_anchor_to_world,
        anchor.s_anchor_to_world,
    )
    R_world = quaternion_to_matrix(world_params["rotation"])[0]
    assert torch.allclose(R_world, anchor.R_anchor_to_world @ R_local, atol=1e-5)


def test_anchor_world_rotation_backward_is_finite_at_identity_local_gaussian():
    anchor = AnchorLocalMap.create(0, device="cpu")
    anchor.R_anchor_to_world = so3_exp(torch.tensor([0.2, -0.3, 0.1], dtype=torch.float32))
    params = {
        "xyz": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "f_dc": torch.zeros(1, 1, 3),
        "f_rest": torch.zeros(1, 15, 3),
        "opacity": torch.zeros(1, 1),
        "scaling": torch.log(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    anchor.gaussian_model.append(params, anchor_id=0)
    anchor.gaussian_model.params["rotation"]["val"].requires_grad_(True)
    world_rotation = anchor.gaussian_model.world_params(
        anchor.R_anchor_to_world,
        anchor.t_anchor_to_world,
        anchor.s_anchor_to_world,
    )["rotation"]
    world_rotation.square().sum().backward()
    assert torch.isfinite(anchor.gaussian_model.params["rotation"]["val"].grad).all()
