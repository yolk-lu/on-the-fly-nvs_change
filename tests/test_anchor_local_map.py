import torch

from poses.parallax_geometry import so3_exp
from scene.anchor_local_map import AnchorLocalMap


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
    pts = torch.tensor([[0.2, 0.3, 0.4], [-1.0, 2.0, 3.0]])
    assert torch.allclose(anchor.world_to_local(anchor.local_to_world(pts)), pts, atol=1e-6)
