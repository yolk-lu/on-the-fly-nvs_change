import torch

from scene.adaptive_tsdf import AdaptiveTSDF
from scene.local_gaussian_model import LocalGaussianModel
from scene.tsdf_fusion import TSDFFusion
from utils import inverse_sigmoid


def test_integrate_optimized_gaussians_preserves_interface_without_fusing():
    model = LocalGaussianModel.empty(sh_degree=1, device="cpu")
    model.append(
        {
            "xyz": torch.zeros(2, 3),
            "f_dc": torch.zeros(2, 1, 3),
            "f_rest": torch.zeros(2, 3, 3),
            "opacity": inverse_sigmoid(torch.tensor([[0.8], [0.02]], dtype=torch.float32)),
            "scaling": torch.zeros(2, 3),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        },
        anchor_id=0,
    )
    tsdf = AdaptiveTSDF(base_voxel_size=0.25, device="cpu")

    stats = TSDFFusion().integrate_optimized_gaussians(tsdf, model, min_opacity=0.05, max_samples=10)

    assert stats["mode"] == "disabled"
    assert stats["reason"] == "tsdf_backend_removed"
    assert stats["candidates"] == 2
    assert stats["integrated"] == 0
    assert stats["voxel_count_after"] == stats["voxel_count_before"] == 0
    assert tsdf.keys.shape == (0, 3)


def test_integrate_depth_is_noop():
    tsdf = AdaptiveTSDF(device="cpu")
    fusion = TSDFFusion()

    fusion.integrate_depth(
        tsdf,
        torch.ones(1, 4, 4),
        torch.ones(1, 4, 4),
        torch.eye(3),
        torch.zeros(3),
        torch.eye(3),
        torch.zeros(3),
        1.0,
        torch.zeros(2),
    )

    assert tsdf.keys.shape == (0, 3)
