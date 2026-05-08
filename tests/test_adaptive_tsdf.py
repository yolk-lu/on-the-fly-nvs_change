import torch

from scene.adaptive_tsdf import AdaptiveTSDF


def test_adaptive_tsdf_refines_high_variance_voxels():
    tsdf = AdaptiveTSDF(base_voxel_size=1.0, num_levels=3, device="cpu")
    pts = torch.tensor(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [2.1, 0.0, 0.0], [2.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    sdf = torch.tensor([0.0, 1.0, 0.05, 0.06], dtype=torch.float32)
    tsdf.integrate_samples(pts, sdf, torch.ones_like(sdf))
    query = tsdf.query(torch.tensor([[0.1, 0.1, 0.1], [2.1, 0.0, 0.0]], dtype=torch.float32))
    assert query.valid.all()
    assert query.level[0] >= query.level[1]
    assert torch.isfinite(query.tsdf).all()
