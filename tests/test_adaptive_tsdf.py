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


def test_adaptive_tsdf_hash_lookup_and_active_blocks():
    tsdf = AdaptiveTSDF(base_voxel_size=0.5, num_levels=2, device="cpu")
    pts = torch.tensor([[0.1, 0.1, 0.1], [1.2, 0.0, 0.0], [-0.6, 0.0, 0.0]], dtype=torch.float32)
    sdf = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    tsdf.integrate_samples(pts, sdf, torch.ones_like(sdf))
    keys = tsdf._keys_for_points(pts)
    idx, valid = tsdf.lookup_keys(keys)
    assert valid.all()
    assert torch.allclose(tsdf.tsdf_mean[idx], sdf)
    assert tsdf.hashes.shape[0] == tsdf.keys.shape[0]
    assert tsdf.active_block_keys(block_size=4).shape[1] == 3
