import torch

from scene.adaptive_tsdf import AdaptiveTSDF


def test_adaptive_tsdf_keeps_compatibility_fields_without_storing_voxels():
    tsdf = AdaptiveTSDF(base_voxel_size=1.0, num_levels=3, device="cpu")
    pts = torch.tensor([[0.1, 0.1, 0.1], [2.1, 0.0, 0.0]], dtype=torch.float32)
    sdf = torch.tensor([0.0, 1.0], dtype=torch.float32)

    tsdf.integrate_samples(pts, sdf, torch.ones_like(sdf))
    query = tsdf.query(pts)

    assert tsdf.keys.shape == (0, 3)
    assert tsdf.hashes.shape == (0,)
    assert tsdf.tsdf_mean.shape == (0,)
    assert not query.valid.any()
    assert torch.count_nonzero(query.tsdf).item() == 0


def test_adaptive_tsdf_helper_interfaces_remain_available():
    tsdf = AdaptiveTSDF(base_voxel_size=0.5, num_levels=2, device="cpu")
    pts = torch.tensor([[0.1, 0.1, 0.1], [1.2, 0.0, 0.0], [-0.6, 0.0, 0.0]], dtype=torch.float32)

    keys = tsdf._keys_for_points(pts)
    idx, valid = tsdf.lookup_keys(keys)

    assert keys.shape == (3, 3)
    assert idx.shape == (3,)
    assert not valid.any()
    assert tsdf.active_block_keys(block_size=4).shape == (0, 3)
