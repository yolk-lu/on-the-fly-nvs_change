import torch

from scene.scale_alignment import GridScaleAligner


def test_grid_scale_aligner_estimates_global_inverse_depth_ratio():
    aligner = GridScaleAligner(grid_size=4, min_samples_per_cell=4)
    ref = torch.ones(1, 1, 8, 8) * 3.0
    new = torch.ones(1, 1, 8, 8)
    conf = torch.ones(1, 1, 8, 8)
    result = aligner.estimate(ref, new, conf, conf)
    assert result.num_valid_cells == 4
    assert torch.allclose(result.scale_map[result.valid_mask], torch.full((4,), 3.0))
    assert torch.allclose(result.global_scale, torch.tensor(3.0))
