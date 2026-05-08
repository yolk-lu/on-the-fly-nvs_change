import torch

from scene.render_guard import RenderGuard


def test_render_guard_filters_nonfinite_and_large_screen_gaussians():
    params = {
        "xyz": torch.tensor([[0.0, 0.0, 5.0], [float("nan"), 0.0, 1.0], [0.0, 0.0, 0.01]]),
        "f_dc": torch.zeros(3, 1, 3),
        "f_rest": torch.zeros(3, 15, 3),
        "opacity": torch.zeros(3, 1),
        "scaling": torch.tensor([[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]]),
        "rotation": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
    }
    result = RenderGuard(f=500.0, max_screen_px=100.0).filter(params, torch.zeros(3))
    assert result.mask.tolist() == [True, False, False]
    assert result.params["xyz"].shape[0] == 1
