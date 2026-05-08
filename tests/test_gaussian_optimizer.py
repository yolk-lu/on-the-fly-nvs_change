import torch

from scene.local_gaussian_model import LocalGaussianModel
from scene.gaussian_optimizer import GaussianOptimizer


def test_gaussian_optimizer_anisotropy_loss_penalizes_extreme_scale():
    model = LocalGaussianModel.empty(device="cpu")
    model.append(
        {
            "xyz": torch.zeros(2, 3),
            "f_dc": torch.zeros(2, 1, 3),
            "f_rest": torch.zeros(2, 15, 3),
            "opacity": torch.zeros(2, 1),
            "scaling": torch.log(torch.tensor([[1.0, 1.0, 1.0], [1.0, 20.0, 1.0]])),
            "rotation": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float32),
        },
        anchor_id=0,
    )
    opt = GaussianOptimizer(model.params)
    loss = opt.anisotropy_loss(max_ratio=8.0)
    assert torch.isfinite(loss)
    assert loss > 0
