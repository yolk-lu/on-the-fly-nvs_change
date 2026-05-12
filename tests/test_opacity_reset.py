import torch

from scene.opacity_reset import ViewDiversityOpacityReset


def test_opacity_reset_is_localized_to_visible_unstable_points():
    params = {"opacity": {"val": torch.zeros(4, 1)}}
    resetter = ViewDiversityOpacityReset(min_views=2, grad_quantile=0.5, reset_opacity=0.01)
    result = resetter.apply(
        params,
        visible_mask=torch.tensor([True, True, False, True]),
        view_counts=torch.tensor([0, 4, 0, 1]),
        grad_norm=torch.tensor([30.0, 20.0, 10.0, 1.0]),
        tsdf_valid_mask=torch.tensor([False, False, False, True]),
    )
    assert result.reset_mask.tolist() == [True, False, False, False]
    assert result.num_reset == 1
    assert params["opacity"]["val"][0].sigmoid().item() < 0.011
