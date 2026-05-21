import json

import torch

from scene.anchor_local_map import AnchorLocalMap
from scene.anchor_local_renderer import AnchorLocalRenderer, AnchorRenderResult
import scene.progressive_scene_model as progressive_scene_model_module
from scene.progressive_scene_model import ProgressiveSceneModel
from pipeline.reconstruction_controller import ReconstructionController


def _append_gaussian(anchor, xyz):
    n = xyz.shape[0]
    rest_dim = anchor.gaussian_model.params["f_rest"]["val"].shape[1]
    extension = {
        "xyz": xyz,
        "f_dc": torch.zeros(n, 1, 3, device=xyz.device),
        "f_rest": torch.zeros(n, rest_dim, 3, device=xyz.device),
        "opacity": torch.zeros(n, 1, device=xyz.device),
        "scaling": torch.full((n, 3), -4.0, device=xyz.device),
        "rotation": torch.zeros(n, 4, device=xyz.device),
    }
    extension["rotation"][:, 0] = 1.0
    anchor.gaussian_model.append(extension, anchor.anchor_id)


def test_collect_anchor_batch_keeps_anchor_ownership():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor0 = AnchorLocalMap.create(0, device=device)
    anchor1 = AnchorLocalMap.create(1, device=device)
    anchor1.t_anchor_to_world = torch.tensor([10.0, 0.0, 0.0], device=device)
    _append_gaussian(anchor0, torch.tensor([[0.0, 0.0, 5.0]], device=device))
    _append_gaussian(anchor1, torch.tensor([[1.0, 0.0, 5.0]], device=device))

    renderer = AnchorLocalRenderer(64, 48, 50.0, device=device)
    batch = renderer.collect_anchor_batch([anchor0, anchor1])

    assert batch.n == 2
    assert batch.anchor_ids.tolist() == [0, 1]
    assert torch.allclose(batch.params["xyz"][0], torch.tensor([0.0, 0.0, 5.0], device=device))
    assert torch.allclose(batch.params["xyz"][1], torch.tensor([11.0, 0.0, 5.0], device=device))


def test_render_guard_filters_bad_anchor_gaussians_before_rasterizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor = AnchorLocalMap.create(0, device=device)
    _append_gaussian(
        anchor,
        torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [float("nan"), 0.0, 5.0],
            ],
            device=device,
        ),
    )

    renderer = AnchorLocalRenderer(64, 48, 50.0, device=device)
    batch = renderer.collect_anchor_batch([anchor])
    guard = renderer.guard.filter(batch.params, torch.zeros(3, device=device))

    assert guard.mask.tolist() == [True, False]
    assert guard.reason_counts["nonfinite"] == 1


def test_renderer_limits_rasterized_gaussians_before_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor = AnchorLocalMap.create(0, device=device)
    _append_gaussian(
        anchor,
        torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [0.1, 0.0, 5.0],
                [0.2, 0.0, 5.0],
            ],
            device=device,
        ),
    )
    renderer = AnchorLocalRenderer(64, 48, 50.0, max_rasterized_gaussians=2, device=device)
    batch = renderer.collect_anchor_batch([anchor])
    guard = renderer.guard.filter(batch.params, torch.zeros(3, device=device))
    kept = torch.nonzero(guard.mask, as_tuple=False).flatten()
    selected = renderer._select_raster_subset({key: value[kept] for key, value in batch.params.items()}, torch.zeros(3, device=device), 2)

    assert selected.shape[0] == 2


def test_camera_space_filter_rejects_behind_camera_and_far_frustum_points():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor = AnchorLocalMap.create(0, device=device)
    _append_gaussian(
        anchor,
        torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, -1.0],
                [100.0, 0.0, 5.0],
            ],
            device=device,
        ),
    )
    renderer = AnchorLocalRenderer(64, 48, 50.0, device=device)
    batch = renderer.collect_anchor_batch([anchor])

    mask, counts = renderer._camera_space_filter(batch.params, torch.eye(4, device=device))

    assert mask.tolist() == [True, False, False]
    assert counts["bad_camera_depth"] == 1
    assert counts["bad_frustum"] == 1


def test_progressive_scene_model_save_writes_required_outputs(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = ReconstructionController(device=device)
    anchor = controller.create_anchor()
    _append_gaussian(anchor, torch.tensor([[0.0, 0.0, 5.0]], device=device))
    model = ProgressiveSceneModel(controller, 64, 48, 50.0, device=device)

    metrics = model.save(str(tmp_path), keyframes=[], reconstruction_time=2.0, n_frames=4)

    assert metrics["num anchors"] == 1
    assert (tmp_path / "metadata.json").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["pose_graph_optimization"]["enabled"]
    assert (tmp_path / "point_clouds" / "anchor_0.ply").exists()
    assert (tmp_path / "anchor_states" / "anchor_0.pt").exists()
    assert (tmp_path / "tsdf" / "anchor_0.pt").exists()
    assert (tmp_path / "colmap" / "cameras.bin").exists()
    assert (tmp_path / "colmap" / "images.bin").exists()


def test_merge_anchor_gaussians_reduces_voxel_duplicates():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = ReconstructionController(device=device)
    anchor = controller.create_anchor()
    _append_gaussian(
        anchor,
        torch.tensor(
            [
                [0.00, 0.00, 1.0],
                [0.01, 0.01, 1.0],
                [1.00, 0.00, 1.0],
            ],
            device=device,
        ),
    )
    model = ProgressiveSceneModel(controller, 64, 48, 50.0, device=device)

    stats = model.merge_anchor_gaussians(anchor, voxel_size=0.1, target_max=2)

    assert stats["before"] == 3
    assert stats["after"] == 2
    assert anchor.gaussian_model.n == 2
    assert anchor.gaussian_model.anchor_ids.tolist() == [anchor.anchor_id, anchor.anchor_id]


def test_depth_loss_ignores_unrendered_background(monkeypatch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = ReconstructionController(device=device)
    controller.create_anchor()
    model = ProgressiveSceneModel(controller, 8, 8, 10.0, device=device)
    monkeypatch.setattr(
        progressive_scene_model_module,
        "fused_ssim",
        lambda left, right: torch.ones((), dtype=left.dtype, device=left.device),
    )

    class _Keyframe:
        def get_Rt(self):
            return torch.eye(4, device=device)

        def get_centre(self, approx=False):
            return torch.zeros(3, device=device)

    class _Frame:
        image = torch.zeros(3, 8, 8, device=device)
        mono_idepth = torch.ones(1, 1, 8, 8, device=device)
        mono_depth_conf = torch.ones(1, 1, 8, 8, device=device)
        mask = None

    result = AnchorRenderResult(
        render=torch.zeros(3, 8, 8, device=device),
        invdepth=torch.zeros(1, 8, 8, device=device),
        main_gaussian_id=torch.zeros(1, 8, 8, device=device, dtype=torch.int32),
        radii=torch.zeros(0, device=device, dtype=torch.int32),
        visibility_filter=torch.zeros(0, device=device, dtype=torch.bool),
        screenspace_points=torch.zeros(0, 3, device=device),
        anchor_ids=torch.zeros(0, device=device, dtype=torch.long),
        local_indices=torch.zeros(0, device=device, dtype=torch.long),
        kept_indices=torch.zeros(0, device=device, dtype=torch.long),
        guard_reason_counts={},
    )
    monkeypatch.setattr(model, "render_from_keyframe", lambda keyframe, active_anchor_ids=None: result)

    losses = model.loss_from_keyframe(_Keyframe(), _Frame(), active_anchor_ids=[0])

    assert losses.depth.item() == 0.0
    assert losses.depth_valid_pixels.item() == 0.0
    assert losses.ssim_weight.item() == 0.0


def test_normalized_inverse_depth_loss_is_scale_shift_invariant():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = ReconstructionController(device=device)
    controller.create_anchor()
    model = ProgressiveSceneModel(controller, 8, 8, 10.0, device=device)
    target = torch.arange(1, 17, dtype=torch.float32, device=device).view(1, 4, 4)
    pred = target * 2.0 + 3.0
    valid = torch.ones_like(target, dtype=torch.bool)

    loss = model._normalized_inverse_depth_loss(pred, target, valid)

    assert loss.item() < 0.01


def test_anchor_regularization_can_be_limited_to_active_anchor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = ReconstructionController(device=device)
    anchor0 = controller.create_anchor()
    anchor1 = controller.create_anchor()
    _append_gaussian(anchor0, torch.tensor([[0.0, 0.0, 1.0]], device=device))
    _append_gaussian(anchor1, torch.tensor([[0.0, 0.0, 1.0]], device=device))
    anchor0.gaussian_model.params["scaling"]["val"].fill_(0.0)
    anchor1.gaussian_model.params["scaling"]["val"][:] = torch.tensor([[0.0, 3.0, 0.0]], device=device)
    model = ProgressiveSceneModel(controller, 8, 8, 10.0, device=device)

    _, all_anisotropy = model.anchor_regularization_losses()
    _, active_anisotropy = model.anchor_regularization_losses(active_anchor_ids=[0])

    assert all_anisotropy > active_anisotropy
    assert active_anisotropy.item() == 0.0
