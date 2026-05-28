import torch

from pipeline.frame_state import FrameState
from pipeline.reconstruction_controller import AnchorPoseUpdate, ReconstructionController
from poses.parallax_geometry import so3_exp


def _frame(frame_id: int, scale: float = 1.0) -> FrameState:
    return FrameState(
        image=torch.zeros(3, 8, 8),
        info={"is_test": False, "name": f"{frame_id}.png"},
        desc_kpts=object(),
        dense_features=None,
        mono_idepth=torch.ones(1, 1, 8, 8) * scale,
        mono_depth_conf=torch.ones(1, 1, 8, 8),
        frame_id=frame_id,
    )


def test_mapping_worker_consumes_keyframe_tasks():
    handled = []

    def callback(task):
        handled.append((task.frame.frame_id, task.anchor_id, task.kind))

    controller = ReconstructionController(device="cpu", mapping_callback=callback)
    anchor = controller.create_anchor()
    controller.start_mapping_worker()
    assert controller.enqueue_mapping_task(_frame(1), anchor.anchor_id)
    controller.mapping_queue.join()
    controller.stop_mapping_worker()
    assert handled == [(1, anchor.anchor_id, "keyframe")]
    assert controller.mapping_errors == []


def test_anchor_pose_updates_use_covariance_rotation_path():
    controller = ReconstructionController(device="cpu")
    anchor = controller.create_anchor()
    anchor.gaussian_model.append(
        {
            "xyz": torch.zeros(1, 3),
            "f_dc": torch.zeros(1, 1, 3),
            "f_rest": torch.zeros(1, 15, 3),
            "opacity": torch.zeros(1, 1),
            "scaling": torch.log(torch.tensor([[0.1, 0.2, 0.3]])),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        },
        anchor_id=anchor.anchor_id,
    )
    R_new = so3_exp(torch.tensor([0.1, -0.2, 0.05]))
    t_new = torch.tensor([1.0, 2.0, 3.0])
    controller.apply_anchor_pose_updates([AnchorPoseUpdate(anchor.anchor_id, R_new, t_new)])
    T = controller.read_anchor_pose(anchor.anchor_id)
    assert torch.allclose(T[:3, :3], R_new, atol=1e-6)
    assert torch.allclose(T[:3, 3], t_new, atol=1e-6)
    assert anchor.covariance_positive_mask().all()


def test_create_active_anchor_records_grid_scale_alignment():
    controller = ReconstructionController(device="cpu")
    ref = _frame(0, scale=2.0)
    new = _frame(1, scale=1.0)
    anchor = controller.create_active_anchor(torch.eye(3), torch.zeros(3), ref, new)
    assert anchor.scale_alignment.num_valid_cells > 0
    assert torch.allclose(anchor.scale_alignment.global_scale, torch.tensor(2.0), atol=1e-6)


def test_anchor_budget_status_uses_gaussian_and_keyframe_limits_with_tsdf_noop():
    controller = ReconstructionController(device="cpu")
    anchor = controller.create_anchor()
    anchor.gaussian_model.append(
        {
            "xyz": torch.zeros(2, 3),
            "f_dc": torch.zeros(2, 1, 3),
            "f_rest": torch.zeros(2, 15, 3),
            "opacity": torch.zeros(2, 1),
            "scaling": torch.zeros(2, 3),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        },
        anchor_id=anchor.anchor_id,
    )
    controller.add_frame(_frame(0), anchor)
    controller.add_frame(_frame(1), anchor)
    anchor.tsdf.integrate_samples(torch.zeros(3, 3), torch.zeros(3), torch.ones(3))

    status = controller.anchor_budget_status(
        torch.zeros(3),
        max_gaussians=2,
        max_keyframes=2,
        max_tsdf_voxels=1,
    )

    assert status["should_roll"]
    assert "max_anchor_gaussians" in status["reasons"]
    assert "max_anchor_keyframes" in status["reasons"]
    assert "max_anchor_tsdf_voxels" not in status["reasons"]
    assert status["num_tsdf_voxels"] == 0
