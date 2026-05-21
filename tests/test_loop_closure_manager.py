from types import SimpleNamespace

import torch

from pipeline.frame_state import FrameState
from pipeline.keyframe_store import TrackingKeyframeStore
from pipeline.loop_closure_manager import LoopClosureManager
from pipeline.reconstruction_controller import ReconstructionController
from poses.feature_detector import DescribedKeypoints


def _desc() -> DescribedKeypoints:
    kpts = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), indexing="xy"), dim=-1).reshape(-1, 2).float()
    feats = torch.ones(kpts.shape[0], 4)
    return DescribedKeypoints(kpts, feats)


def _frame(frame_id: int) -> FrameState:
    return FrameState(
        image=torch.zeros(3, 16, 16),
        info={"is_test": False, "frame_id": frame_id},
        desc_kpts=_desc(),
        dense_features=None,
        mono_idepth=torch.ones(1, 1, 16, 16),
        mono_depth_conf=torch.ones(1, 1, 16, 16),
        frame_id=frame_id,
    )


def _store() -> TrackingKeyframeStore:
    args = SimpleNamespace(pyr_levels=1, lod_min=1, num_prev_keyframes_check=3)
    matcher = SimpleNamespace(evaluate_match=lambda left, right: torch.tensor(1.0))
    return TrackingKeyframeStore(16, 16, args, matcher, triangulator=SimpleNamespace(n_cams=2), device="cpu")


def _controller_with_three_anchors():
    controller = ReconstructionController(device="cpu")
    a0 = controller.create_anchor()
    a1 = controller.create_anchor()
    a2 = controller.create_anchor()
    a0.t_anchor_to_world = torch.tensor([0.0, 0.0, 0.0])
    a1.t_anchor_to_world = torch.tensor([20.0, 0.0, 0.0])
    a2.t_anchor_to_world = torch.tensor([0.2, 0.0, 0.0])
    for anchor in (a0, a1, a2):
        controller.graph.add_node(anchor.anchor_id, anchor.T_anchor_to_world)
    return controller


def test_loop_closure_manager_accepts_verified_non_adjacent_anchor():
    controller = _controller_with_three_anchors()
    store = _store()
    for i, anchor in enumerate(controller.anchors):
        frame = _frame(i)
        store.add(frame, torch.eye(4), 20.0, index=i)
        controller.add_frame(frame, anchor)

    pts = _desc().kpts

    def matcher(_, __):
        return SimpleNamespace(kpts=pts, kpts_other=pts + torch.tensor([2.0, 1.0]))

    manager = LoopClosureManager(min_anchor_gap=2, max_candidates=2)
    before = len(controller.graph.edges)
    results = manager.check_anchor_rollover(2, store, controller, matcher)

    assert len(results) == 1
    assert results[0].accepted
    assert len(controller.graph.edges) == before + 1
    assert controller.graph.edges[-1].kind == "loop"
    assert controller.graph.edges[-1].src == 2
    assert controller.graph.edges[-1].dst == 0
    assert manager.summary()["pose_graph_optimization"]["status"] == "enabled"


def test_loop_closure_manager_rejects_failed_verification_without_graph_edge():
    controller = _controller_with_three_anchors()
    store = _store()
    for i, anchor in enumerate(controller.anchors):
        frame = _frame(i)
        store.add(frame, torch.eye(4), 20.0, index=i)
        controller.add_frame(frame, anchor)

    def matcher(_, __):
        return SimpleNamespace(kpts=torch.zeros(3, 2), kpts_other=torch.zeros(3, 2))

    manager = LoopClosureManager(min_anchor_gap=2, max_candidates=2)
    before = len(controller.graph.edges)
    results = manager.check_anchor_rollover(2, store, controller, matcher)

    assert len(results) == 1
    assert not results[0].accepted
    assert results[0].reason == "not_enough_matches"
    assert len(controller.graph.edges) == before
