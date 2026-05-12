from types import SimpleNamespace

import torch

from pipeline.frame_state import FrameState
from scene.loop_verifier import LoopClosureVerifier


def _frame(frame_id: int) -> FrameState:
    return FrameState(
        image=torch.zeros(3, 8, 8),
        info={"is_test": False},
        desc_kpts=object(),
        dense_features=None,
        mono_idepth=torch.ones(1, 1, 8, 8),
        mono_depth_conf=torch.ones(1, 1, 8, 8),
        frame_id=frame_id,
    )


def test_loop_verifier_accepts_homography_consistent_matches():
    verifier = LoopClosureVerifier(min_matches=8, min_inlier_ratio=0.8)
    pts = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
        dtype=torch.float32,
    )

    def matcher(_, __):
        return SimpleNamespace(kpts=pts, kpts_other=pts + torch.tensor([5.0, 2.0]))

    result = verifier.verify(_frame(0), _frame(1), matcher)
    assert result.accepted
    assert result.num_matches == 8


def test_loop_verifier_rejects_too_few_matches():
    verifier = LoopClosureVerifier(min_matches=8)

    def matcher(_, __):
        return SimpleNamespace(kpts=torch.zeros(3, 2), kpts_other=torch.zeros(3, 2))

    result = verifier.verify(_frame(0), _frame(1), matcher)
    assert not result.accepted
    assert result.reason == "not_enough_matches"
