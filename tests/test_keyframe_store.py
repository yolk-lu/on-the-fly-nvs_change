from types import SimpleNamespace

import torch

from pipeline.frame_state import FrameState
from pipeline.keyframe_store import TrackingKeyframeStore
from poses.feature_detector import DescribedKeypoints


def _desc(offset: float = 0.0) -> DescribedKeypoints:
    kpts = torch.tensor([[1.0 + offset, 2.0], [3.0 + offset, 4.0], [5.0 + offset, 6.0]], dtype=torch.float32)
    feats = torch.ones(3, 4, dtype=torch.float32)
    return DescribedKeypoints(kpts, feats)


def _frame(frame_id: int, offset: float = 0.0) -> FrameState:
    return FrameState(
        image=torch.zeros(3, 8, 8),
        info={"is_test": False, "frame_id": frame_id, "name": f"{frame_id:04d}.png"},
        desc_kpts=_desc(offset),
        dense_features=None,
        mono_idepth=torch.ones(1, 1, 8, 8),
        mono_depth_conf=torch.ones(1, 1, 8, 8),
        frame_id=frame_id,
    )


class _Matcher:
    def evaluate_match(self, left, right):
        return torch.tensor(float(left.kpts[:, 0].mean() - right.kpts[:, 0].mean()).__abs__())


def _store() -> TrackingKeyframeStore:
    args = SimpleNamespace(pyr_levels=1, lod_min=1, num_prev_keyframes_check=3)
    return TrackingKeyframeStore(8, 8, args, _Matcher(), triangulator=SimpleNamespace(n_cams=2), device="cpu")


def test_tracking_keyframe_store_adds_pose_and_colmap_export():
    store = _store()
    Rt = torch.eye(4)
    Rt[0, 3] = -2.0
    keyframe = store.add(_frame(10), Rt, 12.0, index=0)

    assert len(store.keyframes) == 1
    assert torch.allclose(keyframe.get_Rt(), Rt)
    assert torch.allclose(keyframe.get_centre(), torch.tensor([2.0, 0.0, 0.0]))
    assert keyframe.to_json()["f"] == 12.0
    camera, image = keyframe.to_colmap(1)
    assert camera.params[0] == 12.0
    assert image.name == "0010.png"


def test_tracking_keyframe_store_prev_selection_uses_match_score():
    store = _store()
    for i, offset in enumerate([0.0, 10.0, 20.0]):
        Rt = torch.eye(4)
        Rt[0, 3] = float(i)
        store.add(_frame(i, offset), Rt, 10.0, index=i)

    query = _desc(0.0)
    prev = store.get_prev_keyframes(2, update_3dpts=False, desc_kpts=query)

    assert len(prev) == 2
    assert {kf.index for kf in prev} == {1, 2}


def test_tracking_keyframe_store_refreshes_after_pose_update():
    store = _store()
    kf0 = store.add(_frame(0), torch.eye(4), 10.0, index=0)
    Rt1 = torch.eye(4)
    Rt1[0, 3] = -5.0
    store.add(_frame(1), Rt1, 10.0, index=1)

    new_Rt = torch.eye(4)
    new_Rt[0, 3] = -1.0
    kf0.set_Rt(new_Rt)
    store.refresh_after_pose_updates()

    assert torch.allclose(kf0.get_centre(approx=True), torch.tensor([1.0, 0.0, 0.0]))
    assert store.approx_cam_centres.shape == (2, 3)
