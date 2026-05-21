import torch

from pipeline.frame_state import FrameState
from pipeline.place_recognition import AnchorDescriptorIndex, AdaptiveSimilarityThreshold, DINOv2GlobalDescriptorExtractor


class _Keyframe:
    def __init__(self, image, index, frame_id):
        self.index = index
        self.frame = FrameState(
            image=image,
            info={},
            desc_kpts=None,
            dense_features=None,
            mono_idepth=torch.ones(1, 1, image.shape[-2], image.shape[-1]),
            mono_depth_conf=torch.ones(1, 1, image.shape[-2], image.shape[-1]),
            frame_id=frame_id,
        )


def test_descriptor_index_uses_adaptive_threshold_for_candidates():
    extractor = DINOv2GlobalDescriptorExtractor(descriptor_dim=8, use_dinov2=False, device="cpu")
    index = AnchorDescriptorIndex(
        extractor=extractor,
        threshold=AdaptiveSimilarityThreshold(min_similarity=0.1),
        top_k=4,
    )
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    base = torch.stack([xx, yy, xx * yy], dim=0)
    index.add_keyframe(_Keyframe(base, 0, 0), anchor_id=0)
    index.add_keyframe(_Keyframe(torch.zeros(3, 16, 16), 1, 1), anchor_id=1)
    index.add_keyframe(_Keyframe(base * 0.95, 2, 2), anchor_id=2)
    candidates = index.query_anchor(2, exclude_anchor_window=2)
    assert [candidate.anchor_id for candidate in candidates] == [0]
    assert candidates[0].score >= candidates[0].threshold


def test_descriptor_index_can_query_transient_frame():
    extractor = DINOv2GlobalDescriptorExtractor(descriptor_dim=8, use_dinov2=False, device="cpu")
    index = AnchorDescriptorIndex(
        extractor=extractor,
        threshold=AdaptiveSimilarityThreshold(min_similarity=0.1),
        top_k=4,
    )
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    base = torch.stack([xx, yy, xx * yy], dim=0)
    index.add_keyframe(_Keyframe(base, 0, 0), anchor_id=0)
    index.add_keyframe(_Keyframe(torch.zeros(3, 16, 16), 1, 1), anchor_id=1)

    query_frame = FrameState(
        image=base * 0.98,
        info={},
        desc_kpts=None,
        dense_features=None,
        mono_idepth=torch.ones(1, 1, 16, 16),
        mono_depth_conf=torch.ones(1, 1, 16, 16),
        frame_id=3,
    )

    candidates = index.query_frame(query_frame, current_anchor_id=3, exclude_anchor_window=2)

    assert candidates[0].anchor_id == 0
    assert candidates[0].keyframe_index == 0
    assert candidates[0].score >= candidates[-1].score
