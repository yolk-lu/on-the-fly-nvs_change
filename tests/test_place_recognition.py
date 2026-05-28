import torch

from pipeline.frame_state import FrameState
from pipeline.place_recognition import AnchorDescriptorIndex, AdaptiveSimilarityThreshold, DINOv2GlobalDescriptorExtractor


class _Keyframe:
    def __init__(self, image, index, frame_id, n_3d=0):
        class _Desc:
            def __init__(self, n):
                self.has_pt3d = torch.ones(n, dtype=torch.bool)

        self.index = index
        self.desc_kpts = _Desc(n_3d)
        self.frame = FrameState(
            image=image,
            info={},
            desc_kpts=None,
            dense_features=None,
            mono_idepth=torch.ones(1, 1, image.shape[-2], image.shape[-1]),
            mono_depth_conf=torch.ones(1, 1, image.shape[-2], image.shape[-1]),
            frame_id=frame_id,
        )

    def get_centre(self, approx=True):
        return torch.tensor([float(self.index), 0.0, 0.0])


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


def test_relocalization_query_returns_keyframe_candidates_without_anchor_exclusion():
    extractor = DINOv2GlobalDescriptorExtractor(descriptor_dim=8, use_dinov2=False, device="cpu")
    index = AnchorDescriptorIndex(
        extractor=extractor,
        threshold=AdaptiveSimilarityThreshold(min_similarity=0.99),
        top_k=4,
    )
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    base = torch.stack([xx, yy, xx * yy], dim=0)
    index.add_keyframe(_Keyframe(base, 0, 0, n_3d=32), anchor_id=0)
    index.add_keyframe(_Keyframe(torch.zeros(3, 16, 16), 1, 1, n_3d=0), anchor_id=1)

    query_frame = FrameState(
        image=base * 0.99,
        info={},
        desc_kpts=None,
        dense_features=None,
        mono_idepth=torch.ones(1, 1, 16, 16),
        mono_depth_conf=torch.ones(1, 1, 16, 16),
        frame_id=11,
    )

    candidates = index.query_frame_for_relocalization(query_frame, top_k=2)

    assert candidates[0].keyframe_index == 0
    assert candidates[0].anchor_id == 0
    assert candidates[0].triangulated_points == 32
    assert candidates[0].score >= candidates[1].score


def test_vlad_centroids_freeze_and_descriptors_are_stable():
    from pipeline.place_recognition import VLADAggregator

    aggregator = VLADAggregator(num_clusters=2, descriptor_dim=8, warmup_keyframes=2)
    extractor = DINOv2GlobalDescriptorExtractor(aggregator=aggregator, descriptor_dim=8, use_dinov2=False, device="cpu")
    index = AnchorDescriptorIndex(extractor=extractor, threshold=AdaptiveSimilarityThreshold(min_similarity=0.1))
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    base = torch.stack([xx, yy, xx * yy], dim=0)

    index.add_keyframe(_Keyframe(base, 0, 0, n_3d=8), anchor_id=0)
    assert not extractor.aggregator.is_ready
    index.add_keyframe(_Keyframe(base * 0.9, 1, 1, n_3d=8), anchor_id=0)
    assert extractor.aggregator.is_ready

    d1 = extractor.extract(base)
    d2 = extractor.extract(base)
    assert torch.allclose(d1, d2, atol=1e-8)
    assert all(record["vlad_ready"] for record in index.records)
