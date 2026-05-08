from __future__ import annotations

import torch

from pipeline.frame_state import FrameState
from poses.feature_detector import Detector
from poses.matcher import Matcher, Matches
from scene.dense_extractor import DenseExtractor
from scene.mono_depth import MonoDepthEstimator


class ObservationBuilder:
    """Build FrameState objects while preserving existing detector/depth modules."""

    def __init__(
        self,
        detector: Detector,
        depth_estimator: MonoDepthEstimator,
        dense_extractor: DenseExtractor | None = None,
        matcher: Matcher | None = None,
    ):
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.dense_extractor = dense_extractor
        self.matcher = matcher

    @torch.no_grad()
    def build(self, image: torch.Tensor, info: dict, frame_id: int) -> FrameState:
        desc_kpts = self.detector(image)
        dense_features = None if self.dense_extractor is None else self.dense_extractor(image)
        mono_idepth, mono_conf = self.depth_estimator(image)
        mask = info.get("mask", None)
        frame_info = dict(info)
        frame_info["frame_id"] = int(frame_id)
        return FrameState(
            image=image,
            info=frame_info,
            desc_kpts=desc_kpts,
            dense_features=dense_features,
            mono_idepth=mono_idepth,
            mono_depth_conf=mono_conf,
            frame_id=int(frame_id),
            mask=mask,
        )

    @torch.no_grad()
    def match(self, left: FrameState, right: FrameState, **kwargs) -> Matches:
        if self.matcher is None:
            raise RuntimeError("ObservationBuilder.match requires a Matcher instance")
        return self.matcher(left.desc_kpts, right.desc_kpts, **kwargs)
