from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from poses.feature_detector import DescribedKeypoints


@dataclass
class FrameState:
    """Immutable-ish observation bundle for the anchor-local pipeline."""

    image: torch.Tensor
    info: dict[str, Any]
    desc_kpts: DescribedKeypoints
    dense_features: torch.Tensor | None
    mono_idepth: torch.Tensor
    mono_depth_conf: torch.Tensor
    frame_id: int
    mask: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.image.device

    @property
    def height(self) -> int:
        return int(self.image.shape[-2])

    @property
    def width(self) -> int:
        return int(self.image.shape[-1])

    def to(self, device: str | torch.device) -> "FrameState":
        dense = None if self.dense_features is None else self.dense_features.to(device)
        mask = None if self.mask is None else self.mask.to(device)
        self.desc_kpts.to(device)
        return FrameState(
            image=self.image.to(device),
            info=self.info,
            desc_kpts=self.desc_kpts,
            dense_features=dense,
            mono_idepth=self.mono_idepth.to(device),
            mono_depth_conf=self.mono_depth_conf.to(device),
            frame_id=self.frame_id,
            mask=mask,
        )
