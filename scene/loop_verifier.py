from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from pipeline.frame_state import FrameState


@dataclass
class LoopVerificationResult:
    accepted: bool
    num_matches: int
    inlier_ratio: float
    reason: str


class LoopClosureVerifier:
    """Feature and homography gate before adding loop edges to AnchorGraph."""

    def __init__(self, min_matches: int = 64, min_inlier_ratio: float = 0.35, ransac_px: float = 3.0):
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ransac_px = float(ransac_px)

    @torch.no_grad()
    def verify(
        self,
        left: FrameState,
        right: FrameState,
        matcher_fn: Callable[[FrameState, FrameState], object],
    ) -> LoopVerificationResult:
        matches = matcher_fn(left, right)
        kpts = getattr(matches, "kpts", None)
        kpts_other = getattr(matches, "kpts_other", None)
        if kpts is None or kpts_other is None:
            return LoopVerificationResult(False, 0, 0.0, "missing_match_tensors")
        n_matches = int(kpts.shape[0])
        if n_matches < self.min_matches:
            return LoopVerificationResult(False, n_matches, 0.0, "not_enough_matches")
        inlier_ratio = self._homography_inlier_ratio(kpts.float(), kpts_other.float())
        accepted = inlier_ratio >= self.min_inlier_ratio
        return LoopVerificationResult(
            accepted=accepted,
            num_matches=n_matches,
            inlier_ratio=float(inlier_ratio),
            reason="" if accepted else "homography_rejected",
        )

    def _homography_inlier_ratio(self, kpts: torch.Tensor, kpts_other: torch.Tensor) -> float:
        try:
            import cv2

            H, mask = cv2.findHomography(
                kpts.detach().cpu().numpy(),
                kpts_other.detach().cpu().numpy(),
                cv2.RANSAC,
                self.ransac_px,
            )
            if H is None or mask is None:
                return 0.0
            return float(mask.reshape(-1).mean())
        except Exception:
            displacement = torch.linalg.vector_norm(kpts - kpts_other, dim=-1)
            if displacement.numel() == 0:
                return 0.0
            median = displacement.median()
            mad = (displacement - median).abs().median().clamp_min(1e-6)
            return float(((displacement - median).abs() < 3.0 * mad).float().mean().item())
