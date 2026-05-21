from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from pipeline.anchor_pose_graph_optimizer import AnchorPoseGraphOptimizer, AnchorPoseGraphResult
from pipeline.frame_state import FrameState
from pipeline.keyframe_store import TrackingKeyframeStore
from pipeline.place_recognition import AnchorDescriptorIndex, AnchorRetrievalCandidate
from pipeline.reconstruction_controller import ReconstructionController


@dataclass
class LoopClosureEvent:
    src_anchor_id: int
    dst_anchor_id: int
    src_frame_id: int
    dst_frame_id: int
    accepted: bool
    num_matches: int
    inlier_ratio: float
    reason: str
    retrieval_score: float = 0.0
    retrieval_threshold: float = 0.0
    edge_weight: float = 0.0
    pgo_converged: bool = False
    pgo_initial_residual: float = 0.0
    pgo_final_residual: float = 0.0
    pgo_reason: str = "not_triggered"


class LoopClosureManager:
    """Candidate discovery and geometric verification for anchor loop edges."""

    def __init__(
        self,
        min_anchor_gap: int = 2,
        max_candidates: int = 4,
        proximity_factor: float = 2.5,
        bbox_margin: float = 5.0,
        place_index: AnchorDescriptorIndex | None = None,
        pose_graph_optimizer: AnchorPoseGraphOptimizer | None = None,
    ):
        self.min_anchor_gap = int(min_anchor_gap)
        self.max_candidates = int(max_candidates)
        self.proximity_factor = float(proximity_factor)
        self.bbox_margin = float(bbox_margin)
        self.place_index = place_index
        self.pose_graph_optimizer = pose_graph_optimizer or AnchorPoseGraphOptimizer()
        self.events: list[LoopClosureEvent] = []
        self.pose_graph_optimization_status = "enabled"
        self.last_pose_graph_result: AnchorPoseGraphResult | None = None

    def check_anchor_rollover(
        self,
        new_anchor_id: int,
        store: TrackingKeyframeStore,
        controller: ReconstructionController,
        matcher_fn: Callable[[FrameState, FrameState], object],
    ) -> list[LoopClosureEvent]:
        if len(controller.anchors) < 3:
            return []
        src_id = int(new_anchor_id)
        candidates = self._retrieval_candidates(src_id) + self._fallback_candidates(src_id, controller)
        seen: set[int] = set()
        events: list[LoopClosureEvent] = []
        for candidate in candidates:
            dst_id = int(candidate.anchor_id)
            if dst_id in seen:
                continue
            seen.add(dst_id)
            src_kf = self._representative_keyframe(src_id, store, controller)
            dst_kf = self._representative_keyframe(dst_id, store, controller, candidate.keyframe_index)
            if src_kf is None or dst_kf is None:
                continue
            T_src_to_dst = self._relative_transform(src_id, dst_id, controller)
            result = controller.verify_and_add_loop_edge(
                src_id,
                dst_id,
                src_kf.frame,
                dst_kf.frame,
                matcher_fn,
                T_src_to_dst,
                weight=1.0,
            )
            pgo_result = None
            edge_weight = max(float(result.inlier_ratio), 1e-6) if result.accepted else 0.0
            if result.accepted:
                # Convert geometric confidence into a stored graph weight after the edge is inserted.
                controller.graph.edges[-1].weight = edge_weight
                pgo_result = controller.optimize_anchor_graph(self.pose_graph_optimizer)
                self.last_pose_graph_result = pgo_result
            event = LoopClosureEvent(
                src_anchor_id=src_id,
                dst_anchor_id=dst_id,
                src_frame_id=int(src_kf.frame.frame_id),
                dst_frame_id=int(dst_kf.frame.frame_id),
                accepted=bool(result.accepted),
                num_matches=int(result.num_matches),
                inlier_ratio=float(result.inlier_ratio),
                reason=result.reason,
                retrieval_score=float(candidate.score),
                retrieval_threshold=float(candidate.threshold),
                edge_weight=float(edge_weight),
                pgo_converged=bool(getattr(pgo_result, "converged", False)),
                pgo_initial_residual=float(getattr(pgo_result, "initial_residual", 0.0)),
                pgo_final_residual=float(getattr(pgo_result, "final_residual", 0.0)),
                pgo_reason=str(getattr(pgo_result, "reason", "not_triggered")),
            )
            self.events.append(event)
            events.append(event)
        return events

    def check_frame_candidates(
        self,
        src_anchor_id: int,
        src_frame: FrameState,
        candidates: list[AnchorRetrievalCandidate],
        store: TrackingKeyframeStore,
        controller: ReconstructionController,
        matcher_fn: Callable[[FrameState, FrameState], object],
    ) -> list[LoopClosureEvent]:
        if len(candidates) == 0:
            return []
        src_id = int(src_anchor_id)
        events: list[LoopClosureEvent] = []
        seen: set[int] = set()
        for candidate in candidates[: self.max_candidates]:
            dst_id = int(candidate.anchor_id)
            if dst_id in seen or dst_id == src_id or abs(dst_id - src_id) < self.min_anchor_gap:
                continue
            seen.add(dst_id)
            dst_kf = self._representative_keyframe(dst_id, store, controller, candidate.keyframe_index)
            if dst_kf is None:
                continue
            T_src_to_dst = self._relative_transform(src_id, dst_id, controller)
            result = controller.verify_and_add_loop_edge(
                src_id,
                dst_id,
                src_frame,
                dst_kf.frame,
                matcher_fn,
                T_src_to_dst,
                weight=1.0,
            )
            pgo_result = None
            edge_weight = max(float(result.inlier_ratio), 1e-6) if result.accepted else 0.0
            if result.accepted:
                controller.graph.edges[-1].weight = edge_weight
                pgo_result = controller.optimize_anchor_graph(self.pose_graph_optimizer)
                self.last_pose_graph_result = pgo_result
            event = LoopClosureEvent(
                src_anchor_id=src_id,
                dst_anchor_id=dst_id,
                src_frame_id=int(src_frame.frame_id),
                dst_frame_id=int(dst_kf.frame.frame_id),
                accepted=bool(result.accepted),
                num_matches=int(result.num_matches),
                inlier_ratio=float(result.inlier_ratio),
                reason=result.reason,
                retrieval_score=float(candidate.score),
                retrieval_threshold=float(candidate.threshold),
                edge_weight=float(edge_weight),
                pgo_converged=bool(getattr(pgo_result, "converged", False)),
                pgo_initial_residual=float(getattr(pgo_result, "initial_residual", 0.0)),
                pgo_final_residual=float(getattr(pgo_result, "final_residual", 0.0)),
                pgo_reason=str(getattr(pgo_result, "reason", "not_triggered")),
            )
            self.events.append(event)
            events.append(event)
        return events

    def summary(self) -> dict:
        last = self.last_pose_graph_result
        return {
            "pose_graph_optimization": {
                "status": self.pose_graph_optimization_status,
                "last_converged": bool(getattr(last, "converged", False)),
                "last_initial_residual": float(getattr(last, "initial_residual", 0.0)),
                "last_final_residual": float(getattr(last, "final_residual", 0.0)),
                "last_reason": str(getattr(last, "reason", "not_run")),
            },
            "num_events": len(self.events),
            "num_accepted": sum(1 for event in self.events if event.accepted),
            "events": [event.__dict__ for event in self.events],
        }

    def _retrieval_candidates(self, src_id: int) -> list[AnchorRetrievalCandidate]:
        if self.place_index is None:
            return []
        return self.place_index.query_anchor(
            src_id,
            exclude_anchor_window=self.min_anchor_gap,
            top_k=self.max_candidates,
        )

    def _fallback_candidates(self, src_id: int, controller: ReconstructionController) -> list[AnchorRetrievalCandidate]:
        src_anchor = controller.anchors[src_id]
        src_min, src_max = self._anchor_bounds(src_anchor)
        src_min = src_min.to(controller.device)
        src_max = src_max.to(controller.device)
        src_centre = src_anchor.t_anchor_to_world.to(controller.device)
        scored: list[tuple[float, int]] = []
        for anchor in controller.anchors:
            dst_id = int(anchor.anchor_id)
            if dst_id == src_id or abs(dst_id - src_id) < self.min_anchor_gap:
                continue
            dst_min, dst_max = self._anchor_bounds(anchor)
            dst_min = dst_min.to(controller.device)
            dst_max = dst_max.to(controller.device)
            overlaps = bool(((src_min - self.bbox_margin) <= (dst_max + self.bbox_margin)).all().item())
            overlaps = overlaps and bool(((dst_min - self.bbox_margin) <= (src_max + self.bbox_margin)).all().item())
            dist = torch.linalg.vector_norm(src_centre - anchor.t_anchor_to_world.to(controller.device)).item()
            radius = max(self._anchor_radius(src_min, src_max), self._anchor_radius(dst_min, dst_max), 1.0)
            if overlaps or dist <= self.proximity_factor * radius:
                scored.append((float(dist), dst_id))
        scored.sort(key=lambda item: item[0])
        return [
            AnchorRetrievalCandidate(anchor_id=anchor_id, score=0.0, threshold=0.0, keyframe_index=-1)
            for _, anchor_id in scored[: self.max_candidates]
        ]

    @staticmethod
    def _representative_keyframe(
        anchor_id: int,
        store: TrackingKeyframeStore,
        controller: ReconstructionController,
        preferred_index: int = -1,
    ):
        if preferred_index >= 0:
            keyframe = store.by_index(int(preferred_index))
            if keyframe is not None:
                return keyframe
        anchor = controller.anchors[int(anchor_id)]
        for frame_id in reversed(anchor.keyframe_ids):
            keyframe = store.by_frame_id(int(frame_id))
            if keyframe is not None:
                return keyframe
        return None

    @staticmethod
    def _relative_transform(src_id: int, dst_id: int, controller: ReconstructionController) -> torch.Tensor:
        T_src = controller.read_anchor_pose(src_id).to(controller.device)
        T_dst = controller.read_anchor_pose(dst_id).to(controller.device)
        return torch.linalg.inv(T_src) @ T_dst

    @staticmethod
    def _anchor_bounds(anchor) -> tuple[torch.Tensor, torch.Tensor]:
        device = anchor.t_anchor_to_world.device
        if anchor.gaussian_model.n == 0:
            centre = anchor.t_anchor_to_world.to(device)
            return centre - 0.5, centre + 0.5
        xyz = anchor.gaussian_model.world_params(
            anchor.R_anchor_to_world,
            anchor.t_anchor_to_world,
            anchor.s_anchor_to_world,
        )["xyz"]
        finite = torch.isfinite(xyz).all(dim=1)
        if not finite.any():
            centre = anchor.t_anchor_to_world.to(device)
            return centre - 0.5, centre + 0.5
        xyz = xyz[finite]
        return xyz.min(dim=0).values, xyz.max(dim=0).values

    @staticmethod
    def _anchor_radius(box_min: torch.Tensor, box_max: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(box_max - box_min).item() * 0.5)
