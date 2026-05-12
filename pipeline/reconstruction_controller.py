from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Callable

import torch

from pipeline.concurrency import ReadWriteLock
from pipeline.frame_state import FrameState
from scene.anchor_chunk_manager import AnchorChunkManager
from scene.anchor_graph import AnchorGraph
from scene.anchor_local_map import AnchorLocalMap
from scene.loop_verifier import LoopClosureVerifier, LoopVerificationResult
from scene.scale_alignment import GridScaleAligner, ScaleAlignmentResult


@dataclass
class MappingTask:
    frame: FrameState
    anchor_id: int
    kind: str = "keyframe"


@dataclass
class AnchorPoseUpdate:
    anchor_id: int
    R_anchor_to_world: torch.Tensor
    t_anchor_to_world: torch.Tensor


class ReconstructionController:
    """Pose-decoupling owner for tracking, mapping, and global anchor state."""

    def __init__(
        self,
        sh_degree: int = 3,
        max_active_anchors: int = 3,
        device: str | torch.device = "cuda",
        mapping_callback: Callable[[MappingTask], None] | None = None,
        mapping_queue_size: int = 128,
    ):
        self.sh_degree = int(sh_degree)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.anchors: list[AnchorLocalMap] = []
        self.anchor_locks: dict[int, ReadWriteLock] = {}
        self.graph = AnchorGraph()
        self.chunk_manager = AnchorChunkManager(max_active_anchors=max_active_anchors)
        self.frames: list[FrameState] = []
        self.scale_aligner = GridScaleAligner()
        self.loop_verifier = LoopClosureVerifier()
        self.mapping_callback = mapping_callback
        self.mapping_queue: queue.Queue[MappingTask | None] = queue.Queue(maxsize=int(mapping_queue_size))
        self.mapping_thread: threading.Thread | None = None
        self.mapping_errors: list[str] = []

    def create_anchor(
        self,
        R_anchor_to_world: torch.Tensor | None = None,
        t_anchor_to_world: torch.Tensor | None = None,
        reference_frame: FrameState | None = None,
        new_frame: FrameState | None = None,
    ) -> AnchorLocalMap:
        anchor = AnchorLocalMap.create(len(self.anchors), sh_degree=self.sh_degree, device=self.device)
        if R_anchor_to_world is not None:
            anchor.R_anchor_to_world = R_anchor_to_world.to(self.device)
        if t_anchor_to_world is not None:
            anchor.t_anchor_to_world = t_anchor_to_world.to(self.device)
        scale_alignment = None
        if reference_frame is not None and new_frame is not None:
            scale_alignment = self.align_inter_anchor_scale(reference_frame, new_frame)
            anchor.scale_alignment = scale_alignment
        self.anchors.append(anchor)
        self.anchor_locks[anchor.anchor_id] = ReadWriteLock()
        self.graph.add_node(anchor.anchor_id, anchor.T_anchor_to_world)
        if len(self.anchors) > 1:
            prev_T = self.anchors[-2].T_anchor_to_world.to(self.device)
            T = torch.linalg.inv(prev_T) @ anchor.T_anchor_to_world
            self.graph.add_sequential_edge(self.anchors[-2].anchor_id, anchor.anchor_id, T)
        return anchor

    def add_frame(self, frame: FrameState, anchor: AnchorLocalMap | None = None) -> None:
        self.frames.append(frame)
        if anchor is not None:
            with self.anchor_locks[anchor.anchor_id].write_lock():
                anchor.keyframe_ids.append(frame.frame_id)

    def active_anchor(self) -> AnchorLocalMap:
        if len(self.anchors) == 0:
            return self.create_anchor()
        return self.anchors[-1]

    def update_active_set(self, cam_centre_world: torch.Tensor) -> list[int]:
        return self.chunk_manager.select_active(self.anchors, cam_centre_world.to(self.device))

    def align_inter_anchor_scale(self, reference_frame: FrameState, new_frame: FrameState) -> ScaleAlignmentResult:
        return self.scale_aligner.estimate(
            reference_frame.mono_idepth,
            new_frame.mono_idepth,
            reference_frame.mono_depth_conf,
            new_frame.mono_depth_conf,
        )

    def should_roll_anchor(
        self,
        cam_centre_world: torch.Tensor,
        max_anchor_radius: float = 25.0,
        min_keyframes: int = 20,
    ) -> bool:
        if len(self.anchors) == 0:
            return True
        anchor = self.active_anchor()
        with self.anchor_locks[anchor.anchor_id].read_lock():
            dist = torch.linalg.vector_norm(cam_centre_world.to(anchor.device) - anchor.t_anchor_to_world)
            enough_frames = len(anchor.keyframe_ids) >= int(min_keyframes)
        return bool(enough_frames and dist.item() > float(max_anchor_radius))

    def seal_active_anchor(self) -> int:
        anchor = self.active_anchor()
        with self.anchor_locks[anchor.anchor_id].write_lock():
            anchor.to("cpu")
        return anchor.anchor_id

    def create_active_anchor(
        self,
        R_anchor_to_world: torch.Tensor,
        t_anchor_to_world: torch.Tensor,
        reference_frame: FrameState | None = None,
        new_frame: FrameState | None = None,
    ) -> AnchorLocalMap:
        if len(self.anchors) > 0:
            self.seal_active_anchor()
        return self.create_anchor(R_anchor_to_world, t_anchor_to_world, reference_frame, new_frame)

    def apply_anchor_pose_updates(self, updates: list[AnchorPoseUpdate]) -> None:
        for update in updates:
            anchor = self.anchors[int(update.anchor_id)]
            with self.anchor_locks[anchor.anchor_id].write_lock():
                anchor.update_pose_with_covariance_rotation(
                    update.R_anchor_to_world.to(anchor.device),
                    update.t_anchor_to_world.to(anchor.device),
                )
                self.graph.add_node(anchor.anchor_id, anchor.T_anchor_to_world)

    def read_anchor_pose(self, anchor_id: int) -> torch.Tensor:
        anchor = self.anchors[int(anchor_id)]
        with self.anchor_locks[anchor.anchor_id].read_lock():
            return anchor.T_anchor_to_world.detach().clone()

    def enqueue_mapping_task(self, frame: FrameState, anchor_id: int | None = None, kind: str = "keyframe") -> bool:
        if anchor_id is None:
            anchor_id = self.active_anchor().anchor_id
        task = MappingTask(frame=frame, anchor_id=int(anchor_id), kind=str(kind))
        try:
            self.mapping_queue.put_nowait(task)
            return True
        except queue.Full:
            return False

    def start_mapping_worker(self) -> None:
        if self.mapping_thread is not None and self.mapping_thread.is_alive():
            return
        self.mapping_thread = threading.Thread(target=self._mapping_loop, daemon=True)
        self.mapping_thread.start()

    def stop_mapping_worker(self) -> None:
        if self.mapping_thread is None:
            return
        self.mapping_queue.put(None)
        self.mapping_thread.join()
        self.mapping_thread = None

    def _mapping_loop(self) -> None:
        while True:
            task = self.mapping_queue.get()
            if task is None:
                self.mapping_queue.task_done()
                break
            try:
                if self.mapping_callback is not None:
                    self.mapping_callback(task)
            except Exception as exc:
                self.mapping_errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                self.mapping_queue.task_done()

    def verify_and_add_loop_edge(
        self,
        src_anchor_id: int,
        dst_anchor_id: int,
        left_frame: FrameState,
        right_frame: FrameState,
        matcher_fn: Callable[[FrameState, FrameState], object],
        T_src_to_dst: torch.Tensor,
        weight: float = 1.0,
    ) -> LoopVerificationResult:
        result = self.loop_verifier.verify(left_frame, right_frame, matcher_fn)
        if result.accepted:
            self.graph.add_loop_edge(src_anchor_id, dst_anchor_id, T_src_to_dst.to(self.device), weight=weight)
        return result

    def state_summary(self) -> dict:
        return {
            "num_anchors": len(self.anchors),
            "num_frames": len(self.frames),
            "num_edges": len(self.graph.edges),
            "active_anchor_ids": list(self.chunk_manager.active_anchor_ids),
            "num_gaussians": sum(anchor.gaussian_model.n for anchor in self.anchors),
            "mapping_queue_size": self.mapping_queue.qsize(),
            "mapping_errors": list(self.mapping_errors),
        }
