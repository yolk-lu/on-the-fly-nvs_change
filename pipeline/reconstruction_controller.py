from __future__ import annotations

import torch

from pipeline.frame_state import FrameState
from scene.anchor_chunk_manager import AnchorChunkManager
from scene.anchor_graph import AnchorGraph
from scene.anchor_local_map import AnchorLocalMap


class ReconstructionController:
    """State owner for the anchor-local reconstruction pipeline."""

    def __init__(self, sh_degree: int = 3, max_active_anchors: int = 3, device: str | torch.device = "cuda"):
        self.sh_degree = int(sh_degree)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.anchors: list[AnchorLocalMap] = []
        self.graph = AnchorGraph()
        self.chunk_manager = AnchorChunkManager(max_active_anchors=max_active_anchors)
        self.frames: list[FrameState] = []

    def create_anchor(self, R_anchor_to_world: torch.Tensor | None = None, t_anchor_to_world: torch.Tensor | None = None) -> AnchorLocalMap:
        anchor = AnchorLocalMap.create(len(self.anchors), sh_degree=self.sh_degree, device=self.device)
        if R_anchor_to_world is not None:
            anchor.R_anchor_to_world = R_anchor_to_world.to(self.device)
        if t_anchor_to_world is not None:
            anchor.t_anchor_to_world = t_anchor_to_world.to(self.device)
        self.anchors.append(anchor)
        self.graph.add_node(anchor.anchor_id, anchor.T_anchor_to_world)
        if len(self.anchors) > 1:
            T = torch.linalg.inv(self.anchors[-2].T_anchor_to_world) @ anchor.T_anchor_to_world
            self.graph.add_sequential_edge(self.anchors[-2].anchor_id, anchor.anchor_id, T)
        return anchor

    def add_frame(self, frame: FrameState, anchor: AnchorLocalMap | None = None) -> None:
        self.frames.append(frame)
        if anchor is not None:
            anchor.keyframe_ids.append(frame.frame_id)

    def active_anchor(self) -> AnchorLocalMap:
        if len(self.anchors) == 0:
            return self.create_anchor()
        return self.anchors[-1]

    def update_active_set(self, cam_centre_world: torch.Tensor) -> list[int]:
        return self.chunk_manager.select_active(self.anchors, cam_centre_world.to(self.device))

    def state_summary(self) -> dict:
        return {
            "num_anchors": len(self.anchors),
            "num_frames": len(self.frames),
            "num_edges": len(self.graph.edges),
            "active_anchor_ids": list(self.chunk_manager.active_anchor_ids),
            "num_gaussians": sum(anchor.gaussian_model.n for anchor in self.anchors),
        }
