from __future__ import annotations

import torch


class AnchorChunkManager:
    """Move active anchors to GPU and inactive anchors to CPU."""

    def __init__(self, max_active_anchors: int = 3):
        self.max_active_anchors = int(max_active_anchors)
        self.active_anchor_ids: list[int] = []

    @torch.no_grad()
    def select_active(self, anchors: list, cam_centre_world: torch.Tensor) -> list[int]:
        if len(anchors) == 0:
            self.active_anchor_ids = []
            return []
        positions = torch.stack([anchor.t_anchor_to_world.to(cam_centre_world.device) for anchor in anchors], dim=0)
        dist = torch.linalg.vector_norm(positions - cam_centre_world[None], dim=-1)
        ids = torch.argsort(dist)[: self.max_active_anchors].cpu().tolist()
        self.active_anchor_ids = [int(anchors[i].anchor_id) for i in ids]
        active_set = set(self.active_anchor_ids)
        for anchor in anchors:
            anchor.to("cuda" if anchor.anchor_id in active_set else "cpu")
        return self.active_anchor_ids
