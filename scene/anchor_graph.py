from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


def sim3_scale_from_matrix(T: torch.Tensor) -> torch.Tensor:
    linear = T[:3, :3]
    det = torch.linalg.det(linear).abs().clamp_min(1e-12)
    return det.pow(1.0 / 3.0)


@dataclass
class AnchorGraphNode:
    anchor_id: int
    T_anchor_to_world: torch.Tensor
    s_anchor_to_world: float = 1.0


@dataclass
class AnchorGraphEdge:
    src: int
    dst: int
    T_src_to_dst: torch.Tensor
    weight: float = 1.0
    kind: str = "sequential"
    scale_src_to_dst: float = 1.0


class AnchorGraph:
    """Pose-graph skeleton with sequential edges and loop-edge hook."""

    def __init__(self):
        self.nodes: dict[int, AnchorGraphNode] = {}
        self.edges: list[AnchorGraphEdge] = []

    def add_node(self, anchor_id: int, T_anchor_to_world: torch.Tensor) -> None:
        scale = float(sim3_scale_from_matrix(T_anchor_to_world.detach()).item())
        self.nodes[int(anchor_id)] = AnchorGraphNode(int(anchor_id), T_anchor_to_world.detach().clone(), scale)

    def add_sequential_edge(self, src: int, dst: int, T_src_to_dst: torch.Tensor, weight: float = 1.0) -> None:
        scale = float(sim3_scale_from_matrix(T_src_to_dst.detach()).item())
        self.edges.append(AnchorGraphEdge(int(src), int(dst), T_src_to_dst.detach().clone(), float(weight), "sequential", scale))

    def add_loop_edge(self, src: int, dst: int, T_src_to_dst: torch.Tensor, weight: float = 1.0) -> None:
        scale = float(sim3_scale_from_matrix(T_src_to_dst.detach()).item())
        self.edges.append(AnchorGraphEdge(int(src), int(dst), T_src_to_dst.detach().clone(), float(weight), "loop", scale))

    def to_json(self) -> dict:
        return {
            "nodes": [
                {
                    "anchor_id": n.anchor_id,
                    "T_anchor_to_world": n.T_anchor_to_world.cpu().tolist(),
                    "s_anchor_to_world": n.s_anchor_to_world,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "T_src_to_dst": e.T_src_to_dst.cpu().tolist(),
                    "scale_src_to_dst": e.scale_src_to_dst,
                    "weight": e.weight,
                    "kind": e.kind,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_json(cls, payload: dict, device: str | torch.device = "cuda") -> "AnchorGraph":
        graph = cls()
        for node in payload.get("nodes", []):
            graph.add_node(node["anchor_id"], torch.tensor(node["T_anchor_to_world"], dtype=torch.float32, device=device))
        for edge in payload.get("edges", []):
            T = torch.tensor(edge["T_src_to_dst"], dtype=torch.float32, device=device)
            graph.edges.append(
                AnchorGraphEdge(
                    int(edge["src"]),
                    int(edge["dst"]),
                    T,
                    float(edge.get("weight", 1.0)),
                    str(edge.get("kind", "sequential")),
                    float(edge.get("scale_src_to_dst", sim3_scale_from_matrix(T).item())),
                )
            )
        return graph
