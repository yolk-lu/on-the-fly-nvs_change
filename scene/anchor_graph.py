from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class AnchorGraphNode:
    anchor_id: int
    T_anchor_to_world: torch.Tensor


@dataclass
class AnchorGraphEdge:
    src: int
    dst: int
    T_src_to_dst: torch.Tensor
    weight: float = 1.0
    kind: str = "sequential"


class AnchorGraph:
    """Pose-graph skeleton with sequential edges and loop-edge hook."""

    def __init__(self):
        self.nodes: dict[int, AnchorGraphNode] = {}
        self.edges: list[AnchorGraphEdge] = []

    def add_node(self, anchor_id: int, T_anchor_to_world: torch.Tensor) -> None:
        self.nodes[int(anchor_id)] = AnchorGraphNode(int(anchor_id), T_anchor_to_world.detach().clone())

    def add_sequential_edge(self, src: int, dst: int, T_src_to_dst: torch.Tensor, weight: float = 1.0) -> None:
        self.edges.append(AnchorGraphEdge(int(src), int(dst), T_src_to_dst.detach().clone(), float(weight), "sequential"))

    def add_loop_edge(self, src: int, dst: int, T_src_to_dst: torch.Tensor, weight: float = 1.0) -> None:
        self.edges.append(AnchorGraphEdge(int(src), int(dst), T_src_to_dst.detach().clone(), float(weight), "loop"))

    def to_json(self) -> dict:
        return {
            "nodes": [
                {"anchor_id": n.anchor_id, "T_anchor_to_world": n.T_anchor_to_world.cpu().tolist()}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "T_src_to_dst": e.T_src_to_dst.cpu().tolist(),
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
            graph.edges.append(
                AnchorGraphEdge(
                    int(edge["src"]),
                    int(edge["dst"]),
                    torch.tensor(edge["T_src_to_dst"], dtype=torch.float32, device=device),
                    float(edge.get("weight", 1.0)),
                    str(edge.get("kind", "sequential")),
                )
            )
        return graph
