import torch

from pipeline.anchor_pose_graph_optimizer import AnchorPoseGraphOptimizer, compose_sim3
from scene.anchor_graph import AnchorGraph


def _T(scale=1.0, t=(0.0, 0.0, 0.0)):
    return compose_sim3(torch.log(torch.tensor(float(scale))), torch.eye(3, dtype=torch.float64), torch.tensor(t, dtype=torch.float64)).float()


def test_sequential_only_graph_residual_is_stable():
    graph = AnchorGraph()
    graph.add_node(0, _T())
    graph.add_node(1, _T(t=(1.0, 0.0, 0.0)))
    graph.add_sequential_edge(0, 1, _T(t=(1.0, 0.0, 0.0)))
    result = AnchorPoseGraphOptimizer(max_iterations=10).optimize(graph)
    assert result.converged
    assert result.final_residual < 1e-10


def test_synthetic_loop_graph_residual_decreases():
    graph = AnchorGraph()
    graph.add_node(0, _T())
    graph.add_node(1, _T(t=(1.2, 0.0, 0.0)))
    graph.add_node(2, _T(t=(2.4, 0.0, 0.0)))
    graph.add_sequential_edge(0, 1, _T(t=(1.0, 0.0, 0.0)))
    graph.add_sequential_edge(1, 2, _T(t=(1.0, 0.0, 0.0)))
    graph.add_loop_edge(0, 2, _T(t=(2.0, 0.0, 0.0)), weight=2.0)
    result = AnchorPoseGraphOptimizer(max_iterations=120, lr=0.04).optimize(graph)
    assert result.converged
    assert result.final_residual < result.initial_residual
    assert all(update.anchor_id != 0 for update in result.updates)


def test_invalid_edge_does_not_emit_updates():
    graph = AnchorGraph()
    graph.add_node(0, _T())
    graph.add_node(1, _T(t=(1.0, 0.0, 0.0)))
    bad = _T(t=(1.0, 0.0, 0.0))
    bad[0, 0] = float("nan")
    graph.edges.append(type("Edge", (), {"src": 0, "dst": 1, "T_src_to_dst": bad, "weight": 1.0})())
    result = AnchorPoseGraphOptimizer().optimize(graph)
    assert not result.converged
    assert result.updates == []


def test_sim3_scale_loop_corrects_anchor_scale_jump():
    graph = AnchorGraph()
    graph.add_node(0, _T())
    graph.add_node(1, _T(scale=1.5, t=(1.0, 0.0, 0.0)))
    graph.add_sequential_edge(0, 1, _T(scale=1.0, t=(1.0, 0.0, 0.0)), weight=2.0)
    result = AnchorPoseGraphOptimizer(max_iterations=160, lr=0.03).optimize(graph)
    update = next(item for item in result.updates if item.anchor_id == 1)
    assert result.converged
    assert abs(float(update.s_anchor_to_world) - 1.0) < 0.1
