from __future__ import annotations

from dataclasses import dataclass

import torch

from pipeline.reconstruction_controller import AnchorPoseUpdate
from poses.parallax_geometry import so3_exp, so3_log, vee
from scene.anchor_graph import AnchorGraph


@dataclass
class AnchorPoseGraphResult:
    updates: list[AnchorPoseUpdate]
    initial_residual: float
    final_residual: float
    converged: bool
    reason: str


class AnchorPoseGraphOptimizer:
    """Small dense Sim(3) pose graph optimizer for anchor-level corrections."""

    def __init__(
        self,
        max_iterations: int = 80,
        lr: float = 0.05,
        gauge_anchor_id: int = 0,
        min_improvement: float = 1e-9,
        max_update_scale: float = 100.0,
    ):
        self.max_iterations = int(max_iterations)
        self.lr = float(lr)
        self.gauge_anchor_id = int(gauge_anchor_id)
        self.min_improvement = float(min_improvement)
        self.max_update_scale = float(max_update_scale)
        self.last_result: AnchorPoseGraphResult | None = None

    def optimize(self, graph: AnchorGraph) -> AnchorPoseGraphResult:
        if len(graph.nodes) == 0:
            return self._result([], 0.0, 0.0, False, "empty_graph")
        if len(graph.edges) == 0:
            return self._result([], 0.0, 0.0, True, "no_edges")
        if self.gauge_anchor_id not in graph.nodes:
            return self._result([], float("inf"), float("inf"), False, "missing_gauge_anchor")

        device = next(iter(graph.nodes.values())).T_anchor_to_world.device
        dtype = torch.float64
        anchor_ids = sorted(graph.nodes.keys())
        gauge_idx = anchor_ids.index(self.gauge_anchor_id)
        init_log_s = []
        init_phi = []
        init_t = []
        for anchor_id in anchor_ids:
            s, R, t = decompose_sim3(graph.nodes[anchor_id].T_anchor_to_world.to(device=device, dtype=dtype))
            init_log_s.append(torch.log(s.clamp_min(1e-8)))
            init_phi.append(so3_log(R))
            init_t.append(t)
        log_s0 = torch.stack(init_log_s)
        phi0 = torch.stack(init_phi)
        t0 = torch.stack(init_t)

        train_mask = torch.ones(len(anchor_ids), dtype=torch.bool, device=device)
        train_mask[gauge_idx] = False
        x0 = torch.cat([t0[train_mask].reshape(-1), phi0[train_mask].reshape(-1), log_s0[train_mask].reshape(-1)])
        x = torch.nn.Parameter(x0.detach().clone())
        edges = [
            (
                anchor_ids.index(edge.src),
                anchor_ids.index(edge.dst),
                edge.T_src_to_dst.to(device=device, dtype=dtype),
                max(float(edge.weight), 1e-8),
            )
            for edge in graph.edges
            if edge.src in graph.nodes and edge.dst in graph.nodes and torch.isfinite(edge.T_src_to_dst).all()
        ]
        if len(edges) == 0:
            return self._result([], float("inf"), float("inf"), False, "no_finite_edges")

        def unpack(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            n_train = int(train_mask.sum().item())
            t = t0.clone()
            phi = phi0.clone()
            log_s = log_s0.clone()
            cursor = 0
            t[train_mask] = vec[cursor : cursor + 3 * n_train].reshape(n_train, 3)
            cursor += 3 * n_train
            phi[train_mask] = vec[cursor : cursor + 3 * n_train].reshape(n_train, 3)
            cursor += 3 * n_train
            log_s[train_mask] = vec[cursor : cursor + n_train]
            return log_s, phi, t

        def loss_fn(vec: torch.Tensor) -> torch.Tensor:
            log_s, phi, t = unpack(vec)
            mats = [compose_sim3(log_s[i], so3_exp(phi[i]), t[i]) for i in range(len(anchor_ids))]
            residuals = []
            for src_idx, dst_idx, measured, weight in edges:
                if not torch.isfinite(measured).all():
                    continue
                pred = torch.linalg.inv(mats[src_idx]) @ mats[dst_idx]
                err = torch.linalg.inv(measured) @ pred
                res = sim3_residual(err) * weight**0.5
                residuals.append(res)
            if len(residuals) == 0:
                return torch.tensor(float("inf"), dtype=dtype, device=device)
            stacked = torch.cat(residuals)
            if not torch.isfinite(stacked).all():
                return torch.tensor(float("inf"), dtype=dtype, device=device)
            return (stacked.square().mean())

        with torch.no_grad():
            initial_loss = loss_fn(x).detach()
        if not torch.isfinite(initial_loss):
            return self._result([], float("inf"), float("inf"), False, "initial_residual_not_finite")

        optimizer = torch.optim.Adam([x], lr=self.lr)
        best_x = x.detach().clone()
        best_loss = initial_loss.detach().clone()
        for _ in range(self.max_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(x)
            if not torch.isfinite(loss):
                break
            loss.backward()
            if x.grad is None:
                break
            if not torch.isfinite(x.grad).all():
                x.grad = torch.nan_to_num(x.grad, nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.step()
            with torch.no_grad():
                x.clamp_(-self.max_update_scale, self.max_update_scale)
                current = loss_fn(x)
                if torch.isfinite(current) and current < best_loss:
                    best_loss = current.detach().clone()
                    best_x = x.detach().clone()

        final_loss = best_loss.detach()
        improved = final_loss <= initial_loss - self.min_improvement
        if not torch.isfinite(final_loss):
            return self._result([], float(initial_loss.item()), float("inf"), False, "final_residual_not_finite")
        if not improved and float(initial_loss.item()) > 1e-12:
            return self._result([], float(initial_loss.item()), float(final_loss.item()), False, "residual_not_decreased")

        log_s, phi, t = unpack(best_x.to(device=device, dtype=dtype))
        updates = []
        for i, anchor_id in enumerate(anchor_ids):
            if anchor_id == self.gauge_anchor_id:
                continue
            R = so3_exp(phi[i]).to(dtype=torch.float32)
            updates.append(
                AnchorPoseUpdate(
                    anchor_id=int(anchor_id),
                    R_anchor_to_world=R,
                    t_anchor_to_world=t[i].to(dtype=torch.float32),
                    s_anchor_to_world=torch.exp(log_s[i]).to(dtype=torch.float32),
                )
            )
        return self._result(updates, float(initial_loss.item()), float(final_loss.item()), True, "converged")

    def _result(
        self,
        updates: list[AnchorPoseUpdate],
        initial: float,
        final: float,
        converged: bool,
        reason: str,
    ) -> AnchorPoseGraphResult:
        result = AnchorPoseGraphResult(updates, float(initial), float(final), bool(converged), str(reason))
        self.last_result = result
        return result


def decompose_sim3(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    linear = T[:3, :3]
    scale = torch.linalg.det(linear).abs().clamp_min(1e-12).pow(1.0 / 3.0)
    R_raw = linear / scale
    u, _, vh = torch.linalg.svd(R_raw)
    R = u @ vh
    return scale, R, T[:3, 3]


def compose_sim3(log_scale: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    linear = torch.exp(log_scale).clamp_min(1e-8) * R
    upper = torch.cat([linear, t.reshape(3, 1)], dim=1)
    lower = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=R.dtype, device=R.device)
    return torch.cat([upper, lower], dim=0)


def sim3_residual(T_error: torch.Tensor) -> torch.Tensor:
    linear = T_error[:3, :3]
    scale = torch.linalg.det(linear).abs().clamp_min(1e-12).pow(1.0 / 3.0)
    R = linear / scale
    rot_residual = 0.5 * vee(R - R.transpose(-1, -2))
    return torch.cat([T_error[:3, 3], rot_residual, torch.log(scale.clamp_min(1e-8)).reshape(1)])
