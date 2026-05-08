from __future__ import annotations

from dataclasses import dataclass

import torch

from poses.feature_detector import DescribedKeypoints
from poses.parallax_geometry import relative_pose_lr
from poses.parallax_mini_ba import ParallaxMiniBA, ParallaxBAResult


@dataclass
class ParallaxPoseInitializationResult:
    left_R: torch.Tensor
    left_t: torch.Tensor
    right_R: torch.Tensor
    right_t: torch.Tensor
    point_state: torch.Tensor
    inlier_mask: torch.Tensor
    residual: torch.Tensor
    failure_reason: str = ""


class ParallaxPoseInitializer:
    """
    Minimal, side-effect-free entry point for ParallaxBA pose initialization.

    Existing feature detectors and matchers remain responsible for producing
    DescribedKeypoints and matches. This class only consumes already-built
    observations and returns optimized pose/state tensors; it does not mutate
    scene, keyframe, or anchor state.
    """

    def __init__(self, f: float | torch.Tensor, centre: torch.Tensor, max_reproj_error: float = 5.0, iters: int = 10):
        self.f = f
        self.centre = centre
        self.max_reproj_error = float(max_reproj_error)
        self.solver = ParallaxMiniBA(iters=iters)
        self.last_failure_reason = ""

    @staticmethod
    def initial_state_from_match(
        uv_l: torch.Tensor,
        uv_r: torch.Tensor,
        R_l: torch.Tensor,
        t_l: torch.Tensor,
        R_r: torch.Tensor,
        t_r: torch.Tensor,
        f: torch.Tensor | float,
        centre: torch.Tensor,
    ) -> torch.Tensor:
        del uv_r
        _, baseline_l = relative_pose_lr(R_l, t_l, R_r, t_r)
        ray = torch.cat([(uv_l - centre) / f, torch.ones_like(uv_l[..., :1])], dim=-1)
        ray = ray / torch.linalg.vector_norm(ray, dim=-1, keepdim=True).clamp_min(1e-8)
        theta = torch.atan2(ray[..., 0], ray[..., 2])
        phi = torch.atan2(
            ray[..., 1],
            torch.linalg.vector_norm(ray[..., [0, 2]], dim=-1).clamp_min(1e-8),
        )
        baseline_norm = torch.linalg.vector_norm(baseline_l, dim=-1)
        alpha = torch.full_like(theta, 0.05)
        alpha = torch.where(baseline_norm > 1e-8, alpha, torch.full_like(alpha, 0.2))
        return torch.stack([theta, phi, alpha], dim=-1)

    def initialize_pair(
        self,
        left: DescribedKeypoints,
        right: DescribedKeypoints,
        R_l: torch.Tensor,
        t_l: torch.Tensor,
        R_r: torch.Tensor,
        t_r: torch.Tensor,
        match_key: int,
    ) -> ParallaxPoseInitializationResult:
        if match_key not in left.matches:
            self.last_failure_reason = "missing_matches"
            empty = torch.zeros(0, dtype=torch.bool, device=left.kpts.device)
            return ParallaxPoseInitializationResult(R_l, t_l, R_r, t_r, torch.empty(0, 3, device=left.kpts.device), empty, torch.empty(0, 4, device=left.kpts.device), self.last_failure_reason)

        matches = left.matches[match_key]
        uv_l = matches.kpts
        uv_r = matches.kpts_other
        if uv_l.shape[0] == 0:
            self.last_failure_reason = "empty_matches"
            empty = torch.zeros(0, dtype=torch.bool, device=left.kpts.device)
            return ParallaxPoseInitializationResult(R_l, t_l, R_r, t_r, torch.empty(0, 3, device=left.kpts.device), empty, torch.empty(0, 4, device=left.kpts.device), self.last_failure_reason)

        del right
        R_l_b = R_l.expand(uv_l.shape[0], 3, 3).clone()
        t_l_b = t_l.expand(uv_l.shape[0], 3).clone()
        R_r_b = R_r.expand(uv_l.shape[0], 3, 3).clone()
        t_r_b = t_r.expand(uv_l.shape[0], 3).clone()
        y = self.initial_state_from_match(uv_l, uv_r, R_l_b, t_l_b, R_r_b, t_r_b, self.f, self.centre)
        result: ParallaxBAResult = self.solver.solve(R_l_b, t_l_b, R_r_b, t_r_b, y, uv_l, uv_r, self.f, self.centre)
        err = torch.linalg.vector_norm(result.residual.view(-1, 2, 2), dim=-1).mean(dim=-1)
        inliers = torch.isfinite(err) & (err < self.max_reproj_error) & result.converged
        self.last_failure_reason = "" if inliers.any() else "no_parallax_inliers"
        return ParallaxPoseInitializationResult(
            result.R_l,
            result.t_l,
            result.R_r,
            result.t_r,
            result.y,
            inliers,
            result.residual,
            self.last_failure_reason,
        )
