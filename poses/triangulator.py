#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import warnings
import os
import math

from poses.feature_detector import DescribedKeypoints
from utils import depth2points, pts2px

def matches_to_points(uv, uv_matched, R, t, f, centre):
    p2 = t[None]  # [1, 3]
    d1 = depth2points(uv, 1, f, centre)

    d2 = depth2points(uv_matched, 1, f, centre)
    d2 = torch.matmul(d2, R.T)  # Transform d2 by the rotation matrix

    # Normalize directions
    d1 = d1 / torch.linalg.vector_norm(d1, dim=-1, keepdim=True)
    d2 = d2 / torch.linalg.vector_norm(d2, dim=-1, keepdim=True)

    # Compute the normal vector and its secondary vector
    n = torch.cross(d1, d2, dim=-1)  # [N, 3]\
    n2 = torch.cross(d2, n, dim=-1)  # [N, 3]\

    # Compute distances
    dist = torch.matmul(n2, p2.T) / torch.bmm(n2.unsqueeze(1), d1.unsqueeze(-1)).squeeze(-1)

    # # Compute pose direction and angles
    angles = torch.acos(torch.sum(d1 * d2, dim=1))  # Angle between d1 and d2

    # Compute 3D points
    xyz = d1 * dist

    # Compute the views' disambiguation capability
    xyz1 = torch.matmul(d1 - p2, R)
    uv1 = pts2px(xyz1, f, centre)
    xyz2 = torch.matmul(d1 * 10 - p2, R)
    uv2 = pts2px(xyz2, f, centre)
    disambiguation = torch.linalg.vector_norm(uv1 - uv2, dim=-1)

    # Transform xyz to matched coordinates
    xyz_matched = torch.matmul(xyz - p2, R)
    expected_uv_matched = pts2px(xyz_matched, f, centre)

    # Compute reprojection error
    error = torch.linalg.vector_norm(expected_uv_matched - uv_matched, dim=-1)

    # Mark invalid points
    invalid = (xyz.isnan() | xyz.isinf()).any(dim=-1) | angles.isnan()
    angles = torch.where(invalid, 0, angles)

    # Return 3D points, disambiguation, and reprojection error
    return xyz, disambiguation, error


def _safe_unit(v, eps=1e-8):
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _project_safe(xyz, f, centre):
    xyz_safe = xyz.clone()
    z = xyz_safe[..., 2:3]
    z_safe = torch.where(z.abs() > 1e-6, z, z.sign().clamp_min(0.0) + 1e-6)
    xyz_safe[..., 2:3] = z_safe
    return pts2px(xyz_safe, f, centre)


def _direction_to_angles(d):
    d = _safe_unit(d)
    theta = torch.atan2(d[..., 0], d[..., 2])
    phi = torch.atan2(d[..., 1], torch.linalg.vector_norm(d[..., [0, 2]], dim=-1).clamp_min(1e-8))
    return torch.stack([theta, phi], dim=-1)


def _angles_to_direction(theta, phi):
    cos_phi = torch.cos(phi)
    return torch.stack(
        [
            cos_phi * torch.sin(theta),
            torch.sin(phi),
            cos_phi * torch.cos(theta),
        ],
        dim=-1,
    )


def _parallax_state_to_xyz(state, baseline):
    theta = state[:, 0]
    phi = state[:, 1]
    alpha = torch.exp(state[:, 2]).clamp(1e-5, math.pi - 1e-4)

    d = _safe_unit(_angles_to_direction(theta, phi))
    b_norm = torch.linalg.vector_norm(baseline, dim=-1).clamp_min(1e-6)
    b_dir = baseline / b_norm[:, None]
    gamma = torch.acos((d * b_dir).sum(dim=-1).clamp(-1.0 + 1e-5, 1.0 - 1e-5))
    max_alpha = (math.pi - gamma - 1e-4).clamp_min(1e-4)
    alpha = torch.minimum(alpha, max_alpha)

    depth = b_norm * torch.sin(gamma + alpha) / torch.sin(alpha).clamp_min(1e-5)
    depth = torch.nan_to_num(depth, nan=1.0, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
    return d * depth[:, None]


def _parallax_residual_from_xyz(xyz, uv, uvs_others, rel_Rs, rel_ts, f, centre, ref_weight):
    ref_res = (_project_safe(xyz, f, centre) - uv) * float(ref_weight)

    if uvs_others.shape[0] == 0:
        return ref_res

    xyz_other = torch.einsum("cni,cij->cnj", xyz[None] - rel_ts[:, None], rel_Rs)
    other_proj = _project_safe(xyz_other, f, centre)
    other_mask = (uvs_others >= 0).all(dim=-1)[..., None]
    other_res = torch.where(other_mask, other_proj - uvs_others, torch.zeros_like(other_proj))
    return torch.cat([ref_res, other_res.permute(1, 0, 2).reshape(xyz.shape[0], -1)], dim=-1)


def refine_points_parallax(
    uv,
    uvs_others,
    rel_Rts,
    f,
    centre,
    xyz_init,
    best_cam_idx,
    best_disambiguation,
    valid_mask,
    max_error,
    iters=4,
    ref_weight=0.25,
):
    """
    Point-only ParallaxBA refinement.

    Internally optimizes each point as (theta, phi, log(alpha)) against fixed
    camera poses, then returns xyz so the renderer/optimizer storage remains
    unchanged. This is the first low-risk step toward the parallax-angle Mini-BA
    flow described in Problem.md.
    """
    if xyz_init.numel() == 0 or rel_Rts.shape[0] == 0:
        return xyz_init, valid_mask

    active = (best_cam_idx >= 0) & torch.isfinite(xyz_init).all(dim=-1)
    active &= torch.linalg.vector_norm(xyz_init, dim=-1) > 1e-6
    if not active.any():
        return xyz_init, valid_mask

    rel_Rs = rel_Rts[:, :3, :3]
    rel_ts = rel_Rts[:, :3, 3]
    selected_baseline = rel_ts[best_cam_idx.clamp_min(0)]
    baseline_ok = torch.linalg.vector_norm(selected_baseline, dim=-1) > 1e-6
    active &= baseline_ok
    if not active.any():
        return xyz_init, valid_mask

    d_init = _safe_unit(xyz_init)
    angles = _direction_to_angles(d_init)
    cam_to_point = _safe_unit(xyz_init - selected_baseline)
    alpha = torch.acos((-d_init * cam_to_point).sum(dim=-1).clamp(-1.0 + 1e-5, 1.0 - 1e-5))
    alpha = torch.nan_to_num(alpha, nan=1e-3, posinf=1e-3, neginf=1e-3).clamp(1e-5, math.pi - 1e-4)
    state = torch.cat([angles, alpha.log().unsqueeze(-1)], dim=-1)

    residual_init = _parallax_residual_from_xyz(
        xyz_init, uv, uvs_others, rel_Rs, rel_ts, f, centre, ref_weight
    )
    err_init = residual_init.square().mean(dim=-1)
    lm = 1e-3
    eps = torch.tensor([1e-3, 1e-3, 1e-2], device=state.device, dtype=state.dtype)

    state = state.clone()
    for _ in range(max(int(iters), 0)):
        xyz = _parallax_state_to_xyz(state, selected_baseline)
        residual = _parallax_residual_from_xyz(
            xyz, uv, uvs_others, rel_Rs, rel_ts, f, centre, ref_weight
        )

        jac_cols = []
        for param_id in range(3):
            state_eps = state.clone()
            state_eps[:, param_id] += eps[param_id]
            xyz_eps = _parallax_state_to_xyz(state_eps, selected_baseline)
            residual_eps = _parallax_residual_from_xyz(
                xyz_eps, uv, uvs_others, rel_Rs, rel_ts, f, centre, ref_weight
            )
            jac_cols.append((residual_eps - residual) / eps[param_id])
        J = torch.stack(jac_cols, dim=-1)

        H = torch.bmm(J.transpose(1, 2), J)
        H.diagonal(dim1=-2, dim2=-1).add_(lm)
        g = torch.bmm(J.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
        delta = torch.linalg.solve_ex(H, g.unsqueeze(-1))[0].squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.25, 0.25)

        candidate = state - delta
        xyz_candidate = _parallax_state_to_xyz(candidate, selected_baseline)
        residual_candidate = _parallax_residual_from_xyz(
            xyz_candidate, uv, uvs_others, rel_Rs, rel_ts, f, centre, ref_weight
        )
        improved = residual_candidate.square().mean(dim=-1) <= residual.square().mean(dim=-1)
        state = torch.where((active & improved)[:, None], candidate, state)

    xyz_refined = _parallax_state_to_xyz(state, selected_baseline)
    residual_refined = _parallax_residual_from_xyz(
        xyz_refined, uv, uvs_others, rel_Rs, rel_ts, f, centre, ref_weight
    )
    err_refined = residual_refined.square().mean(dim=-1)
    accept = active & torch.isfinite(xyz_refined).all(dim=-1)
    accept &= xyz_refined[:, 2] > 1e-6
    accept &= err_refined <= torch.maximum(err_init, max_error.square())

    low_parallax_ok = best_disambiguation > (max_error * 20.0)
    refined_norm = torch.linalg.vector_norm(xyz_refined, dim=-1)
    init_norm = torch.linalg.vector_norm(xyz_init, dim=-1)
    bounded_refined = (xyz_refined[:, 2] > 1e-6) & (xyz_refined[:, 2] < 1e4) & (refined_norm < 1e4)
    bounded_init = (xyz_init[:, 2] > 1e-6) & (xyz_init[:, 2] < 1e4) & (init_norm < 1e4)
    accept &= low_parallax_ok & bounded_refined
    accept_direct = active & low_parallax_ok & bounded_init & (err_init <= max_error.square())
    xyz_out = torch.where(accept[:, None], xyz_refined, xyz_init)
    valid_out = valid_mask | accept | accept_direct
    return xyz_out, valid_out

class TriangulatorInternal(nn.Module):
    def __init__(self, use_parallax_ba=False, parallax_iters=4, parallax_ref_weight=0.25):
        super().__init__()
        self.use_parallax_ba = bool(use_parallax_ba)
        self.parallax_iters = int(parallax_iters)
        self.parallax_ref_weight = float(parallax_ref_weight)

    @torch.no_grad()
    def forward(self, uv, uvs_others, Rt, Rts_others, f, centre, max_error, min_dis):
        n_pts = uv.shape[0]
        kpts3d = torch.zeros(n_pts, 3, device="cuda")
        best_disambiguation = torch.zeros(n_pts, device="cuda")
        best_cam_idx = -torch.ones(n_pts, device="cuda", dtype=torch.long)
        Rts_others_inv = torch.linalg.inv_ex(Rts_others)[0]
        rel_Rts = torch.empty_like(Rts_others)

        for cam_idx in range(uvs_others.shape[0]):
            uv_other = uvs_others[cam_idx]
            Rt_other_inv = Rts_others_inv[cam_idx]
            rel_Rt = Rt @ Rt_other_inv
            rel_Rts[cam_idx] = rel_Rt
            kpts3dTmp, disTmp, error = matches_to_points(uv, uv_other, rel_Rt[:3, :3], rel_Rt[:3, 3], f, centre)
            validMask = (kpts3dTmp[:, 2] > 1e-6) * (disTmp > best_disambiguation) * (error < max_error)
            validMask *= uv_other.min(dim=-1).values > 0

            kpts3d = torch.where(validMask.unsqueeze(-1), kpts3dTmp, kpts3d)
            best_disambiguation = torch.where(validMask, disTmp, best_disambiguation)
            best_cam_idx = torch.where(validMask, torch.full_like(best_cam_idx, cam_idx), best_cam_idx)

        valid = best_disambiguation > min_dis
        if self.use_parallax_ba:
            kpts3d, valid = refine_points_parallax(
                uv,
                uvs_others,
                rel_Rts,
                f,
                centre,
                kpts3d,
                best_cam_idx,
                best_disambiguation,
                valid,
                max_error,
                iters=self.parallax_iters,
                ref_weight=self.parallax_ref_weight,
            )

        depth = kpts3d[:, 2].clone()
        kpts3d = (kpts3d - Rt[None, :3, 3]) @ Rt[:3, :3]
        return kpts3d, depth, best_disambiguation, valid
    
class Triangulator():
    @torch.no_grad()
    def __init__(
        self,
        n_pts,
        n_cams,
        max_error,
        use_parallax_ba=False,
        parallax_iters=4,
        parallax_ref_weight=0.25,
    ):
        self.n_cams = n_cams
        self.model = TriangulatorInternal(
            use_parallax_ba=use_parallax_ba,
            parallax_iters=parallax_iters,
            parallax_ref_weight=parallax_ref_weight,
        ).eval().cuda()
        uv = torch.rand(n_pts, 2, device="cuda")
        uvs_others = torch.rand(n_cams, n_pts, 2, device="cuda")
        Rt = torch.eye(4, device="cuda")
        Rts_others = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        f = torch.rand(1, device="cuda") + 1e-1 # avoid zero focal length
        centre = torch.rand(2, device="cuda")
        self.max_error = torch.tensor(max_error, device="cuda")
        self.min_dis = torch.tensor(max_error * 30, device="cuda")

        use_cuda_graph = (
            not use_parallax_ba
            and os.environ.get("OTFNVS_TRIANGULATOR_CUDA_GRAPH", "1") not in ("0", "false", "False")
        )
        if use_cuda_graph:
            try:
                self.model = torch.cuda.make_graphed_callables(
                    self.model, (uv, uvs_others, Rt, Rts_others, f, centre, self.max_error, self.min_dis)
                )
            except RuntimeError as error:
                warnings.warn(
                    "Triangulator CUDA graph capture failed; falling back to eager execution. "
                    f"Set OTFNVS_TRIANGULATOR_CUDA_GRAPH=0 to disable graph capture. Error: {error}",
                    RuntimeWarning,
                )

    def __call__(self, uv, uvs_others, Rt, Rts_others, f, centre):
        return self.model(uv, uvs_others, Rt, Rts_others, f, centre, self.max_error, self.min_dis)
    
    def prepare_matches(self, desc_kpts: DescribedKeypoints):
        """
        Organize the sets of matches for the triangulation.
        Select the top matches based on the number of matches available for each keyframe.
        """
        uv = desc_kpts.kpts
        uvs_others = -torch.ones(self.n_cams, uv.shape[0], 2, device="cuda")
        n_matches = torch.tensor([matches.idx.shape[0] for matches in desc_kpts.matches.values()])
        kf_indices = torch.tensor(list(desc_kpts.matches.keys()))
        chosen_ids = torch.topk(n_matches, min(self.n_cams, n_matches.shape[0])).indices
        chosen_kfs_ids = kf_indices[chosen_ids].tolist()

        for i, index in enumerate(chosen_kfs_ids):
            matches = desc_kpts.matches[index]
            uvs_others[i, matches.idx, :] = matches.kpts_other
        return uv, uvs_others, chosen_kfs_ids
