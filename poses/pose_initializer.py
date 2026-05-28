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
import math
import csv
import os
from typing import Optional

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from poses.velocity_estimator import LSFVelocityEstimator
from poses.vggt_pose_prior import VGGTPosePrior
from utils import fov2focal, depth2points, sixD2mtx
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType


def build_lsf_weights(order: int, window: int, device=None, dtype=torch.float32):
    # Backward-compatible helper; core implementation lives in velocity_estimator.py
    return LSFVelocityEstimator.build_weights(order, window, device=device, dtype=dtype)


def lsf_velocity_from_centers(centers: torch.Tensor, weights: torch.Tensor, dt: float = 1.0):
    # Backward-compatible helper; core implementation lives in velocity_estimator.py
    return LSFVelocityEstimator.velocity_from_centers(centers, weights, dt=dt)


class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers

        # Initialize the focal length
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)

        # Failure logging
        self.failure_log = []
        self.last_failure_reason = ""
        self.last_velocity_debug = {
            "velocity_lsf_norm": 0.0,
            "velocity_curr_norm": 0.0,
            "velocity_jump_ratio": 0.0,
            "velocity_angle_deg": 0.0,
        }
        self.last_geom_debug = {
            "total_2d3d_matches": 0,
            "pnp_inliers": 0,
            "miniba_inliers": 0,
            "residual": 0.0,
        }
        self.last_geom_valid_Rt = None

        # Camera track extrapolation settings
        self.use_track_extrapolation = getattr(args, 'use_track_extrapolation', False)
        self.track_extrapolation_window = getattr(args, 'track_extrapolation_window', 3)
        self.use_vggt_pose_prior = bool(getattr(args, "use_vggt_pose_prior", False))
        self.vggt_pose_prior_path = str(getattr(args, "vggt_pose_prior_path", ""))
        self.vggt_pose_prior = None
        if self.use_vggt_pose_prior and self.vggt_pose_prior_path:
            try:
                self.vggt_pose_prior = VGGTPosePrior(self.vggt_pose_prior_path, device="cuda")
                print(
                    f"[PosePrior] Loaded VGGT trajectory prior from {self.vggt_pose_prior_path}"
                )
                print(
                    f"[PosePrior] Intrinsics available: {self.vggt_pose_prior.has_intrinsics()}"
                )
            except Exception as e:
                print(f"[PosePrior] Failed to load VGGT pose prior: {e}")
                self.vggt_pose_prior = None

        # LSF velocity-gate settings
        self.pose_use_lsf_velocity_gate = getattr(args, "pose_use_lsf_velocity_gate", False)
        self.pose_lsf_order = int(getattr(args, "pose_lsf_order", 2))
        self.pose_lsf_window = int(getattr(args, "pose_lsf_window", 8))
        self.pose_jump_max_ratio = float(getattr(args, "pose_jump_max_ratio", 0.35))
        self.pose_vel_angle_max_deg = float(getattr(args, "pose_vel_angle_max_deg", 25.0))
        self.pose_lsf_force_accept_after = int(getattr(args, "pose_lsf_force_accept_after", 0))
        self.lsf_consecutive_rejections = 0
        self.lsf_estimator = None
        if self.pose_use_lsf_velocity_gate:
            self.lsf_estimator = LSFVelocityEstimator(
                order=self.pose_lsf_order,
                window=self.pose_lsf_window,
                dt=1.0,
                device="cuda",
                dtype=torch.float32,
            )

    def _extrapolate_pose(self, keyframes, frame_uid: Optional[int] = None):
        """
        Extrapolate the next camera pose from recent keyframe trajectory.
        Returns (Rs6D_init, ts_init) as initial values for PnP.
        Falls back to keyframes[0] pose if extrapolation is unreliable.
        """
        if self.vggt_pose_prior is not None and frame_uid is not None:
            Rt_prior = self.vggt_pose_prior.get_rt(int(frame_uid))
            if Rt_prior is not None:
                return Rt_prior[:3, :2], Rt_prior[:3, 3]

        if not self.use_track_extrapolation or len(keyframes) < 2:
            return keyframes[0].rW2C, keyframes[0].tW2C

        try:
            # Collect recent poses (sorted by proximity, take first few)
            n = min(self.track_extrapolation_window, len(keyframes))
            recent_kfs = keyframes[:n]

            # Get translations and rotations
            ts = torch.stack([kf.tW2C for kf in recent_kfs])  # [n, 3]
            Rs = torch.stack([sixD2mtx(kf.rW2C[None])[0] for kf in recent_kfs])  # [n, 3, 3]

            # Linear velocity extrapolation on translation
            if n >= 2:
                # Use the two closest keyframes to estimate velocity
                v = ts[0] - ts[1]  # velocity from second-closest to closest
                t_pred = ts[0] + v
            else:
                t_pred = ts[0]

            # Rotation extrapolation: R_pred = R_delta @ R_closest
            if n >= 2:
                R_delta = Rs[0] @ Rs[1].T
                R_pred = R_delta @ Rs[0]
                # Force R_pred back onto SO(3) manifold via QR decomposition
                # Without this, floating-point drift produces non-orthogonal matrices → NaN xyz
                Q, R_qr = torch.linalg.qr(R_pred)
                R_pred = Q * torch.sign(torch.diag(R_qr)).unsqueeze(0)
            else:
                R_pred = Rs[0]

            # Sanity check: if the predicted translation is too far, fallback
            pred_dist = torch.norm(t_pred - ts[0])
            avg_dist = torch.norm(ts[0] - ts[min(1, n - 1)])
            if pred_dist > 5 * avg_dist + 1e-6:
                return keyframes[0].rW2C, keyframes[0].tW2C

            # Check for NaN/Inf in the extrapolated pose
            if not torch.isfinite(R_pred).all() or not torch.isfinite(t_pred).all():
                return keyframes[0].rW2C, keyframes[0].tW2C

            # Convert R_pred back to 6D representation (first two columns)
            Rs6D_pred = R_pred[:, :2]  # [3, 2]
            return Rs6D_pred, t_pred

        except Exception:
            return keyframes[0].rW2C, keyframes[0].tW2C

    def _get_vggt_intrinsics_prior(self, frame_uid: Optional[int] = None):
        if self.vggt_pose_prior is None or frame_uid is None:
            return None
        intr = self.vggt_pose_prior.get_intrinsics(
            int(frame_uid),
            target_width=self.width,
            target_height=self.height,
        )
        return intr

    def get_vggt_pose_prior_rt(self, frame_uid: Optional[int] = None):
        if self.vggt_pose_prior is None or frame_uid is None:
            return None
        return self.vggt_pose_prior.get_rt(int(frame_uid))

    def get_vggt_intrinsics_prior(self, frame_uid: Optional[int] = None):
        return self._get_vggt_intrinsics_prior(frame_uid)

    def _lock_translation_to_vggt_prior(self, Rt: torch.Tensor, frame_uid: Optional[int] = None) -> torch.Tensor:
        prior_Rt = self.get_vggt_pose_prior_rt(frame_uid)
        if prior_Rt is None:
            return Rt
        prior_centre = self._camera_center_from_rt(prior_Rt).to(device=Rt.device, dtype=Rt.dtype)
        if not torch.isfinite(prior_centre).all():
            return Rt
        locked = Rt.detach().clone()
        locked[:3, 3] = -(locked[:3, :3] @ prior_centre)
        return locked

    @staticmethod
    def _camera_center_from_rt(Rt: torch.Tensor):
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        return -R.T @ t

    def _collect_mixed_lsf_centers(self, prev_keyframes: list[Keyframe], all_keyframes: list[Keyframe] = None):
        """
        Mixed-window policy:
        1) temporal window from latest contiguous registered keyframes
        2) supplement with high-match prev_keyframes if temporal window is short
        Returns centers sorted by keyframe index (oldest -> newest).
        """
        target_M = self.pose_lsf_window
        selected = {}

        if all_keyframes is None:
            all_keyframes = prev_keyframes

        # Primary temporal window
        temporal = sorted(all_keyframes, key=lambda kf: kf.index)
        temporal = temporal[-target_M:]
        for kf in temporal:
            selected[kf.index] = kf

        # Secondary match-based supplement
        if len(selected) < target_M:
            for kf in prev_keyframes:
                if kf.index not in selected:
                    selected[kf.index] = kf
                if len(selected) >= target_M:
                    break

        selected_kfs = sorted(selected.values(), key=lambda kf: kf.index)
        if len(selected_kfs) > target_M:
            selected_kfs = selected_kfs[-target_M:]

        if len(selected_kfs) == 0:
            return torch.empty(0, 3, device="cuda")

        centers = torch.stack(
            [self._camera_center_from_rt(kf.get_Rt().detach()) for kf in selected_kfs],
            dim=0,
        )
        return centers

    def _lsf_velocity(self, centers: torch.Tensor, dt: float = 1.0):
        if self.lsf_estimator is None:
            return torch.zeros(3, device=centers.device, dtype=centers.dtype)
        return self.lsf_estimator.estimate(centers, dt=dt)

    def _velocity_consistency_gate(
        self,
        Rt_candidate: torch.Tensor,
        prev_keyframes: list[Keyframe],
        all_keyframes: list[Keyframe] = None,
        dt: float = 1.0,
    ):
        metrics = {
            "velocity_lsf_norm": 0.0,
            "velocity_curr_norm": 0.0,
            "velocity_jump_ratio": 0.0,
            "velocity_angle_deg": 0.0,
        }
        if not self.pose_use_lsf_velocity_gate:
            return True, metrics
        if all_keyframes is None:
            all_keyframes = prev_keyframes
        if len(all_keyframes) == 0:
            return True, metrics

        centers = self._collect_mixed_lsf_centers(prev_keyframes, all_keyframes)
        if centers.shape[0] <= self.pose_lsf_order + 1:
            return True, metrics

        c_last = self._camera_center_from_rt(all_keyframes[-1].get_Rt().detach())
        c_cand = self._camera_center_from_rt(Rt_candidate)
        metrics.update(
            self.lsf_estimator.compare_with_current(
                centers=centers,
                c_last=c_last,
                c_candidate=c_cand,
                dt=dt,
            )
        )
        jump_ratio = metrics["velocity_jump_ratio"]
        angle_deg = metrics["velocity_angle_deg"]
        passed = (
            jump_ratio <= self.pose_jump_max_ratio
            and angle_deg <= self.pose_vel_angle_max_deg
        )
        return passed, metrics

    def _log_failure(
        self,
        frame_index: int,
        total_2d3d_matches: int,
        pnp_inliers: int,
        miniba_inliers: int,
        residual: float,
        threshold: int,
        reason_code: str,
        retry_count: int = 0,
        velocity_debug: Optional[dict] = None,
    ):
        if velocity_debug is None:
            velocity_debug = self.last_velocity_debug
        self.failure_log.append(
            {
                "frame_index": frame_index,
                "total_2d3d_matches": total_2d3d_matches,
                "pnp_inliers": pnp_inliers,
                "miniba_inliers": miniba_inliers,
                "residual": residual,
                "threshold": threshold,
                "velocity_lsf_norm": float(velocity_debug.get("velocity_lsf_norm", 0.0)),
                "velocity_curr_norm": float(velocity_debug.get("velocity_curr_norm", 0.0)),
                "velocity_jump_ratio": float(velocity_debug.get("velocity_jump_ratio", 0.0)),
                "velocity_angle_deg": float(velocity_debug.get("velocity_angle_deg", 0.0)),
                "reason_code": reason_code,
                "retry_count": int(retry_count),
            }
        )

    def save_failure_log(self, path: str):
        """Save failure log to CSV file."""
        if not self.failure_log:
            return
        csv_path = os.path.join(path, "failure_log.csv")
        os.makedirs(path, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_index",
                    "total_2d3d_matches",
                    "pnp_inliers",
                    "miniba_inliers",
                    "residual",
                    "threshold",
                    "velocity_lsf_norm",
                    "velocity_curr_norm",
                    "velocity_jump_ratio",
                    "velocity_angle_deg",
                    "reason_code",
                    "retry_count",
                ],
            )
            writer.writeheader()
            writer.writerows(self.failure_log)
        print(f"Failure log saved to {csv_path} ({len(self.failure_log)} entries)")

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        num_kpts_per_cam = [desc_kpts.kpts.shape[0] for desc_kpts in desc_kpts_list]
        unused_kpts_mask = [
            torch.ones(num_kpts, device='cuda', dtype=torch.bool)
            for num_kpts in num_kpts_per_cam
        ]
        for k in range(n_primary_cam):
            num_kpts_k = num_kpts_per_cam[k]
            idx_occurrences = torch.zeros(num_kpts_k, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            n_selected = min(npts_per_primary_cam, int(idx_occurrences.sum().item()))
            selected_indices = torch.multinomial(idx_occurrences.float(), n_selected, replacement=False)

            selected_mask = torch.zeros(num_kpts_k, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(n_selected, device="cuda")
            all_aligned_ids = torch.zeros(num_kpts_k, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:n_selected, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:n_selected, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l][idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(num_kpts_per_cam[l], device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(num_kpts_per_cam[l], device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list: list[DescribedKeypoints], rebooting=False):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list. 
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        f_init = (torch.tensor([self.f_init], device="cuda"))
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ## Scale to 0.1 average translation
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
        all_keyframes: list[Keyframe] = None,
        retry_count: int = 0,
        frame_uid: Optional[int] = None,
    ):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        # `index` is the prospective keyframe slot id and must be used for match-cache keys.
        # `frame_uid` is only for diagnostics/logging to avoid repeated frame-id confusion.
        kf_match_id = int(index)
        frame_log_id = int(frame_uid) if frame_uid is not None else int(index)
        self.last_failure_reason = ""
        self.last_velocity_debug = {
            "velocity_lsf_norm": 0.0,
            "velocity_curr_norm": 0.0,
            "velocity_jump_ratio": 0.0,
            "velocity_angle_deg": 0.0,
        }
        self.last_geom_debug = {
            "total_2d3d_matches": 0,
            "pnp_inliers": 0,
            "miniba_inliers": 0,
            "residual": 0.0,
        }
        self.last_geom_valid_Rt = None
        if all_keyframes is None:
            all_keyframes = keyframes

        if len(keyframes) == 0:
            self.last_failure_reason = "no_prev_keyframes"
            self._log_failure(
                frame_index=frame_log_id,
                total_2d3d_matches=0,
                pnp_inliers=0,
                miniba_inliers=0,
                residual=0.0,
                threshold=self.min_num_inliers,
                reason_code=self.last_failure_reason,
                retry_count=retry_count,
            )
            return None

        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        for keyframe in keyframes:
            matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=kf_match_id, kID_other=keyframe.index)

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        if len(xyz) == 0:
            self.last_failure_reason = "no_2d3d_matches"
            self._log_failure(
                frame_index=frame_log_id,
                total_2d3d_matches=0,
                pnp_inliers=0,
                miniba_inliers=0,
                residual=0.0,
                threshold=self.min_num_inliers,
                reason_code=self.last_failure_reason,
                retry_count=retry_count,
            )
            return None

        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)
        total_matches_count = len(xyz)

        # P4P needs at least 4 2D-3D correspondences.
        if total_matches_count < self.PnPRANSAC.m:
            self.last_failure_reason = "insufficient_pnp_points"
            self._log_failure(
                frame_index=frame_log_id,
                total_2d3d_matches=total_matches_count,
                pnp_inliers=0,
                miniba_inliers=0,
                residual=0.0,
                threshold=self.min_num_inliers,
                reason_code=self.last_failure_reason,
                retry_count=retry_count,
            )
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(kf_match_id, None)
            return None

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        Rs6D_init, ts_init = self._extrapolate_pose(keyframes, frame_uid=frame_uid)
        focal_for_frame = self.f
        centre_for_frame = self.centre
        intr_prior = self._get_vggt_intrinsics_prior(frame_uid=frame_uid)
        if intr_prior is not None:
            focal_for_frame = intr_prior["focal"]
            centre_for_frame = intr_prior["centre"]

        Rt, inliers = self.PnPRANSAC(
            uvs,
            xyz,
            focal_for_frame,
            centre_for_frame,
            Rs6D_init,
            ts_init,
            confs,
        )
        pnp_inliers = int(inliers.sum().item())

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        if len(xyz) == 0:
            self.last_failure_reason = "no_pnp_inliers"
            self._log_failure(
                frame_index=frame_log_id,
                total_2d3d_matches=total_matches_count,
                pnp_inliers=pnp_inliers,
                miniba_inliers=0,
                residual=0.0,
                threshold=self.min_num_inliers,
                reason_code=self.last_failure_reason,
                retry_count=retry_count,
            )
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(kf_match_id, None)
            return None

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the initialization
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(
            Rs6D,
            ts,
            focal_for_frame,
            xyz_ba,
            centre_for_frame,
            uvs_ba.reshape(-1),
        )
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]
        Rt = self._lock_translation_to_vggt_prior(Rt, frame_uid)

        # Check if we have sufficiently many inliers
        n_inliers = mask.sum().item()
        # Compute per-inlier mean reprojection residual
        residual = (r * mask).abs().sum() / max(n_inliers, 1)
        residual_f = float(residual.item())
        self.last_geom_debug = {
            "total_2d3d_matches": int(total_matches_count),
            "pnp_inliers": int(pnp_inliers),
            "miniba_inliers": int(n_inliers),
            "residual": float(residual_f),
        }

        geom_ok = False
        if is_test or n_inliers > self.min_num_inliers:
            geom_ok = True
        elif n_inliers >= self.min_num_inliers // 2 and residual < self.max_pnp_error * 0.5:
            geom_ok = True
            # print(
            #     f"[Accepted low-inlier pose] frame {frame_log_id}: "
            #     f"inliers={n_inliers}, residual={residual_f:.2f}px (threshold: {self.min_num_inliers})"
            # )

        if not geom_ok:
            self.last_failure_reason = "low_inliers_or_high_residual"
            print(
                f"Too few inliers for pose initialization (frame {frame_log_id}): "
                f"inliers={n_inliers}, residual={residual_f:.2f}px"
            )
            self._log_failure(
                frame_index=frame_log_id,
                total_2d3d_matches=total_matches_count,
                pnp_inliers=pnp_inliers,
                miniba_inliers=int(n_inliers),
                residual=residual_f,
                threshold=self.min_num_inliers,
                reason_code=self.last_failure_reason,
                retry_count=retry_count,
            )
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(kf_match_id, None)
            return None

        # Geometry-valid pose candidate (before velocity gate), used for debug and queue-full rescue.
        self.last_geom_valid_Rt = Rt.detach().clone()

        lsf_ok, lsf_metrics = self._velocity_consistency_gate(
            Rt_candidate=Rt,
            prev_keyframes=keyframes,
            all_keyframes=all_keyframes,
            dt=1.0,
        )
        self.last_velocity_debug = lsf_metrics
        if not lsf_ok:
            self.lsf_consecutive_rejections += 1
            if (
                self.pose_lsf_force_accept_after > 0
                and self.lsf_consecutive_rejections >= self.pose_lsf_force_accept_after
            ):
                print(
                    f"[PoseLSF] force-accept frame {frame_log_id} after "
                    f"{self.lsf_consecutive_rejections} consecutive LSF rejections "
                    f"(jump={lsf_metrics['velocity_jump_ratio']:.3f}, "
                    f"angle={lsf_metrics['velocity_angle_deg']:.1f}deg)."
                )
                self.lsf_consecutive_rejections = 0
                return Rt
            self.last_failure_reason = "lsf_velocity_gate"
            self._log_failure(
                frame_index=frame_log_id,
                total_2d3d_matches=total_matches_count,
                pnp_inliers=pnp_inliers,
                miniba_inliers=int(n_inliers),
                residual=residual_f,
                threshold=self.min_num_inliers,
                reason_code=self.last_failure_reason,
                retry_count=retry_count,
                velocity_debug=lsf_metrics,
            )
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(kf_match_id, None)
            return None

        self.lsf_consecutive_rejections = 0
        return Rt
