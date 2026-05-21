from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from fused_ssim import fused_ssim
from dataloaders.read_write_model import write_model
from pipeline.reconstruction_controller import ReconstructionController
from scene.anchor_local_renderer import AnchorLocalRenderer, AnchorRenderResult
from scene.local_gaussian_model import GAUSSIAN_KEYS
from scene.tsdf_losses import anisotropy_regularization, tsdf_surface_loss


@dataclass
class ProgressiveLoss:
    total: torch.Tensor
    rgb: torch.Tensor
    ssim: torch.Tensor
    depth: torch.Tensor
    tsdf: torch.Tensor
    anisotropy: torch.Tensor
    depth_valid_pixels: torch.Tensor
    visible_gaussians: torch.Tensor
    render_coverage: torch.Tensor
    ssim_weight: torch.Tensor


class ProgressiveSceneModel:
    """Pipeline-owned scene model for anchor-local rendering and losses."""

    def __init__(
        self,
        controller: ReconstructionController,
        width: int,
        height: int,
        f: torch.Tensor | float,
        sh_degree: int = 3,
        lambda_dssim: float = 0.2,
        depth_loss_weight: float = 1e-2,
        depth_valid_epsilon: float = 1e-6,
        tsdf_loss_weight: float = 1e-3,
        anisotropy_loss_weight: float = 1e-4,
        max_gaussian_aspect_ratio: float = 8.0,
        max_rasterized_gaussians: int = 60_000,
        rgb_visible_weight: float = 0.85,
        ssim_min_coverage: float = 0.08,
        depth_conf_min: float = 0.35,
        robust_loss_epsilon: float = 1e-3,
        lr_by_name: dict[str, float] | None = None,
        device: str | torch.device = "cuda",
    ):
        self.controller = controller
        self.width = int(width)
        self.height = int(height)
        self.lambda_dssim = float(lambda_dssim)
        self.depth_loss_weight = float(depth_loss_weight)
        self.depth_valid_epsilon = float(depth_valid_epsilon)
        self.tsdf_loss_weight = float(tsdf_loss_weight)
        self.anisotropy_loss_weight = float(anisotropy_loss_weight)
        self.max_gaussian_aspect_ratio = float(max_gaussian_aspect_ratio)
        self.rgb_visible_weight = float(rgb_visible_weight)
        self.ssim_min_coverage = float(ssim_min_coverage)
        self.depth_conf_min = float(depth_conf_min)
        self.robust_loss_epsilon = float(robust_loss_epsilon)
        self.renderer = AnchorLocalRenderer(
            width,
            height,
            f,
            sh_degree=sh_degree,
            max_rasterized_gaussians=max_rasterized_gaussians,
            device=device,
        )
        self.lr_by_name = {} if lr_by_name is None else dict(lr_by_name)
        self.optimizers: dict[int, tuple[int, torch.optim.Optimizer]] = {}
        self.last_render_debug: dict = {}

    def update_intrinsics(self, f: torch.Tensor | float) -> None:
        self.renderer.update_intrinsics(f)

    def render_from_keyframe(self, keyframe, active_anchor_ids: list[int] | None = None) -> AnchorRenderResult:
        view_matrix = keyframe.get_Rt().transpose(0, 1)
        return self.render(view_matrix, keyframe.get_centre(approx=True), active_anchor_ids=active_anchor_ids)

    def render(
        self,
        view_matrix: torch.Tensor,
        cam_centre_world: torch.Tensor,
        active_anchor_ids: list[int] | None = None,
    ) -> AnchorRenderResult:
        if active_anchor_ids is None:
            active_anchor_ids = self.controller.update_active_set(cam_centre_world)
        anchors = [self.controller.anchors[int(anchor_id)] for anchor_id in active_anchor_ids]
        result = self.renderer.render(anchors, view_matrix)
        self.last_render_debug = self._render_debug(result)
        return result

    def loss_from_keyframe(self, keyframe, frame_state, active_anchor_ids: list[int] | None = None) -> ProgressiveLoss:
        result = self.render_from_keyframe(keyframe, active_anchor_ids=active_anchor_ids)
        gt_image = frame_state.image.to(result.render.device)
        mono_idepth = frame_state.mono_idepth.to(result.invdepth.device)
        mono_conf = frame_state.mono_depth_conf.to(result.invdepth.device)
        if mono_idepth.ndim == 4:
            mono_idepth = mono_idepth[0]
        if mono_conf.ndim == 4:
            mono_conf = mono_conf[0]
        if mono_idepth.shape[-2:] != result.invdepth.shape[-2:]:
            mono_idepth = F.interpolate(mono_idepth[None], result.invdepth.shape[-2:], mode="bilinear", align_corners=True)[0]
        if mono_conf.shape[-2:] != result.invdepth.shape[-2:]:
            mono_conf = F.interpolate(mono_conf[None], result.invdepth.shape[-2:], mode="bilinear", align_corners=True)[0]
        mask = None if frame_state.mask is None else frame_state.mask.to(result.render.device)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:1]
            mask = mask.to(result.render.device).float()
            supervision_mask = mask.clamp(0.0, 1.0)
        else:
            supervision_mask = torch.ones_like(result.invdepth)

        invdepth = result.invdepth
        visible_mask = (
            torch.isfinite(invdepth)
            & (invdepth > self.depth_valid_epsilon)
            & (supervision_mask > 0.5)
        )
        render_coverage = visible_mask.float().mean()
        rgb_loss = self._visibility_aware_rgb_loss(result.render, gt_image, supervision_mask, visible_mask)
        ssim_loss = 1 - fused_ssim((result.render * supervision_mask)[None], (gt_image * supervision_mask)[None])
        ssim_scale = (render_coverage / max(self.ssim_min_coverage, 1e-6)).clamp(0.0, 1.0).detach()
        effective_ssim_weight = torch.as_tensor(self.lambda_dssim, device=rgb_loss.device, dtype=rgb_loss.dtype) * ssim_scale
        valid_depth = (
            visible_mask
            & torch.isfinite(mono_idepth)
            & (mono_idepth > self.depth_valid_epsilon)
            & torch.isfinite(mono_conf)
            & (mono_conf > self.depth_conf_min)
        )
        if int(valid_depth.sum().item()) >= 16:
            depth_loss = self._normalized_inverse_depth_loss(invdepth, mono_idepth, valid_depth)
        else:
            depth_loss = invdepth.sum() * 0.0
        tsdf_loss, anisotropy_loss = self.anchor_regularization_losses(active_anchor_ids=active_anchor_ids)
        total = (
            effective_ssim_weight * ssim_loss
            + (1.0 - effective_ssim_weight) * rgb_loss
            + self.depth_loss_weight * depth_loss
            + self.tsdf_loss_weight * tsdf_loss
            + self.anisotropy_loss_weight * anisotropy_loss
        )
        return ProgressiveLoss(
            total,
            rgb_loss,
            ssim_loss,
            depth_loss,
            tsdf_loss,
            anisotropy_loss,
            valid_depth.sum().to(total.dtype),
            result.visibility_filter.sum().to(total.dtype),
            render_coverage.to(total.dtype),
            effective_ssim_weight.to(total.dtype),
        )

    def _visibility_aware_rgb_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        supervision_mask: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        diff = self._charbonnier(rendered - target)
        mask3 = supervision_mask.expand_as(diff)
        denom = mask3.sum().clamp_min(1.0)
        full_loss = (diff * mask3).sum() / denom
        visible3 = visible_mask.expand_as(diff)
        if visible3.any():
            visible_loss = diff[visible3].mean()
            visible_weight = min(max(self.rgb_visible_weight, 0.0), 1.0)
            return visible_weight * visible_loss + (1.0 - visible_weight) * full_loss
        return full_loss

    def _normalized_inverse_depth_loss(
        self,
        invdepth: torch.Tensor,
        mono_idepth: torch.Tensor,
        valid_depth: torch.Tensor,
    ) -> torch.Tensor:
        pred = invdepth[valid_depth]
        target = mono_idepth[valid_depth]
        pred_center = pred.detach().median()
        target_center = target.detach().median()
        pred_scale = (pred.detach() - pred_center).abs().median().clamp_min(self.depth_valid_epsilon)
        target_scale = (target.detach() - target_center).abs().median().clamp_min(self.depth_valid_epsilon)
        pred_norm = (pred - pred_center) / pred_scale
        target_norm = (target - target_center) / target_scale
        return self._charbonnier(pred_norm - target_norm).mean()

    def _charbonnier(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.square() + self.robust_loss_epsilon * self.robust_loss_epsilon)

    def optimization_step(self, keyframe, frame_state, anchor_id: int) -> dict | None:
        return self.optimization_step_multiview(anchor_id, [(keyframe, frame_state)])

    def optimization_step_multiview(self, anchor_id: int, view_items: list[tuple[object, object]]) -> dict | None:
        anchor = self.controller.anchors[int(anchor_id)]
        if anchor.gaussian_model.n == 0:
            return None
        view_items = [(kf, frame) for kf, frame in view_items if kf is not None and frame is not None]
        if len(view_items) == 0:
            return None
        optimizer = self._optimizer_for_anchor(anchor)
        optimizer.zero_grad(set_to_none=True)
        per_view_losses = [
            self.loss_from_keyframe(kf, frame, active_anchor_ids=[anchor.anchor_id])
            for kf, frame in view_items
        ]
        totals = torch.stack([loss.total for loss in per_view_losses])
        total_loss = totals.mean()
        if not torch.isfinite(total_loss.detach()).all():
            self.last_render_debug["nonfinite_loss"] = True
            return None
        if not total_loss.requires_grad:
            self.last_render_debug["loss_without_grad"] = True
            return self._loss_stats(self._average_losses(per_view_losses))
        try:
            total_loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            grad_stats = self._grad_stats(anchor)
            torch.nn.utils.clip_grad_norm_(self._anchor_parameters(anchor), max_norm=10.0)
            optimizer.step()
        except RuntimeError as exc:
            if not self._is_cuda_backward_error(exc):
                raise
            self.last_render_debug["backward_cuda_error"] = f"{type(exc).__name__}: {exc}"
            self.last_render_debug["backward_skipped_step"] = True
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return None
        self._sanitize_anchor_params(anchor)
        losses = self._average_losses(per_view_losses)
        stats = self._loss_stats(losses)
        stats.update(grad_stats)
        stats["num_views"] = int(len(view_items))
        self.last_render_debug["num_optimization_views"] = int(len(view_items))
        return stats

    def optimization_loop(
        self,
        keyframe,
        frame_state,
        anchor_id: int,
        n_iters: int,
        view_items: list[tuple[object, object]] | None = None,
    ) -> dict | None:
        if view_items is None:
            view_items = [(keyframe, frame_state)]
        records = []
        for _ in range(int(n_iters)):
            stats = self.optimization_step_multiview(anchor_id, view_items)
            if stats is not None:
                records.append(stats)
        if len(records) == 0:
            return None
        keys = records[0].keys()
        return {key: float(sum(row[key] for row in records) / len(records)) for key in keys}

    def anchor_regularization_losses(self, active_anchor_ids: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.renderer.device
        total_tsdf = torch.zeros((), device=device)
        total_anisotropy = torch.zeros((), device=device)
        count = 0
        if active_anchor_ids is None:
            anchors = self.controller.anchors
        else:
            anchors = [self.controller.anchors[int(anchor_id)] for anchor_id in active_anchor_ids]
        for anchor in anchors:
            params = anchor.gaussian_model.params
            if params["xyz"]["val"].shape[0] == 0:
                continue
            points_local = params["xyz"]["val"]
            tsdf_loss = tsdf_surface_loss(points_local, anchor.tsdf)
            query = anchor.tsdf.query(points_local)
            tsdf_mask = query.valid & (query.weight >= 1.0)
            if tsdf_mask.any():
                voxel_size = anchor.tsdf.base_voxel_size
                voxel_keys = torch.floor(points_local[tsdf_mask] / voxel_size).detach()
                voxel_centres = (voxel_keys + 0.5) * voxel_size
                tsdf_loss = tsdf_loss + 0.01 * (points_local[tsdf_mask] - voxel_centres).square().sum(dim=-1).mean()
            total_tsdf = total_tsdf + tsdf_loss
            total_anisotropy = total_anisotropy + anisotropy_regularization(
                params["scaling"]["val"], max_ratio=self.max_gaussian_aspect_ratio
            )
            count += 1
        if count == 0:
            return total_tsdf, total_anisotropy
        return total_tsdf / count, total_anisotropy / count

    @staticmethod
    def _average_losses(losses: list[ProgressiveLoss]) -> ProgressiveLoss:
        if len(losses) == 1:
            return losses[0]
        return ProgressiveLoss(
            total=torch.stack([loss.total for loss in losses]).mean(),
            rgb=torch.stack([loss.rgb for loss in losses]).mean(),
            ssim=torch.stack([loss.ssim for loss in losses]).mean(),
            depth=torch.stack([loss.depth for loss in losses]).mean(),
            tsdf=torch.stack([loss.tsdf for loss in losses]).mean(),
            anisotropy=torch.stack([loss.anisotropy for loss in losses]).mean(),
            depth_valid_pixels=torch.stack([loss.depth_valid_pixels for loss in losses]).mean(),
            visible_gaussians=torch.stack([loss.visible_gaussians for loss in losses]).mean(),
            render_coverage=torch.stack([loss.render_coverage for loss in losses]).mean(),
            ssim_weight=torch.stack([loss.ssim_weight for loss in losses]).mean(),
        )

    def state_summary(self) -> dict:
        anchor_summaries = []
        for anchor in self.controller.anchors:
            anchor_summaries.append(
                {
                    "anchor_id": int(anchor.anchor_id),
                    "s_anchor_to_world": float(anchor.s_anchor_to_world.detach().cpu().item()),
                    "num_gaussians": int(anchor.gaussian_model.n),
                    "num_keyframes": len(anchor.keyframe_ids),
                    "num_tsdf_voxels": int(anchor.tsdf.keys.shape[0]),
                }
            )
        return {
            "num_anchors": len(self.controller.anchors),
            "num_gaussians": sum(item["num_gaussians"] for item in anchor_summaries),
            "anchors": anchor_summaries,
            "last_render_debug": dict(self.last_render_debug),
        }

    @torch.no_grad()
    def save(self, path: str, keyframes: list, reconstruction_time: float = 0.0, n_frames: int = 0) -> dict:
        os.makedirs(path, exist_ok=True)
        pcd_path = os.path.join(path, "point_clouds")
        state_path = os.path.join(path, "anchor_states")
        tsdf_path = os.path.join(path, "tsdf")
        os.makedirs(pcd_path, exist_ok=True)
        os.makedirs(state_path, exist_ok=True)
        os.makedirs(tsdf_path, exist_ok=True)

        for anchor in self.controller.anchors:
            self.save_anchor(path, anchor)

        metrics = {
            "num anchors": len(self.controller.anchors),
            "num keyframes": len(keyframes),
            "num gaussians": sum(anchor.gaussian_model.n for anchor in self.controller.anchors),
        }
        if reconstruction_time > 0:
            metrics["time"] = float(reconstruction_time)
            if n_frames > 0:
                metrics["FPS"] = float(n_frames / reconstruction_time)

        metadata = {
            **metrics,
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.renderer.sh_degree,
                "f": self.renderer.f,
            },
            "anchors": [
                {
                    "anchor_id": int(anchor.anchor_id),
                    "R_anchor_to_world": anchor.R_anchor_to_world.detach().cpu().numpy().tolist(),
                    "t_anchor_to_world": anchor.t_anchor_to_world.detach().cpu().numpy().tolist(),
                    "s_anchor_to_world": float(anchor.s_anchor_to_world.detach().cpu().item()),
                    "keyframe_ids": list(anchor.keyframe_ids),
                    "num_gaussians": anchor.gaussian_model.n,
                    "num_tsdf_voxels": int(anchor.tsdf.keys.shape[0]),
                }
                for anchor in self.controller.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in keyframes],
            "graph": self.controller.graph.to_json() if hasattr(self.controller.graph, "to_json") else self.controller.state_summary(),
            "pose_graph_optimization": self.controller.pose_graph_summary()
            if hasattr(self.controller, "pose_graph_summary")
            else "not_implemented",
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        cameras = {}
        images = {}
        for index, keyframe in enumerate(keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics

    @torch.no_grad()
    def save_anchor(self, path: str, anchor) -> None:
        pcd_path = os.path.join(path, "point_clouds")
        state_path = os.path.join(path, "anchor_states")
        tsdf_path = os.path.join(path, "tsdf")
        os.makedirs(pcd_path, exist_ok=True)
        os.makedirs(state_path, exist_ok=True)
        os.makedirs(tsdf_path, exist_ok=True)
        self._save_anchor_ply(anchor, os.path.join(pcd_path, f"anchor_{anchor.anchor_id}.ply"))
        torch.save(
            {
                "anchor_id": int(anchor.anchor_id),
                "R_anchor_to_world": anchor.R_anchor_to_world.detach().cpu(),
                "t_anchor_to_world": anchor.t_anchor_to_world.detach().cpu(),
                "s_anchor_to_world": anchor.s_anchor_to_world.detach().cpu(),
                "keyframe_ids": list(anchor.keyframe_ids),
                "gaussian_params": {
                    key: value["val"].detach().cpu()
                    for key, value in anchor.gaussian_model.params.items()
                },
                "anchor_ids": None if anchor.gaussian_model.anchor_ids is None else anchor.gaussian_model.anchor_ids.detach().cpu(),
            },
            os.path.join(state_path, f"anchor_{anchor.anchor_id}.pt"),
        )
        torch.save(
            {
                "base_voxel_size": anchor.tsdf.base_voxel_size,
                "num_levels": anchor.tsdf.num_levels,
                "keys": anchor.tsdf.keys.detach().cpu(),
                "hashes": anchor.tsdf.hashes.detach().cpu(),
                "tsdf_mean": anchor.tsdf.tsdf_mean.detach().cpu(),
                "m2": anchor.tsdf.m2.detach().cpu(),
                "weight": anchor.tsdf.weight.detach().cpu(),
                "level": anchor.tsdf.level.detach().cpu(),
            },
            os.path.join(tsdf_path, f"anchor_{anchor.anchor_id}.pt"),
        )

    @torch.no_grad()
    def merge_anchor_gaussians(self, anchor, voxel_size: float, target_max: int = 0, max_rounds: int = 4) -> dict:
        if voxel_size <= 0 or anchor.gaussian_model.n == 0:
            return {"merged": False, "before": anchor.gaussian_model.n, "after": anchor.gaussian_model.n, "voxel_size": voxel_size}
        original = int(anchor.gaussian_model.n)
        before = original
        current_voxel = float(voxel_size)
        rounds = 0
        while rounds < int(max_rounds):
            after = self._merge_anchor_once(anchor, current_voxel)
            rounds += 1
            if int(target_max) <= 0 or after <= int(target_max) or after == before:
                break
            before = after
            current_voxel *= 2.0
        self.optimizers.pop(anchor.anchor_id, None)
        return {
            "merged": True,
            "before": int(original),
            "after": int(anchor.gaussian_model.n),
            "voxel_size": float(current_voxel),
            "rounds": int(rounds),
        }

    def _merge_anchor_once(self, anchor, voxel_size: float) -> int:
        params = anchor.gaussian_model.params
        xyz = params["xyz"]["val"]
        if xyz.shape[0] == 0:
            return 0
        keys = torch.floor(xyz / float(voxel_size)).to(torch.int64)
        _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
        n_out = int(counts.shape[0])
        if n_out == xyz.shape[0]:
            return int(xyz.shape[0])
        for key in GAUSSIAN_KEYS:
            value = params[key]["val"]
            out = torch.zeros((n_out, *value.shape[1:]), dtype=value.dtype, device=value.device)
            out.index_add_(0, inverse.to(value.device), value)
            denom = counts.to(value.device, value.dtype).clamp_min(1).reshape(-1, *([1] * (value.ndim - 1)))
            params[key]["val"] = (out / denom).contiguous()
        rot = params["rotation"]["val"]
        norm = torch.linalg.vector_norm(rot, dim=-1, keepdim=True).clamp_min(1e-8)
        params["rotation"]["val"] = (rot / norm).contiguous()
        if anchor.gaussian_model.anchor_ids is not None:
            anchor.gaussian_model.anchor_ids = torch.full(
                (n_out,),
                int(anchor.anchor_id),
                dtype=torch.long,
                device=anchor.gaussian_model.device,
            )
        return n_out

    def _save_anchor_ply(self, anchor, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        world_params = anchor.gaussian_model.world_params(
            anchor.R_anchor_to_world,
            anchor.t_anchor_to_world,
            anchor.s_anchor_to_world,
        )
        xyz = self._to_numpy(world_params["xyz"])
        normals = np.zeros_like(xyz)
        f_dc = self._to_numpy(world_params["f_dc"].detach().transpose(1, 2).flatten(start_dim=1))
        f_rest = self._to_numpy(world_params["f_rest"].detach().transpose(1, 2).flatten(start_dim=1))
        opacity = self._to_numpy(world_params["opacity"])
        scaling = self._to_numpy(world_params["scaling"])
        rotation = self._to_numpy(world_params["rotation"])
        names = self._ply_attribute_names(anchor)
        dtype_full = [(attribute, "f4") for attribute in names]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if xyz.shape[0] > 0:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scaling, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _ply_attribute_names(anchor) -> list[str]:
        params = anchor.gaussian_model.params
        names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(params["f_dc"]["val"].shape[2]):
            names.append(f"f_dc_{i}")
        for i in range(params["f_rest"]["val"].shape[1] * params["f_rest"]["val"].shape[2]):
            names.append(f"f_rest_{i}")
        names.append("opacity")
        for i in range(params["scaling"]["val"].shape[1]):
            names.append(f"scale_{i}")
        for i in range(params["rotation"]["val"].shape[1]):
            names.append(f"rot_{i}")
        return names

    def _optimizer_for_anchor(self, anchor) -> torch.optim.Optimizer:
        existing = self.optimizers.get(anchor.anchor_id)
        if existing is not None and existing[0] == anchor.gaussian_model.n:
            return existing[1]
        params = []
        for name, item in anchor.gaussian_model.params.items():
            value = item["val"]
            if value.numel() == 0:
                continue
            value.requires_grad_(True)
            params.append({"params": [value], "lr": self.lr_by_name.get(name, self._default_lr(name))})
        if len(params) == 0:
            raise RuntimeError(f"Anchor {anchor.anchor_id} has no optimizable Gaussian parameters")
        optimizer = torch.optim.Adam(params, betas=(0.5, 0.99), eps=1e-15)
        self.optimizers[anchor.anchor_id] = (anchor.gaussian_model.n, optimizer)
        return optimizer

    @staticmethod
    def _default_lr(name: str) -> float:
        return {
            "xyz": 5e-5,
            "f_dc": 5e-3,
            "f_rest": 2.5e-4,
            "opacity": 1e-1,
            "scaling": 1e-2,
            "rotation": 2e-3,
        }.get(name, 1e-3)

    @staticmethod
    def _anchor_parameters(anchor) -> list[torch.Tensor]:
        return [
            item["val"]
            for item in anchor.gaussian_model.params.values()
            if item["val"].requires_grad and item["val"].numel() > 0
        ]

    @torch.no_grad()
    def _sanitize_anchor_params(self, anchor) -> None:
        params = anchor.gaussian_model.params
        if anchor.gaussian_model.n == 0:
            return
        finite = anchor.gaussian_model.finite_mask()
        if not finite.all():
            anchor.gaussian_model.prune(finite)
            self.optimizers.pop(anchor.anchor_id, None)
        if anchor.gaussian_model.n == 0:
            return
        params["scaling"]["val"].clamp_(min=-20.0, max=8.0)
        params["opacity"]["val"].clamp_(min=-12.0, max=12.0)
        rot = params["rotation"]["val"]
        norm = torch.linalg.vector_norm(rot, dim=-1, keepdim=True)
        identity = torch.zeros_like(rot)
        identity[:, 0] = 1.0
        params["rotation"]["val"].copy_(torch.where(norm > 1e-8, rot / norm.clamp_min(1e-8), identity))

    @staticmethod
    def _loss_stats(losses: ProgressiveLoss) -> dict:
        return {
            "total": float(losses.total.detach().item()),
            "l1": float(losses.rgb.detach().item()),
            "rgb": float(losses.rgb.detach().item()),
            "ssim": float(losses.ssim.detach().item()),
            "depth": float(losses.depth.detach().item()),
            "tsdf": float(losses.tsdf.detach().item()),
            "anisotropy": float(losses.anisotropy.detach().item()),
            "depth_valid_pixels": float(losses.depth_valid_pixels.detach().item()),
            "visible_gaussians": float(losses.visible_gaussians.detach().item()),
            "render_coverage": float(losses.render_coverage.detach().item()),
            "ssim_weight": float(losses.ssim_weight.detach().item()),
        }

    @staticmethod
    def _grad_stats(anchor) -> dict:
        stats = {}
        for name, item in anchor.gaussian_model.params.items():
            grad = item["val"].grad
            if grad is None or grad.numel() == 0:
                stats[f"grad_{name}_mean"] = 0.0
                stats[f"grad_{name}_max"] = 0.0
                continue
            finite = torch.isfinite(grad)
            if not finite.any():
                stats[f"grad_{name}_mean"] = float("inf")
                stats[f"grad_{name}_max"] = float("inf")
                continue
            abs_grad = grad[finite].abs()
            stats[f"grad_{name}_mean"] = float(abs_grad.mean().detach().item())
            stats[f"grad_{name}_max"] = float(abs_grad.max().detach().item())
        return stats

    @staticmethod
    def _render_debug(result: AnchorRenderResult) -> dict:
        visible = result.visibility_filter
        visible_anchor_ids = result.anchor_ids[visible] if result.anchor_ids.numel() == visible.numel() else result.anchor_ids[:0]
        anchor_hist = {}
        if visible_anchor_ids.numel() > 0:
            ids, counts = torch.unique(visible_anchor_ids.detach().cpu(), return_counts=True)
            anchor_hist = {str(int(i.item())): int(c.item()) for i, c in zip(ids, counts)}
        return {
            "num_input_gaussians": int(result.anchor_ids.shape[0]),
            "num_visible_gaussians": int(visible.sum().item()),
            "num_positive_radii": int((result.radii > 0).sum().item()),
            "num_raster_limited": int(result.raster_limit_count),
            "guard_reason_counts": dict(result.guard_reason_counts),
            "visible_anchor_histogram": anchor_hist,
            "cuda_error": result.cuda_error,
        }

    @staticmethod
    def _is_cuda_backward_error(exc: RuntimeError) -> bool:
        text = str(exc).lower()
        return "cuda" in text or "invalid configuration argument" in text or "illegal memory access" in text
