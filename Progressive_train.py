from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm

from dataloaders.image_dataset import ImageDataset
from dataloaders.stream_dataset import StreamDataset
from pipeline.gaussian_spawn_policy import GaussianSpawnPolicy
from pipeline.keyframe_store import TrackingKeyframe, TrackingKeyframeStore
from pipeline.loop_closure_manager import LoopClosureManager
from pipeline.observation_builder import ObservationBuilder
from pipeline.place_recognition import AnchorDescriptorIndex, DINOv2GlobalDescriptorExtractor
from pipeline.progressive_config import ProgressiveConfig, parse_progressive_config
from pipeline.reconstruction_controller import MappingTask, ReconstructionController
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from poses.guided_mvs import GuidedMVS
from resource_tracker import ResourceTracker
from scene.dense_extractor import DenseExtractor
from scene.mono_depth import MonoDepthEstimator
from scene.progressive_scene_model import ProgressiveSceneModel
from scene.tsdf_fusion import TSDFFusion
from utils import RGB2SH, align_mean_up_fwd, depth2points, inverse_sigmoid, make_torch_sampler, pts2px


@dataclass
class ProgressiveState:
    n_keyframes: int = 0
    focal_px: float = 0.0
    needs_reboot: bool = False
    last_reboot: int = 0
    loss_step_idx: int = 0


def _parse_progressive_args(argv: list[str]) -> tuple[ProgressiveConfig, list[str]]:
    cfg = parse_progressive_config(argv)
    return cfg, [argv[0]]


def _append_loss_record(records: list[dict], stats: dict | None, phase: str, lod: int, step_idx: int) -> int:
    if stats is None:
        return step_idx
    records.append(
        {
            "step": step_idx,
            "phase": phase,
            "lod": int(lod),
            "total": float(stats["total"]),
            "l1": float(stats["l1"]),
            "ssim": float(stats["ssim"]),
            "depth": float(stats["depth"]),
            "tsdf": float(stats.get("tsdf", 0.0)),
            "anisotropy": float(stats.get("anisotropy", 0.0)),
            "depth_valid_pixels": float(stats.get("depth_valid_pixels", 0.0)),
            "visible_gaussians": float(stats.get("visible_gaussians", 0.0)),
            "render_coverage": float(stats.get("render_coverage", 0.0)),
            "ssim_weight": float(stats.get("ssim_weight", 0.0)),
            "num_views": float(stats.get("num_views", 0.0)),
        }
    )
    return step_idx + 1


def _save_loss_records_and_plot(records: list[dict], out_dir: str) -> None:
    if len(records) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_records.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "phase",
                "lod",
                "total",
                "l1",
                "ssim",
                "depth",
                "tsdf",
                "anisotropy",
                "depth_valid_pixels",
                "visible_gaussians",
                "render_coverage",
                "ssim_weight",
                "num_views",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [row["step"] for row in records]
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(steps, [row["total"] for row in records], color="tab:blue", linewidth=1.2, label="total")
        axes[0].set_ylabel("Total Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(steps, [row["l1"] for row in records], color="tab:orange", linewidth=1.0, label="l1")
        axes[1].plot(steps, [row["ssim"] for row in records], color="tab:green", linewidth=1.0, label="ssim")
        axes[1].plot(steps, [row["depth"] for row in records], color="tab:red", linewidth=1.0, label="depth")
        axes[1].set_ylabel("Component Loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[LossPlot] Skip plotting due to error: {exc}")


def _save_lod_completion_marker(out_dir: str, lod_step: int, reconstruction_time: float, lod_gate: dict, metrics: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    marker_path = os.path.join(out_dir, f"lod_{lod_step}_complete.json")
    payload = {
        "lod": int(lod_step),
        "reconstruction_time": float(reconstruction_time),
        "lod_gate": {
            key: (float(value) if isinstance(value, (float, int)) else value)
            for key, value in lod_gate.items()
        },
        "metrics": {
            key: (float(value) if isinstance(value, (float, int)) else value)
            for key, value in metrics.items()
        },
        "checkpoint_note": "single_lod_reuses_initial_full_scene_save",
    }
    with open(marker_path, "w") as f:
        json.dump(payload, f, indent=2)
    return marker_path


def _collect_output_status(model_path: str) -> dict:
    if not model_path:
        return {"model_path": "", "complete": False, "missing": ["model_path"]}
    checks = {
        "metadata_json": os.path.exists(os.path.join(model_path, "metadata.json")),
        "point_clouds_dir": os.path.isdir(os.path.join(model_path, "point_clouds")),
        "anchor_ply": len(glob.glob(os.path.join(model_path, "point_clouds", "anchor_*.ply"))) > 0,
        "anchor_states_dir": os.path.isdir(os.path.join(model_path, "anchor_states")),
        "anchor_state_pt": len(glob.glob(os.path.join(model_path, "anchor_states", "anchor_*.pt"))) > 0,
        "tsdf_dir": os.path.isdir(os.path.join(model_path, "tsdf")),
        "tsdf_pt": len(glob.glob(os.path.join(model_path, "tsdf", "anchor_*.pt"))) > 0,
        "colmap_dir": os.path.isdir(os.path.join(model_path, "colmap")),
        "colmap_cameras": os.path.exists(os.path.join(model_path, "colmap", "cameras.bin")),
        "colmap_images": os.path.exists(os.path.join(model_path, "colmap", "images.bin")),
        "resource_stats": os.path.exists(os.path.join(model_path, "resource_stats.txt")),
        "loss_records": os.path.exists(os.path.join(model_path, "loss_records.csv")),
        "loss_curve": os.path.exists(os.path.join(model_path, "loss_curve.png")),
        "test_renders_dir": os.path.isdir(os.path.join(model_path, "test_renders")),
        "test_render_images": len(glob.glob(os.path.join(model_path, "test_renders", "*_render.png"))) > 0,
        "lod_completion_marker": len(glob.glob(os.path.join(model_path, "lod_*_complete.json"))) > 0,
    }
    optional = {"loss_records", "loss_curve", "test_renders_dir", "test_render_images"}
    required_missing = [name for name, ok in checks.items() if not ok and name not in optional]
    return {
        "model_path": model_path,
        "checks": checks,
        "missing": required_missing,
        "complete": len(required_missing) == 0,
    }


def _recent_keyframe_centre_median(keyframes: list[TrackingKeyframe], fallback: torch.Tensor, window: int) -> torch.Tensor:
    recent = keyframes[-max(1, int(window)) :]
    centres = [
        keyframe.get_centre(approx=True).detach().to(device=fallback.device, dtype=fallback.dtype)
        for keyframe in recent
    ]
    centres = [centre for centre in centres if centre.shape == fallback.shape and torch.isfinite(centre).all()]
    if len(centres) == 0:
        return fallback.detach().clone()
    return torch.stack(centres, dim=0).median(dim=0).values


def _write_manifest(
    model_path: str,
    cfg: ProgressiveConfig,
    status: str,
    started_at: float,
    error: str = "",
    extra: dict | None = None,
) -> None:
    if not model_path:
        return
    os.makedirs(model_path, exist_ok=True)
    output_status = _collect_output_status(model_path)
    manifest_status = status
    if status == "completed" and not output_status["complete"]:
        manifest_status = "completed_with_missing_outputs"
    payload = {
        "run_label": cfg.run_label,
        "status": manifest_status,
        "started_at_unix": float(started_at),
        "finished_at_unix": float(time.time()),
        "elapsed_sec": float(time.time() - started_at),
        "progressive": {
            "backend_mode": cfg.backend_mode,
            "trainer_entrypoint": "Progressive_train.py",
            "overlap_mode": cfg.overlap_mode,
            "overlap_optimization": "reserved_not_implemented",
            "anchor_radius": cfg.anchor_radius,
            "anchor_min_keyframes": cfg.anchor_min_keyframes,
            "async_mapping": cfg.async_mapping,
            "tsdf_loss_weight": cfg.tsdf_loss_weight,
            "anisotropy_loss_weight": cfg.anisotropy_loss_weight,
            "local_spawn_max": cfg.local_spawn_max,
            "local_spawn_target": cfg.local_spawn_target,
            "surface_sample_floor": cfg.surface_sample_floor,
            "low_frequency_spawn_fraction": cfg.low_frequency_spawn_fraction,
            "edge_probability_threshold": cfg.edge_probability_threshold,
            "spawn_opacity_init": cfg.spawn_opacity_init,
            "mvs_depth_consistency_idepth": cfg.mvs_depth_consistency_idepth,
            "mvs_mono_fallback_fraction": cfg.mvs_mono_fallback_fraction,
            "mvs_mono_fallback_max_fraction": cfg.mvs_mono_fallback_max_fraction,
            "mvs_target_ratio": cfg.mvs_target_ratio,
            "mono_fallback_min_points": cfg.mono_fallback_min_points,
            "tsdf_fusion_mode": cfg.tsdf_fusion_mode,
            "delayed_tsdf_min_opacity": cfg.delayed_tsdf_min_opacity,
            "delayed_tsdf_max_samples": cfg.delayed_tsdf_max_samples,
            "max_rasterized_gaussians": cfg.max_rasterized_gaussians,
            "anchor_render_check_every": cfg.anchor_render_check_every,
            "anchor_iterations": cfg.anchor_iterations,
            "anchor_train_views": cfg.anchor_train_views,
            "save_test_renders": cfg.save_test_renders,
            "test_render_every": cfg.test_render_every,
            "test_render_dir": cfg.test_render_dir,
            "loop_max_candidates": cfg.loop_check_max_candidates,
            "loop_min_anchor_gap": cfg.loop_min_anchor_gap,
            "depth_valid_epsilon": cfg.depth_valid_epsilon,
            "pose_graph_optimization": "sim3_enabled_always",
            "completion_gate": "Progressive_train.py exits successfully and writes reconstruction outputs",
        },
        "progressive_config": cfg.to_manifest(),
        "outputs": output_status,
        "error": error,
    }
    if extra:
        payload["summary"] = extra
    with open(os.path.join(model_path, cfg.manifest_name), "w") as f:
        json.dump(payload, f, indent=2)


class ProgressiveTrainer:
    def __init__(self, args: ProgressiveConfig, cfg: ProgressiveConfig):
        self.args = args
        self.cfg = cfg
        self.dataset = None
        self.is_stream = False
        self.height = 0
        self.width = 0
        self.max_error = 0.0
        self.min_displacement = 0.0
        self.matcher = None
        self.triangulator = None
        self.pose_initializer = None
        self.dense_extractor = None
        self.depth_estimator = None
        self.observation_builder = None
        self.keyframe_store = None
        self.anchor_scene_model = None
        self.loop_manager = None
        self.place_index = None
        self.detector = None
        self.controller = None
        self.tsdf_fusion = TSDFFusion()
        self.spawn_policy = None
        self.tracker = ResourceTracker()
        self.state = ProgressiveState()
        self.metrics: dict = {}
        self.loss_records: list[dict] = []
        self.anchor_seal_events: list[dict] = []
        self.pending_pose_queue: deque[dict] = deque()
        self.bootstrap_keyframe_dicts: list[dict] = []
        self.bootstrap_frames: list = []
        self.bootstrap_desc_kpts: list = []
        self.prev_frame = None
        self.frame_states: dict[int, object] = {}
        self.current_lod = int(getattr(args, "lod_min", 1))
        self.saved_test_render_frames: set[int] = set()

    def initialize(self) -> None:
        if "://" in self.args.source_path:
            self.dataset = StreamDataset(self.args.source_path, self.args.downsampling)
            self.is_stream = True
        else:
            self.dataset = ImageDataset(self.args)
            self.is_stream = False

        self.height, self.width = self.dataset.get_image_size()
        self.max_error = max(self.args.match_max_error * self.width, 1.5)
        self.min_displacement = max(self.args.min_displacement * self.width, 30)

        print("Initializing Progressive modules and running just in time compilation, may take a while...")
        self.matcher = Matcher(
            self.args.fundmat_samples,
            self.max_error,
            sem_weight=self.args.sem_weight if self.args.use_semantic_features else 0.0,
            matcher_backend=self.args.matcher_backend,
            feature_backend=self.args.feature_backend,
            lightglue_filter_threshold=self.args.lightglue_filter_threshold,
            lightglue_depth_confidence=self.args.lightglue_depth_confidence,
            lightglue_width_confidence=self.args.lightglue_width_confidence,
        )
        self.triangulator = Triangulator(
            self.args.num_kpts,
            self.args.num_prev_keyframes_miniba_incr,
            self.max_error,
            use_parallax_ba=self.args.use_parallax_ba,
            parallax_iters=self.args.parallax_ba_iters,
            parallax_ref_weight=self.args.parallax_ref_weight,
        )
        self.pose_initializer = PoseInitializer(
            self.width, self.height, self.triangulator, self.matcher, 2 * self.max_error, self.args
        )
        self.state.focal_px = float(self.pose_initializer.f_init)
        self.dense_extractor = DenseExtractor(self.width, self.height)
        self.depth_estimator = MonoDepthEstimator(self.width, self.height)
        self.keyframe_store = TrackingKeyframeStore(
            self.width,
            self.height,
            self.args,
            self.matcher,
            self.triangulator,
            device="cuda",
        )
        self.keyframe_store.f = self.state.focal_px
        self.guided_mvs = GuidedMVS(self.args)
        semantic_extractor = None
        if self.args.use_semantic_features:
            from poses.semantic_extractor import SemanticExtractor

            semantic_extractor = SemanticExtractor(self.width, self.height, sem_dim=self.args.sem_feat_dim)
            print(f"Semantic features enabled (dim={self.args.sem_feat_dim}, weight={self.args.sem_weight})")
        self.detector = Detector(
            self.args.num_kpts,
            self.width,
            self.height,
            semantic_extractor=semantic_extractor,
            feature_backend=self.args.feature_backend,
        )
        self.observation_builder = ObservationBuilder(
            detector=self.detector,
            depth_estimator=self.depth_estimator,
            dense_extractor=self.dense_extractor,
            matcher=self.matcher,
        )
        self.controller = ReconstructionController(
            sh_degree=self.args.sh_degree,
            max_active_anchors=3,
            device="cuda",
            mapping_callback=self._mapping_step,
        )
        self.controller.create_anchor()
        self.place_index = AnchorDescriptorIndex(
            extractor=DINOv2GlobalDescriptorExtractor(
                descriptor_dim=64,
                use_dinov2=os.environ.get("PROGRESSIVE_USE_DINOV2", "1") != "0",
                device="cuda",
            ),
            top_k=self.cfg.loop_check_max_candidates,
        )
        self.loop_manager = LoopClosureManager(
            min_anchor_gap=self.cfg.loop_min_anchor_gap,
            max_candidates=self.cfg.loop_check_max_candidates,
            place_index=self.place_index,
        )
        self.anchor_scene_model = ProgressiveSceneModel(
            controller=self.controller,
            width=self.width,
            height=self.height,
            f=self.state.focal_px,
            sh_degree=self.args.sh_degree,
            lambda_dssim=self.args.lambda_dssim,
            depth_loss_weight=self.args.depth_loss_weight_init,
            depth_valid_epsilon=self.args.depth_valid_epsilon,
            tsdf_loss_weight=self.cfg.tsdf_loss_weight,
            anisotropy_loss_weight=self.cfg.anisotropy_loss_weight,
            max_gaussian_aspect_ratio=getattr(self.args, "max_gaussian_aspect_ratio", 8.0),
            max_rasterized_gaussians=self.cfg.max_rasterized_gaussians,
            rgb_visible_weight=self.cfg.rgb_visible_weight,
            ssim_min_coverage=self.cfg.ssim_min_coverage,
            depth_conf_min=self.cfg.depth_conf_min,
            robust_loss_epsilon=self.cfg.robust_loss_epsilon,
            lr_by_name={
                "xyz": self.args.position_lr_init,
                "f_dc": self.args.feature_lr,
                "f_rest": self.args.feature_lr / 20.0,
                "opacity": self.args.opacity_lr,
                "scaling": self.args.scaling_lr,
                "rotation": self.args.rotation_lr,
            },
        )
        if self.cfg.async_mapping:
            self.controller.start_mapping_worker()

    def run(self) -> dict:
        self.initialize()
        print(f"[ProgressiveTrain] backend={self.cfg.backend_mode}")
        print(
            "[ProgressiveTrain] overlap=reserved_not_implemented"
            if self.cfg.overlap_mode == "reserved"
            else "[ProgressiveTrain] overlap=off"
        )
        print(f"Starting Progressive reconstruction for {self.args.source_path}")
        reconstruction_start_time = time.time()
        pbar = tqdm(range(0, len(self.dataset)))

        for frame_id in pbar:
            self._process_frame(frame_id, pbar)

        reconstruction_time = time.time() - reconstruction_start_time
        return self._finalize(reconstruction_time)

    def _process_frame(self, frame_id: int, pbar) -> None:
        self.tracker.start("Load")
        image, info = self.dataset.getnext()
        frame = self.observation_builder.build(image, info, int(frame_id))
        info = frame.info
        desc_kpts = frame.desc_kpts
        self.frame_states[int(frame.frame_id)] = frame

        if self.state.n_keyframes == 0:
            self.bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            self.bootstrap_frames = [frame]
            self.bootstrap_desc_kpts = [desc_kpts]
            self.prev_frame = frame
            self.state.n_keyframes += 1
            self.tracker.stop()
            return

        curr_prev_matches = self.observation_builder.match(frame, self.prev_frame)
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        should_add_keyframe = (
            dist.median() > self.min_displacement
            and len(curr_prev_matches.kpts) > self.args.min_num_inliers
        )
        should_add_keyframe |= info["is_test"]
        self.tracker.stop()

        if not should_add_keyframe:
            should_add_keyframe = self._promote_loop_candidate_keyframe(frame)

        if should_add_keyframe:
            extra_registered = self._register_keyframe_candidate(frame)
            should_add_keyframe = extra_registered >= 0
        else:
            extra_registered = 0

        if should_add_keyframe:
            self.state.n_keyframes += 1 + extra_registered
            if not info["is_test"]:
                self.prev_frame = frame
            self._evaluate_and_checkpoint(frame_id)
            self._update_progress_bar(pbar)

    def _register_keyframe_candidate(self, frame) -> int:
        image, info, desc_kpts = frame.image, frame.info, frame.desc_kpts
        if self.state.n_keyframes < self.args.num_keyframes_miniba_bootstrap:
            self.bootstrap_keyframe_dicts.append({"image": image, "info": info})
            self.bootstrap_frames.append(frame)
            self.bootstrap_desc_kpts.append(desc_kpts)

        if self.state.n_keyframes == self.args.num_keyframes_miniba_bootstrap - 1:
            self._bootstrap_scene()
            return 0

        self._maybe_reboot()

        if self.state.n_keyframes >= self.args.num_keyframes_miniba_bootstrap:
            return self._register_incremental_keyframe(frame)
        return 0

    def _bootstrap_scene(self) -> None:
        with self.tracker.track("BAB"):
            Rts, f, _ = self.pose_initializer.initialize_bootstrap(self.bootstrap_desc_kpts)
            Rts, f = self._apply_vggt_bootstrap_prior(Rts, f)
            self.state.focal_px = float(f.detach().cpu().item())
            self.keyframe_store.f = self.state.focal_px
            self.anchor_scene_model.update_intrinsics(f)

        created_keyframes = []
        for index, (keyframe_dict, frame, desc_kpts, Rt) in enumerate(
            zip(self.bootstrap_keyframe_dicts, self.bootstrap_frames, self.bootstrap_desc_kpts, Rts)
        ):
            with self.tracker.track("Add"):
                focal = f
                if self.args.use_colmap_poses:
                    Rt = keyframe_dict["info"]["Rt"]
                    focal = keyframe_dict["info"]["focal"]
                keyframe = self._make_keyframe(frame, Rt, index, focal)
                created_keyframes.append((frame, keyframe))

        for frame, keyframe in created_keyframes:
            self._calibrate_keyframe_depth_from_triangulation(keyframe)
            self._attach_frame_to_controller(frame, keyframe)

        for index in range(self.args.num_keyframes_miniba_bootstrap):
            frame = self.bootstrap_frames[index]
            keyframe = self.keyframe_store.keyframes[index]
            self._optimize_anchor_scene(frame, keyframe, "anchor_bootstrap_opt")

        with self.tracker.track("Opt"):
            stats = self._optimize_scene("bootstrap_opt")
            self.state.loss_step_idx = _append_loss_record(
                self.loss_records,
                stats,
                phase="bootstrap_opt",
                lod=self.current_lod,
                step_idx=self.state.loss_step_idx,
            )
        self.state.last_reboot = self.state.n_keyframes

    def _apply_vggt_bootstrap_prior(self, Rts: torch.Tensor, focal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior_Rts = []
        prior_focal = None
        for frame in self.bootstrap_frames:
            frame_id = int(frame.frame_id)
            prior_Rt = self.pose_initializer.get_vggt_pose_prior_rt(frame_id)
            if prior_Rt is None or not torch.isfinite(prior_Rt).all():
                return Rts, focal
            prior_Rts.append(prior_Rt.to(device=Rts.device, dtype=Rts.dtype))
            if prior_focal is None:
                intrinsics = self.pose_initializer.get_vggt_intrinsics_prior(frame_id)
                if intrinsics is not None:
                    prior_focal = intrinsics["focal"].to(device=focal.device, dtype=focal.dtype)
        print(f"[PosePrior] Using VGGT metric bootstrap poses for {len(prior_Rts)} frames.")
        return torch.stack(prior_Rts, dim=0), focal if prior_focal is None else prior_focal

    def _maybe_reboot(self) -> None:
        if (
            self.args.enable_reboot
            and self.keyframe_store.approx_cam_centres is not None
            and len(self.keyframe_store.keyframes) >= 20
        ):
            last_centers = self.keyframe_store.approx_cam_centres[-20:]
            rel_dist = torch.norm(last_centers[1:] - last_centers[:-1], dim=-1).mean()
            self.state.needs_reboot = (
                rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3
            ) and self.state.n_keyframes - self.state.last_reboot > 50

        if not self.state.needs_reboot:
            return

        bs_kfs = self.keyframe_store.recent(8)
        bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
        in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
        Rts, _, final_residual = self.pose_initializer.initialize_bootstrap(bootstrap_desc_kpts, rebooting=True)
        if final_residual < self.max_error * 0.5:
            Rts = align_mean_up_fwd(Rts, in_Rts)
            for Rt, keyframe in zip(Rts, bs_kfs):
                keyframe.set_Rt(Rt)
            self.keyframe_store.refresh_after_pose_updates()
            self.state.needs_reboot = False
            self.state.last_reboot = self.state.n_keyframes

    def _register_incremental_keyframe(self, frame) -> int:
        image, info, desc_kpts = frame.image, frame.info, frame.desc_kpts
        with self.tracker.track("tri"):
            prev_keyframes = self.keyframe_store.get_prev_keyframes(
                self.args.num_prev_keyframes_miniba_incr, True, desc_kpts
            )
        with self.tracker.track("BAI"):
            Rt = self.pose_initializer.initialize_incremental(
                prev_keyframes,
                desc_kpts,
                self.state.n_keyframes,
                info["is_test"],
                image,
                all_keyframes=self.keyframe_store.keyframes,
                retry_count=0,
                frame_uid=info.get("frame_id"),
            )

        if Rt is None:
            Rt = self._try_global_relocalization(frame, desc_kpts)
            if Rt is None:
                self._queue_failed_pose_candidate(image, info, desc_kpts)
                return -1
            info["pose_source"] = "global_relocalization"

        self._add_initialized_keyframe(image, info, desc_kpts, Rt, self.state.n_keyframes, "incremental_opt")
        return self._retry_pending_keyframes()

    def _try_global_relocalization(self, frame, desc_kpts):
        if self.place_index is None or len(self.keyframe_store.keyframes) == 0:
            frame.info["relocalization"] = {
                "attempted": False,
                "accepted": False,
                "failure_reason": "empty_global_descriptor_index",
            }
            return None
        candidates = self.place_index.query_frame_for_relocalization(
            frame,
            top_k=self.args.relocalization_top_k,
            exclude_self_frame_id=frame.frame_id,
        )
        attempts = []
        selected_keyframes = []
        seen_indices = set()
        for candidate in candidates:
            keyframe = self.keyframe_store.by_index(candidate.keyframe_index)
            if keyframe is None or int(keyframe.index) in seen_indices:
                continue
            desc = keyframe.desc_kpts
            triangulated = 0
            if desc is not None and getattr(desc, "has_pt3d", None) is not None:
                triangulated = int(desc.has_pt3d.sum().detach().cpu().item())
            attempt = {
                "retrieval_backend": "dinov2_vlad" if candidate.vlad_ready else "dinov2_mean_pool",
                "global_descriptor_backend": candidate.descriptor_backend,
                "retrieval_score": float(candidate.score),
                "retrieval_threshold": float(candidate.threshold),
                "candidate_keyframe_index": int(candidate.keyframe_index),
                "candidate_anchor_id": int(candidate.anchor_id),
                "candidate_frame_id": int(candidate.frame_id),
                "candidate_triangulated_points": int(triangulated),
                "accepted": False,
                "failure_reason": "",
            }
            if triangulated < int(self.args.relocalization_min_triangulated):
                attempt["failure_reason"] = "relocalization_failed_no_3d"
                attempts.append(attempt)
                continue
            selected_keyframes.append(keyframe)
            seen_indices.add(int(keyframe.index))
            attempts.append(attempt)
            if len(selected_keyframes) >= int(self.args.relocalization_max_keyframes):
                break
        if len(selected_keyframes) == 0:
            frame.info["relocalization"] = {
                "attempted": True,
                "accepted": False,
                "failure_reason": "relocalization_failed_no_3d",
                "candidates": attempts,
            }
            return None
        for keyframe in selected_keyframes:
            keyframe.update_3dpts(self.keyframe_store.keyframes)
        with self.tracker.track("Reloc"):
            Rt = self.pose_initializer.initialize_incremental(
                selected_keyframes,
                desc_kpts,
                self.state.n_keyframes,
                frame.info["is_test"],
                frame.image,
                all_keyframes=self.keyframe_store.keyframes,
                retry_count=0,
                frame_uid=frame.info.get("frame_id"),
            )
        geom = dict(getattr(self.pose_initializer, "last_geom_debug", {}))
        accepted = Rt is not None
        for attempt in attempts:
            if int(attempt.get("candidate_keyframe_index", -1)) in {int(kf.index) for kf in selected_keyframes}:
                attempt["local_matches"] = int(geom.get("total_2d3d_matches", 0))
                attempt["pnp_inliers"] = int(geom.get("pnp_inliers", 0))
                attempt["miniba_inliers"] = int(geom.get("miniba_inliers", 0))
                attempt["accepted"] = bool(accepted)
                attempt["failure_reason"] = "" if accepted else f"relocalization_failed_{self.pose_initializer.last_failure_reason}"
        frame.info["relocalization"] = {
            "attempted": True,
            "accepted": bool(accepted),
            "failure_reason": "" if accepted else f"relocalization_failed_{self.pose_initializer.last_failure_reason}",
            "pose_source": "global_relocalization" if accepted else "",
            "num_candidates": int(len(candidates)),
            "num_selected_keyframes": int(len(selected_keyframes)),
            "candidates": attempts,
            "pose_initialization": geom,
        }
        if accepted:
            frame.info["pose_source"] = "global_relocalization"
            print(
                "[Relocalization] "
                f"frame={frame.frame_id} accepted "
                f"selected={len(selected_keyframes)} "
                f"pnp={geom.get('pnp_inliers', 0)} "
                f"miniba={geom.get('miniba_inliers', 0)}"
            )
        return Rt

    def _add_initialized_keyframe(self, image, info: dict, desc_kpts, Rt, index: int, phase: str) -> None:
        with self.tracker.track("Add"):
            if self.args.use_colmap_poses:
                Rt = info["Rt"]
            focal_device = Rt.device if torch.is_tensor(Rt) else ("cuda" if torch.cuda.is_available() else "cpu")
            keyframe = self._make_keyframe(
                self.frame_states.get(int(info.get("frame_id", index))),
                Rt,
                index,
                torch.as_tensor(self.state.focal_px, device=focal_device),
            )
            frame = self.frame_states.get(int(info.get("frame_id", index)))
            if frame is not None:
                frame.info["pose_initialization"] = dict(getattr(self.pose_initializer, "last_geom_debug", {}))
                self._calibrate_keyframe_depth_from_triangulation(keyframe)
                self._attach_frame_to_controller(frame, keyframe)

        frame = self.frame_states.get(int(info.get("frame_id", index)))
        if frame is not None:
            self._optimize_anchor_scene(frame, keyframe, f"anchor_{phase}")

        with self.tracker.track("Opt"):
            stats = self._optimize_scene(phase)
            self.state.loss_step_idx = _append_loss_record(
                self.loss_records,
                stats,
                phase=phase,
                lod=self.current_lod,
                step_idx=self.state.loss_step_idx,
            )

    def _queue_failed_pose_candidate(self, image, info: dict, desc_kpts) -> None:
        if (
            self.pose_initializer.last_failure_reason == "lsf_velocity_gate"
            and not info["is_test"]
            and self.args.pose_retry_max_attempts > 0
        ):
            pending_item = {
                "image": image,
                "info": info,
                "desc_kpts": desc_kpts,
                "retry_count": 1,
            }
            if len(self.pending_pose_queue) < self.args.pose_retry_queue_size:
                self.pending_pose_queue.append(pending_item)
            else:
                print(f"[PoseRetry] Queue full ({self.args.pose_retry_queue_size}), dropping frame {info.get('frame_id')}.")

    def _optimize_anchor_scene(self, frame, keyframe: TrackingKeyframe, phase: str) -> None:
        if self.cfg.anchor_iterations <= 0 or self.cfg.async_mapping:
            return
        anchor_id = int(frame.info.get("progressive_anchor_id", -1))
        if anchor_id < 0 or anchor_id >= len(self.controller.anchors):
            return
        anchor = self.controller.anchors[anchor_id]
        if anchor.gaussian_model.n == 0:
            return
        view_items = self._select_anchor_training_views(anchor, keyframe, frame)
        with self.tracker.track("AnchorOpt"):
            stats = self.anchor_scene_model.optimization_loop(
                keyframe,
                frame,
                anchor_id=anchor_id,
                n_iters=self.cfg.anchor_iterations,
                view_items=view_items,
            )
        self.state.loss_step_idx = _append_loss_record(
            self.loss_records,
            stats,
            phase=phase,
            lod=self.current_lod,
            step_idx=self.state.loss_step_idx,
        )
        status = self._anchor_budget_status(anchor, keyframe.get_centre(approx=True).detach())
        if status["should_roll"] and anchor.anchor_id == self.controller.active_anchor().anchor_id:
            self._seal_anchor_and_start_next(anchor, keyframe.get_centre(approx=True).detach(), self.prev_frame, frame, status)

    def _anchor_budget_status(self, anchor, cam_centre: torch.Tensor | None = None) -> dict:
        if anchor.anchor_id != self.controller.active_anchor().anchor_id:
            return {"should_roll": False, "reasons": ["inactive_anchor"], "anchor_id": int(anchor.anchor_id)}
        return self.controller.anchor_budget_status(
            cam_centre,
            max_anchor_radius=self.cfg.anchor_radius,
            min_keyframes=self.cfg.anchor_min_keyframes,
            max_gaussians=self.cfg.max_anchor_gaussians,
            max_keyframes=self.cfg.max_anchor_keyframes,
            max_tsdf_voxels=self.cfg.max_anchor_tsdf_voxels,
            max_vram_mb=self.cfg.max_anchor_vram_mb,
        )

    def _seal_anchor_and_start_next(self, anchor, cam_centre: torch.Tensor, reference_frame, new_frame, status: dict) -> None:
        anchor_origin = self._recent_anchor_origin(cam_centre)
        if anchor.gaussian_model.n > 0 and self.cfg.anchor_final_iterations > 0 and len(anchor.keyframe_ids) > 0:
            last_frame_id = int(anchor.keyframe_ids[-1])
            last_keyframe = self.keyframe_store.by_frame_id(last_frame_id)
            last_frame = self.frame_states.get(last_frame_id)
            if last_keyframe is not None and last_frame is not None:
                with self.tracker.track("AnchorFinalOpt"):
                    self.anchor_scene_model.optimization_loop(
                        last_keyframe,
                        last_frame,
                        anchor_id=anchor.anchor_id,
                        n_iters=self.cfg.anchor_final_iterations,
                        view_items=self._select_anchor_training_views(anchor, last_keyframe, last_frame),
                    )
        merge_stats = self.anchor_scene_model.merge_anchor_gaussians(
            anchor,
            voxel_size=self.cfg.anchor_merge_voxel_size,
            target_max=self.cfg.anchor_merge_target_gaussians,
        )
        delayed_tsdf_stats = {}
        if self.cfg.tsdf_fusion_mode == "delayed_gaussian":
            with self.tracker.track("DelayedTSDF"):
                delayed_tsdf_stats = self._integrate_delayed_tsdf(anchor)
        if self.args.model_path:
            with self.tracker.track("AnchorSealSave"):
                self.anchor_scene_model.save_anchor(self.args.model_path, anchor)
        event = {
            "anchor_id": int(anchor.anchor_id),
            "reasons": list(status.get("reasons", [])),
            "budget_status": status,
            "merge": merge_stats,
            "delayed_tsdf": delayed_tsdf_stats,
            "raw_rollover_centre": cam_centre.detach().cpu().tolist(),
            "new_anchor_origin": anchor_origin.detach().cpu().tolist(),
            "anchor_origin_median_window": int(self.cfg.anchor_origin_median_window),
        }
        self.anchor_seal_events.append(event)
        print(
            "[AnchorSeal] "
            f"anchor={anchor.anchor_id} reasons={event['reasons']} "
            f"gaussians={status.get('num_gaussians', 0)} "
            f"merge_after={merge_stats.get('after', anchor.gaussian_model.n)} "
            f"delayed_tsdf={delayed_tsdf_stats.get('integrated', 0)}"
        )
        self.controller.create_active_anchor(
            torch.eye(3, device=anchor_origin.device, dtype=anchor_origin.dtype),
            anchor_origin,
            reference_frame=reference_frame,
            new_frame=new_frame,
        )
        if self.loop_manager is not None:
            self.loop_manager.optimize_sequential_rollover(self.controller)

    def _integrate_delayed_tsdf(self, anchor) -> dict:
        if getattr(anchor, "_delayed_tsdf_fused", False):
            return dict(getattr(anchor, "_delayed_tsdf_stats", {"mode": "delayed_gaussian", "skipped": "already_fused"}))
        stats = self.tsdf_fusion.integrate_optimized_gaussians(
            anchor.tsdf,
            anchor.gaussian_model,
            anchor,
            min_opacity=self.cfg.delayed_tsdf_min_opacity,
            max_samples=self.cfg.delayed_tsdf_max_samples,
        )
        anchor._delayed_tsdf_fused = True
        anchor._delayed_tsdf_stats = stats
        return stats

    def _recent_anchor_origin(self, cam_centre: torch.Tensor) -> torch.Tensor:
        return _recent_keyframe_centre_median(
            self.keyframe_store.keyframes,
            cam_centre,
            self.cfg.anchor_origin_median_window,
        )

    def _attach_frame_to_controller(self, frame, keyframe: TrackingKeyframe) -> None:
        cam_centre = keyframe.get_centre(approx=True).detach()
        anchor_origin = self._recent_anchor_origin(cam_centre)
        active_anchor = self.controller.active_anchor()
        rolled_anchor = False
        if len(active_anchor.keyframe_ids) == 0:
            with self.controller.anchor_locks[active_anchor.anchor_id].write_lock():
                active_anchor.t_anchor_to_world = anchor_origin.to(active_anchor.device)
                self.controller.graph.add_node(active_anchor.anchor_id, active_anchor.T_anchor_to_world)
        else:
            budget_status = self._anchor_budget_status(active_anchor, cam_centre)
            if budget_status["should_roll"]:
                self._seal_anchor_and_start_next(active_anchor, cam_centre, self.prev_frame, frame, budget_status)
                active_anchor = self.controller.active_anchor()
                rolled_anchor = True

        self.controller.add_frame(frame, active_anchor)
        frame.info["progressive_anchor_id"] = active_anchor.anchor_id
        frame.info["progressive_keyframe_index"] = int(keyframe.index)
        if self.place_index is not None:
            self.place_index.add_keyframe(keyframe, active_anchor.anchor_id)
        if rolled_anchor:
            self._check_loop_closure(active_anchor.anchor_id)
        self._check_frame_loop_closure(frame, active_anchor.anchor_id)
        if self.cfg.async_mapping:
            queued = self.controller.enqueue_mapping_task(frame, active_anchor.anchor_id)
            if not queued:
                self.controller.mapping_errors.append(f"mapping_queue_full:{frame.frame_id}")
        else:
            self._mapping_step(MappingTask(frame=frame, anchor_id=active_anchor.anchor_id))

    def _check_loop_closure(self, new_anchor_id: int) -> None:
        if self.loop_manager is None:
            return

        def matcher_fn(left, right):
            return self.observation_builder.match(left, right, remove_outliers=False)

        results = self.loop_manager.check_anchor_rollover(
            new_anchor_id,
            self.keyframe_store,
            self.controller,
            matcher_fn,
        )
        for result in results:
            status = "accepted" if result.accepted else result.reason
            print(
                "[LoopClosure] "
                f"anchor={new_anchor_id} status={status} "
                f"matches={result.num_matches} inlier={result.inlier_ratio:.3f}"
            )

    def _promote_loop_candidate_keyframe(self, frame) -> bool:
        if self.place_index is None or self.loop_manager is None or self.controller is None:
            return False
        if len(self.controller.anchors) < 3:
            return False
        active_anchor_id = int(self.controller.active_anchor().anchor_id)
        candidates = self.place_index.query_frame(
            frame,
            current_anchor_id=active_anchor_id,
            exclude_anchor_window=self.cfg.loop_min_anchor_gap,
            top_k=self.cfg.loop_check_max_candidates,
        )
        if len(candidates) == 0:
            return False

        def matcher_fn(left, right):
            return self.observation_builder.match(left, right, remove_outliers=False)

        accepted_candidates = []
        for candidate in candidates:
            dst_keyframe = self.keyframe_store.by_index(candidate.keyframe_index)
            if dst_keyframe is None:
                continue
            result = self.controller.loop_verifier.verify(frame, dst_keyframe.frame, matcher_fn)
            if result.accepted:
                accepted_candidates.append(candidate)
                frame.info["progressive_loop_hint"] = {
                    "candidate_anchor_id": int(candidate.anchor_id),
                    "candidate_keyframe_index": int(candidate.keyframe_index),
                    "retrieval_score": float(candidate.score),
                    "retrieval_threshold": float(candidate.threshold),
                    "num_matches": int(result.num_matches),
                    "inlier_ratio": float(result.inlier_ratio),
                }
                break
        if len(accepted_candidates) == 0:
            return False
        frame.info["progressive_loop_candidates"] = accepted_candidates
        return True

    def _check_frame_loop_closure(self, frame, active_anchor_id: int) -> None:
        if self.loop_manager is None:
            return
        candidates = frame.info.get("progressive_loop_candidates", [])
        if len(candidates) == 0:
            return

        def matcher_fn(left, right):
            return self.observation_builder.match(left, right, remove_outliers=False)

        results = self.loop_manager.check_frame_candidates(
            active_anchor_id,
            frame,
            candidates,
            self.keyframe_store,
            self.controller,
            matcher_fn,
        )
        for result in results:
            status = "accepted" if result.accepted else result.reason
            print(
                "[LoopClosureFrame] "
                f"anchor={active_anchor_id} status={status} "
                f"dst={result.dst_anchor_id} matches={result.num_matches} "
                f"inlier={result.inlier_ratio:.3f}"
            )

    @torch.no_grad()
    def _mapping_step(self, task: MappingTask) -> None:
        anchor = self.controller.anchors[int(task.anchor_id)]
        frame = task.frame
        keyframe_index = int(frame.info.get("progressive_keyframe_index", -1))
        keyframe = self.keyframe_store.by_index(keyframe_index)
        if keyframe is None:
            return
        calibrated_depth = bool(frame.info.get("mono_depth_calibrated", False))
        Rt = keyframe.get_Rt().detach()
        R_w2c = Rt[:3, :3]
        t_w2c = Rt[:3, 3]
        R_cam_to_world = R_w2c.T
        t_cam_to_world = -R_w2c.T @ t_w2c
        rendered_image = None
        rendered_invdepth = None
        if anchor.gaussian_model.n > 0:
            with self.controller.anchor_locks[anchor.anchor_id].read_lock():
                result = self.anchor_scene_model.render_from_keyframe(keyframe, active_anchor_ids=[anchor.anchor_id])
                if not result.cuda_error:
                    rendered_image = result.render.detach()
                    rendered_invdepth = result.invdepth.detach()
        with self.controller.anchor_locks[anchor.anchor_id].write_lock():
            self._spawn_triangulated_keypoint_gaussians(anchor, keyframe, R_cam_to_world, t_cam_to_world)
            if calibrated_depth or not self.cfg.require_calibrated_depth:
                if self.cfg.tsdf_fusion_mode == "immediate_depth":
                    self.tsdf_fusion.integrate_depth(
                        anchor.tsdf,
                        frame.mono_idepth,
                        frame.mono_depth_conf,
                        anchor.R_world_to_anchor,
                        anchor.t_world_to_anchor,
                        R_cam_to_world,
                        t_cam_to_world,
                        self.keyframe_store.f,
                        self.keyframe_store.centre,
                    )
                self._spawn_anchor_local_gaussians(
                    anchor,
                    frame,
                    R_cam_to_world,
                    t_cam_to_world,
                    rendered_image=rendered_image,
                    rendered_invdepth=rendered_invdepth,
                )
            else:
                self.controller.mapping_errors.append(f"uncalibrated_dense_depth_skip:{frame.frame_id}")
        self._maybe_anchor_render_check(frame, anchor)
        self._maybe_save_test_render(frame, keyframe, anchor)

    @staticmethod
    def _image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
        image = image.detach()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 2:
            image = image[None]
        image = image[:3].clamp(0.0, 1.0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    @staticmethod
    def _gray_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
        image = image.detach()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3:
            image = image[0]
        image = image.float().cpu()
        finite = torch.isfinite(image)
        if bool(finite.any()):
            finite_vals = image[finite]
            lo = torch.quantile(finite_vals, 0.02)
            hi = torch.quantile(finite_vals, 0.98)
            if float((hi - lo).abs()) < 1e-12:
                hi = lo + 1.0
            image = ((image - lo) / (hi - lo)).clamp(0.0, 1.0)
            image = torch.where(finite, image, torch.zeros_like(image))
        else:
            image = torch.zeros_like(image)
        return (image.numpy() * 255.0).round().astype(np.uint8)

    @staticmethod
    def _write_png(path: str, image: np.ndarray) -> None:
        import struct
        import zlib

        image = np.asarray(image, dtype=np.uint8)
        if image.ndim == 2:
            height, width = image.shape
            color_type = 0
            raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
        elif image.ndim == 3 and image.shape[2] == 3:
            height, width, _ = image.shape
            color_type = 2
            raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
        else:
            raise ValueError(f"Unsupported PNG array shape: {image.shape}")

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        payload = b"\x89PNG\r\n\x1a\n"
        payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        payload += chunk(b"IDAT", zlib.compress(raw, level=6))
        payload += chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(payload)

    @staticmethod
    def _json_safe(value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): ProgressiveTrainer._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ProgressiveTrainer._json_safe(v) for v in value]
        return value

    @torch.no_grad()
    def _maybe_save_test_render(self, frame, keyframe: TrackingKeyframe, anchor) -> None:
        if not self.cfg.save_test_renders:
            return
        if not frame.info.get("is_test", False):
            return
        every = max(1, int(self.cfg.test_render_every))
        if int(frame.frame_id) % every != 0:
            return
        if int(frame.frame_id) in self.saved_test_render_frames:
            return
        if anchor.gaussian_model.n <= 0:
            return
        with self.tracker.track("TestRenderSave"):
            with self.controller.anchor_locks[anchor.anchor_id].read_lock():
                result = self.anchor_scene_model.render_from_keyframe(keyframe, active_anchor_ids=[anchor.anchor_id])
                debug = dict(self.anchor_scene_model.last_render_debug)
        if result.cuda_error:
            self.controller.mapping_errors.append(f"test_render_cuda:{frame.frame_id}:{result.cuda_error}")
            return

        out_dir = os.path.join(self.args.model_path, self.cfg.test_render_dir)
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"frame_{int(frame.frame_id):05d}_kf_{int(keyframe.index):05d}_anchor_{int(anchor.anchor_id):03d}"
        render = result.render.detach().clamp(0.0, 1.0)
        gt = frame.image.detach().to(render.device).clamp(0.0, 1.0)
        error = (render - gt).abs().mean(dim=0, keepdim=True)

        self._write_png(os.path.join(out_dir, f"{prefix}_render.png"), self._image_tensor_to_uint8(render))
        self._write_png(os.path.join(out_dir, f"{prefix}_gt.png"), self._image_tensor_to_uint8(gt))
        self._write_png(os.path.join(out_dir, f"{prefix}_abs_error.png"), self._gray_tensor_to_uint8(error))
        self._write_png(os.path.join(out_dir, f"{prefix}_invdepth.png"), self._gray_tensor_to_uint8(result.invdepth))
        metadata = {
            "frame_id": int(frame.frame_id),
            "keyframe_index": int(keyframe.index),
            "anchor_id": int(anchor.anchor_id),
            "image_name": frame.info.get("name", ""),
            "num_anchor_gaussians": int(anchor.gaussian_model.n),
            "mean_abs_error": float(error.mean().detach().cpu().item()),
            "render_debug": debug,
            "pose_initialization": frame.info.get("pose_initialization", {}),
            "mono_depth_alignment": frame.info.get("mono_depth_alignment", {}),
        }
        with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(self._json_safe(metadata), f, indent=2)
        self.saved_test_render_frames.add(int(frame.frame_id))
        print(f"[TestRender] saved frame={frame.frame_id} anchor={anchor.anchor_id} dir={out_dir}")

    @torch.no_grad()
    def _spawn_triangulated_keypoint_gaussians(
        self,
        anchor,
        keyframe: TrackingKeyframe,
        R_cam_to_world: torch.Tensor,
        t_cam_to_world: torch.Tensor,
    ) -> None:
        desc = keyframe.desc_kpts
        valid = desc.has_pt3d & torch.isfinite(desc.pts3d).all(dim=-1) & torch.isfinite(desc.depth) & (desc.depth > 1e-6)
        if not valid.any():
            return
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        max_sparse = min(int(self.cfg.local_spawn_max), 2048)
        if valid_idx.shape[0] > max_sparse:
            conf = desc.pts_conf[valid_idx].float()
            conf = torch.where(torch.isfinite(conf), conf, torch.zeros_like(conf))
            valid_idx = valid_idx[torch.topk(conf, k=max_sparse, largest=True).indices]
        xyz_cam = desc.pts3d[valid_idx].to(keyframe.device)
        uv = desc.kpts[valid_idx].to(keyframe.device)
        depth = desc.depth[valid_idx].to(keyframe.device).clamp_min(1e-6)
        xyz_world = (R_cam_to_world @ xyz_cam.T).T + t_cam_to_world[None]
        xyz_local = anchor.world_to_local(xyz_world)
        finite = torch.isfinite(xyz_local).all(dim=-1) & (depth > 1e-6)
        if not finite.any():
            return
        xyz_local = xyz_local[finite]
        uv = uv[finite]
        depth = depth[finite]
        sampler = make_torch_sampler(uv.view(1, 1, -1, 2), keyframe.width, keyframe.height)
        colors = torch.nn.functional.grid_sample(
            keyframe.frame.image[None].to(keyframe.device),
            sampler,
            mode="bilinear",
            align_corners=True,
        )[0, :, 0, :].T.contiguous()
        n_new = xyz_local.shape[0]
        rest_dim = anchor.gaussian_model.params["f_rest"]["val"].shape[1]
        world_scale = (depth / float(self.keyframe_store.f)).clamp(1e-5, 1.0)
        local_scale = (world_scale / anchor.s_anchor_to_world.to(world_scale).clamp_min(1e-8)).clamp(1e-6, 1e6)
        extension = {
            "xyz": xyz_local.contiguous(),
            "f_dc": RGB2SH(colors[:, None, :]).contiguous(),
            "f_rest": torch.zeros(n_new, rest_dim, 3, device=xyz_local.device),
            "opacity": inverse_sigmoid(torch.full((n_new, 1), self.cfg.spawn_opacity_init, device=xyz_local.device)),
            "scaling": torch.log(local_scale).unsqueeze(-1).repeat(1, 3).contiguous(),
            "rotation": torch.zeros(n_new, 4, device=xyz_local.device),
        }
        extension["rotation"][:, 0] = 1
        anchor.gaussian_model.append(extension, anchor.anchor_id)

    @torch.no_grad()
    def _spawn_anchor_local_gaussians(
        self,
        anchor,
        frame,
        R_cam_to_world: torch.Tensor,
        t_cam_to_world: torch.Tensor,
        rendered_image: torch.Tensor | None = None,
        rendered_invdepth: torch.Tensor | None = None,
    ) -> None:
        if frame.info.get("is_test", False):
            return
        if self.spawn_policy is None:
            self.spawn_policy = GaussianSpawnPolicy(
                width=self.width,
                height=self.height,
                f=self.keyframe_store.f,
                centre=self.keyframe_store.centre,
                init_proba_scaler=self.args.init_proba_scaler,
                target_samples=self.cfg.local_spawn_target,
                surface_sample_floor=self.cfg.surface_sample_floor,
                low_frequency_fraction=self.cfg.low_frequency_spawn_fraction,
                edge_probability_threshold=self.cfg.edge_probability_threshold,
            )
        spawn = self.spawn_policy.sample(
            frame,
            rendered_image=rendered_image,
            rendered_invdepth=rendered_invdepth,
        )
        if spawn.uv.shape[0] == 0:
            return
        if spawn.uv.shape[0] > self.cfg.local_spawn_max:
            keep = torch.randperm(spawn.uv.shape[0], device=spawn.uv.device)[: self.cfg.local_spawn_max]
            f_dc = spawn.f_dc[keep]
            uv = spawn.uv[keep]
            sample_probability = spawn.init_probability[keep]
        else:
            f_dc = spawn.f_dc
            uv = spawn.uv
            sample_probability = spawn.init_probability
        keyframe_index = int(frame.info.get("progressive_keyframe_index", -1))
        keyframe = self.keyframe_store.by_index(keyframe_index)
        if keyframe is None:
            return
        depth, valid_depth, mvs_stats = self._resolve_spawn_depths_with_mvs(keyframe, frame, uv, sample_probability)
        if not valid_depth.any():
            self.anchor_scene_model.last_render_debug["last_spawn"] = self._spawn_diagnostics(
                anchor,
                spawn,
                uv,
                torch.empty(0, device=uv.device),
                torch.empty(0, 3, device=uv.device),
                torch.empty(0, 3, device=uv.device),
                torch.empty(0, dtype=torch.bool, device=uv.device),
                R_cam_to_world,
                t_cam_to_world,
                extra_stats=mvs_stats,
            )
            return
        uv = uv[valid_depth]
        f_dc = f_dc[valid_depth]
        depth = depth[valid_depth]
        sample_probability = sample_probability[valid_depth]
        xyz_cam = depth2points(uv, depth[:, None], self.keyframe_store.f, self.keyframe_store.centre)
        xyz_world = (R_cam_to_world @ xyz_cam.T).T + t_cam_to_world[None]
        xyz_local = anchor.world_to_local(xyz_world)
        n_new = xyz_local.shape[0]
        rest_dim = anchor.gaussian_model.params["f_rest"]["val"].shape[1]
        world_scale = (depth / (float(self.keyframe_store.f) * torch.sqrt(sample_probability.clamp_min(1e-3)))).clamp(1e-6, 1e6)
        local_scale = (world_scale / anchor.s_anchor_to_world.to(world_scale).clamp_min(1e-8)).clamp(1e-6, 1e6)
        extension = {
            "xyz": xyz_local.contiguous(),
            "f_dc": f_dc.contiguous(),
            "f_rest": torch.zeros(n_new, rest_dim, 3, device=xyz_local.device),
            "opacity": inverse_sigmoid(torch.full((n_new, 1), self.cfg.spawn_opacity_init, device=xyz_local.device)),
            "scaling": torch.log(local_scale).unsqueeze(-1).repeat(1, 3).contiguous(),
            "rotation": torch.zeros(n_new, 4, device=xyz_local.device),
        }
        extension["rotation"][:, 0] = 1
        finite = torch.ones(n_new, dtype=torch.bool, device=xyz_local.device)
        for tensor in extension.values():
            finite &= torch.isfinite(tensor.flatten(1)).all(dim=1)
        if finite.any():
            anchor.gaussian_model.append({key: value[finite].contiguous() for key, value in extension.items()}, anchor.anchor_id)
        self.anchor_scene_model.last_render_debug["last_spawn"] = self._spawn_diagnostics(
            anchor,
            spawn,
            uv,
            depth,
            xyz_world,
            xyz_local,
            finite,
            R_cam_to_world,
            t_cam_to_world,
            extra_stats=mvs_stats,
        )

    @torch.no_grad()
    def _resolve_spawn_depths_with_mvs(
        self,
        keyframe: TrackingKeyframe,
        frame,
        uv: torch.Tensor,
        sample_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        mono_depth, mono_conf = self._sample_keyframe_mono_depth_and_conf(keyframe, uv)
        mono_valid = (
            torch.isfinite(mono_depth)
            & (mono_depth > 1e-6)
            & torch.isfinite(mono_conf)
            & (mono_conf >= self.cfg.depth_conf_min)
        )
        mvs_depth = torch.full_like(mono_depth, -1.0)
        accurate_mask = torch.zeros_like(mono_valid)
        prev_keyframes = []
        if self.guided_mvs is not None:
            prev_keyframes = self.keyframe_store.get_prev_keyframes(
                self.guided_mvs.n_cams + 1,
                update_3dpts=False,
                desc_kpts=frame.desc_kpts,
            )
            prev_keyframes = [kf for kf in prev_keyframes if int(kf.index) != int(keyframe.index)]
            prev_keyframes = [
                kf
                for kf in prev_keyframes
                if kf.feat_map is not None and torch.isfinite(kf.get_Rt()).all()
            ][: self.guided_mvs.n_cams]
        if (
            self.guided_mvs is not None
            and len(prev_keyframes) == self.guided_mvs.n_cams
            and keyframe.feat_map is not None
            and uv.numel() > 0
        ):
            try:
                mvs_depth, accurate_mask = self.guided_mvs(uv.contiguous(), keyframe, prev_keyframes)
            except Exception as exc:
                self.controller.mapping_errors.append(f"guided_mvs_failed:{frame.frame_id}:{type(exc).__name__}")
                mvs_depth = torch.full_like(mono_depth, -1.0)
                accurate_mask = torch.zeros_like(mono_valid)

        mvs_valid = torch.isfinite(mvs_depth) & (mvs_depth > 1e-6)
        idepth_mvs = 1.0 / mvs_depth.clamp_min(1e-6)
        idepth_mono = 1.0 / mono_depth.clamp_min(1e-6)
        depth_consistent = (idepth_mvs - idepth_mono).abs() < float(self.cfg.mvs_depth_consistency_idepth)
        reliable_mvs = mvs_valid & accurate_mask & depth_consistent
        fallback_mono = (~reliable_mvs) & mono_valid
        total = max(int(uv.shape[0]), 1)
        reliable_count = int(reliable_mvs.sum().item())
        mvs_ratio = float(reliable_count / total)
        adaptive_fraction = self._adaptive_mono_fallback_fraction(
            mvs_ratio,
            self.cfg.mvs_mono_fallback_fraction,
            self.cfg.mvs_mono_fallback_max_fraction,
            self.cfg.mvs_target_ratio,
        )
        cap_before_valid = max(
            int(round(adaptive_fraction * float(total))),
            max(0, int(self.cfg.mono_fallback_min_points)),
        )
        fallback_available = int(fallback_mono.sum().item())
        max_fallback = min(cap_before_valid, fallback_available)
        if max_fallback <= 0:
            fallback_mono &= False
        elif fallback_available > max_fallback:
            scores = (mono_conf * sample_probability.to(mono_conf.device)).masked_fill(~fallback_mono, -1.0)
            keep_idx = torch.topk(scores, k=max_fallback, largest=True).indices
            capped = torch.zeros_like(fallback_mono)
            capped[keep_idx] = True
            fallback_mono = capped

        valid = reliable_mvs | fallback_mono
        depth = torch.where(reliable_mvs, mvs_depth, mono_depth)
        stats = {
            "sampled_before_mvs": int(uv.shape[0]),
            "prev_keyframes_used": int(len(prev_keyframes)),
            "reliable_mvs": int(reliable_count),
            "fallback_mono": int(fallback_mono.sum().item()),
            "mvs_valid": int(mvs_valid.sum().item()),
            "mvs_accurate": int(accurate_mask.sum().item()),
            "depth_consistent": int(depth_consistent.sum().item()),
            "mono_valid": int(mono_valid.sum().item()),
            "mvs_ratio": float(mvs_ratio),
            "mono_fallback_ratio": float(fallback_mono.float().sum().item() / total),
            "depth_consistent_ratio": float(depth_consistent.float().sum().item() / total),
            "mono_fallback_cap": int(max_fallback),
            "mono_fallback_base_fraction": float(max(0.0, min(1.0, float(self.cfg.mvs_mono_fallback_fraction)))),
            "mono_fallback_adaptive_fraction": float(adaptive_fraction),
            "mono_fallback_max_fraction": float(max(0.0, min(1.0, float(self.cfg.mvs_mono_fallback_max_fraction)))),
            "mvs_target_ratio": float(max(0.0, float(self.cfg.mvs_target_ratio))),
            "mono_fallback_min_points": int(self.cfg.mono_fallback_min_points),
            "mono_fallback_cap_before_valid_count": int(cap_before_valid),
        }
        return depth, valid, stats

    @staticmethod
    def _adaptive_mono_fallback_fraction(
        mvs_ratio: float,
        base_fraction: float,
        max_fraction: float,
        target_mvs_ratio: float,
    ) -> float:
        base = max(0.0, min(1.0, float(base_fraction)))
        max_allowed = max(base, min(1.0, float(max_fraction)))
        target = max(float(target_mvs_ratio), 1e-8)
        ratio = max(0.0, min(1.0, float(mvs_ratio)))
        shortage = max(0.0, min(1.0, (target - ratio) / target))
        return float(base + (max_allowed - base) * shortage)

    @torch.no_grad()
    def _sample_keyframe_mono_depth_and_conf(self, keyframe: TrackingKeyframe, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if uv.numel() == 0:
            return torch.empty(0, device=keyframe.device), torch.empty(0, device=keyframe.device)
        sampler = make_torch_sampler(uv.view(1, 1, -1, 2), keyframe.width, keyframe.height)
        mono_idepth = torch.nn.functional.grid_sample(
            keyframe.get_mono_idepth()[None],
            sampler,
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0]
        mono_conf = torch.nn.functional.grid_sample(
            keyframe.mono_depth_conf,
            sampler,
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0]
        return 1.0 / mono_idepth.clamp_min(1e-6), mono_conf

    @torch.no_grad()
    def _spawn_diagnostics(
        self,
        anchor,
        spawn,
        uv: torch.Tensor,
        depth: torch.Tensor,
        xyz_world: torch.Tensor,
        xyz_local: torch.Tensor,
        finite: torch.Tensor,
        R_cam_to_world: torch.Tensor,
        t_cam_to_world: torch.Tensor,
        extra_stats: dict | None = None,
    ) -> dict:
        stats = dict(getattr(spawn, "stats", {}))
        if extra_stats:
            stats.update(extra_stats)
        stats["kept_after_cap"] = int(uv.shape[0])
        stats["finite_after_transform"] = int(finite.sum().item()) if finite.numel() else 0
        diag_mask = finite & torch.isfinite(xyz_world).all(dim=1) & torch.isfinite(xyz_local).all(dim=1)
        if uv.numel() == 0 or not diag_mask.any():
            return stats
        uv = uv[diag_mask]
        xyz_world = xyz_world[diag_mask]
        xyz_local = xyz_local[diag_mask]
        xyz_cam_check = ((xyz_world - t_cam_to_world[None]) @ R_cam_to_world).contiguous()
        uv_reprojected = pts2px(xyz_cam_check, self.keyframe_store.f, self.keyframe_store.centre.to(xyz_cam_check.device))
        reproj_error = torch.linalg.vector_norm(uv_reprojected - uv.to(uv_reprojected.device), dim=-1)
        query = anchor.tsdf.query(xyz_local)
        valid_tsdf = query.valid & (query.weight > 0)
        stats.update(
            {
                "reprojection_error_mean_px": float(reproj_error.mean().detach().item()),
                "reprojection_error_p95_px": float(torch.quantile(reproj_error.float(), 0.95).detach().item()),
                "tsdf_valid_ratio": float(valid_tsdf.float().mean().detach().item()) if valid_tsdf.numel() else 0.0,
                "tsdf_weight_mean": float(query.weight[valid_tsdf].mean().detach().item()) if valid_tsdf.any() else 0.0,
                "tsdf_abs_mean": float(query.tsdf[valid_tsdf].abs().mean().detach().item()) if valid_tsdf.any() else 0.0,
                "camera_depth_min": float(xyz_cam_check[:, 2].min().detach().item()),
                "camera_depth_median": float(xyz_cam_check[:, 2].median().detach().item()),
                "camera_depth_max": float(xyz_cam_check[:, 2].max().detach().item()),
                "local_bbox_min": xyz_local.min(dim=0).values.detach().cpu().tolist(),
                "local_bbox_max": xyz_local.max(dim=0).values.detach().cpu().tolist(),
            }
        )
        return stats

    def _select_anchor_training_views(self, anchor, current_keyframe: TrackingKeyframe, current_frame) -> list[tuple[TrackingKeyframe, object]]:
        max_views = max(1, int(self.cfg.anchor_train_views))
        selected: list[tuple[TrackingKeyframe, object]] = [(current_keyframe, current_frame)]
        seen = {int(current_keyframe.frame.frame_id)}
        for frame_id in reversed(anchor.keyframe_ids):
            if len(selected) >= max_views:
                break
            frame_id = int(frame_id)
            if frame_id in seen:
                continue
            keyframe = self.keyframe_store.by_frame_id(frame_id)
            frame = self.frame_states.get(frame_id)
            if keyframe is None or frame is None:
                continue
            selected.append((keyframe, frame))
            seen.add(frame_id)
        return selected

    @torch.no_grad()
    def _maybe_anchor_render_check(self, frame, anchor) -> None:
        if self.cfg.anchor_render_check_every <= 0:
            return
        if len(anchor.keyframe_ids) % self.cfg.anchor_render_check_every != 0:
            return
        keyframe_index = int(frame.info.get("progressive_keyframe_index", -1))
        keyframe = self.keyframe_store.by_index(keyframe_index)
        if keyframe is None:
            return
        with self.tracker.track("AnchorRender"):
            result = self.anchor_scene_model.render_from_keyframe(keyframe, active_anchor_ids=[anchor.anchor_id])
        debug = self.anchor_scene_model.last_render_debug
        if result.cuda_error:
            self.controller.mapping_errors.append(f"anchor_render_cuda:{frame.frame_id}:{result.cuda_error}")
        if debug:
            spawn_debug = debug.get("last_spawn", {})
            print(
                "[AnchorRender] "
                f"frame={frame.frame_id} anchor={anchor.anchor_id} "
                f"input={debug.get('num_input_gaussians', 0)} "
                f"visible={debug.get('num_visible_gaussians', 0)} "
                f"radii_pos={debug.get('num_positive_radii', 0)} "
                f"raster_limited={debug.get('num_raster_limited', 0)} "
                f"guard={debug.get('guard_reason_counts', {})}"
            )
            if spawn_debug:
                print(
                    "[GaussianSpawn] "
                    f"frame={frame.frame_id} anchor={anchor.anchor_id} "
                    f"spawned={spawn_debug.get('spawned', 0)} "
                    f"kept={spawn_debug.get('kept_after_cap', 0)} "
                    f"finite={spawn_debug.get('finite_after_transform', 0)} "
                    f"mvs={spawn_debug.get('mvs_ratio', 0.0):.3f} "
                    f"mono_fb={spawn_debug.get('mono_fallback_ratio', 0.0):.3f} "
                    f"tsdf_valid={spawn_debug.get('tsdf_valid_ratio', 0.0):.3f} "
                    f"reproj_p95={spawn_debug.get('reprojection_error_p95_px', 0.0):.3f}px "
                    f"depth_med={spawn_debug.get('camera_depth_median', 0.0):.3f}"
                )

    def _retry_pending_keyframes(self) -> int:
        extra_registered = 0
        retries = min(self.args.pose_retry_per_success, len(self.pending_pose_queue))
        for _ in range(retries):
            pending = self.pending_pose_queue.popleft()
            retry_index = self.state.n_keyframes + 1 + extra_registered
            with self.tracker.track("tri_retry"):
                retry_prev_keyframes = self.keyframe_store.get_prev_keyframes(
                    self.args.num_prev_keyframes_miniba_incr, True, pending["desc_kpts"]
                )
            with self.tracker.track("BAI_retry"):
                Rt_retry = self.pose_initializer.initialize_incremental(
                    retry_prev_keyframes,
                    pending["desc_kpts"],
                    retry_index,
                    pending["info"]["is_test"],
                    pending["image"],
                    all_keyframes=self.keyframe_store.keyframes,
                    retry_count=pending["retry_count"],
                    frame_uid=pending["info"].get("frame_id", retry_index),
                )

            if Rt_retry is not None:
                self._add_initialized_keyframe(
                    pending["image"],
                    pending["info"],
                    pending["desc_kpts"],
                    Rt_retry,
                    retry_index,
                    "incremental_retry_opt",
                )
                extra_registered += 1
            elif (
                self.pose_initializer.last_failure_reason == "lsf_velocity_gate"
                and pending["retry_count"] < self.args.pose_retry_max_attempts
                and len(self.pending_pose_queue) < self.args.pose_retry_queue_size
            ):
                pending["retry_count"] += 1
                self.pending_pose_queue.append(pending)
        return extra_registered

    @torch.no_grad()
    def _calibrate_keyframe_depth_from_triangulation(self, keyframe: TrackingKeyframe) -> None:
        keyframe.update_3dpts(self.keyframe_store.keyframes)
        desc = keyframe.desc_kpts
        valid = desc.has_pt3d & torch.isfinite(desc.depth) & (desc.depth > 1e-6) & (desc.pts_conf > 0)
        stats = {
            "method": "triangulated_keypoint_median_inverse_depth_scale",
            "num_candidates": int(valid.sum().item()),
            "triangulated_points": int(valid.sum().item()),
            "required": int(self.cfg.depth_scale_min_samples),
            "calibrated": False,
            "depth_scale_method": "uncalibrated",
        }
        if int(valid.sum().item()) < int(self.cfg.depth_scale_min_samples):
            if self._inherit_recent_depth_scale(keyframe, stats):
                return
            keyframe.frame.info["mono_depth_alignment"] = stats
            keyframe.frame.info["mono_depth_calibrated"] = False
            return

        uv = desc.kpts[valid].to(keyframe.device)
        sampler = make_torch_sampler(uv.view(1, 1, -1, 2), keyframe.width, keyframe.height)
        mono = torch.nn.functional.grid_sample(
            keyframe.mono_idepth,
            sampler,
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0]
        mono_conf = torch.nn.functional.grid_sample(
            keyframe.mono_depth_conf,
            sampler,
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0]
        target_idepth = 1.0 / desc.depth[valid].to(keyframe.device).clamp_min(1e-6)
        ratios = target_idepth / mono.clamp_min(1e-6)
        ratio_valid = (
            torch.isfinite(ratios)
            & (ratios > 0)
            & torch.isfinite(mono_conf)
            & (mono_conf >= self.cfg.scale.min_conf)
        )
        if int(ratio_valid.sum().item()) < int(self.cfg.depth_scale_min_samples):
            stats["num_valid_ratios"] = int(ratio_valid.sum().item())
            if self._inherit_recent_depth_scale(keyframe, stats):
                return
            keyframe.frame.info["mono_depth_alignment"] = stats
            keyframe.frame.info["mono_depth_calibrated"] = False
            return

        ratios = ratios[ratio_valid].float()
        if ratios.numel() < 8:
            median = ratios.median()
            mad = (ratios - median).abs().median().clamp_min(1e-8)
            robust = ratios[(ratios - median).abs() <= 5.0 * mad]
            q10 = ratios.min()
            q90 = ratios.max()
            filter_name = "median_5mad"
        else:
            q10 = torch.quantile(ratios, 0.10)
            q90 = torch.quantile(ratios, 0.90)
            robust = ratios[(ratios >= q10) & (ratios <= q90)]
            filter_name = "q10_q90"
        if robust.numel() == 0:
            robust = ratios
        scale = robust.median().clamp(
            min=float(self.cfg.min_depth_scale),
            max=float(self.cfg.max_depth_scale),
        )
        keyframe.apply_mono_idepth_calibration(scale)
        stats.update(
            {
                "calibrated": True,
                "num_valid_ratios": int(ratios.numel()),
                "num_robust_ratios": int(robust.numel()),
                "scale": float(scale.detach().item()),
                "depth_scale_method": "direct_calibrated",
                "robust_filter": filter_name,
                "ratio_q10": float(q10.detach().item()),
                "ratio_q90": float(q90.detach().item()),
            }
        )
        keyframe.frame.info["mono_depth_alignment"] = stats
        keyframe.frame.info["mono_depth_calibrated"] = True

    def _inherit_recent_depth_scale(self, keyframe: TrackingKeyframe, stats: dict) -> bool:
        for prev in reversed(self.keyframe_store.keyframes[:-1]):
            prev_stats = prev.frame.info.get("mono_depth_alignment", {})
            if not prev.frame.info.get("mono_depth_calibrated", False):
                continue
            scale = prev_stats.get("scale", None)
            if scale is None:
                continue
            scale_t = torch.as_tensor(scale, dtype=keyframe.mono_idepth.dtype, device=keyframe.device)
            if not torch.isfinite(scale_t).all() or float(scale_t.item()) <= 0:
                continue
            keyframe.apply_mono_idepth_calibration(scale_t)
            stats.update(
                {
                    "calibrated": True,
                    "depth_scale_method": "inherited",
                    "inherited_depth_scale": True,
                    "scale": float(scale_t.detach().cpu().item()),
                    "source_keyframe_index": int(prev.index),
                    "source_frame_id": int(prev.frame.frame_id),
                }
            )
            keyframe.frame.info["mono_depth_alignment"] = stats
            keyframe.frame.info["mono_depth_calibrated"] = True
            return True
        return False

    def _make_keyframe(self, frame, Rt, index: int, focal) -> TrackingKeyframe:
        if frame is None:
            raise RuntimeError(f"Missing FrameState for keyframe index {index}")
        focal = self._focal_tensor(focal, Rt)
        keyframe = self.keyframe_store.add_keyframe(frame, Rt, focal, index)
        self.state.focal_px = float(keyframe.f.detach().cpu().item())
        return keyframe

    @staticmethod
    def _focal_tensor(focal, Rt=None) -> torch.Tensor:
        if torch.is_tensor(focal):
            f = focal.detach().clone() if not focal.requires_grad else focal
        else:
            device = Rt.device if torch.is_tensor(Rt) else ("cuda" if torch.cuda.is_available() else "cpu")
            f = torch.tensor([float(focal)], device=device)
        if f.ndim == 0:
            f = f.reshape(1)
        else:
            f = f.flatten()[:1]
        return f.contiguous()

    def _optimize_scene(self, phase: str) -> dict | None:
        del phase
        return None

    def _evaluate_and_checkpoint(self, frame_id: int) -> None:
        if frame_id % self.args.save_every == 0 and self.args.save_every > 0:
            self.anchor_scene_model.save(
                os.path.join(self.args.model_path, "progress", f"{frame_id:05d}"),
                self.keyframe_store.keyframes,
                n_frames=len(self.dataset),
            )

    def _update_progress_bar(self, pbar) -> None:
        bar_postfix = []
        for key, value in self.metrics.items():
            bar_postfix += [self._format_metric(key, value)]
        if self.args.display_runtimes:
            for key, value in self.tracker.stats.items():
                if value["count"] > 0:
                    bar_postfix += [f"\033[35m{key}:{1000 * value['time'] / value['count']:.1f}\033[0m"]
        bar_postfix += [
            f"\033[36mFocal:{self.state.focal_px:.1f}",
            f"\033[36mKeyframes:{self.state.n_keyframes}\033[0m",
            f"\033[36mGaussians:{self.anchor_scene_model.state_summary()['num_gaussians']}\033[0m",
            f"\033[36mAnchors:{len(self.controller.anchors)}\033[0m",
        ]
        pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    def _finalize(self, reconstruction_time: float) -> dict:
        if self.cfg.async_mapping:
            self.controller.mapping_queue.join()
            self.controller.stop_mapping_worker()
        if self.cfg.tsdf_fusion_mode == "delayed_gaussian":
            for anchor in self.controller.anchors:
                with self.tracker.track("DelayedTSDFFinal"):
                    stats = self._integrate_delayed_tsdf(anchor)
                if int(stats.get("integrated", 0)) > 0:
                    self.anchor_seal_events.append(
                        {
                            "anchor_id": int(anchor.anchor_id),
                            "reasons": ["finalize_delayed_tsdf"],
                            "delayed_tsdf": stats,
                        }
                    )
        print("Saving the Progressive reconstruction to:", self.args.model_path)
        metrics = self.anchor_scene_model.save(
            self.args.model_path,
            self.keyframe_store.keyframes,
            reconstruction_time,
            len(self.dataset),
        )
        print(
            ", ".join(
                f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}"
                for metric, value in metrics.items()
            )
        )
        self.pose_initializer.save_failure_log(self.args.model_path)

        self.tracker.print_stats()
        self.tracker.save_stats(os.path.join(self.args.model_path, "resource_stats.txt"))
        _save_loss_records_and_plot(self.loss_records, self.args.model_path)
        lod_gate = {
            "ready": True,
            "reason": "progressive_anchor_local_single_lod",
            "pose_stability": 1.0,
            "projection_error_mean": 0.0,
        }
        _save_lod_completion_marker(self.args.model_path, self.current_lod, reconstruction_time, lod_gate, metrics)

        return {
            "num_keyframes": self.state.n_keyframes,
            "num_anchors": len(self.controller.anchors),
            "pipeline_state": self.controller.state_summary(),
            "anchor_scene_state": self.anchor_scene_model.state_summary(),
            "anchor_seal_events": list(self.anchor_seal_events),
            "loop_closure": self.loop_manager.summary() if self.loop_manager is not None else {},
            "num_loss_records": len(self.loss_records),
            "reconstruction_time": reconstruction_time,
        }

    @staticmethod
    def _format_metric(key: str, value) -> str:
        if isinstance(value, (float, int)):
            return f"\033[31m{key}:{value:.2f}\033[0m"
        return f"\033[31m{key}:{value}\033[0m"

def main() -> None:
    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    np.random.seed(0)

    args, _ = _parse_progressive_args(sys.argv)
    cfg = args
    started_at = time.time()
    try:
        summary = ProgressiveTrainer(args, cfg).run()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            _write_manifest(args.model_path, cfg, "completed", started_at)
        else:
            _write_manifest(args.model_path, cfg, "failed", started_at, error=f"SystemExit({exc.code})")
        raise
    except Exception as exc:
        _write_manifest(args.model_path, cfg, "failed", started_at, error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        _write_manifest(args.model_path, cfg, "completed", started_at, extra=summary)


if __name__ == "__main__":
    main()
