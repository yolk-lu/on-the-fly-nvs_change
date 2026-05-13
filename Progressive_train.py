from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm

from args import get_args
from dataloaders.image_dataset import ImageDataset
from dataloaders.stream_dataset import StreamDataset
from gaussianviewer import GaussianViewer
from graphdecoviewer.types import ViewerMode
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from resource_tracker import ResourceTracker
from scene.LoD_utils import lod_progressive_ready
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from utils import align_mean_up_fwd
from webviewer.webviewer import WebViewer


@dataclass
class ProgressiveRunConfig:
    overlap_mode: str
    backend_mode: str
    manifest_name: str
    run_label: str


@dataclass
class ProgressiveState:
    n_keyframes: int = 0
    focal_px: float = 0.0
    needs_reboot: bool = False
    last_reboot: int = 0
    loss_step_idx: int = 0


def _parse_progressive_args(argv: list[str]) -> tuple[ProgressiveRunConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--progressive_overlap_mode",
        choices=["reserved", "off"],
        default="reserved",
        help="Reserve overlap-region optimization hooks. Actual overlap optimization is not implemented yet.",
    )
    parser.add_argument(
        "--progressive_backend_mode",
        choices=["progressive_scene_model"],
        default="progressive_scene_model",
        help="Training backend owned by Progressive_train.py. Anchor-local backend will replace this surface later.",
    )
    parser.add_argument(
        "--progressive_manifest_name",
        default="progressive_manifest.json",
        help="Manifest filename written under model_path after training.",
    )
    parser.add_argument(
        "--progressive_run_label",
        default="progressive_train",
        help="Human-readable run label stored in the manifest.",
    )
    progressive_args, remaining = parser.parse_known_args(argv[1:])
    cfg = ProgressiveRunConfig(
        overlap_mode=progressive_args.progressive_overlap_mode,
        backend_mode=progressive_args.progressive_backend_mode,
        manifest_name=progressive_args.progressive_manifest_name,
        run_label=progressive_args.progressive_run_label,
    )
    return cfg, [argv[0], *remaining]


def _parse_training_args(clean_argv: list[str]):
    old_argv = sys.argv
    sys.argv = clean_argv
    try:
        return get_args()
    finally:
        sys.argv = old_argv


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
            f, fieldnames=["step", "phase", "lod", "total", "l1", "ssim", "depth"]
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
        "colmap_dir": os.path.isdir(os.path.join(model_path, "colmap")),
        "colmap_cameras": os.path.exists(os.path.join(model_path, "colmap", "cameras.bin")),
        "colmap_images": os.path.exists(os.path.join(model_path, "colmap", "images.bin")),
        "resource_stats": os.path.exists(os.path.join(model_path, "resource_stats.txt")),
        "loss_records": os.path.exists(os.path.join(model_path, "loss_records.csv")),
        "loss_curve": os.path.exists(os.path.join(model_path, "loss_curve.png")),
        "lod_completion_marker": len(glob.glob(os.path.join(model_path, "lod_*_complete.json"))) > 0,
    }
    optional = {"loss_records", "loss_curve"}
    required_missing = [name for name, ok in checks.items() if not ok and name not in optional]
    return {
        "model_path": model_path,
        "checks": checks,
        "missing": required_missing,
        "complete": len(required_missing) == 0,
    }


def _write_manifest(
    model_path: str,
    cfg: ProgressiveRunConfig,
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
            "completion_gate": "Progressive_train.py exits successfully and writes reconstruction outputs",
        },
        "outputs": output_status,
        "error": error,
    }
    if extra:
        payload["summary"] = extra
    with open(os.path.join(model_path, cfg.manifest_name), "w") as f:
        json.dump(payload, f, indent=2)


class ProgressiveTrainer:
    def __init__(self, args, cfg: ProgressiveRunConfig):
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
        self.scene_model = None
        self.detector = None
        self.viewer = None
        self.viewer_thread = None
        self.web_server = None
        self.web_server_thread = None
        self.tracker = ResourceTracker()
        self.state = ProgressiveState()
        self.metrics: dict = {}
        self.loss_records: list[dict] = []
        self.pending_pose_queue: deque[dict] = deque()
        self.bootstrap_keyframe_dicts: list[dict] = []
        self.bootstrap_desc_kpts: list = []
        self.prev_desc_kpts = None

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
        self.scene_model = SceneModel(self.width, self.height, self.args, self.matcher)

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
        self._initialize_viewer()

    def _initialize_viewer(self) -> None:
        if self.args.viewer_mode in ["server", "local"]:
            viewer_mode = ViewerMode.SERVER if self.args.viewer_mode == "server" else ViewerMode.LOCAL
            self.viewer = GaussianViewer.from_scene_model(self.scene_model, viewer_mode)
            self.viewer_thread = Thread(target=self.viewer.run, args=(self.args.ip, self.args.port), daemon=True)
            self.viewer_thread.start()
            self.viewer.throttling = True
        elif self.args.viewer_mode == "web":
            ip = "0.0.0.0"
            self.web_server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
            self.web_server_thread = Thread(target=self.web_server.serve_forever, daemon=True)
            self.web_server_thread.start()
            print(f"Visit http://{ip}:8000/webviewer to for the viewer")
            self.viewer = WebViewer(self.scene_model, self.args.ip, self.args.port)
            self.viewer_thread = Thread(target=self.viewer.run, daemon=True)
            self.viewer_thread.start()

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
            if not self._viewer_allows_frame(pbar):
                break
            self._process_frame(frame_id, pbar)

        reconstruction_time = time.time() - reconstruction_start_time
        return self._finalize(reconstruction_time)

    def _viewer_allows_frame(self, pbar) -> bool:
        if self.args.viewer_mode != "web":
            return True
        self.viewer.trainer_state = "running"
        while self.viewer.state == "stop":
            pbar.set_postfix_str("\033[31mPaused. Press the Start button in the webviewer\033[0m")
            time.sleep(0.1)
        if self.viewer.state == "finish":
            self.viewer.trainer_state = "finish"
            return False
        return True

    def _process_frame(self, frame_id: int, pbar) -> None:
        self.tracker.start("Load")
        image, info = self.dataset.getnext()
        info["frame_id"] = int(frame_id)
        desc_kpts = self.detector(image)

        if self.state.n_keyframes == 0:
            self.bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            self.bootstrap_desc_kpts = [desc_kpts]
            self.prev_desc_kpts = desc_kpts
            self.state.n_keyframes += 1
            self.tracker.stop()
            return

        curr_prev_matches = self.matcher(desc_kpts, self.prev_desc_kpts)
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        should_add_keyframe = (
            dist.median() > self.min_displacement
            and len(curr_prev_matches.kpts) > self.args.min_num_inliers
        )
        should_add_keyframe |= info["is_test"]
        self.tracker.stop()

        if should_add_keyframe:
            extra_registered = self._register_keyframe_candidate(image, info, desc_kpts)
            should_add_keyframe = extra_registered >= 0
        else:
            extra_registered = 0

        if should_add_keyframe:
            with self.tracker.track("anc"):
                self.scene_model.place_anchor_if_needed()
            self.state.n_keyframes += 1 + extra_registered
            if not info["is_test"]:
                self.prev_desc_kpts = desc_kpts
            self._evaluate_and_checkpoint(frame_id)
            self._update_progress_bar(pbar)

    def _register_keyframe_candidate(self, image, info: dict, desc_kpts) -> int:
        if self.state.n_keyframes < self.args.num_keyframes_miniba_bootstrap:
            self.bootstrap_keyframe_dicts.append({"image": image, "info": info})
            self.bootstrap_desc_kpts.append(desc_kpts)

        if self.state.n_keyframes == self.args.num_keyframes_miniba_bootstrap - 1:
            self._bootstrap_scene()
            return 0

        self._maybe_reboot()

        if self.state.n_keyframes >= self.args.num_keyframes_miniba_bootstrap:
            return self._register_incremental_keyframe(image, info, desc_kpts)
        return 0

    def _bootstrap_scene(self) -> None:
        with self.tracker.track("BAB"):
            Rts, f, _ = self.pose_initializer.initialize_bootstrap(self.bootstrap_desc_kpts)
            self.state.focal_px = float(f.detach().cpu().item())

        for index, (keyframe_dict, desc_kpts, Rt) in enumerate(
            zip(self.bootstrap_keyframe_dicts, self.bootstrap_desc_kpts, Rts)
        ):
            with self.tracker.track("Add"):
                focal = f
                if self.args.use_colmap_poses:
                    Rt = keyframe_dict["info"]["Rt"]
                    focal = keyframe_dict["info"]["focal"]
                keyframe = self._make_keyframe(
                    keyframe_dict["image"], keyframe_dict["info"], desc_kpts, Rt, index, focal
                )
                self.scene_model.add_keyframe(keyframe, focal)

        if self.args.viewer_mode not in ["none", "web"]:
            self.viewer.reset_intrinsics("point_view")

        for index in range(self.args.num_keyframes_miniba_bootstrap):
            with self.tracker.track("Init"):
                self.scene_model.add_new_gaussians(index)

        with self.tracker.track("Opt"):
            stats = self._optimize_scene("bootstrap_opt")
            self.state.loss_step_idx = _append_loss_record(
                self.loss_records,
                stats,
                phase="bootstrap_opt",
                lod=self.scene_model.current_lod,
                step_idx=self.state.loss_step_idx,
            )
        self.state.last_reboot = self.state.n_keyframes

    def _maybe_reboot(self) -> None:
        if (
            self.args.enable_reboot
            and self.scene_model.approx_cam_centres is not None
            and len(self.scene_model.anchors)
        ):
            last_centers = self.scene_model.approx_cam_centres[-20:]
            rel_dist = torch.norm(last_centers[1:] - last_centers[:-1], dim=-1).mean()
            self.state.needs_reboot = (
                rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3
            ) and self.state.n_keyframes - self.state.last_reboot > 50

        if not self.state.needs_reboot:
            return

        bs_kfs = self.scene_model.keyframes[-8:]
        bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
        in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
        Rts, _, final_residual = self.pose_initializer.initialize_bootstrap(bootstrap_desc_kpts, rebooting=True)
        if final_residual < self.max_error * 0.5:
            Rts = align_mean_up_fwd(Rts, in_Rts)
            for Rt, keyframe in zip(Rts, bs_kfs):
                keyframe.set_Rt(Rt)
            self.scene_model.reset()
            for i in range(3, 0, -1):
                self.scene_model.add_new_gaussians(-i)
            for _ in range(3 * self.args.num_iterations):
                stats = self.scene_model.optimization_step()
                self.state.loss_step_idx = _append_loss_record(
                    self.loss_records,
                    stats,
                    phase="reboot_opt",
                    lod=self.scene_model.current_lod,
                    step_idx=self.state.loss_step_idx,
                )
            self.state.needs_reboot = False
            self.state.last_reboot = self.state.n_keyframes

    def _register_incremental_keyframe(self, image, info: dict, desc_kpts) -> int:
        with self.tracker.track("tri"):
            prev_keyframes = self.scene_model.get_prev_keyframes(
                self.args.num_prev_keyframes_miniba_incr, True, desc_kpts
            )
        with self.tracker.track("BAI"):
            Rt = self.pose_initializer.initialize_incremental(
                prev_keyframes,
                desc_kpts,
                self.state.n_keyframes,
                info["is_test"],
                image,
                all_keyframes=self.scene_model.keyframes,
                retry_count=0,
                frame_uid=info.get("frame_id"),
            )

        if Rt is None:
            self._queue_failed_pose_candidate(image, info, desc_kpts)
            return -1

        self._add_initialized_keyframe(image, info, desc_kpts, Rt, self.state.n_keyframes, "incremental_opt")
        return self._retry_pending_keyframes()

    def _add_initialized_keyframe(self, image, info: dict, desc_kpts, Rt, index: int, phase: str) -> None:
        with self.tracker.track("Add"):
            if self.args.use_colmap_poses:
                Rt = info["Rt"]
            focal_device = Rt.device if torch.is_tensor(Rt) else ("cuda" if torch.cuda.is_available() else "cpu")
            keyframe = self._make_keyframe(
                image,
                info,
                desc_kpts,
                Rt,
                index,
                torch.as_tensor(self.state.focal_px, device=focal_device),
            )
            self.scene_model.add_keyframe(keyframe)

        with self.tracker.track("Init"):
            self.scene_model.add_new_gaussians()

        with self.tracker.track("Opt"):
            stats = self._optimize_scene(phase)
            self.state.loss_step_idx = _append_loss_record(
                self.loss_records,
                stats,
                phase=phase,
                lod=self.scene_model.current_lod,
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

    def _retry_pending_keyframes(self) -> int:
        extra_registered = 0
        retries = min(self.args.pose_retry_per_success, len(self.pending_pose_queue))
        for _ in range(retries):
            pending = self.pending_pose_queue.popleft()
            retry_index = self.state.n_keyframes + 1 + extra_registered
            with self.tracker.track("tri_retry"):
                retry_prev_keyframes = self.scene_model.get_prev_keyframes(
                    self.args.num_prev_keyframes_miniba_incr, True, pending["desc_kpts"]
                )
            with self.tracker.track("BAI_retry"):
                Rt_retry = self.pose_initializer.initialize_incremental(
                    retry_prev_keyframes,
                    pending["desc_kpts"],
                    retry_index,
                    pending["info"]["is_test"],
                    pending["image"],
                    all_keyframes=self.scene_model.keyframes,
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
                with self.tracker.track("anc_retry"):
                    self.scene_model.place_anchor_if_needed()
                extra_registered += 1
            elif (
                self.pose_initializer.last_failure_reason == "lsf_velocity_gate"
                and pending["retry_count"] < self.args.pose_retry_max_attempts
                and len(self.pending_pose_queue) < self.args.pose_retry_queue_size
            ):
                pending["retry_count"] += 1
                self.pending_pose_queue.append(pending)
        return extra_registered

    def _make_keyframe(self, image, info: dict, desc_kpts, Rt, index: int, focal):
        return Keyframe(
            image,
            info,
            desc_kpts,
            Rt,
            index,
            focal,
            self.dense_extractor,
            self.depth_estimator,
            self.triangulator,
            self.args,
        )

    def _optimize_scene(self, phase: str) -> dict | None:
        if self.is_stream:
            self.scene_model.optimize_async(self.args.num_iterations)
            return None
        return self.scene_model.optimization_loop(self.args.num_iterations)

    def _evaluate_and_checkpoint(self, frame_id: int) -> None:
        if (
            self.state.n_keyframes % self.args.test_frequency == 0
            and self.args.test_frequency > 0
            and (self.args.test_hold > 0 or self.args.eval_poses)
        ):
            self.metrics = self.scene_model.evaluate(self.args.eval_poses)

        if frame_id % self.args.save_every == 0 and self.args.save_every > 0:
            self.scene_model.save(os.path.join(self.args.model_path, "progress", f"{frame_id:05d}"))

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
            f"\033[36mGaussians:{self.scene_model.n_active_gaussians}\033[0m",
            f"\033[36mAnchors:{len(self.scene_model.anchors)}\033[0m",
        ]
        pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    def _finalize(self, reconstruction_time: float) -> dict:
        self.scene_model.enable_inference_mode()
        print("Saving the Progressive reconstruction to:", self.args.model_path)
        metrics = self.scene_model.save(self.args.model_path, reconstruction_time, len(self.dataset))
        print(
            ", ".join(
                f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}"
                for metric, value in metrics.items()
            )
        )
        self.pose_initializer.save_failure_log(self.args.model_path)

        reconstruction_time = self._run_lod_finetune(reconstruction_time, metrics)
        self.scene_model.inference_mode = True
        self.tracker.print_stats()
        self.tracker.save_stats(os.path.join(self.args.model_path, "resource_stats.txt"))
        _save_loss_records_and_plot(self.loss_records, self.args.model_path)

        if self.args.viewer_mode != "none":
            self._keep_viewer_alive()

        return {
            "num_keyframes": self.state.n_keyframes,
            "num_anchors": len(self.scene_model.anchors),
            "num_loss_records": len(self.loss_records),
            "reconstruction_time": reconstruction_time,
        }

    def _run_lod_finetune(self, reconstruction_time: float, metrics: dict) -> float:
        finetune_epochs_per_level = 0
        if len(self.args.save_at_finetune_epoch) > 0:
            finetune_epochs_per_level = max(self.args.save_at_finetune_epoch)
        elif self.args.lod_max > self.args.lod_min:
            finetune_epochs_per_level = 10

        current_lod_step = self.scene_model.current_lod
        while current_lod_step <= self.args.lod_max:
            if finetune_epochs_per_level > 0:
                print(f"\n--- Progressive Training LoD {current_lod_step} ---")
                torch.cuda.empty_cache()
                self.scene_model.inference_mode = False
                pbar = tqdm(range(0, finetune_epochs_per_level), desc=f"Fine tuning LoD {current_lod_step}")
                for epoch in pbar:
                    epoch_start_time = time.time()
                    with self.tracker.track(f"LoD_{current_lod_step}"):
                        stats = self.scene_model.finetune_epoch()
                        self.state.loss_step_idx = _append_loss_record(
                            self.loss_records,
                            stats,
                            phase=f"finetune_lod_{current_lod_step}",
                            lod=current_lod_step,
                            step_idx=self.state.loss_step_idx,
                        )
                    reconstruction_time += time.time() - epoch_start_time
                    if epoch + 1 in self.args.save_at_finetune_epoch:
                        torch.cuda.empty_cache()
                        self.scene_model.inference_mode = True
                        metrics = self.scene_model.save(
                            os.path.join(self.args.model_path, f"lod_{current_lod_step}_epoch_{epoch + 1}"),
                            reconstruction_time,
                        )
                        pbar.set_postfix_str(",".join(self._format_metric(k, v) for k, v in metrics.items()))
                        self.scene_model.inference_mode = False
                        torch.cuda.empty_cache()

            lod_gate = lod_progressive_ready(
                self.scene_model,
                min_pose_stability=self.args.lod_min_pose_stability,
                max_projection_error_px=self.args.lod_max_projection_error_px,
                pose_window=self.args.lod_pose_window,
            )
            print(
                f"[LoD Gate] L{current_lod_step}: ready={lod_gate['ready']} "
                f"pose={lod_gate['pose_stability']:.3f}, proj_mean={lod_gate['projection_error_mean']:.3f}px"
            )

            extra_epochs = 0
            while (
                current_lod_step < self.args.lod_max
                and not lod_gate["ready"]
                and extra_epochs < self.args.lod_progressive_max_extra_epochs
            ):
                extra_epochs += 1
                self.scene_model.inference_mode = False
                epoch_start_time = time.time()
                with self.tracker.track(f"LoD_{current_lod_step}_extra"):
                    stats = self.scene_model.finetune_epoch()
                    self.state.loss_step_idx = _append_loss_record(
                        self.loss_records,
                        stats,
                        phase=f"finetune_lod_{current_lod_step}_extra",
                        lod=current_lod_step,
                        step_idx=self.state.loss_step_idx,
                    )
                reconstruction_time += time.time() - epoch_start_time
                lod_gate = lod_progressive_ready(
                    self.scene_model,
                    min_pose_stability=self.args.lod_min_pose_stability,
                    max_projection_error_px=self.args.lod_max_projection_error_px,
                    pose_window=self.args.lod_pose_window,
                )
                print(
                    f"[LoD Gate][extra {extra_epochs}/{self.args.lod_progressive_max_extra_epochs}] "
                    f"ready={lod_gate['ready']} pose={lod_gate['pose_stability']:.3f}, "
                    f"proj_mean={lod_gate['projection_error_mean']:.3f}px"
                )

            save_dir = self._lod_save_dir(current_lod_step)
            self.scene_model.inference_mode = True
            print(f"Saving LoD {current_lod_step} completion checkpoint to {save_dir}")
            if self.args.lod_min == self.args.lod_max:
                marker_path = _save_lod_completion_marker(save_dir, current_lod_step, reconstruction_time, lod_gate, metrics)
                print(f"Single-LoD checkpoint reuses the initial full scene save; completion marker written to {marker_path}")
            else:
                self.scene_model.save(save_dir, reconstruction_time)
                _save_lod_completion_marker(save_dir, current_lod_step, reconstruction_time, lod_gate, metrics)
            self.scene_model.inference_mode = False

            if current_lod_step < self.args.lod_max:
                if not lod_gate["ready"]:
                    print(
                        f"[LoD Gate] Max extra epochs reached at LoD {current_lod_step} "
                        f"(reason={lod_gate['reason']}). Progressing to next LoD."
                    )
                print(f"Increasing LoD from {current_lod_step} to {current_lod_step + 1}")
                with self.tracker.track(f"IncLoD_{current_lod_step}"):
                    self.scene_model.increase_lod()
                current_lod_step = self.scene_model.current_lod
            else:
                break

        return reconstruction_time

    def _lod_save_dir(self, current_lod_step: int) -> str:
        if self.args.lod_min == self.args.lod_max:
            return self.args.model_path
        base_parent = os.path.dirname(os.path.normpath(self.args.model_path))
        base_name = os.path.basename(os.path.normpath(self.args.model_path))
        lod_tag = f"LoD-{self.args.lod_min}-{current_lod_step}-{self.args.lod_max}"
        return os.path.join(base_parent, f"{base_name}_{lod_tag}")

    @staticmethod
    def _format_metric(key: str, value) -> str:
        if isinstance(value, (float, int)):
            return f"\033[31m{key}:{value:.2f}\033[0m"
        return f"\033[31m{key}:{value}\033[0m"

    def _keep_viewer_alive(self) -> None:
        if self.args.viewer_mode == "web":
            while True:
                time.sleep(1)
        self.viewer.throttling = False
        while self.viewer.running:
            time.sleep(1)


def main() -> None:
    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    np.random.seed(0)

    cfg, clean_argv = _parse_progressive_args(sys.argv)
    args = _parse_training_args(clean_argv)
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
