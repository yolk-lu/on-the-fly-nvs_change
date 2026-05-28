from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass


@dataclass
class DataConfig:
    source_path: str
    images_dir: str = "images"
    masks_dir: str = ""
    num_loader_threads: int = 4
    downsampling: float = -1.0
    pyr_levels: int = 2
    min_displacement: float = 0.03
    start_at: int = 0
    test_hold: int = -1
    eval_poses: bool = False
    use_colmap_poses: bool = False


@dataclass
class FeatureConfig:
    num_kpts: int = int(4096 * 1.5)
    feature_backend: str = "xfeat"
    matcher_backend: str = "mnn"
    lightglue_filter_threshold: float = 0.1
    lightglue_depth_confidence: float = 0.95
    lightglue_width_confidence: float = 0.99
    match_max_error: float = 2e-3
    fundmat_samples: int = 1000
    min_num_inliers: int = 20
    use_semantic_features: bool = False
    sem_feat_dim: int = 128
    sem_weight: float = 0.3


@dataclass
class PoseConfig:
    num_keyframes_miniba_bootstrap: int = 8
    num_pts_miniba_bootstrap: int = 2000
    iters_miniba_bootstrap: int = 200
    fix_focal: bool = False
    init_focal: float = -1.0
    init_fov: float = -1.0
    num_prev_keyframes_miniba_incr: int = 6
    num_prev_keyframes_check: int = 20
    pnpransac_samples: int = 2000
    num_pts_miniba_incr: int = 2000
    iters_miniba_incr: int = 20
    use_parallax_ba: bool = True
    parallax_ba_iters: int = 4
    parallax_ref_weight: float = 0.25
    enable_reboot: bool = False
    pose_retry_max_attempts: int = 0
    pose_retry_queue_size: int = 16
    pose_retry_per_success: int = 0
    use_vggt_pose_prior: bool = False
    vggt_pose_prior_path: str = ""
    pose_use_lsf_velocity_gate: bool = False
    pose_lsf_order: int = 2
    pose_lsf_window: int = 8
    pose_jump_max_ratio: float = 0.35
    pose_vel_angle_max_deg: float = 25.0
    pose_lsf_force_accept_after: int = 0
    pose_retry_rescue_when_full: bool = False
    relocalization_top_k: int = 16
    relocalization_min_triangulated: int = 4
    relocalization_max_keyframes: int = 6


@dataclass
class AnchorConfig:
    sh_degree: int = 3
    anchor_radius: float = 5.0
    anchor_min_keyframes: int = 20
    anchor_origin_median_window: int = 5
    max_active_keyframes: int = 200
    max_anchor_gaussians: int = 250_000
    max_anchor_keyframes: int = 80
    max_anchor_tsdf_voxels: int = 0
    max_anchor_vram_mb: float = 0.0
    anchor_final_iterations: int = 5
    anchor_merge_voxel_size: float = 0.05
    anchor_merge_target_gaussians: int = 150_000
    async_mapping: bool = False
    local_spawn_max: int = 16_384
    local_spawn_target: int = 8_192
    surface_sample_floor: float = 0.01
    low_frequency_spawn_fraction: float = 0.15
    edge_probability_threshold: float = 0.02
    spawn_opacity_init: float = 0.06
    mvs_depth_consistency_idepth: float = 0.25
    mvs_mono_fallback_fraction: float = 0.25
    mvs_mono_fallback_max_fraction: float = 0.50
    mvs_target_ratio: float = 0.35
    mono_fallback_min_points: int = 2048
    tsdf_fusion_mode: str = "disabled"
    delayed_tsdf_min_opacity: float = 0.05
    delayed_tsdf_max_samples: int = 250_000
    max_rasterized_gaussians: int = 30_000
    anchor_render_check_every: int = 25
    anchor_iterations: int = 20
    anchor_train_views: int = 4
    loop_check_max_candidates: int = 4
    loop_min_anchor_gap: int = 2
    overlap_mode: str = "reserved"
    backend_mode: str = "progressive_scene_model"


@dataclass
class LossConfig:
    lambda_dssim: float = 0.2
    use_last_frame_proba: float = 0.2
    depth_loss_weight_init: float = 3e-2
    depth_loss_weight_decay: float = 0.9
    depth_valid_epsilon: float = 1e-6
    tsdf_loss_weight: float = 0.0
    anisotropy_loss_weight: float = 1e-4
    max_gaussian_aspect_ratio: float = 8.0
    rgb_visible_weight: float = 0.85
    ssim_min_coverage: float = 0.08
    depth_conf_min: float = 0.35
    robust_loss_epsilon: float = 1e-3


@dataclass
class OptimizerConfig:
    lr_poses: float = 1e-4
    lr_exposure: float = 5e-4
    lr_depth_scale_offset: float = 1e-4
    position_lr_init: float = 5e-5
    position_lr_decay: float = 1 - 2e-5
    feature_lr: float = 0.005
    opacity_lr: float = 0.1
    scaling_lr: float = 0.01
    rotation_lr: float = 0.002


@dataclass
class SpawnConfig:
    init_proba_scaler: float = 2.0


@dataclass
class OutputConfig:
    model_path: str = ""
    save_every: int = -1
    display_runtimes: bool = False
    smoke_only: bool = False
    input_model: str = "sparse/0"
    input_format: str = ".bin"
    output_model: str | None = None
    output_format: str = ".bin"
    manifest_name: str = "progressive_manifest.json"
    run_label: str = "progressive_train"
    lod_min: int = 1
    lod_max: int = 1
    save_test_renders: bool = True
    test_render_every: int = 1
    test_render_dir: str = "test_renders"


@dataclass
class ScaleConfig:
    scale_formula: str = "global_scale=median(reference_idepth/new_idepth), s_new=s_prev*global_scale"
    min_conf: float = 0.35
    grid_size: int = 16
    min_samples_per_cell: int = 8
    min_triangulated_samples: int = 3
    min_depth_scale: float = 1e-3
    max_depth_scale: float = 1e3
    require_calibrated_depth: bool = True


class ProgressiveConfig:
    def __init__(
        self,
        data: DataConfig,
        feature: FeatureConfig,
        pose: PoseConfig,
        anchor: AnchorConfig,
        loss: LossConfig,
        optimizer: OptimizerConfig,
        spawn: SpawnConfig,
        output: OutputConfig,
        scale: ScaleConfig | None = None,
    ):
        self.data = data
        self.feature = feature
        self.pose = pose
        self.anchor = anchor
        self.loss = loss
        self.optimizer = optimizer
        self.spawn = spawn
        self.output = output
        self.scale = scale or ScaleConfig()
        self._expose_legacy_fields()

    def _expose_legacy_fields(self) -> None:
        for group in (self.data, self.feature, self.pose, self.anchor, self.loss, self.optimizer, self.spawn, self.output):
            for key, value in asdict(group).items():
                setattr(self, key, value)
        self.progressive_overlap_mode = self.anchor.overlap_mode
        self.progressive_backend_mode = self.anchor.backend_mode
        self.progressive_manifest_name = self.output.manifest_name
        self.progressive_run_label = self.output.run_label
        self.progressive_anchor_radius = self.anchor.anchor_radius
        self.progressive_anchor_min_keyframes = self.anchor.anchor_min_keyframes
        self.progressive_anchor_origin_median_window = self.anchor.anchor_origin_median_window
        self.progressive_max_anchor_gaussians = self.anchor.max_anchor_gaussians
        self.progressive_max_anchor_keyframes = self.anchor.max_anchor_keyframes
        self.progressive_max_anchor_tsdf_voxels = self.anchor.max_anchor_tsdf_voxels
        self.progressive_max_anchor_vram_mb = self.anchor.max_anchor_vram_mb
        self.progressive_anchor_final_iterations = self.anchor.anchor_final_iterations
        self.progressive_anchor_merge_voxel_size = self.anchor.anchor_merge_voxel_size
        self.progressive_anchor_merge_target_gaussians = self.anchor.anchor_merge_target_gaussians
        self.progressive_async_mapping = self.anchor.async_mapping
        self.progressive_tsdf_loss_weight = self.loss.tsdf_loss_weight
        self.progressive_anisotropy_loss_weight = self.loss.anisotropy_loss_weight
        self.progressive_rgb_visible_weight = self.loss.rgb_visible_weight
        self.progressive_ssim_min_coverage = self.loss.ssim_min_coverage
        self.progressive_depth_conf_min = self.loss.depth_conf_min
        self.progressive_robust_loss_epsilon = self.loss.robust_loss_epsilon
        self.progressive_local_spawn_max = self.anchor.local_spawn_max
        self.progressive_local_spawn_target = self.anchor.local_spawn_target
        self.progressive_surface_sample_floor = self.anchor.surface_sample_floor
        self.progressive_low_frequency_spawn_fraction = self.anchor.low_frequency_spawn_fraction
        self.progressive_edge_probability_threshold = self.anchor.edge_probability_threshold
        self.progressive_spawn_opacity_init = self.anchor.spawn_opacity_init
        self.progressive_mvs_depth_consistency_idepth = self.anchor.mvs_depth_consistency_idepth
        self.progressive_mvs_mono_fallback_fraction = self.anchor.mvs_mono_fallback_fraction
        self.progressive_mvs_mono_fallback_max_fraction = self.anchor.mvs_mono_fallback_max_fraction
        self.progressive_mvs_target_ratio = self.anchor.mvs_target_ratio
        self.progressive_mono_fallback_min_points = self.anchor.mono_fallback_min_points
        self.progressive_tsdf_fusion_mode = self.anchor.tsdf_fusion_mode
        self.progressive_delayed_tsdf_min_opacity = self.anchor.delayed_tsdf_min_opacity
        self.progressive_delayed_tsdf_max_samples = self.anchor.delayed_tsdf_max_samples
        self.progressive_max_rasterized_gaussians = self.anchor.max_rasterized_gaussians
        self.progressive_anchor_render_check_every = self.anchor.anchor_render_check_every
        self.progressive_anchor_iterations = self.anchor.anchor_iterations
        self.progressive_anchor_train_views = self.anchor.anchor_train_views
        self.progressive_loop_max_candidates = self.anchor.loop_check_max_candidates
        self.progressive_loop_min_anchor_gap = self.anchor.loop_min_anchor_gap
        self.progressive_depth_scale_min_samples = self.scale.min_triangulated_samples
        self.progressive_min_depth_scale = self.scale.min_depth_scale
        self.progressive_max_depth_scale = self.scale.max_depth_scale
        self.progressive_require_calibrated_depth = self.scale.require_calibrated_depth
        self.depth_scale_min_samples = self.scale.min_triangulated_samples
        self.min_depth_scale = self.scale.min_depth_scale
        self.max_depth_scale = self.scale.max_depth_scale
        self.require_calibrated_depth = self.scale.require_calibrated_depth

    def to_manifest(self) -> dict:
        return {
            "data": asdict(self.data),
            "feature": asdict(self.feature),
            "pose": asdict(self.pose),
            "anchor": asdict(self.anchor),
            "loss": asdict(self.loss),
            "optimizer": asdict(self.optimizer),
            "spawn": asdict(self.spawn),
            "output": asdict(self.output),
            "scale": asdict(self.scale),
        }


def parse_progressive_config(argv: list[str]) -> ProgressiveConfig:
    parser = argparse.ArgumentParser(description="Progressive anchor-local training")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-i", "--images_dir", default="images")
    parser.add_argument("--masks_dir", default="")
    parser.add_argument("--num_loader_threads", type=int, default=4)
    parser.add_argument("--downsampling", type=float, default=-1.0)
    parser.add_argument("--pyr_levels", type=int, default=2)
    parser.add_argument("--min_displacement", type=float, default=0.03)
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--test_hold", type=int, default=-1)
    parser.add_argument("--eval_poses", action="store_true")
    parser.add_argument("--use_colmap_poses", action="store_true")

    parser.add_argument("--num_kpts", type=int, default=int(4096 * 1.5))
    parser.add_argument("--feature_backend", choices=["xfeat", "superpoint", "disk", "sift", "aliked"], default="xfeat")
    parser.add_argument("--matcher_backend", choices=["mnn", "lightglue"], default="mnn")
    parser.add_argument("--lightglue_filter_threshold", type=float, default=0.1)
    parser.add_argument("--lightglue_depth_confidence", type=float, default=0.95)
    parser.add_argument("--lightglue_width_confidence", type=float, default=0.99)
    parser.add_argument("--match_max_error", type=float, default=2e-3)
    parser.add_argument("--fundmat_samples", type=int, default=1000)
    parser.add_argument("--min_num_inliers", type=int, default=20)
    parser.add_argument("--use_semantic_features", action="store_true")
    parser.add_argument("--sem_feat_dim", type=int, default=128)
    parser.add_argument("--sem_weight", type=float, default=0.3)

    parser.add_argument("--num_keyframes_miniba_bootstrap", type=int, default=8)
    parser.add_argument("--num_pts_miniba_bootstrap", type=int, default=2000)
    parser.add_argument("--iters_miniba_bootstrap", type=int, default=200)
    parser.add_argument("--fix_focal", action="store_true")
    parser.add_argument("--init_focal", type=float, default=-1.0)
    parser.add_argument("--init_fov", type=float, default=-1.0)
    parser.add_argument("--num_prev_keyframes_miniba_incr", type=int, default=6)
    parser.add_argument("--num_prev_keyframes_check", type=int, default=20)
    parser.add_argument("--pnpransac_samples", type=int, default=2000)
    parser.add_argument("--num_pts_miniba_incr", type=int, default=2000)
    parser.add_argument("--iters_miniba_incr", type=int, default=20)
    parser.add_argument("--use_parallax_ba", action="store_true", default=True)
    parser.add_argument("--no_parallax_ba", dest="use_parallax_ba", action="store_false")
    parser.add_argument("--parallax_ba_iters", type=int, default=4)
    parser.add_argument("--parallax_ref_weight", type=float, default=0.25)
    parser.add_argument("--enable_reboot", action="store_true")
    parser.add_argument("--use_vggt_pose_prior", action="store_true")
    parser.add_argument("--vggt_pose_prior_path", default="")
    parser.add_argument("--relocalization_top_k", type=int, default=16)
    parser.add_argument("--relocalization_min_triangulated", type=int, default=4)
    parser.add_argument("--relocalization_max_keyframes", type=int, default=6)

    parser.add_argument("--progressive_overlap_mode", choices=["reserved", "off"], default="reserved")
    parser.add_argument("--progressive_backend_mode", choices=["progressive_scene_model"], default="progressive_scene_model")
    parser.add_argument("--max_active_keyframes", type=int, default=200)
    parser.add_argument("--progressive_anchor_radius", type=float, default=5.0)
    parser.add_argument("--progressive_anchor_min_keyframes", type=int, default=20)
    parser.add_argument("--progressive_anchor_origin_median_window", type=int, default=5)
    parser.add_argument("--progressive_max_anchor_gaussians", type=int, default=250_000)
    parser.add_argument("--progressive_max_anchor_keyframes", type=int, default=80)
    parser.add_argument("--progressive_max_anchor_tsdf_voxels", type=int, default=0)
    parser.add_argument("--progressive_max_anchor_vram_mb", type=float, default=0.0)
    parser.add_argument("--progressive_anchor_final_iterations", type=int, default=5)
    parser.add_argument("--progressive_anchor_merge_voxel_size", type=float, default=0.05)
    parser.add_argument("--progressive_anchor_merge_target_gaussians", type=int, default=150_000)
    parser.add_argument("--progressive_async_mapping", action="store_true")
    parser.add_argument("--progressive_local_spawn_max", type=int, default=16_384)
    parser.add_argument("--progressive_local_spawn_target", type=int, default=8_192)
    parser.add_argument("--progressive_surface_sample_floor", type=float, default=0.01)
    parser.add_argument("--progressive_low_frequency_spawn_fraction", type=float, default=0.15)
    parser.add_argument("--progressive_edge_probability_threshold", type=float, default=0.02)
    parser.add_argument("--progressive_spawn_opacity_init", type=float, default=0.06)
    parser.add_argument("--progressive_mvs_depth_consistency_idepth", type=float, default=0.25)
    parser.add_argument("--progressive_mvs_mono_fallback_fraction", type=float, default=0.25)
    parser.add_argument("--progressive_mvs_mono_fallback_max_fraction", type=float, default=0.50)
    parser.add_argument("--progressive_mvs_target_ratio", type=float, default=0.35)
    parser.add_argument("--progressive_mono_fallback_min_points", type=int, default=2048)
    parser.add_argument(
        "--progressive_tsdf_fusion_mode",
        choices=["disabled", "delayed_gaussian", "immediate_depth"],
        default="disabled",
    )
    parser.add_argument("--progressive_delayed_tsdf_min_opacity", type=float, default=0.05)
    parser.add_argument("--progressive_delayed_tsdf_max_samples", type=int, default=250_000)
    parser.add_argument("--progressive_max_rasterized_gaussians", type=int, default=30_000)
    parser.add_argument("--progressive_anchor_render_check_every", type=int, default=25)
    parser.add_argument("--progressive_anchor_iterations", type=int, default=20)
    parser.add_argument("--progressive_anchor_train_views", type=int, default=4)
    parser.add_argument("--progressive_loop_max_candidates", type=int, default=4)
    parser.add_argument("--progressive_loop_min_anchor_gap", type=int, default=2)

    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--use_last_frame_proba", type=float, default=0.2)
    parser.add_argument("--depth_loss_weight_init", type=float, default=3e-2)
    parser.add_argument("--depth_loss_weight_decay", type=float, default=0.9)
    parser.add_argument("--progressive_depth_valid_epsilon", type=float, default=1e-6)
    parser.add_argument("--progressive_tsdf_loss_weight", type=float, default=0.0)
    parser.add_argument("--progressive_anisotropy_loss_weight", type=float, default=1e-4)
    parser.add_argument("--max_gaussian_aspect_ratio", type=float, default=8.0)
    parser.add_argument("--progressive_rgb_visible_weight", type=float, default=0.85)
    parser.add_argument("--progressive_ssim_min_coverage", type=float, default=0.08)
    parser.add_argument("--progressive_depth_conf_min", type=float, default=0.35)
    parser.add_argument("--progressive_robust_loss_epsilon", type=float, default=1e-3)
    parser.add_argument("--lr_poses", type=float, default=1e-4)
    parser.add_argument("--lr_exposure", type=float, default=5e-4)
    parser.add_argument("--lr_depth_scale_offset", type=float, default=1e-4)
    parser.add_argument("--position_lr_init", type=float, default=5e-5)
    parser.add_argument("--position_lr_decay", type=float, default=1 - 2e-5)
    parser.add_argument("--feature_lr", type=float, default=0.005)
    parser.add_argument("--opacity_lr", type=float, default=0.1)
    parser.add_argument("--scaling_lr", type=float, default=0.01)
    parser.add_argument("--rotation_lr", type=float, default=0.002)
    parser.add_argument("--init_proba_scaler", type=float, default=2.0)
    parser.add_argument("--sh_degree", type=int, default=3)

    parser.add_argument("-m", "--model_path", default="")
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument("--display_runtimes", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--input_model", default="sparse/0")
    parser.add_argument("--input_format", default=".bin")
    parser.add_argument("--output_model", default=None)
    parser.add_argument("--output_format", default=".bin")
    parser.add_argument("--progressive_manifest_name", default="progressive_manifest.json")
    parser.add_argument("--progressive_run_label", default="progressive_train")
    parser.add_argument("--progressive_lod", type=int, default=1)
    parser.add_argument("--save_test_renders", action="store_true", default=True)
    parser.add_argument("--no_save_test_renders", dest="save_test_renders", action="store_false")
    parser.add_argument("--test_render_every", type=int, default=1)
    parser.add_argument("--test_render_dir", default="test_renders")
    parser.add_argument("--progressive_depth_scale_min_samples", type=int, default=3)
    parser.add_argument("--progressive_min_depth_scale", type=float, default=1e-3)
    parser.add_argument("--progressive_max_depth_scale", type=float, default=1e3)
    parser.add_argument("--progressive_require_calibrated_depth", action="store_true", default=True)
    parser.add_argument("--no_progressive_require_calibrated_depth", dest="progressive_require_calibrated_depth", action="store_false")

    parsed = parser.parse_args(argv[1:])
    if parsed.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        parsed.model_path = f"results/{i:06d}"

    return ProgressiveConfig(
        data=DataConfig(
            source_path=parsed.source_path,
            images_dir=parsed.images_dir,
            masks_dir=parsed.masks_dir,
            num_loader_threads=parsed.num_loader_threads,
            downsampling=parsed.downsampling,
            pyr_levels=parsed.pyr_levels,
            min_displacement=parsed.min_displacement,
            start_at=parsed.start_at,
            test_hold=parsed.test_hold,
            eval_poses=parsed.eval_poses,
            use_colmap_poses=parsed.use_colmap_poses,
        ),
        feature=FeatureConfig(
            num_kpts=parsed.num_kpts,
            feature_backend=parsed.feature_backend,
            matcher_backend=parsed.matcher_backend,
            lightglue_filter_threshold=parsed.lightglue_filter_threshold,
            lightglue_depth_confidence=parsed.lightglue_depth_confidence,
            lightglue_width_confidence=parsed.lightglue_width_confidence,
            match_max_error=parsed.match_max_error,
            fundmat_samples=parsed.fundmat_samples,
            min_num_inliers=parsed.min_num_inliers,
            use_semantic_features=parsed.use_semantic_features,
            sem_feat_dim=parsed.sem_feat_dim,
            sem_weight=parsed.sem_weight,
        ),
        pose=PoseConfig(
            num_keyframes_miniba_bootstrap=parsed.num_keyframes_miniba_bootstrap,
            num_pts_miniba_bootstrap=parsed.num_pts_miniba_bootstrap,
            iters_miniba_bootstrap=parsed.iters_miniba_bootstrap,
            fix_focal=parsed.fix_focal,
            init_focal=parsed.init_focal,
            init_fov=parsed.init_fov,
            num_prev_keyframes_miniba_incr=parsed.num_prev_keyframes_miniba_incr,
            num_prev_keyframes_check=parsed.num_prev_keyframes_check,
            pnpransac_samples=parsed.pnpransac_samples,
            num_pts_miniba_incr=parsed.num_pts_miniba_incr,
            iters_miniba_incr=parsed.iters_miniba_incr,
            use_parallax_ba=parsed.use_parallax_ba,
            parallax_ba_iters=parsed.parallax_ba_iters,
            parallax_ref_weight=parsed.parallax_ref_weight,
            enable_reboot=parsed.enable_reboot,
            use_vggt_pose_prior=parsed.use_vggt_pose_prior,
            vggt_pose_prior_path=parsed.vggt_pose_prior_path,
            relocalization_top_k=parsed.relocalization_top_k,
            relocalization_min_triangulated=parsed.relocalization_min_triangulated,
            relocalization_max_keyframes=parsed.relocalization_max_keyframes,
        ),
        anchor=AnchorConfig(
            sh_degree=parsed.sh_degree,
            anchor_radius=parsed.progressive_anchor_radius,
            anchor_min_keyframes=parsed.progressive_anchor_min_keyframes,
            anchor_origin_median_window=parsed.progressive_anchor_origin_median_window,
            max_active_keyframes=parsed.max_active_keyframes,
            max_anchor_gaussians=parsed.progressive_max_anchor_gaussians,
            max_anchor_keyframes=parsed.progressive_max_anchor_keyframes,
            max_anchor_tsdf_voxels=parsed.progressive_max_anchor_tsdf_voxels,
            max_anchor_vram_mb=parsed.progressive_max_anchor_vram_mb,
            anchor_final_iterations=parsed.progressive_anchor_final_iterations,
            anchor_merge_voxel_size=parsed.progressive_anchor_merge_voxel_size,
            anchor_merge_target_gaussians=parsed.progressive_anchor_merge_target_gaussians,
            async_mapping=parsed.progressive_async_mapping,
            local_spawn_max=parsed.progressive_local_spawn_max,
            local_spawn_target=parsed.progressive_local_spawn_target,
            surface_sample_floor=parsed.progressive_surface_sample_floor,
            low_frequency_spawn_fraction=parsed.progressive_low_frequency_spawn_fraction,
            edge_probability_threshold=parsed.progressive_edge_probability_threshold,
            spawn_opacity_init=parsed.progressive_spawn_opacity_init,
            mvs_depth_consistency_idepth=parsed.progressive_mvs_depth_consistency_idepth,
            mvs_mono_fallback_fraction=parsed.progressive_mvs_mono_fallback_fraction,
            mvs_mono_fallback_max_fraction=parsed.progressive_mvs_mono_fallback_max_fraction,
            mvs_target_ratio=parsed.progressive_mvs_target_ratio,
            mono_fallback_min_points=parsed.progressive_mono_fallback_min_points,
            tsdf_fusion_mode=parsed.progressive_tsdf_fusion_mode,
            delayed_tsdf_min_opacity=parsed.progressive_delayed_tsdf_min_opacity,
            delayed_tsdf_max_samples=parsed.progressive_delayed_tsdf_max_samples,
            max_rasterized_gaussians=parsed.progressive_max_rasterized_gaussians,
            anchor_render_check_every=parsed.progressive_anchor_render_check_every,
            anchor_iterations=parsed.progressive_anchor_iterations,
            anchor_train_views=parsed.progressive_anchor_train_views,
            loop_check_max_candidates=parsed.progressive_loop_max_candidates,
            loop_min_anchor_gap=parsed.progressive_loop_min_anchor_gap,
            overlap_mode=parsed.progressive_overlap_mode,
            backend_mode=parsed.progressive_backend_mode,
        ),
        loss=LossConfig(
            lambda_dssim=parsed.lambda_dssim,
            use_last_frame_proba=parsed.use_last_frame_proba,
            depth_loss_weight_init=parsed.depth_loss_weight_init,
            depth_loss_weight_decay=parsed.depth_loss_weight_decay,
            depth_valid_epsilon=parsed.progressive_depth_valid_epsilon,
            tsdf_loss_weight=parsed.progressive_tsdf_loss_weight,
            anisotropy_loss_weight=parsed.progressive_anisotropy_loss_weight,
            max_gaussian_aspect_ratio=parsed.max_gaussian_aspect_ratio,
            rgb_visible_weight=parsed.progressive_rgb_visible_weight,
            ssim_min_coverage=parsed.progressive_ssim_min_coverage,
            depth_conf_min=parsed.progressive_depth_conf_min,
            robust_loss_epsilon=parsed.progressive_robust_loss_epsilon,
        ),
        optimizer=OptimizerConfig(
            lr_poses=parsed.lr_poses,
            lr_exposure=parsed.lr_exposure,
            lr_depth_scale_offset=parsed.lr_depth_scale_offset,
            position_lr_init=parsed.position_lr_init,
            position_lr_decay=parsed.position_lr_decay,
            feature_lr=parsed.feature_lr,
            opacity_lr=parsed.opacity_lr,
            scaling_lr=parsed.scaling_lr,
            rotation_lr=parsed.rotation_lr,
        ),
        spawn=SpawnConfig(init_proba_scaler=parsed.init_proba_scaler),
        output=OutputConfig(
            model_path=parsed.model_path,
            save_every=parsed.save_every,
            display_runtimes=parsed.display_runtimes,
            smoke_only=parsed.smoke_only,
            input_model=parsed.input_model,
            input_format=parsed.input_format,
            output_model=parsed.output_model,
            output_format=parsed.output_format,
            manifest_name=parsed.progressive_manifest_name,
            run_label=parsed.progressive_run_label,
            lod_min=parsed.progressive_lod,
            lod_max=parsed.progressive_lod,
            save_test_renders=parsed.save_test_renders,
            test_render_every=parsed.test_render_every,
            test_render_dir=parsed.test_render_dir,
        ),
        scale=ScaleConfig(
            min_triangulated_samples=parsed.progressive_depth_scale_min_samples,
            min_depth_scale=parsed.progressive_min_depth_scale,
            max_depth_scale=parsed.progressive_max_depth_scale,
            require_calibrated_depth=parsed.progressive_require_calibrated_depth,
        ),
    )
