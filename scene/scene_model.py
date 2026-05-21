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

from argparse import Namespace
import gc
import os
import json
import math
import csv
import threading
import time
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision

import lpips
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distIndex2
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.guided_mvs import GuidedMVS
from scene.optimizers import SparseGaussianAdam
from scene.keyframe import Keyframe
from scene.anchor import Anchor
from scene.LoD_utils import (
    
    projected_major_axis_px,
    screenspace_clamp,
    validate_merged_gaussians,
)
from utils import (
    RGB2SH,
    depth2points,
    focal2fov,
    get_lapla_norm,
    getProjectionMatrix,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    psnr,
    rotation_distance,
)
from dataloaders.read_write_model import write_model


class SceneModel:
    """
    Scene Model class that contains the scene's Gaussians, anchors, keyframes, and methods for rendering and optimization.
    """
    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher: Matcher = None,
        inference_mode: bool = False,
    ):
        """
        Args:
            width: Width of the image.
            height: Height of the image.
            args: Arguments for the scene model. Should always have anchor_overlap, and training parameters if inference_mode is False.
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.width = width
        self.height = height
        self.matcher = matcher
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")
        self.anchor_overlap = getattr(args, "anchor_overlap", 0.0)
        self.model_path = getattr(args, "model_path", "/tmp/lod_debug_fallback")
        self.progressive_loss_hook = None
        self.progressive_tsdf_loss_weight = float(getattr(args, "progressive_tsdf_loss_weight", 0.0))
        self.progressive_anisotropy_loss_weight = float(getattr(args, "progressive_anisotropy_loss_weight", 0.0))
        self.optimization_thread = None
        self.debug_log_lock = threading.Lock()

        try:
            import sys

            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            self.lpips = lpips.LPIPS(net="vgg").cuda()
            sys.stdout = original_stdout
        except:
            self.lpips = None

        if not inference_mode:
            self.current_lod = getattr(args, "lod_min", 1)
            self.num_prev_keyframes_check = args.num_prev_keyframes_check
            self.active_sh_degree = args.sh_degree
            self.max_sh_degree = args.sh_degree
            self.lambda_dssim = args.lambda_dssim
            self.init_proba_scaler = args.init_proba_scaler
            self.max_active_keyframes = args.max_active_keyframes
            self.use_last_frame_proba = args.use_last_frame_proba
            self.active_frames_cpu = []
            self.active_frames_gpu = []
            self.guided_mvs = GuidedMVS(args)
            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            ## Initialize Gaussian parameters
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": args.position_lr_init,
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device="cuda"),
                    "lr": args.feature_lr,
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device="cuda",
                    ),
                    "lr": args.feature_lr / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": args.scaling_lr,
                },
                "rotation": {
                    "val": torch.empty(0, 4, device="cuda"),
                    "lr": args.rotation_lr,
                },
                "opacity": {
                    "val": torch.empty(0, 1, device="cuda"),
                    "lr": args.opacity_lr,
                },
            }
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            ## Initialize optimizer
            self.reset_optimizer()

        self.keyframes = []
        self.anchor_weights = [1.0]
        self.f = 0.7 * width
        self.init_intrinsics()

        self.approx_cam_centres = None
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
        self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        self.lock = threading.Lock()
        self.inference_mode = inference_mode

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )

    def verify_no_nans(self, context=""):
        for k, v in self.gaussian_params.items():
            if torch.isnan(v["val"]).any() or torch.isinf(v["val"]).any():
                print(f"\n[FATAL] NaN/Inf detected in {k} at '{context}'!!!\n")
                import sys; sys.exit(1)
        if len(self.keyframes) > 0 and torch.isnan(self.keyframes[-1].get_Rt()).any():
            print(f"\n[FATAL] NaN detected in Camera Poses at '{context}'!!!\n")
            import sys; sys.exit(1)

    def reset_optimizer(self):
        for key in self.gaussian_params:
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict
        )

    @property
    def xyz(self):
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]

    @classmethod
    def from_scene(cls, scene_dir: str, args):
        with open(os.path.join(scene_dir, "metadata.json")) as f:
            metadata = json.load(f)

        width = metadata["config"]["width"]
        height = metadata["config"]["height"]
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.active_sh_degree = metadata["config"]["sh_degree"]
        scene_model.max_sh_degree = metadata["config"]["sh_degree"]

        # Load anchors
        scene_model.anchors = []
        for i in range(len(metadata["anchors"])):
            scene_model.anchors.append(
                Anchor.from_ply(
                    os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
                    torch.tensor(metadata["anchors"][i]["position"]),
                    metadata["config"]["sh_degree"],
                )
            )

        scene_model.active_anchor = scene_model.anchors[0]

        # Load keyframes
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(metadata["keyframes"][i], i, width, height)
            scene_model.add_keyframe(keyframe)

        return scene_model

    @property
    def first_active_frame(self):
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        return self.last_active_frame - self.first_active_frame + 1

    def optimization_step(self, finetuning=False):
        if len(self.xyz) == 0:
            return
        # Select which keyframe to train on
        # We train on the latest keyframe with self.use_last_frame_proba probability or a random keyframe otherwise
        if (
            np.random.rand() > self.use_last_frame_proba
            or self.last_trained_id == -1
            or finetuning
        ):
            keyframe_id = np.random.choice(self.active_frames_gpu)
        else:
            keyframe_id = -1
        keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl

        # Zero gradients
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        # Render image and depth
        render_pkg = self.render_from_id(
            keyframe_id, pyr_lvl=lvl, bg=torch.rand(3, device="cuda")
        )
        
        if render_pkg["radii"].max() == 0:
            
            if not hasattr(self, 'latest_loss_stats'):
                self.latest_loss_stats = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "depth": 0.0}
            return self.latest_loss_stats

        image = render_pkg["render"].contiguous()
        invdepth = render_pkg["invdepth"].contiguous()

        gt_image = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)

        # Mask image and depth if necessary
        if keyframe.mask_pyr is not None:
            image = image * keyframe.mask_pyr[lvl]
            gt_image = gt_image * keyframe.mask_pyr[lvl]
            invdepth = invdepth * keyframe.mask_pyr[lvl]
            mono_idepth = mono_idepth * keyframe.mask_pyr[lvl]

        # Loss
        l1_loss = (image - gt_image).abs().mean()
        ssim_loss = 1 - fused_ssim(image[None], gt_image[None])
        depth_loss = (invdepth - mono_idepth).abs().mean()
        progressive_tsdf_loss = depth_loss * 0.0
        progressive_anisotropy_loss = depth_loss * 0.0
        if self.progressive_loss_hook is not None:
            progressive_losses = self.progressive_loss_hook(self)
            progressive_tsdf_loss = progressive_losses.get("tsdf", progressive_tsdf_loss)
            progressive_anisotropy_loss = progressive_losses.get("anisotropy", progressive_anisotropy_loss)
        # Measure pure depth impact on xyz gradients
        self.optimizer.zero_grad()
        (keyframe.depth_loss_weight * depth_loss).backward(retain_graph=True)
        depth_xyz_grad_norm = 0.0
        if self.gaussian_params["xyz"]["val"].grad is not None:
            depth_xyz_grad_norm = float(self.gaussian_params["xyz"]["val"].grad.detach().norm(p=2).item())

        self.optimizer.zero_grad()

        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
            + self.progressive_tsdf_loss_weight * progressive_tsdf_loss
            + self.progressive_anisotropy_loss_weight * progressive_anisotropy_loss
        )
        loss.backward()

        # Optimizers
        with torch.no_grad():
            # Pose optimization
            keyframe.step()

            # Skip the scene optimization if the current keyframe is a test keyframe
            if not keyframe.info["is_test"]:
                # Scene Gaussian optimization
                self.optimizer.step(
                    render_pkg["visibility_filter"], render_pkg["radii"].shape[0]
                )

            keyframe.latest_invdepth = render_pkg["invdepth"].detach()

        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id
        
        self.verify_no_nans("optimization_step End")
        
        self.latest_loss_stats = {
            "total": float(loss.detach().item()),
            "l1": float(l1_loss.detach().item()),
            "ssim": float(ssim_loss.detach().item()),
            "depth": float(depth_loss.detach().item()),
            "tsdf": float(progressive_tsdf_loss.detach().item()),
            "anisotropy": float(progressive_anisotropy_loss.detach().item()),
            "depth_xyz_grad": depth_xyz_grad_norm,
        }
        return self.latest_loss_stats

    def optimization_loop(self, n_iters: int, run_until_interupt: bool = False):
        """
        Runs at least n_iters optimization steps.
        If run_until_interupt, also runs until join_optimization_thread is called (Useful to run the optimization until the next keyframe is added in streaming mode).
        """
        self.interupt_optimization = False
        i = 0
        loss_stats = []
        while i < n_iters or (run_until_interupt and not self.interupt_optimization): 
            stats = self.optimization_step()
            if stats is not None:
                loss_stats.append(stats)
            i += 1
            
        import numpy as np
        if len(loss_stats) == 0:
            return None
        keys = ["total", "l1", "ssim", "depth"]
        return {key: float(np.mean([item[key] for item in loss_stats])) for key in keys}
        
    def join_optimization_thread(self):
        """
        Interupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None
    
    def optimize_async(self, n_iters: int):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters, True)
        )
        self.optimization_thread.start()

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Harmonizes the exposure matrices of test keyframes by averaging the exposure of the previous and next keyframes."""
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.info["is_test"]:
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                keyframe.exposure = (
                    self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                ) / 2

    @torch.no_grad()
    def evaluate(self, eval_poses=False, with_LPIPS=False, all=False):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()

        # Compute image quality metrics
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            metrics["LPIPS"] = 0
        n_test_frames = 0
        start_index = 0 if all else self.active_anchor.keyframe_ids[0]
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.info["is_test"]:
                gt_image = keyframe.image_pyr[0].cuda()
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                mask = (
                    keyframe.mask_pyr[0].cuda()
                    if keyframe.mask_pyr is not None
                    else torch.ones_like(image[:1] > 0)
                )
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                metrics["PSNR"] += psnr(image[mask], gt_image[mask])
                metrics["SSIM"] += fused_ssim(
                    image[None], gt_image[None], train=False
                ).item()
                if with_LPIPS and self.lpips is not None:
                    metrics["LPIPS"] += self.lpips(image[None], gt_image[None]).item()
                n_test_frames += 1

        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}

        # Compute pose errors
        if eval_poses:
            Rts = self.get_Rts()
            gt_Rts = self.get_gt_Rts(align=False)
            if len(Rts) == len(gt_Rts):
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)
                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                metrics["R°"] = R_error.mean().item() * 180 / math.pi
                metrics["t"] = t_error.mean().item()

        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)

        for keyframe in self.keyframes:
            if keyframe.info["is_test"]:
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Combine visualizations (Render, Depth, Normal, Laplace)
                img = keyframe.image_pyr[0].cuda()
                img_down = F.avg_pool2d(img, 2)
                img_down = F.interpolate(img_down[None], (self.height, self.width), mode="bilinear", align_corners=True)[0]
                init_proba = get_lapla_norm(img_down, self.disc_kernel)
                proba_img = init_proba / init_proba.max().clamp_min(1e-5)
                proba_img_3c = proba_img.expand(3, -1, -1)

                # Use Monocular Depth (from UniDepth / DepthAnything) instead of Rendered Depth
                mono_depth_raw = keyframe.mono_idepth.cuda()
                while mono_depth_raw.dim() > 2:
                    mono_depth_raw = mono_depth_raw.squeeze(0)
                if mono_depth_raw.dim() == 2:
                    mono_depth_raw = mono_depth_raw.unsqueeze(0)
                
                # Resize mono depth to match RGB resolution (H, W)
                mono_depth_up = F.interpolate(mono_depth_raw.unsqueeze(0), size=(self.height, self.width), mode="bilinear", align_corners=True)[0]
                
                depth_norm = mono_depth_up / mono_depth_up.max().clamp_min(1e-5)
                depth_img_3c = depth_norm.expand(3, -1, -1)

                # Compute Monocular Fake Normals using Depth Gradients
                dy, dx = torch.gradient(mono_depth_up.squeeze(0))
                # For inverse depth/depth, gradient directions might differ slightly in sign, but this is for viewing.
                normal = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=0)
                normal = normal / torch.linalg.vector_norm(normal, dim=0, keepdim=True).clamp_min(1e-5)
                normal_3c = (normal + 1) / 2.0

                render_img = render_pkg["render"].detach().clamp(0, 1)
                
                grid = torchvision.utils.make_grid([render_img, depth_img_3c, normal_3c, proba_img_3c], nrow=2)
                grid_name = "combined_" + keyframe.info["name"]
                if grid_name.endswith((".png", ".jpg", ".jpeg")):
                    grid_name = grid_name.rsplit(".", 1)[0] + ".png"
                torchvision.utils.save_image(grid, os.path.join(out_dir, grid_name))

                is_jpeg = os.path.splitext(keyframe.info["name"])[-1].lower() in [
                    ".jpg",
                    ".jpeg",
                ]
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.info["name"]), image, write_flag
                )

    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        scaling_modifier=1,
        bg=torch.zeros(3, device="cuda"),
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().transpose(0, 1)
        scale = 2**pyr_lvl
        width, height = self.width // scale, self.height // scale
        render_pkg = self.render(width, height, view_matrix, scaling_modifier, bg)
        render_pkg["render"] = (
            keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
        ) + keyframe.exposure[:3, 3, None]
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3, device="cuda"),
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
    ):
        cam_centre = view_matrix.detach().inverse()[3, :3]

        # Use the scene's intrinsic parameters if not provided
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # Use the provided FOV values
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)
                .transpose(0, 1)
                .cuda()
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,
            projection_matrix,
            self.active_sh_degree,
            cam_centre,
            False,
            False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        with self.lock:
            # Load and blend anchors if in inference mode 
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(cam_centre, self.anchors, self.anchor_overlap)
            screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)
            if self.xyz.shape[0] > 0:
                # Set constant scaling and opacity to visualize the Gaussians' positions in the top view
                if top_view:
                    scaling = torch.ones_like(self.scaling) * scaling_modifier
                    opacity = torch.ones_like(self.opacity)
                else:
                    # Use tested screenspace_clamp (log-space → log-space, never mutates stored params)
                    # Then exp() to physical-space for the rasterizer
                    scaling = torch.exp(screenspace_clamp(
                        self.gaussian_params["scaling"]["val"],
                        self.xyz, cam_centre, self.f, max_screen_px=200.0
                    ))
                    opacity = self.opacity
                    
                raw_rot = self.gaussian_params["rotation"]["val"].contiguous()
                rot_norm = torch.linalg.vector_norm(raw_rot, dim=-1, keepdim=True)
                identity_rot = torch.zeros_like(raw_rot)
                identity_rot[:, 0] = 1
                actual_rot = torch.where(
                    rot_norm > 1e-8,
                    raw_rot / rot_norm.clamp_min(1e-8),
                    identity_rot,
                ).contiguous()
                if torch.isnan(scaling).any() or torch.isinf(scaling).any():
                    print("\n[FATAL] Scaling passed to rasterizer contains NaN or Inf!\n")
                    import sys; sys.exit(1)
                    
                try:
                    color, invdepth, mainGaussID, radii = rasterizer(
                        self.xyz.contiguous(),
                        screenspace_points.contiguous(),
                        opacity.contiguous(),
                        self.f_dc.contiguous(),
                        self.f_rest.contiguous(),
                        scaling.contiguous(),
                        actual_rot,
                        view_matrix.contiguous(),
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"\n[WARNING] Rasterizer CUDA error caught and recovered: {e}\n")
                        torch.cuda.synchronize()
                        color = torch.zeros(3, height, width, device="cuda")
                        invdepth = torch.zeros(1, height, width, device="cuda")
                        mainGaussID = torch.zeros(1, height, width, device="cuda", dtype=torch.int32)
                        radii = torch.zeros(self.xyz.shape[0], device="cuda", dtype=torch.int32)
                    else:
                        raise
            else:
                # If no Gaussians are present, return empty tensors
                color = torch.zeros(3, height, width, device="cuda")
                invdepth = torch.zeros(1, height, width, device="cuda")
                mainGaussID = torch.zeros(
                    1, height, width, device="cuda", dtype=torch.int32
                )
                radii = torch.zeros(1, height, width, device="cuda")
        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
        }

    def get_closest_by_cam(self, cam_centre, k=3):
        closest_anchors = []
        closest_anchors_ids = []
        offset = 0
        approx_cam_centres = self.approx_cam_centres.clone()
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            dists = torch.linalg.norm(approx_cam_centres - cam_centre[None], dim=-1)
            min_dist, min_id = torch.min(dists, dim=0)

            if min_dist < 1e9:
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        approx_cam_centres[
                            anchor.keyframe_ids[0] : anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids

    @torch.no_grad()
    def get_prev_keyframes(self, n: int, update_3dpts: bool, desc_kpts: DescribedKeypoints = None):
        """
        Get the n previous keyframes that are the closest to the last
        If desc_kpts is not None, we find the previous keyframes that have the most matches with desc_kpts. The search window is given by self.num_prev_keyframes_check
        """
        # Make sure the optimization thread is not running
        self.join_optimization_thread()

        # Look for the previous keyframes with the most matches with desc_kpts (if provided)
        if desc_kpts is not None and len(self.keyframes) > n:
            n_ckecks = min(self.num_prev_keyframes_check, len(self.keyframes))
            keyframes_indices_to_check = self.sorted_frame_indices[:n_ckecks]
            n_matches = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(
                    self.keyframes[index].desc_kpts, desc_kpts
                )
            _, top_indices = torch.topk(n_matches, n)
            prev_keyframes_indices = keyframes_indices_to_check[top_indices.cpu()]
        # If desc_kpts is not provided, we take the n closest keyframes
        else:
            prev_keyframes_indices = self.sorted_frame_indices[:n]
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # Re-run triangulation if necessary
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes

    def get_Rts(self):
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
        }

    def reset(self, keyframe_id: int = -1):
        """Remove the Gaussians that are visible in the given keyframe."""
        valid_mask = self.opacity[:, 0] > 0.05
        render_pkg = self.render_from_id(keyframe_id)
        valid_mask[render_pkg["visibility_filter"]] = False
        self.optimizer.add_and_prune(self.make_dummy_ext_tensor(), valid_mask)

    @torch.no_grad()
    def _sample_mono_depth_and_conf(self, keyframe: Keyframe, uv: torch.Tensor):
        """Sample aligned monocular depth/confidence at UV coordinates."""
        if uv.shape[0] == 0:
            return (
                torch.empty(0, device="cuda"),
                torch.empty(0, device="cuda"),
            )
        sampler = make_torch_sampler(uv, self.width, self.height)
        mono_idepth = F.grid_sample(
            keyframe.get_mono_idepth()[None],
            sampler[None, None],
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0]
        mono_conf = F.grid_sample(
            keyframe.mono_depth_conf,
            sampler[None, None],
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0]
        mono_depth = 1 / mono_idepth.clamp(1e-6, 1e6)
        return mono_depth, mono_conf

    def _append_gaussian_init_debug(self, row: dict):
        out_dir = os.path.join(self.model_path, "debug")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "gaussian_init_debug.csv")
        fieldnames = [
            "time_sec",
            "keyframe_id",
            "keyframe_name",
            "mode",
            "level",
            "sampled_uv",
            "valid_after_depth_conf",
            "reliable_mvs",
            "fallback_mono",
            "after_occlusion",
            "match_pts_added",
            "new_gaussians_added",
            "sample_proba_mean",
            "sample_proba_max",
            "init_proba_mean",
            "penalty_mean",
            "rendered_depth_used",
        ]
        with self.debug_log_lock:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    @torch.no_grad()
    def _filter_new_gaussian_tensors(
        self,
        extension_tensors: dict[str, torch.Tensor],
        cam_centre: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        xyz = extension_tensors["xyz"]
        if xyz.shape[0] == 0:
            return extension_tensors, torch.zeros(0, device=xyz.device, dtype=torch.bool)

        valid = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.bool)
        for tensor in extension_tensors.values():
            valid &= torch.isfinite(tensor.flatten(1)).all(dim=1)

        dist = torch.linalg.vector_norm(xyz - cam_centre[None], dim=-1)
        scaling = torch.exp(extension_tensors["scaling"].clamp(-20.0, 20.0))
        screen_size = self.f * scaling.max(dim=-1).values / dist.clamp_min(1e-6)
        max_screen_size = 0.25 * max(self.width, self.height)

        valid &= torch.isfinite(dist) & (dist > 1e-5) & (dist < 1e3)
        valid &= torch.isfinite(screen_size) & (screen_size > 0) & (screen_size < max_screen_size)
        return {key: tensor[valid].contiguous() for key, tensor in extension_tensors.items()}, valid

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        keyframe = self.keyframes[keyframe_id]
        ## align the keyframe's depth
        if keyframe.desc_kpts.has_pt3d.sum() == 0:
            keyframe.update_3dpts(self.keyframes)
        keyframe.align_depth()

        # Skip if the keyframe is a test keyframe
        if keyframe.info["is_test"]:
            return

        ## Get the pixel-wise probability to add a Gaussian
        img = keyframe.image_pyr[0]
        init_proba = get_lapla_norm(img, self.disc_kernel) # eq. 1

        if keyframe.mask_pyr is not None:
            dilated_mask = (
                F.conv2d(
                    keyframe.mask_pyr[0][None].float(), self.disc_kernel, padding="same"
                )[0, 0]
                >= 0.99
            )
            init_proba *= dilated_mask

        ## Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        if self.xyz.shape[0] > 0:
            self.verify_no_nans("add_new_gaussians Before Render")
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"].detach()
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            penalty = get_lapla_norm(render, self.disc_kernel)

        ## Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        
        sample_proba = (init_proba - penalty).clamp(0, 1)
        sample_mask = torch.rand_like(init_proba) < sample_proba # eq. 3

        sampled_uv = self.uv[sample_mask]
        n_sampled_uv = int(sampled_uv.shape[0])
        n_reliable_mvs = 0
        n_fallback_mono = 0
        ## Initialize positions
        # Get the samples' depth with guided stereo matching
        prev_KFs = self.get_prev_keyframes(
            self.guided_mvs.n_cams + 1, update_3dpts=False
        )
        for i, prev_keyframe in enumerate(prev_KFs):
            if keyframe.index == prev_keyframe.index:
                prev_KFs.pop(i)
                break
        if sampled_uv.shape[0] > 0:
            depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs)
            mono_depth, mono_conf = self._sample_mono_depth_and_conf(keyframe, sampled_uv)
            mvs_valid = (depth > 1e-6) & torch.isfinite(depth)
            mono_valid = (mono_depth > 1e-6) & torch.isfinite(mono_depth) & (mono_conf > 0.35)
            idepth_mvs = 1 / depth.clamp_min(1e-6)
            idepth_mono = 1 / mono_depth.clamp_min(1e-6)
            depth_consistent = (idepth_mvs - idepth_mono).abs() < 0.25
            reliable_mvs = mvs_valid & accurate_mask & depth_consistent
            fallback_mono = (~reliable_mvs) & mono_valid
            n_reliable_mvs = int(reliable_mvs.sum().item())
            n_fallback_mono = int(fallback_mono.sum().item())
            depth = torch.where(reliable_mvs, depth, mono_depth)
            accurate_mask = reliable_mvs
            valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) & (reliable_mvs | fallback_mono)
        else:
            depth = torch.empty(0, device="cuda")
            accurate_mask = torch.empty(0, device="cuda", dtype=torch.bool)
            valid_mask = accurate_mask
        sample_mask[sample_mask.clone()] = valid_mask
        depth = depth[valid_mask]
        sampled_uv = sampled_uv[valid_mask]
        accurate_mask = accurate_mask[valid_mask]
        n_valid_after_depth_conf = int(depth.shape[0])

        # Remove Gaussians that are coarser than the newpoints
        if len(self.xyz) > 0:
            main_gaussians_map = render_pkg["mainGaussID"]
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            valid_gs_mask[ids] = counts < 10
            with self.lock:
                self.optimizer.add_and_prune(
                    self.make_dummy_ext_tensor(), valid_gs_mask
                )
            render_pkg = self.render_from_id(keyframe_id)
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

        # Check for occlusions
        if rendered_depth is not None:
            valid_mask = depth < rendered_depth[sample_mask]
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]
        n_after_occlusion = int(depth.shape[0])

        # Get the samples' 3D positions
        new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
        new_pts = (new_pts - keyframe.get_t()) @ keyframe.get_R()
        # Add points from matching
        match_pts = keyframe.desc_kpts.pts3d[keyframe.desc_kpts.has_pt3d]
        n_match_pts = int(match_pts.shape[0])
        new_pts = torch.cat([new_pts, match_pts], dim=0)
        n_new_gaussians = int(new_pts.shape[0])

        ## Initialize Colour
        f_dc = img[:, sample_mask]
        match_sampler = keyframe.desc_kpts.kpts[keyframe.desc_kpts.has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        match_colors = F.grid_sample(
            img[None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(3, -1)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        ## Initialize Scales
        sampled_init_proba = init_proba[sample_mask]
        match_init_proba = F.grid_sample(
            init_proba[None, None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)
        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 0.02 * ~accurate_mask[..., None]
        )
        # High opacity for triangulated Gaussians
        opacities[sampled_uv.shape[0] :] *= 0.2
        opacities = inverse_sigmoid(opacities)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1

        ## Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > 0.05
            valid_gs_mask &= torch.isfinite(self.xyz).all(dim=-1)
            valid_gs_mask &= torch.isfinite(self.gaussian_params["scaling"]["val"]).all(dim=-1)
            valid_gs_mask &= torch.isfinite(self.gaussian_params["opacity"]["val"]).all(dim=-1)

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist.clamp_min(1e-6)
            valid_gs_mask &= screen_size < 0.5 * self.width
        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

        ## Append the new Gaussians
        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        extension_tensors, valid_new_mask = self._filter_new_gaussian_tensors(
            extension_tensors, keyframe.approx_centre
        )
        n_match_pts = int(valid_new_mask[sampled_uv.shape[0] :].sum().item())
        n_new_gaussians = int(extension_tensors["xyz"].shape[0])

        with self.lock:
            self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

        self._append_gaussian_init_debug(
            {
                "time_sec": f"{time.time():.6f}",
                "keyframe_id": keyframe.index,
                "keyframe_name": keyframe.info.get("name", ""),
                "mode": "add_new_gaussians",
                "level": -1,
                "sampled_uv": n_sampled_uv,
                "valid_after_depth_conf": n_valid_after_depth_conf,
                "reliable_mvs": n_reliable_mvs,
                "fallback_mono": n_fallback_mono,
                "after_occlusion": n_after_occlusion,
                "match_pts_added": n_match_pts,
                "new_gaussians_added": n_new_gaussians,
                "sample_proba_mean": float(sample_proba.mean().item()),
                "sample_proba_max": float(sample_proba.max().item()),
                "init_proba_mean": float(init_proba.mean().item()),
                "penalty_mean": float(penalty.mean().item()) if torch.is_tensor(penalty) else float(penalty),
                "rendered_depth_used": int(rendered_depth is not None),
            }
        )

    @torch.no_grad()
    def add_new_gaussians_by_levels(self, keyframe_id: int = -1, level: int = 1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        keyframe = self.keyframes[keyframe_id]
        ## align the keyframe's depth
        if keyframe.desc_kpts.has_pt3d.sum() == 0:
            keyframe.update_3dpts(self.keyframes)
        keyframe.align_depth()

        # Skip if the keyframe is a test keyframe
        if keyframe.info["is_test"]:
            return

        ## Get the pixel-wise probability to add a Gaussian
        img = keyframe.image_pyr[0]
        init_proba = get_lapla_norm(img, self.disc_kernel) # eq. 1

        if keyframe.mask_pyr is not None:
            dilated_mask = (
                F.conv2d(
                    keyframe.mask_pyr[0][None].float(), self.disc_kernel, padding="same"
                )[0, 0]
                >= 0.99
            )
            init_proba *= dilated_mask

        ## Compute the penalty based on the rendering from the new keyframe's point of view
        penalty = 0
        rendered_depth = None
        if self.xyz.shape[0] > 0:
            self.verify_no_nans("add_new_gaussians_by_levels Before Render")
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"].detach()
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            penalty = get_lapla_norm(render, self.disc_kernel)

        ## Define which pixels should become Gaussians
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        
        # Apply Distance-based LoD Density multiplier (reduces far-field spawn to save VRAM)
        from scene.LoD_utils import get_lod_density
        depth_val = 1 / render_pkg["invdepth"][0].clamp_min(1e-8) if self.xyz.shape[0] > 0 else torch.zeros_like(init_proba)
        density_mult = get_lod_density(depth_val[None], far_density_ratio=0.5)
        
        sample_proba = ((init_proba - penalty) * density_mult[0]).clamp(0, 1)
        sample_mask = torch.rand_like(init_proba) < sample_proba # eq. 3

        sampled_uv = self.uv[sample_mask]
        n_sampled_uv = int(sampled_uv.shape[0])
        n_reliable_mvs = 0
        n_fallback_mono = 0
        ## Initialize positions
        # Get the samples' depth with guided stereo matching
        prev_KFs = self.get_prev_keyframes(
            self.guided_mvs.n_cams + 1, update_3dpts=False
        )
        for i, prev_keyframe in enumerate(prev_KFs):
            if keyframe.index == prev_keyframe.index:
                prev_KFs.pop(i)
                break
        if sampled_uv.shape[0] > 0:
            depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs)
            mono_depth, mono_conf = self._sample_mono_depth_and_conf(keyframe, sampled_uv)
            mvs_valid = (depth > 1e-6) & torch.isfinite(depth)
            mono_valid = (mono_depth > 1e-6) & torch.isfinite(mono_depth) & (mono_conf > 0.35)
            idepth_mvs = 1 / depth.clamp_min(1e-6)
            idepth_mono = 1 / mono_depth.clamp_min(1e-6)
            depth_consistent = (idepth_mvs - idepth_mono).abs() < 0.25
            reliable_mvs = mvs_valid & accurate_mask & depth_consistent
            fallback_mono = (~reliable_mvs) & mono_valid
            n_reliable_mvs = int(reliable_mvs.sum().item())
            n_fallback_mono = int(fallback_mono.sum().item())
            depth = torch.where(reliable_mvs, depth, mono_depth)
            accurate_mask = reliable_mvs
            valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) & (reliable_mvs | fallback_mono)
        else:
            depth = torch.empty(0, device="cuda")
            accurate_mask = torch.empty(0, device="cuda", dtype=torch.bool)
            valid_mask = accurate_mask
        sample_mask[sample_mask.clone()] = valid_mask
        depth = depth[valid_mask]
        sampled_uv = sampled_uv[valid_mask]
        accurate_mask = accurate_mask[valid_mask]
        n_valid_after_depth_conf = int(depth.shape[0])

        # Remove Gaussians that are coarser than the newpoints
        if len(self.xyz) > 0:
            main_gaussians_map = render_pkg["mainGaussID"]
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            valid_gs_mask[ids] = counts < 10
            with self.lock:
                self.optimizer.add_and_prune(
                    self.make_dummy_ext_tensor(), valid_gs_mask
                )
            render_pkg = self.render_from_id(keyframe_id)
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

        # Check for occlusions
        if rendered_depth is not None:
            valid_mask = depth < rendered_depth[sample_mask]
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]
        n_after_occlusion = int(depth.shape[0])

        # Get the samples' 3D positions
        new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
        new_pts = (new_pts - keyframe.get_t()) @ keyframe.get_R()
        # Add points from matching
        match_pts = keyframe.desc_kpts.pts3d[keyframe.desc_kpts.has_pt3d]
        n_match_pts = int(match_pts.shape[0])
        new_pts = torch.cat([new_pts, match_pts], dim=0)
        n_new_gaussians = int(new_pts.shape[0])

        ## Initialize Colour
        f_dc = img[:, sample_mask]
        match_sampler = keyframe.desc_kpts.kpts[keyframe.desc_kpts.has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        match_colors = F.grid_sample(
            img[None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(3, -1)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        ## Initialize Scales
        sampled_init_proba = init_proba[sample_mask]
        match_init_proba = F.grid_sample(
            init_proba[None, None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)
        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 0.02 * ~accurate_mask[..., None]
        )
        # High opacity for triangulated Gaussians
        opacities[sampled_uv.shape[0] :] *= 0.2
        opacities = inverse_sigmoid(opacities)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1

        ## Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > 0.05
            valid_gs_mask &= torch.isfinite(self.xyz).all(dim=-1)
            valid_gs_mask &= torch.isfinite(self.gaussian_params["scaling"]["val"]).all(dim=-1)
            valid_gs_mask &= torch.isfinite(self.gaussian_params["opacity"]["val"]).all(dim=-1)

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist.clamp_min(1e-6)
            valid_gs_mask &= screen_size < 0.5 * self.width
        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

        ## Append the new Gaussians
        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        extension_tensors, valid_new_mask = self._filter_new_gaussian_tensors(
            extension_tensors, keyframe.approx_centre
        )
        n_match_pts = int(valid_new_mask[sampled_uv.shape[0] :].sum().item())
        n_new_gaussians = int(extension_tensors["xyz"].shape[0])
        with self.lock:
            self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

        self._append_gaussian_init_debug(
            {
                "time_sec": f"{time.time():.6f}",
                "keyframe_id": keyframe.index,
                "keyframe_name": keyframe.info.get("name", ""),
                "mode": "add_new_gaussians_by_levels",
                "level": int(level),
                "sampled_uv": n_sampled_uv,
                "valid_after_depth_conf": n_valid_after_depth_conf,
                "reliable_mvs": n_reliable_mvs,
                "fallback_mono": n_fallback_mono,
                "after_occlusion": n_after_occlusion,
                "match_pts_added": n_match_pts,
                "new_gaussians_added": n_new_gaussians,
                "sample_proba_mean": float(sample_proba.mean().item()),
                "sample_proba_max": float(sample_proba.max().item()),
                "init_proba_mean": float(init_proba.mean().item()),
                "penalty_mean": float(penalty.mean().item()) if torch.is_tensor(penalty) else float(penalty),
                "rendered_depth_used": int(rendered_depth is not None),
            }
        )


    def init_intrinsics(self):
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )

    def move_rand_keyframe_to_cpu(self):
        """Move a random keyframe to CPU memory"""
        frame_id = np.random.choice(self.active_frames_gpu[:-self.n_kept_frames])
        self.keyframes[frame_id].to("cpu")
        self.active_frames_cpu.append(frame_id)
        self.active_frames_gpu.remove(frame_id) 

    def move_rand_keyframe_to_gpu(self):
        """Move a random keyframe to GPU memory"""
        if len(self.active_frames_cpu) > 0:
            frame_id = np.random.choice(self.active_frames_cpu)
            self.keyframes[frame_id].to("cuda")
            self.active_frames_gpu.insert(0, frame_id)
            self.active_frames_cpu.remove(frame_id) 

    def add_keyframe(self, keyframe: Keyframe, f=None):
        """Add a keyframe to the scene, add and prune Gaussians"""

        # Make sure training is not running 
        self.join_optimization_thread()

        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update intrinsics
        if f is not None:
            self.f = f.item()
            self.init_intrinsics()

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device="cuda", dtype=torch.bool)], dim=0
        )
        gt_pose = keyframe.info.get("Rt", None)
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.info.get("focal", self.f)

        if not self.inference_mode:
            ## Add keyframe to the active anchor
            self.active_anchor.add_keyframe(keyframe)
            self.active_frames_gpu.append(keyframe.index)

            ## Clear memory if there are many keyframes
            if len(self.active_frames_gpu) > self.max_active_keyframes:
                self.move_rand_keyframe_to_cpu()
                # Reshuffle the active keyframes and clear cache
                if len(self.active_frames_cpu) % 5 == 0:
                    self.move_rand_keyframe_to_cpu()
                    self.move_rand_keyframe_to_gpu()

                    gc.collect()
                    torch.cuda.empty_cache()

    def enable_inference_mode(self):
        """Enable inference mode and sets the anchor position to the mean of the active keyframes."""
        self.inference_mode = True
        self.update_anchor()

    def update_anchor(self, n_left_frames: int = 0):
        """Update the anchor position and remove the last n_left_frames keyframes from the active anchor."""
        anchor_position = self.approx_cam_centres[
            self.first_active_frame : self.last_active_frame - n_left_frames
        ].mean(dim=0)
        self.active_anchor.position = anchor_position
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[:-n_left_frames]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                :-n_left_frames
            ]

    def place_anchor_if_needed(self):
        """Check if many Gaussians appear small on the screen. If so, place a new anchor. and merge the Gaussians."""
        small_prop_thresh = 0.4
        k = 3
        self.n_kept_frames = 20
        if (
            self.xyz.shape[0] > 0
            and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                dist = torch.linalg.vector_norm(
                    self.xyz - self.approx_cam_centres[-1][None], dim=-1
                )
                screen_size = self.f * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    small_mask = screen_size < 1.5
                    # Update anchor positions using the camera poses used to optimize it
                    self.update_anchor(self.n_kept_frames)

                    ## Merge fine Gaussians for the current active set
                    # Select a subset and get their nearest neighbours for merging
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    small_screen_size = screen_size[small_mask]

                    if xyz.shape[0] <= k:
                        return

                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    
                    M = int(xyz.shape[0] / (k + 1))
                    if M <= 0:
                        return
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: M]
                    selected_nn_idx = torch.cat([idx[..., None], nn_idx[idx]], dim=-1)
                    
                    # Compute merging weights based on contribution to the rendering
                    weights = small_gaussians["opacity"][selected_nn_idx, 0].sigmoid() * (
                        small_screen_size[selected_nn_idx] ** 2
                    )
                    weights_sum = weights.sum(dim=-1, keepdim=True)
                    valid_mask = torch.ones_like(weights, dtype=torch.bool)
                    weights = torch.where(
                        weights_sum > 1e-8,
                        weights / weights_sum.clamp_min(1e-8),
                        valid_mask.float()
                        / valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0),
                    )
                    weights.unsqueeze_(-1)

                    cluster_xyz = small_gaussians["xyz"][selected_nn_idx, :]
                    merged_xyz = (cluster_xyz * weights).sum(dim=1)
                    cluster_radius = torch.linalg.vector_norm(
                        cluster_xyz - merged_xyz[:, None, :], dim=-1
                    ).amax(dim=1)
                    cluster_scale = (
                        torch.exp(small_gaussians["scaling"][selected_nn_idx, :])
                        .amax(dim=-1)
                        .median(dim=1)
                        .values
                    )
                    cluster_cam_dist = torch.linalg.vector_norm(
                        merged_xyz - self.approx_cam_centres[-1][None], dim=-1
                    ).clamp_min(1e-6)
                    max_merge_radius = torch.maximum(
                        6.0 * cluster_scale,
                        0.02 * cluster_cam_dist,
                    )
                    reliable_cluster_mask = (
                        torch.isfinite(cluster_xyz).all(dim=(1, 2))
                        & torch.isfinite(weights.squeeze(-1)).all(dim=1)
                        & torch.isfinite(cluster_radius)
                        & (cluster_radius <= max_merge_radius)
                    )
                    if not reliable_cluster_mask.any():
                        return

                    selected_nn_idx = selected_nn_idx[reliable_cluster_mask]
                    weights = weights[reliable_cluster_mask]
                    valid_mask = valid_mask[reliable_cluster_mask]
                    merged_xyz = merged_xyz[reliable_cluster_mask]

                    # Align quaternions before averaging so q and -q do not cancel out.
                    rots = small_gaussians["rotation"][selected_nn_idx, :]
                    base_rot = rots[:, 0:1, :]
                    dot_product = (rots * base_rot).sum(dim=-1, keepdim=True)
                    rots = torch.where(dot_product < 0, -rots, rots)
                    merged_rots = (rots * weights).sum(dim=1)
                    rot_norm = torch.linalg.vector_norm(merged_rots, dim=-1, keepdim=True)
                    identity_rot = torch.zeros_like(merged_rots)
                    identity_rot[:, 0] = 1
                    merged_rots = torch.where(
                        rot_norm > 1e-8,
                        merged_rots / rot_norm.clamp_min(1e-8),
                        identity_rot,
                    )

                    valid_counts = valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
                    merged_gaussians = {
                        "xyz": merged_xyz,
                        "f_dc": (
                            small_gaussians["f_dc"][selected_nn_idx, :]
                            * weights.unsqueeze(-1)
                        ).sum(dim=1),
                        "f_rest": (
                            small_gaussians["f_rest"][selected_nn_idx, :]
                            * weights.unsqueeze(-1)
                        ).sum(dim=1),
                        "opacity": inverse_sigmoid(
                            (
                                small_gaussians["opacity"][selected_nn_idx, :].sigmoid()
                                * weights
                            )
                            .sum(dim=1)
                            .clamp(1e-4, 1 - 1e-4)
                        ),
                        "scaling": torch.log(
                            (
                                torch.exp(small_gaussians["scaling"][selected_nn_idx, :])
                                * weights
                                * valid_counts
                            )
                            .sum(dim=1)
                            .clamp_min(1e-8)
                        ),
                        "rotation": merged_rots,
                    }
                    merged_gaussians = validate_merged_gaussians(merged_gaussians)

                    # Offload the previous Gaussians to the CPU
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    ## Add the merged Gaussians to the set of Gaussians and reset the optimizer
                    small_ids = torch.nonzero(small_mask, as_tuple=False).view(-1)
                    merged_small_ids = torch.unique(small_ids[selected_nn_idx.reshape(-1)])
                    keep_existing_mask = torch.ones_like(small_mask, dtype=torch.bool)
                    keep_existing_mask[merged_small_ids] = False
                    with self.lock:
                        self.optimizer.add_and_prune(merged_gaussians, keep_existing_mask)

                    # Create a new active anchor with the merged Gaussians
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        self.approx_cam_centres[-1],
                        self.keyframes[-self.n_kept_frames :],
                    )
                    self.anchors.append(self.active_anchor)
                    self.active_frames_gpu = [kf.index for kf in self.active_anchor.keyframes]
                    self.active_frames_cpu = []

                    # Visualization
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        # Get metrics
        metrics = {
            "num anchors": len(self.anchors),
            "num keyframes": len(self.keyframes),
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        metrics.update(self.evaluate(True, True, True))

        if path == "":
            print("No path provided, skipping save")
            return metrics

        # Save anchors
        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        for index, anchor in enumerate(self.anchors):
            anchor.save_ply(os.path.join(pcd_path, f"anchor_{index}.ply"))

        # Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "f": self.f,
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),
                }
                for anchor in self.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
        }
        metadata = {**metrics, **metadata}

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Save renders of test views
        self.save_test_frames(os.path.join(path, "test_images"))

        # Saving cameras with COLMAP format
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        for index, keyframe in enumerate(self.keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics

    def get_closest_keyframe(
        self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.anchor_weights = np.zeros(len(self.anchors))
        for anchor_id, anchor in enumerate(self.anchors):
            self.active_anchor = anchor
            # Load the anchor and make its parameters optimizable
            anchor.to("cuda", with_keyframes=True)
            self.gaussian_params = anchor.gaussian_params
            self.anchor_weights[anchor_id] = 1
            self.reset_optimizer()

            # # Ensure other anchors are on cpu to save memory
            # if anchor_id >= 1:
            #     self.anchors[anchor_id-1].to("cpu", with_keyframes=True)

            # Optimize the anchor by going through its keyframes
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # Update the anchor and store it on cpu
            anchor.gaussian_params = self.gaussian_params
            self.anchor_weights[anchor_id] = 0
