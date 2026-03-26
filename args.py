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

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Options for data loading and training")

    ## Data and Images
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to the data folder (should have sparse/0/ if using COLMAP or evaluating poses)")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png or jpeg).")
    parser.add_argument('--masks_dir', type=str, default="", 
                        help="If set, source_path/masks_dir is the path to optional masks to apply to the images before computing loss (png).")
    parser.add_argument('--num_loader_threads', type=int, default=4,
                        help="Number of workers to load and prepare input images")
    parser.add_argument('--downsampling', type=float, default=-1.0, help="Downsampling ratio for input images")
    parser.add_argument('--pyr_levels', type=int, default=2,
                        help="Number of pyramid levels. Each level l will downsample the image 2^l times in width and height")
    parser.add_argument('--min_displacement', type=float, default=0.03,
                        help="Minimum median keypoint displacement for a new keyframe to be added. Relative to the image width")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the dataset.")
    
    parser.add_argument('--sh_degree', default=3)

    ## COLMAP options
    parser.add_argument('--eval_poses', action='store_true',
                        help="Compare poses to COLMAP")
    parser.add_argument('--use_colmap_poses', action='store_true',
                        help="Load COLMAP data for pose and intrinsics initialization")
        
    ## Learning Rates
    parser.add_argument('--lr_poses', type=float, default=1e-4, help="Pose learning rate")
    parser.add_argument('--lr_exposure', type=float, default=5e-4, 
                        help="Exposure compensation learning rate")
    parser.add_argument('--lr_depth_scale_offset', type=float, default=1e-4)
    parser.add_argument('--position_lr_init', type=float, default=0.00005, 
                        help="Initial position learning rate")
    parser.add_argument('--position_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for position learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.005, help="Feature learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="Opacity learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.01, help="Scaling learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.002, help="Rotation learning rate")
        
    ## Training schedule and losses
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help="Weight for DSSIM loss")
    parser.add_argument('--num_iterations', type=int, default=10, 
                        help="Number of training iterations per keyframe")
    parser.add_argument('--depth_loss_weight_init', type=float, default=1e-2)
    parser.add_argument('--depth_loss_weight_decay', type=float, default=0.9, 
                        help="Weight decay for depth loss, multiply depth loss weight by this factor every iterations")
    parser.add_argument('--save_at_finetune_epoch', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning.")
    parser.add_argument('--use_last_frame_proba', type=float, default=0.2, 
                        help="Probability of using the last registered frame for each training iteration")

    ## Pose initialization options
    # Matching
    parser.add_argument('--num_kpts', type=int, default=int(4096*1.5),
                        help="Number of keypoints to extract from each image")
    parser.add_argument('--match_max_error', type=float, default=2e-3,
                        help="Maximum reprojection error for matching keypoints, proportion of the image width. This is used to filter outliers and discard points at triangulation.")
    parser.add_argument('--fundmat_samples', type=int, default=2000,
                        help="Maximum number of set of matches used to estimate the fundamental matrix for outlier removal")
    parser.add_argument('--min_num_inliers', type=int, default=25, #50 #100
                        help="The keyframe will be added only if the number of inliers is greater than this value")
    # Initial mini bundle adjustment
    parser.add_argument('--num_keyframes_miniba_bootstrap', type=int, default=8,
                        help="Number of first keyframes accumulated for pose and focal estimation before optimization")
    parser.add_argument('--num_pts_miniba_bootstrap', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_bootstrap', type=int, default=200)
    parser.add_argument('--enable_reboot', action='store_true')
    # Focal estimation
    parser.add_argument('--fix_focal', action='store_true', 
                        help="If set, will use init_focal or init_fov without reoptimizing focal")
    parser.add_argument('--init_focal', type=float, default=-1.0, 
                        help="Initial focal length in pixels. If not set, will use init_fov or be set as 0.7*width of the image if init_fov is also not set")
    parser.add_argument('--init_fov', type=float, default=-1.0, 
                        help="Initial horizontal FoV in degrees. Used only if init_focal is not set")
    # Incremental pose optimization
    parser.add_argument('--num_prev_keyframes_miniba_incr', type=int, default=6,
                        help="Number of previous keyframes for incremental pose initialization")
    parser.add_argument('--num_prev_keyframes_check', type=int, default=20, #20
                        help="Number of previous keyframes to check for matches with new keyframe")
    parser.add_argument('--pnpransac_samples', type=int, default=2000,
                        help="Maximum number of set of 2D-3D matches used to estimate the initial pose and outlier removal")
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_incr', type=int, default=20)

    # Semantic feature matching
    parser.add_argument('--use_semantic_features', action='store_true',
                        help="Enable DINOv2 semantic features for matching")
    parser.add_argument('--semantic_backbone', type=str, default='dinov2',
                        choices=['dinov2', 'dinov3'],
                        help="Backbone for semantic feature extraction")
    parser.add_argument('--sem_feat_dim', type=int, default=128,
                        help="Dimensionality of semantic features after PCA")
    parser.add_argument('--sem_weight', type=float, default=0.3,
                        help="Weight of semantic similarity in matching score (0-1)")
    # Camera track extrapolation
    parser.add_argument('--use_track_extrapolation', action='store_true',
                        help="Use camera trajectory extrapolation for PnP initial pose")
    parser.add_argument('--track_extrapolation_window', type=int, default=3,
                        help="Number of recent keyframes for pose extrapolation")

    ## Gaussian initialization options
    parser.add_argument('--init_proba_scaler', type=float, default=2,
                        help="Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")

    # Anchor management
    parser.add_argument('--anchor_overlap', type=float, default=0.3,
                        help="Size of the overlapping regions when blending between anchors")

    ## Level of Detail (LoD) options
    parser.add_argument('--lod_min', type=int, default=1, help="Minimum level of detail")
    parser.add_argument('--lod_max', type=int, default=1, help="Maximum level of detail")
    parser.add_argument('--lod_reference_max', type=int, default=3,
                        help="Reference max LoD used to scale LoD effects when training fixed single-level runs")
    parser.add_argument('--lod1_scaling_lower_bound', type=float, default=0.001, 
                        help="Lower bound for scaling at LoD 1")
    parser.add_argument('--lod_scaling_ratio', type=float, default=2.0, 
                        help="Ratio for scaling reduction between LoDs")
    parser.add_argument('--increase_lod_num_childs', type=int, default=2, 
                        help="Number of children to spawn when increasing LoD")
    parser.add_argument('--lod_min_sample_ratio', type=float, default=0.4,
                        help="At the coarsest LoD, keep this ratio of edge-based Gaussian spawn probability")
    parser.add_argument('--lod_merge_voxel_factor', type=float, default=4.0,
                        help="Controls voxel size for LoD primitive merging at low levels (higher => fewer, blurrier Gaussians)")
    parser.add_argument('--lod_merge_min_points', type=int, default=256,
                        help="Minimum number of newly spawned Gaussians before applying LoD merge")
    parser.add_argument('--lod_blur_scale_boost', type=float, default=0.25,
                        help="Additional scale boost applied after merging at low LoD to increase blur")
    parser.add_argument('--boundary_penalty_weight', type=float, default=0.35,
                        help="Additional spawn penalty near image borders in add_new_gaussians")
    parser.add_argument('--boundary_penalty_margin_ratio', type=float, default=0.08,
                        help="Border width ratio used for boundary spawn penalty (0~0.5)")
    parser.add_argument('--distance_penalty_weight', type=float, default=0.20,
                        help="Additional spawn penalty for farther rendered depth regions")
    parser.add_argument('--max_gaussian_aspect_ratio', type=float, default=8.0,
                        help="Prune Gaussians whose max/min scale ratio exceeds this threshold")
    parser.add_argument('--save_test_depth', type=int, default=1,
                        help="If 1, save depth visualizations and npy arrays for test frames")
    parser.add_argument('--lod_min_pose_stability', type=float, default=0.55,
                        help="Minimum pose stability score required before increasing LoD")
    parser.add_argument('--lod_max_projection_error_px', type=float, default=3.0,
                        help="Maximum mean projection footprint error (in px) allowed before increasing LoD")
    parser.add_argument('--lod_pose_window', type=int, default=8,
                        help="Number of recent keyframes used for pose stability estimation")
    parser.add_argument('--lod_progressive_max_extra_epochs', type=int, default=3,
                        help="Maximum extra fine-tuning epochs per LoD level when the progression gate is not satisfied")


    ## Keyframe management
    parser.add_argument('--max_active_keyframes', type=int, default=200,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")

    ## Evaluation
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Holdout for test set, will exclude every test_hold image from the Gaussian optimization and use them for testing. The test frames will still be used for training the pose. If set to -1, no keyframes will be excluded from training.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true', 
                        help="Display runtimes for each step in the tqdm bar")

    ## Checkpoint options
    parser.add_argument('-m', '--model_path', default="", 
                        help="Directory to store the renders from test view and checkpoints after training. If not set, will be set to results/xxxxxx.")
    parser.add_argument('--save_every', default=-1, type=int, 
                        help="Frequency of exporting renders w.r.t input frames.")

    ## Viewer
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'web', 'none'], default='none')
    parser.add_argument('--ip', type=str, default="140.119.164.24", 
                        help="IP address of the viewer client, if using server viewer_mode")
    parser.add_argument('--port', type=int, default=6009,
                        help="Port of the viewer client, if using server viewer_mode")

    args = parser.parse_args()

    ## Set the output directory if not specified
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
