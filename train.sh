#!/usr/bin/env sh

export OTFNVS_TRIANGULATOR_CUDA_GRAPH=0
export OTFNVS_MINIBA_CUDA_GRAPH=0
# MatrixCity Aerial
SRC=/home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1
OUT=results/MatrixCity/LoD_distance_based
OUT1=results/MatrixCity/baseline_downsampling_2_vggt_prior

# echo "=== Running Distance-based LoD with Semantic Features ==="
# python train_lod.py \
#     -s ${SRC} \
#     -m ${OUT} \
#     --images_dir images \
#     --downsampling 4 \
#     --test_hold 20 \
#     --lod_min 1 \
#     --lod_max 1 \
#     --use_semantic_features \
#     --use_track_extrapolation


python train_lod.py \
    -s ${SRC} \
    -m ${OUT1} \
    --images_dir images \
    --downsampling 2 \
    --test_hold 20 \
    --lod_min 1 \
    --lod_max 1 \
    --no_pose_use_lsf_velocity_gate \
    --use_vggt_pose_prior \
    --vggt_pose_prior_path /home/cglab/project/on-the-fly-nvs/trajectory_output/vggt_camera_params.csv \
    --max_active_keyframes 150



# Optional:
# --use_track_extrapolation
# --use_semantic_features
