#!/usr/bin/env sh

export OTFNVS_TRIANGULATOR_CUDA_GRAPH=0
export OTFNVS_MINIBA_CUDA_GRAPH=0
# MatrixCity Aerial
SRC=/home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1
OUT=results/MatrixCity/LoD_distance_based
OUT1=results/MatrixCity/LoD_distance_based_downsampling_2_without_semantic_feature

echo "=== Running Distance-based LoD with Semantic Features ==="
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
    --use_track_extrapolation \
    --use_semantic_features 