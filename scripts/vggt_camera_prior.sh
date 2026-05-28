#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cglab/project/on-the-fly-nvs}"
VGGT_PYTHON_BIN="${VGGT_PYTHON_BIN:-python}"
IMAGE_DIR="${IMAGE_DIR:-/home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1_small/images}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/trajectory_output}"

mkdir -p "${OUT_DIR}"
cd "${PROJECT_ROOT}"

"${VGGT_PYTHON_BIN}" scripts/export_vggt_camera_prior.py \
  --image_dir "${IMAGE_DIR}" \
  --output_tum "${OUT_DIR}/vggt_prior.tum" \
  --output_csv "${OUT_DIR}/vggt_prior.csv" \
  --output_camera_csv "${OUT_DIR}/vggt_camera_params.csv" \
  --output_camera_npz "${OUT_DIR}/vggt_camera_params.npz" \
  --chunk_size "${VGGT_CHUNK_SIZE:-16}" \
  --overlap "${VGGT_OVERLAP:-4}" \
  --mode "${VGGT_MODE:-pad}" \
  "$@"
