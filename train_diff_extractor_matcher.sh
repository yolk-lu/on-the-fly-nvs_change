#!/usr/bin/env bash
set -euo pipefail

# Run extractor/matcher ablations with train_lod.py
# Default suite:
#  1) xfeat + mnn (baseline)
#  2) superpoint + lightglue
#  3) disk + lightglue
#  4) sift + lightglue
#  5) aliked + lightglue

PROJECT_ROOT="/home/cglab/project/on-the-fly-nvs"
TRAIN_PY="${PROJECT_ROOT}/train_lod.py"

# Hard-coded dataset/output/common args
SRC="/home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1"
BASE_OUT="${PROJECT_ROOT}/results/MatrixCity/diff_extractor_matcher"
COMMON_ARGS=(
  --images_dir images
  --downsampling 2
  --test_hold 20
  --lod_min 1
  --lod_max 1
  --no_pose_use_lsf_velocity_gate
  --use_vggt_pose_prior
  --vggt_pose_prior_path /home/cglab/project/on-the-fly-nvs/trajectory_output/vggt_camera_params.csv
  --max_active_keyframes 200
)

# Basic operation: activate onthefly_nvs before training.
if [[ "${CONDA_DEFAULT_ENV:-}" != "onthefly_nvs" ]]; then
  if [[ -f "/home/cglab/miniconda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source /home/cglab/miniconda/etc/profile.d/conda.sh
    conda activate onthefly_nvs
  else
    echo "[Warning] conda.sh not found. Please activate environment manually: conda activate onthefly_nvs"
  fi
fi

echo "[Env] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<none>}"
echo "[Run] Source=${SRC}"
echo "[Run] Base output=${BASE_OUT}"

run_one() {
  local name="$1"
  local feature="$2"
  local matcher="$3"
  local out_dir="${BASE_OUT}_${name}"

  echo "================================================================"
  echo "[Ablation] ${name}: feature_backend=${feature}, matcher_backend=${matcher}"
  echo "[Ablation] output -> ${out_dir}"

  python "${TRAIN_PY}" \
    -s "${SRC}" \
    -m "${out_dir}" \
    --feature_backend "${feature}" \
    --matcher_backend "${matcher}" \
    "${COMMON_ARGS[@]}"
}

run_one "xfeat_mnn" "xfeat" "mnn"
run_one "superpoint_lightglue" "superpoint" "lightglue"
run_one "disk_lightglue" "disk" "lightglue"
run_one "sift_lightglue" "sift" "lightglue"
run_one "aliked_lightglue" "aliked" "lightglue"

echo "================================================================"
echo "[Done] All extractor/matcher ablations completed."
