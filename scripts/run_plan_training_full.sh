#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cglab/project/on-the-fly-nvs}"
PYTHON_BIN="${PYTHON_BIN:-/home/cglab/miniconda/envs/onthefly_nvs/bin/python}"
SRC="${SRC:-/home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1}"
RUN_NAME="${RUN_NAME:-plan_full_xfeat_mnn_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/MatrixCity/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/MatrixCity/plan_full_logs}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
TRAIN_PY="${TRAIN_PY:-${PROJECT_ROOT}/Progressive_train.py}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OTFNVS_MINIBA_CUDA_GRAPH="${OTFNVS_MINIBA_CUDA_GRAPH:-0}"
export OTFNVS_RANSAC_MODEL_CHUNK="${OTFNVS_RANSAC_MODEL_CHUNK:-256}"
export OTFNVS_GUIDED_MVS_MAX_POINTS="${OTFNVS_GUIDED_MVS_MAX_POINTS:-4096}"

COMMON_ARGS=(
  -s "${SRC}"
  -m "${OUT_DIR}"
  --feature_backend "${FEATURE_BACKEND:-xfeat}"
  --matcher_backend "${MATCHER_BACKEND:-mnn}"
  --images_dir "${IMAGES_DIR:-images}"
  --downsampling "${DOWNSAMPLING:-2}"
  --test_hold "${TEST_HOLD:-20}"
  --lod_min "${LOD_MIN:-1}"
  --lod_max "${LOD_MAX:-1}"
  --no_pose_use_lsf_velocity_gate
  --use_vggt_pose_prior
  --vggt_pose_prior_path "${VGGT_POSE_PRIOR_PATH:-${PROJECT_ROOT}/trajectory_output/vggt_camera_params.csv}"
  --max_active_keyframes "${MAX_ACTIVE_KEYFRAMES:-200}"
)

if [[ "${USE_PARALLAX_BA:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no_parallax_ba)
else
  COMMON_ARGS+=(--use_parallax_ba)
fi

if [[ "${RUN_MODULE_SMOKE_FIRST:-1}" == "1" ]]; then
  echo "[Preflight] Running module smoke before full training" | tee "${LOG_FILE}"
  "${PYTHON_BIN}" -m pipeline.module_smoke_runner 2>&1 | tee -a "${LOG_FILE}"
fi

{
  echo "[ProgressiveTrainFull] project=${PROJECT_ROOT}"
  echo "[ProgressiveTrainFull] python=${PYTHON_BIN}"
  echo "[ProgressiveTrainFull] train=${TRAIN_PY}"
  echo "[ProgressiveTrainFull] source=${SRC}"
  echo "[ProgressiveTrainFull] output=${OUT_DIR}"
  echo "[ProgressiveTrainFull] log=${LOG_FILE}"
  echo "[ProgressiveTrainFull] overlap optimization is reserved but not implemented in this phase."
  echo "[ProgressiveTrainFull] env PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "[ProgressiveTrainFull] env OTFNVS_MINIBA_CUDA_GRAPH=${OTFNVS_MINIBA_CUDA_GRAPH}"
  echo "[ProgressiveTrainFull] env OTFNVS_RANSAC_MODEL_CHUNK=${OTFNVS_RANSAC_MODEL_CHUNK}"
  echo "[ProgressiveTrainFull] env OTFNVS_GUIDED_MVS_MAX_POINTS=${OTFNVS_GUIDED_MVS_MAX_POINTS}"
  echo "[ProgressiveTrainFull] args=${COMMON_ARGS[*]} ${*}"
  echo

  "${PYTHON_BIN}" "${TRAIN_PY}" --progressive_overlap_mode reserved "${COMMON_ARGS[@]}" "$@"

  echo
  echo "[ProgressiveTrainFull] PASS"
} 2>&1 | tee -a "${LOG_FILE}"
