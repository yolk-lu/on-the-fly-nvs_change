#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cglab/project/on-the-fly-nvs}"
PYTHON_BIN="${PYTHON_BIN:-/home/cglab/miniconda/envs/onthefly_nvs/bin/python}"
SRC="${SRC:-/home/cglab/project/Dataset_opensource/MatrixCity_unzip/block_1_small}"
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
  --progressive_lod "${PROGRESSIVE_LOD:-1}"
  --progressive_anchor_radius "${PROGRESSIVE_ANCHOR_RADIUS:-5.0}"
  --progressive_anchor_min_keyframes "${PROGRESSIVE_ANCHOR_MIN_KEYFRAMES:-20}"
  --progressive_max_anchor_gaussians "${PROGRESSIVE_MAX_ANCHOR_GAUSSIANS:-250000}"
  --progressive_max_anchor_keyframes "${PROGRESSIVE_MAX_ANCHOR_KEYFRAMES:-80}"
  --progressive_max_anchor_tsdf_voxels "${PROGRESSIVE_MAX_ANCHOR_TSDF_VOXELS:-250000}"
  --progressive_max_anchor_vram_mb "${PROGRESSIVE_MAX_ANCHOR_VRAM_MB:-0}"
  --progressive_anchor_final_iterations "${PROGRESSIVE_ANCHOR_FINAL_ITERATIONS:-5}"
  --progressive_anchor_merge_voxel_size "${PROGRESSIVE_ANCHOR_MERGE_VOXEL_SIZE:-0.05}"
  --progressive_anchor_merge_target_gaussians "${PROGRESSIVE_ANCHOR_MERGE_TARGET_GAUSSIANS:-150000}"
  --progressive_local_spawn_max "${PROGRESSIVE_LOCAL_SPAWN_MAX:-16384}"
  --progressive_local_spawn_target "${PROGRESSIVE_LOCAL_SPAWN_TARGET:-8192}"
  --progressive_surface_sample_floor "${PROGRESSIVE_SURFACE_SAMPLE_FLOOR:-0.01}"
  --progressive_low_frequency_spawn_fraction "${PROGRESSIVE_LOW_FREQUENCY_SPAWN_FRACTION:-0.15}"
  --progressive_edge_probability_threshold "${PROGRESSIVE_EDGE_PROBABILITY_THRESHOLD:-0.02}"
  --progressive_spawn_opacity_init "${PROGRESSIVE_SPAWN_OPACITY_INIT:-0.06}"
  --progressive_max_rasterized_gaussians "${PROGRESSIVE_MAX_RASTERIZED_GAUSSIANS:-60000}"
  --progressive_anchor_iterations "${PROGRESSIVE_ANCHOR_ITERATIONS:-20}"
  --progressive_anchor_train_views "${PROGRESSIVE_ANCHOR_TRAIN_VIEWS:-4}"
  --progressive_loop_max_candidates "${PROGRESSIVE_LOOP_MAX_CANDIDATES:-4}"
  --progressive_loop_min_anchor_gap "${PROGRESSIVE_LOOP_MIN_ANCHOR_GAP:-2}"
  --progressive_depth_valid_epsilon "${PROGRESSIVE_DEPTH_VALID_EPSILON:-1e-6}"
  --progressive_rgb_visible_weight "${PROGRESSIVE_RGB_VISIBLE_WEIGHT:-0.85}"
  --progressive_ssim_min_coverage "${PROGRESSIVE_SSIM_MIN_COVERAGE:-0.08}"
  --progressive_depth_conf_min "${PROGRESSIVE_DEPTH_CONF_MIN:-0.35}"
  --progressive_robust_loss_epsilon "${PROGRESSIVE_ROBUST_LOSS_EPSILON:-1e-3}"
  --progressive_depth_scale_min_samples "${PROGRESSIVE_DEPTH_SCALE_MIN_SAMPLES:-16}"
  --progressive_min_depth_scale "${PROGRESSIVE_MIN_DEPTH_SCALE:-1e-3}"
  --progressive_max_depth_scale "${PROGRESSIVE_MAX_DEPTH_SCALE:-1e3}"
)

if [[ "${PROGRESSIVE_REQUIRE_CALIBRATED_DEPTH:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no_progressive_require_calibrated_depth)
else
  COMMON_ARGS+=(--progressive_require_calibrated_depth)
fi

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
