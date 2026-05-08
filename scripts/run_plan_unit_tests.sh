#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cglab/project/on-the-fly-nvs}"
PYTHON_BIN="${PYTHON_BIN:-/home/cglab/miniconda/envs/onthefly_nvs/bin/python}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/plan_tests/unit_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/unit_${STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo "[UnitTest] project=${PROJECT_ROOT}" | tee "${LOG_FILE}"
echo "[UnitTest] python=${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "[UnitTest] log=${LOG_FILE}" | tee -a "${LOG_FILE}"

COMPILE_TARGETS=(
  pipeline/frame_state.py
  pipeline/observation_builder.py
  pipeline/gaussian_spawn_policy.py
  pipeline/reconstruction_controller.py
  pipeline/module_smoke_runner.py
  pipeline/anchor_local_train.py
  poses/rpc_compensator.py
  poses/parallax_geometry.py
  poses/parallax_mini_ba.py
  poses/parallax_pose_initializer.py
  scene/local_gaussian_model.py
  scene/adaptive_tsdf.py
  scene/tsdf_fusion.py
  scene/tsdf_losses.py
  scene/render_guard.py
  scene/gaussian_optimizer.py
  scene/anchor_graph.py
  scene/anchor_chunk_manager.py
  scene/anchor_local_map.py
  tests/test_parallax_geometry.py
  tests/test_parallax_mini_ba.py
  tests/test_adaptive_tsdf.py
  tests/test_anchor_local_map.py
  tests/test_render_guard.py
  tests/test_gaussian_optimizer.py
)

PYTEST_TARGETS=(
  tests/test_parallax_geometry.py
  tests/test_parallax_mini_ba.py
  tests/test_adaptive_tsdf.py
  tests/test_anchor_local_map.py
  tests/test_render_guard.py
  tests/test_gaussian_optimizer.py
  tests/test_pipeline_smoke.py
  tests/test_parallax_pose_initializer.py
  tests/test_anchor_graph.py
)

{
  echo
  echo "[Step 1/2] py_compile"
  "${PYTHON_BIN}" -m py_compile "${COMPILE_TARGETS[@]}"

  echo
  echo "[Step 2/2] pytest"
  "${PYTHON_BIN}" -m pytest "${PYTEST_TARGETS[@]}" -q

  echo
  echo "[UnitTest] PASS"
} 2>&1 | tee -a "${LOG_FILE}"

