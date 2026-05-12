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
  pipeline/concurrency.py
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
  scene/scale_alignment.py
  scene/opacity_reset.py
  scene/loop_verifier.py
  scene/anchor_graph.py
  scene/anchor_chunk_manager.py
  scene/anchor_local_map.py
  tests/test_parallax_geometry.py
  tests/test_parallax_mini_ba.py
  tests/test_adaptive_tsdf.py
  tests/test_anchor_local_map.py
  tests/test_render_guard.py
  tests/test_gaussian_optimizer.py
  tests/test_controller_async.py
  tests/test_scale_alignment.py
  tests/test_opacity_reset.py
  tests/test_loop_verifier.py
)

PYTEST_TARGETS=(
  tests/test_parallax_geometry.py
  tests/test_parallax_mini_ba.py
  tests/test_adaptive_tsdf.py
  tests/test_anchor_local_map.py
  tests/test_render_guard.py
  tests/test_gaussian_optimizer.py
  tests/test_controller_async.py
  tests/test_scale_alignment.py
  tests/test_opacity_reset.py
  tests/test_loop_verifier.py
)

{
  echo
  echo "[Step 1/3] py_compile"
  "${PYTHON_BIN}" -m py_compile "${COMPILE_TARGETS[@]}"

  echo
  echo "[Step 2/3] pytest"
  "${PYTHON_BIN}" -m pytest "${PYTEST_TARGETS[@]}" -q

  echo
  echo "[Step 3/3] module smoke"
  "${PYTHON_BIN}" -m pipeline.module_smoke_runner
  "${PYTHON_BIN}" -m pipeline.anchor_local_train --smoke-only

  echo
  echo "[UnitTest] PASS"
} 2>&1 | tee -a "${LOG_FILE}"
