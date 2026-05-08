#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cglab/project/on-the-fly-nvs}"
cd "${PROJECT_ROOT}"

cat <<'EOF'
Plan implementation test flow
=============================

1. Unit/module tests only:
   scripts/run_plan_unit_tests.sh

2. Full MatrixCity training with default xfeat+mnn:
   scripts/run_plan_training_full.sh

3. Full training with another extractor/matcher:
   FEATURE_BACKEND=aliked MATCHER_BACKEND=lightglue scripts/run_plan_training_full.sh

4. Full training with ParallaxBA disabled for comparison:
   USE_PARALLAX_BA=0 RUN_NAME=plan_full_xfeat_mnn_no_parallax scripts/run_plan_training_full.sh

5. Override dataset/output:
   SRC=/path/to/block_1 OUT_DIR=/path/to/output RUN_NAME=my_run scripts/run_plan_training_full.sh

6. Add extra train_lod.py args after the script:
   scripts/run_plan_training_full.sh --save_every 200 --display_runtimes

Logs:
  Unit logs:     results/plan_tests/unit_logs/
  Training logs: results/MatrixCity/plan_full_logs/

Success condition:
  Unit flow: all pytest targets pass and module smoke prints finite counts.
  Full flow: train_lod.py exits 0, saves reconstruction, metrics, failure log,
             and LoD completion marker when lod_min == lod_max.
EOF

