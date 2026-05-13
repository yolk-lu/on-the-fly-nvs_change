#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cglab/project/on-the-fly-nvs}"
cd "${PROJECT_ROOT}"

cat <<'EOF'
Plan implementation test flow
=============================

1. Unit/module tests only:
   scripts/run_plan_unit_tests.sh

2. Anchor-local module smoke only:
   python -m pipeline.anchor_local_train --smoke-only

3. Full Progressive training, phase 1:
   scripts/run_plan_training_full.sh

   This runs Progressive_train.py with overlap optimization reserved but not
   implemented. The current backend delegates to the existing complete training
   loop so the workflow can finish while anchor-local render/optimizer are being
   replaced.

4. Direct legacy baseline, bypassing Progressive entry:
   TRAIN_PY=/home/cglab/project/on-the-fly-nvs/train_lod.py scripts/run_plan_training_full.sh

5. Progressive training with another extractor/matcher:
   FEATURE_BACKEND=aliked MATCHER_BACKEND=lightglue scripts/run_plan_training_full.sh

6. Progressive training with ParallaxBA disabled:
   USE_PARALLAX_BA=0 RUN_NAME=plan_full_xfeat_mnn_no_parallax scripts/run_plan_training_full.sh

Logs:
  Unit logs:     results/plan_tests/unit_logs/
  Training logs: results/MatrixCity/plan_full_logs/

Success condition:
  Unit flow: all pytest targets pass and module smoke prints finite counts.
  Full phase-1 flow: Progressive_train.py exits 0, writes reconstruction outputs,
                     and writes progressive_manifest.json.
EOF
