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

3. Full anchor-local training:
   Not implemented yet. scripts/run_plan_training_full.sh intentionally exits
   non-zero until the anchor-local scene model, render path, training loop,
   and LoD scheduler/checkpoint wiring are implemented.

4. Legacy train_lod.py compatibility baseline only:
   ALLOW_LEGACY_TRAIN_LOD=1 scripts/run_plan_training_full.sh

5. Legacy baseline with another extractor/matcher:
   ALLOW_LEGACY_TRAIN_LOD=1 FEATURE_BACKEND=aliked MATCHER_BACKEND=lightglue scripts/run_plan_training_full.sh

6. Legacy baseline with ParallaxBA disabled:
   ALLOW_LEGACY_TRAIN_LOD=1 USE_PARALLAX_BA=0 RUN_NAME=plan_full_xfeat_mnn_no_parallax scripts/run_plan_training_full.sh

Logs:
  Unit logs:     results/plan_tests/unit_logs/
  Legacy logs:   results/MatrixCity/plan_full_logs/

Success condition:
  Unit flow: all pytest targets pass and module smoke prints finite counts.
  Full anchor-local flow: currently unavailable. Passing train_lod.py is only
                          a legacy compatibility baseline, not completion.
EOF
