#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <source_path> [base_output_dir] [extra train_lod.py args ...]"
  echo "Example: $0 /data/MatrixCity/block_1 results/MatrixCity/lg_ablation --downsampling 2 --test_hold 20"
  exit 1
fi

SRC="$1"
BASE_OUT="${2:-results/lightglue_ablation}"

if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

EXTRA_ARGS=("$@")
FEATURES=(superpoint disk sift aliked)

for FEAT in "${FEATURES[@]}"; do
  OUT="${BASE_OUT}_${FEAT}"
  echo "[LightGlue Ablation] Running ${FEAT} + LightGlue -> ${OUT}"
  python /home/cglab/project/on-the-fly-nvs/train_lod.py \
    -s "${SRC}" \
    -m "${OUT}" \
    --feature_backend "${FEAT}" \
    --matcher_backend lightglue \
    "${EXTRA_ARGS[@]}"
done
