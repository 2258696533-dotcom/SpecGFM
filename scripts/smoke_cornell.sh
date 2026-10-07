#!/usr/bin/env bash
# Short GPU check of the paper entry. Writes only inside this release tree.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${SPECGFM_PYTHON:-/root/miniconda3/envs/mdgfm/bin/python}"
mkdir -p checkpoints logs

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "No CUDA GPU is visible. Turn on the GPU instance and run this script again."
  exit 1
fi

"$PY" -u run_specgfm.py \
  --mode specgfm \
  --dataset Cornell \
  --seeds 1024 \
  --shot_num 1 \
  -- \
  --epochs 1 \
  --eval_episodes 1
