#!/usr/bin/env bash
# Train MiniGPT on a single GPU (baseline).
# Usage: bash scripts/train_single.sh [micro_batch_size] [epochs]
set -euo pipefail

MICRO_BATCH=${1:-8}
EPOCHS=${2:-3}
CONFIG=${CONFIG:-configs/gpt_small.yaml}

python -m mini_llm.train \
    --strategy single \
    --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --epochs "${EPOCHS}"
