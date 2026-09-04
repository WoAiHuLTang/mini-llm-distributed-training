#!/usr/bin/env bash
# Train MiniGPT with DDP across N GPUs.
# Usage: bash scripts/train_ddp.sh [nproc] [micro_batch_size] [epochs]
set -euo pipefail

NPROC=${1:-2}
MICRO_BATCH=${2:-8}
EPOCHS=${3:-3}
CONFIG=${CONFIG:-configs/gpt_small.yaml}

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29500 \
    -m mini_llm.train \
    --strategy ddp \
    --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --epochs "${EPOCHS}"
