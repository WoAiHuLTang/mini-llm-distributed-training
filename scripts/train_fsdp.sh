#!/usr/bin/env bash
# Train MiniGPT with FSDP across N GPUs.
# Optionally enable activation checkpointing with AC=1.
# Usage: bash scripts/train_fsdp.sh [nproc] [micro_batch_size] [epochs] [ac]
set -euo pipefail

NPROC=${1:-2}
MICRO_BATCH=${2:-8}
EPOCHS=${3:-3}
AC=${4:-0}   # 1 to enable activation checkpointing
CONFIG=${CONFIG:-configs/gpt_small.yaml}

AC_ARGS=""
if [ "${AC}" = "1" ]; then
    AC_ARGS="--use-activation-checkpointing"
fi

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29501 \
    -m mini_llm.train \
    --strategy fsdp \
    --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --epochs "${EPOCHS}" \
    ${AC_ARGS}
