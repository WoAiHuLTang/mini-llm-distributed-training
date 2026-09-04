#!/usr/bin/env bash
# Train MiniGPT with DeepSpeed ZeRO across N GPUs.
# Usage: bash scripts/train_deepspeed.sh [nproc] [micro_batch_size] [epochs] [stage]
#   stage: 2 (ZeRO-2) or 3 (ZeRO-3), default 2
set -euo pipefail

NPROC=${1:-2}
MICRO_BATCH=${2:-8}
EPOCHS=${3:-3}
STAGE=${4:-2}   # 2 = ZeRO-2, 3 = ZeRO-3
CONFIG=${CONFIG:-configs/gpt_small.yaml}

if [ "${STAGE}" = "3" ]; then
    DS_CONFIG="configs/deepspeed_z3.json"
else
    DS_CONFIG="configs/deepspeed_z2.json"
fi

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29502 \
    -m mini_llm.train \
    --strategy deepspeed \
    --config "${CONFIG}" \
    --ds-config "${DS_CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --epochs "${EPOCHS}"
