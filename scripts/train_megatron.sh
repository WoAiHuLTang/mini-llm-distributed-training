#!/usr/bin/env bash
# Train MiniGPT with Megatron TP / PP / SP across N GPUs.
#
# Usage:
#   bash scripts/train_megatron.sh tp    [nproc] [micro_batch] [epochs]   # TP=2
#   bash scripts/train_megatron.sh tp_sp [nproc] [micro_batch] [epochs]   # TP=2 + SP
#   bash scripts/train_megatron.sh pp    [nproc] [micro_batch] [epochs]   # PP=2
#
# Examples:
#   bash scripts/train_megatron.sh tp 2 8 3
#   bash scripts/train_megatron.sh pp 2 8 3
set -euo pipefail

MODE=${1:-tp}            # tp | tp_sp | pp
NPROC=${2:-2}
MICRO_BATCH=${3:-8}
EPOCHS=${4:-3}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-4}
CONFIG=${CONFIG:-configs/gpt_small.yaml}

case "${MODE}" in
    tp)
        STRATEGY="megatron_tp"
        EXTRA_ARGS=(--tp-size "${NPROC}")
        ;;
    tp_sp)
        STRATEGY="megatron_tp"
        EXTRA_ARGS=(--tp-size "${NPROC}" --sequence-parallel)
        ;;
    pp)
        STRATEGY="megatron_pp"
        EXTRA_ARGS=(--pp-size "${NPROC}" --num-microbatches "${NUM_MICROBATCHES}")
        ;;
    *)
        echo "Unknown mode '${MODE}'. Use: tp | tp_sp | pp" >&2
        exit 1
        ;;
esac

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port=29503 \
    -m mini_llm.train \
    --strategy "${STRATEGY}" \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --epochs "${EPOCHS}"
