#!/usr/bin/env bash
# Run the full benchmark matrix and generate plots.
#
# This runs the SAME MiniGPT workload under:
#   single (1 GPU) -> ddp (2 GPU) -> fsdp (2 GPU) -> fsdp+ac (2 GPU)
# and appends results to benchmarks/results/benchmark.csv, then plots them.
#
# Usage: bash scripts/benchmark.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG=${CONFIG:-configs/gpt_small.yaml}
MICRO_BATCH=${MICRO_BATCH:-8}
WARMUP=${WARMUP:-5}
MEASURE=${MEASURE:-20}
CSV=${CSV:-benchmarks/results/benchmark.csv}

echo "==> Running benchmark matrix (micro_batch=${MICRO_BATCH})"

# 1) Single GPU
echo "---- single (1 GPU) ----"
python benchmarks/benchmark.py \
    --strategy single --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --csv "${CSV}"

# 2) DDP (2 GPU)
echo "---- ddp (2 GPU) ----"
torchrun --nproc_per_node=2 --master_port=29510 \
    benchmarks/benchmark.py \
    --strategy ddp --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --csv "${CSV}"

# 3) FSDP (2 GPU)
echo "---- fsdp (2 GPU) ----"
torchrun --nproc_per_node=2 --master_port=29511 \
    benchmarks/benchmark.py \
    --strategy fsdp --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --csv "${CSV}"

# 4) FSDP + activation checkpointing (2 GPU)
echo "---- fsdp + activation checkpointing (2 GPU) ----"
torchrun --nproc_per_node=2 --master_port=29512 \
    benchmarks/benchmark.py \
    --strategy fsdp --config "${CONFIG}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --use-activation-checkpointing \
    --csv "${CSV}"

echo "==> Generating plots"
python benchmarks/plot.py --csv "${CSV}" --outdir benchmarks/results

echo "==> Done. Results in ${CSV} and benchmarks/results/*.png"
