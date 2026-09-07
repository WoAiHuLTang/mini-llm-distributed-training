#!/usr/bin/env bash
# Run the Megatron TP / PP / SP benchmark matrix and generate plots.
#
# This runs the SAME MiniGPT workload under Megatron-style parallelism:
#   megatron_tp (TP=2) -> megatron_tp_sp (TP=2 + SP) -> megatron_pp (PP=2)
# and appends results to benchmarks/results/benchmark.csv, then plots them.
#
# Usage: bash scripts/benchmark_megatron.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG=${CONFIG:-configs/gpt_small.yaml}
MICRO_BATCH=${MICRO_BATCH:-8}
WARMUP=${WARMUP:-5}
MEASURE=${MEASURE:-20}
CSV=${CSV:-benchmarks/results/benchmark.csv}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-4}

echo "==> Running Megatron benchmark matrix (micro_batch=${MICRO_BATCH})"

# 1) Megatron tensor parallel, TP=2 (2 GPU)
echo "---- megatron_tp (TP=2, 2 GPU) ----"
torchrun --nproc_per_node=2 --master_port=29520 \
    benchmarks/benchmark.py \
    --strategy megatron_tp --config "${CONFIG}" \
    --tp-size 2 \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --csv "${CSV}"

# 2) Megatron tensor parallel + sequence parallel, TP=2 + SP (2 GPU)
echo "---- megatron_tp_sp (TP=2 + SP, 2 GPU) ----"
torchrun --nproc_per_node=2 --master_port=29521 \
    benchmarks/benchmark.py \
    --strategy megatron_tp --config "${CONFIG}" \
    --tp-size 2 --sequence-parallel \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --csv "${CSV}"

# 3) Megatron pipeline parallel, PP=2 (2 GPU)
echo "---- megatron_pp (PP=2, 2 GPU) ----"
torchrun --nproc_per_node=2 --master_port=29522 \
    benchmarks/benchmark.py \
    --strategy megatron_pp --config "${CONFIG}" \
    --pp-size 2 --num-microbatches "${NUM_MICROBATCHES}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --warmup-steps "${WARMUP}" --measure-steps "${MEASURE}" \
    --csv "${CSV}"

echo "==> Generating plots"
python benchmarks/plot.py --csv "${CSV}" --outdir benchmarks/results

echo "==> Done. Results in ${CSV} and benchmarks/results/*.png"
