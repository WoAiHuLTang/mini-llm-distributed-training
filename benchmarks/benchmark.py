"""Unified benchmark harness.

Runs the SAME MiniGPT workload under a chosen strategy and records:
    - step time (ms)
    - throughput (tokens/s)
    - peak CUDA memory (GB)
    - loss (sanity check that training is happening)

Results are appended to a CSV so that multiple runs (single / ddp / fsdp /
fsdp+ac) can be compared and plotted.

Usage (single GPU):
    python benchmarks/benchmark.py --strategy single --config configs/gpt_small.yaml

Usage (multi-GPU, via torchrun):
    torchrun --nproc_per_node=2 benchmarks/benchmark.py \
        --strategy ddp --config configs/gpt_small.yaml
    torchrun --nproc_per_node=2 benchmarks/benchmark.py \
        --strategy fsdp --config configs/gpt_small.yaml \
        --use-activation-checkpointing
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Make the src/ layout importable when running as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mini_llm.data import build_dataloaders  # noqa: E402
from mini_llm.model import GPTConfig  # noqa: E402
from mini_llm.trainer import Trainer, TrainerConfig  # noqa: E402
from mini_llm.utils import (  # noqa: E402
    Logger,
    destroy_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    load_yaml,
    synchronize,
)

CSV_HEADER = [
    "strategy",
    "gpus",
    "micro_batch_size",
    "seq_len",
    "hidden_size",
    "num_layers",
    "mixed_precision",
    "activation_checkpointing",
    "memory_gb",
    "tokens_per_s",
    "step_ms",
    "loss",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniGPT distributed benchmark")
    p.add_argument(
        "--strategy",
        choices=["single", "ddp", "fsdp", "deepspeed"],
        default="single",
    )
    p.add_argument("--config", type=str, default="configs/gpt_small.yaml")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)
    p.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--use-activation-checkpointing", action="store_true")
    p.add_argument(
        "--ds-config",
        type=str,
        default=None,
        help="Path to DeepSpeed JSON config (required for --strategy deepspeed)",
    )
    p.add_argument("--csv", type=str, default="benchmarks/results/benchmark.csv")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def append_csv_row(path: str, row: dict) -> None:
    """Append a single row to the CSV (creating header if needed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


def main() -> None:
    args = parse_args()

    # Initialize distributed for ddp/fsdp/deepspeed (torchrun sets env vars).
    if args.strategy in ("ddp", "fsdp", "deepspeed"):
        init_distributed(backend="nccl")
    rank = get_rank()
    world_size = get_world_size()
    logger = Logger(rank)

    cfg_dict = load_yaml(args.config)
    model_cfg = GPTConfig(**cfg_dict.get("model", {}))

    trainer_cfg = TrainerConfig(
        strategy=args.strategy,
        model=model_cfg,
        mixed_precision=args.mixed_precision,
        use_activation_checkpointing=args.use_activation_checkpointing,
        micro_batch_size=args.micro_batch_size,
        ds_config=args.ds_config,
        seed=args.seed,
    )

    # Data. Use a large-enough synthetic set so warmup+measure don't exhaust it.
    distributed = args.strategy in ("ddp", "fsdp", "deepspeed")
    num_train = max((args.warmup_steps + args.measure_steps) * args.micro_batch_size * 2, 4096)
    train_loader, _, train_sampler = build_dataloaders(
        seq_len=model_cfg.seq_len,
        vocab_size=model_cfg.vocab_size,
        micro_batch_size=args.micro_batch_size,
        num_train_samples=num_train,
        num_val_samples=64,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=args.seed,
    )

    trainer = Trainer(trainer_cfg)
    if train_sampler is not None:
        trainer.attach_train_sampler(train_sampler)

    logger.info(
        f"benchmarking strategy={args.strategy} gpus={world_size} "
        f"micro_batch={args.micro_batch_size} "
        f"ac={args.use_activation_checkpointing}"
    )

    # Measure steady-state throughput + peak memory.
    metrics = trainer.benchmark(
        train_loader,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )

    # Derive a strategy label. For DeepSpeed, encode the ZeRO stage from the
    # ds-config filename (e.g. deepspeed_z2.json -> "deepspeed_z2") so that
    # ZeRO-2 and ZeRO-3 runs can be told apart in the CSV / plots.
    strategy_label = args.strategy
    if args.strategy == "deepspeed" and args.ds_config:
        stem = Path(args.ds_config).stem
        if "z3" in stem:
            strategy_label = "deepspeed_z3"
        elif "z2" in stem:
            strategy_label = "deepspeed_z2"
        else:
            strategy_label = f"deepspeed_{stem}"

    # Only rank 0 writes the CSV row (all ranks measured the same workload).
    if is_main_process():
        row = {
            "strategy": strategy_label,
            "gpus": world_size,
            "micro_batch_size": args.micro_batch_size,
            "seq_len": model_cfg.seq_len,
            "hidden_size": model_cfg.hidden_size,
            "num_layers": model_cfg.num_layers,
            "mixed_precision": args.mixed_precision,
            "activation_checkpointing": int(args.use_activation_checkpointing),
            "memory_gb": round(metrics["peak_memory_gb"], 3),
            "tokens_per_s": round(metrics["tokens_per_s"], 1),
            "step_ms": round(metrics["step_ms"], 3),
            "loss": round(metrics["loss"], 4),
        }
        append_csv_row(args.csv, row)
        logger.info(
            f"RESULT strategy={args.strategy} gpus={world_size} "
            f"step_ms={row['step_ms']} tokens/s={row['tokens_per_s']} "
            f"peak_mem={row['memory_gb']}GB loss={row['loss']}"
        )

    synchronize()
    destroy_distributed()


if __name__ == "__main__":
    main()
