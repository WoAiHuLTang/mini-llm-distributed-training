"""Unified training entry point.

Usage (single GPU):
    python -m mini_llm.train --strategy single --config configs/gpt_small.yaml

Usage (multi-GPU, launched via torchrun):
    torchrun --nproc_per_node=2 -m mini_llm.train \
        --strategy ddp --config configs/gpt_small.yaml
    torchrun --nproc_per_node=2 -m mini_llm.train \
        --strategy fsdp --config configs/gpt_small.yaml

This script performs a short real training run (loss decreasing) and prints a
summary. For rigorous, repeatable measurements use ``benchmarks/benchmark.py``.
"""

from __future__ import annotations

import argparse
import os

import torch

from .data import build_dataloaders
from .model import GPTConfig, MiniGPT
from .trainer import Trainer, TrainerConfig
from .utils import (
    Logger,
    destroy_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    load_yaml,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MiniGPT under a strategy")
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--config", type=str, default="configs/gpt_small.yaml")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--use-activation-checkpointing", action="store_true")
    p.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Distributed init if needed (torchrun sets these env vars).
    if args.strategy in ("ddp", "fsdp"):
        init_distributed(backend="nccl")
    rank = get_rank()
    world_size = get_world_size()
    logger = Logger(rank)

    # Load model config from yaml.
    cfg_dict = load_yaml(args.config)
    model_cfg = GPTConfig(**cfg_dict.get("model", {}))

    trainer_cfg = TrainerConfig(
        strategy=args.strategy,
        model=model_cfg,
        lr=args.lr if args.lr is not None else 3e-4,
        mixed_precision=args.mixed_precision,
        use_activation_checkpointing=args.use_activation_checkpointing,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    # Report the true parameter count from the raw (unwrapped) model.
    raw_model = MiniGPT(model_cfg)
    n_params = raw_model.num_parameters()
    logger.info(
        f"strategy={args.strategy} world_size={world_size} "
        f"model_params={n_params:,} layers={model_cfg.num_layers} "
        f"hidden={model_cfg.hidden_size}"
    )

    # Data.
    distributed = args.strategy in ("ddp", "fsdp")
    train_loader, val_loader, train_sampler = build_dataloaders(
        seq_len=model_cfg.seq_len,
        vocab_size=model_cfg.vocab_size,
        micro_batch_size=args.micro_batch_size,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=args.seed,
    )

    trainer = Trainer(trainer_cfg)
    if train_sampler is not None:
        trainer.attach_train_sampler(train_sampler)

    # Training loop.
    for epoch in range(args.epochs):
        trainer.set_epoch(epoch)
        running_loss = 0.0
        n = 0
        for step, batch in enumerate(train_loader):
            m = trainer.train_step(batch)
            running_loss += m["loss"]
            n += 1
            if is_main_process() and (step + 1) % trainer_cfg.log_interval == 0:
                logger.info(
                    f"epoch={epoch} step={step+1} loss={running_loss/n:.4f}"
                )
        val_loss = trainer.evaluate(val_loader)
        if is_main_process():
            logger.info(f"epoch={epoch} done, val_loss={val_loss:.4f}")

    if is_main_process():
        logger.info("training finished successfully")
    synchronize()
    destroy_distributed()


if __name__ == "__main__":
    main()
