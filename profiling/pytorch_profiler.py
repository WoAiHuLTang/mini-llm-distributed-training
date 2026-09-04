"""Profile a training step with PyTorch Profiler.

This captures the per-operator / per-collective breakdown of one training step
under a chosen strategy, so you can see where time goes:
    - forward / backward compute
    - NCCL collectives (AllReduce for DDP; AllGather / ReduceScatter for FSDP)
    - optimizer step

Outputs a Chrome-trace JSON (open in chrome://tracing or Perfetto) and prints a
table of the most expensive operators.

Usage (single GPU):
    python profiling/pytorch_profiler.py --strategy single \
        --config configs/gpt_small.yaml

Usage (multi-GPU):
    torchrun --nproc_per_node=2 profiling/pytorch_profiler.py \
        --strategy ddp --config configs/gpt_small.yaml
    torchrun --nproc_per_node=2 profiling/pytorch_profiler.py \
        --strategy fsdp --config configs/gpt_small.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile MiniGPT training step")
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--config", type=str, default="configs/gpt_small.yaml")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--use-activation-checkpointing", action="store_true")
    p.add_argument("--outdir", type=str, default="profiling/traces")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.strategy in ("ddp", "fsdp"):
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
        seed=args.seed,
    )

    distributed = args.strategy in ("ddp", "fsdp")
    num_train = max((args.warmup_steps + args.profile_steps + 2) * args.micro_batch_size * 2, 4096)
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

    # Warmup.
    for i, batch in enumerate(train_loader):
        if i >= args.warmup_steps:
            break
        trainer.train_step(batch)
    synchronize()

    os.makedirs(args.outdir, exist_ok=True)
    trace_file = os.path.join(args.outdir, f"{args.strategy}_rank{rank}.json")

    logger.info(f"profiling strategy={args.strategy} gpus={world_size} -> {trace_file}")

    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i, batch in enumerate(train_loader):
            if i >= args.profile_steps:
                break
            with record_function(f"step_{i}"):
                trainer.train_step(batch)
            prof.step()
    synchronize()

    prof.export_chrome_trace(trace_file)

    if is_main_process():
        # Print the top operators by CUDA time.
        print("\n===== Top operators by CUDA time =====")
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=25
            )
        )

        # Summarize collective communication time.
        print("\n===== Collective / communication summary =====")
        comm_keys = [
            "allreduce", "all_gather", "allgather", "reduce_scatter",
            "reduce_scatter_", "nccl", "broadcast", "AllReduce", "AllGather",
        ]
        total_cuda = 0.0
        comm_cuda = 0.0
        for evt in prof.key_averages():
            t = evt.cuda_time_total
            total_cuda += t
            name = evt.key.lower()
            if any(k in name for k in comm_keys):
                comm_cuda += t
                print(f"  {evt.key:<40s} {t/1000:8.2f} ms")
        if total_cuda > 0:
            print(f"\n  Total CUDA time: {total_cuda/1000:.2f} ms")
            print(f"  Communication time: {comm_cuda/1000:.2f} ms "
                  f"({comm_cuda/total_cuda*100:.1f}% of CUDA time)")

    synchronize()
    destroy_distributed()


if __name__ == "__main__":
    main()
