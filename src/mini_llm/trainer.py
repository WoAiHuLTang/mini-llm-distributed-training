"""Unified trainer that runs the *same* MiniGPT under different strategies.

Supported strategies:
    - "single": plain model on one GPU (baseline).
    - "ddp":     DistributedDataParallel (full replica per rank).
    - "fsdp":    FullyShardedDataParallel (sharded params/grads/opt states).

The trainer exposes a ``train_step`` that can be driven either by a normal
training loop or by the benchmark harness / profiler, so all experiments share
one code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from .distributed.ddp import wrap_ddp
from .distributed.fsdp import wrap_fsdp
from .model import GPTConfig, MiniGPT
from .utils import (
    Logger,
    get_device,
    get_rank,
    get_world_size,
    is_dist_initialized,
    reset_peak_memory,
    synchronize,
)


@dataclass
class TrainerConfig:
    """Runtime configuration for the trainer."""

    strategy: str = "single"  # single | ddp | fsdp
    # Model
    model: GPTConfig = field(default_factory=GPTConfig)
    # Optimizer / training
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Mixed precision (single/ddp use autocast; fsdp uses its own policy)
    mixed_precision: str = "bf16"  # bf16 | fp16 | none
    # FSDP options
    fsdp_sharding: str = "full_shard"
    use_activation_checkpointing: bool = False
    cpu_offload: bool = False
    # Logging
    log_interval: int = 10
    seed: int = 0


class Trainer:
    """Unified trainer for single / DDP / FSDP strategies."""

    def __init__(
        self,
        cfg: TrainerConfig,
        *,
        device: Optional[torch.device] = None,
        logger: Optional[Logger] = None,
    ):
        self.cfg = cfg
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = device or get_device()
        self.logger = logger or Logger(self.rank)

        torch.manual_seed(cfg.seed + self.rank)

        # Build the model on CPU first, then move / wrap per strategy.
        self.model = MiniGPT(cfg.model)

        if cfg.strategy == "single":
            self.model = self.model.to(self.device)
        elif cfg.strategy == "ddp":
            if not is_dist_initialized():
                raise RuntimeError("DDP requires an initialized process group")
            self.model = self.model.to(self.device)
            self.model = wrap_ddp(self.model, self.device)
        elif cfg.strategy == "fsdp":
            if not is_dist_initialized():
                raise RuntimeError("FSDP requires an initialized process group")
            self.model = wrap_fsdp(
                self.model,
                self.device,
                sharding_strategy=cfg.fsdp_sharding,
                mixed_precision=cfg.mixed_precision,
                use_activation_checkpointing=cfg.use_activation_checkpointing,
                cpu_offload=cfg.cpu_offload,
            )
        else:
            raise ValueError(f"Unknown strategy '{cfg.strategy}'")

        # Optimizer. For FSDP, only the *leaf* FSDP modules own real sharded
        # parameters; inner FSDP modules' params are already flattened into the
        # leaf that wraps them. Collecting every FSDP module's parameters would
        # double-count nested ones, so we only take leaf FSDP modules.
        if cfg.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            all_fsdp = FSDP.fsdp_modules(self.model)
            # A leaf FSDP module has no FSDP children.
            leaf_fsdp = [
                m
                for m in all_fsdp
                if not any(
                    isinstance(c, FSDP) for c in m.modules() if c is not m
                )
            ]
            flat_params = [
                p
                for m in leaf_fsdp
                for p in m.parameters()
                if p.requires_grad
            ]
            self.optimizer = AdamW(
                flat_params,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
                foreach=True,
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
                foreach=True,
            )

        # AMP autocast context for single / ddp.
        self.autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16,
            enabled=(self.device.type == "cuda" and cfg.mixed_precision != "none"),
        )

        self._step = 0
        self._tokens_seen = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one forward + backward + optimizer step. Returns metrics dict."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        target_ids = batch["target_ids"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast_ctx:
            loss = self.model(input_ids, targets=target_ids)

        loss.backward()

        # Clip gradients (FSDP: clip on the flattened params).
        if self.cfg.grad_clip > 0:
            if self.cfg.strategy == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                FSDP.clip_grad_norm_(self.model, self.cfg.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )

        self.optimizer.step()

        tokens = input_ids.numel()
        self._step += 1
        self._tokens_seen += tokens

        return {
            "loss": float(loss.detach().float()),
            "tokens": tokens,
        }

    def evaluate(self, val_loader) -> float:
        """Compute mean validation loss over one pass (no grad)."""
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                with self.autocast_ctx:
                    loss = self.model(input_ids, targets=target_ids)
                total_loss += float(loss.detach().float())
                n += 1
        self.model.train()
        if n == 0:
            return float("nan")
        return total_loss / n

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch on the distributed sampler (if present)."""
        sampler = getattr(self, "_train_sampler", None)
        if sampler is not None:
            sampler.set_epoch(epoch)

    def attach_train_sampler(self, sampler) -> None:
        self._train_sampler = sampler

    # ------------------------------------------------------------------ #
    # Benchmark helpers
    # ------------------------------------------------------------------ #
    def benchmark(
        self,
        loader,
        *,
        warmup_steps: int = 5,
        measure_steps: int = 20,
    ) -> dict[str, float]:
        """Measure steady-state throughput and peak memory.

        Returns a dict with:
            step_ms, tokens_per_s, peak_memory_gb, loss
        """
        reset_peak_memory()

        # Warmup
        for i, batch in enumerate(loader):
            if i >= warmup_steps:
                break
            self.train_step(batch)
        synchronize()

        # Measure
        reset_peak_memory()
        start = time.perf_counter()
        total_tokens = 0
        loss_sum = 0.0
        steps = 0
        for i, batch in enumerate(loader):
            if i >= measure_steps:
                break
            m = self.train_step(batch)
            total_tokens += m["tokens"]
            loss_sum += m["loss"]
            steps += 1
        synchronize()
        elapsed = time.perf_counter() - start

        peak_mem = self._get_peak_memory_gb()

        return {
            "step_ms": (elapsed / steps) * 1000.0 if steps else 0.0,
            "tokens_per_s": total_tokens / elapsed if elapsed else 0.0,
            "peak_memory_gb": peak_mem,
            "loss": loss_sum / steps if steps else float("nan"),
            "steps": steps,
        }

    def _get_peak_memory_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0.0

    # ------------------------------------------------------------------ #
    # State dict helpers (strategy-aware)
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        if self.cfg.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return FSDP.state_dict(self.model)
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if self.cfg.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            FSDP.load_state_dict(self.model, state, strict=True)
        else:
            self.model.load_state_dict(state)
