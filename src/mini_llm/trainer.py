"""Unified trainer that runs the *same* MiniGPT under different strategies.

Supported strategies:
    - "single":      plain model on one GPU (baseline).
    - "ddp":         DistributedDataParallel (full replica per rank).
    - "fsdp":        FullyShardedDataParallel (sharded params/grads/opt states).
    - "deepspeed":   DeepSpeed ZeRO-2 / ZeRO-3 (engine takes over the step).
    - "megatron_tp": Megatron tensor parallelism (TP=2, or TP+SP when
                     ``sequence_parallel`` is set) via ``MegatronMiniGPT``.
    - "megatron_pp": Megatron pipeline parallelism (PP=2) via ``PipelineStage``
                     + a 1F1B ``PipelineRunner``.

The trainer exposes a ``train_step`` that can be driven either by a normal
training loop or by the benchmark harness / profiler, so all experiments share
one code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

from .distributed.ddp import wrap_ddp
from .distributed.deepspeed import wrap_deepspeed
from .distributed.fsdp import wrap_fsdp
from .distributed.megatron_loader import (
    broadcast_minigpt_state_dict,
    load_pp_weights,
    load_tp_weights,
)
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

    strategy: str = "single"  # single | ddp | fsdp | deepspeed | megatron_tp | megatron_pp
    # Model
    model: GPTConfig = field(default_factory=GPTConfig)
    # Optimizer / training
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    micro_batch_size: int = 8
    # Mixed precision (single/ddp use autocast; fsdp uses its own policy)
    mixed_precision: str = "bf16"  # bf16 | fp16 | none
    # FSDP options
    fsdp_sharding: str = "full_shard"
    use_activation_checkpointing: bool = False
    cpu_offload: bool = False
    # DeepSpeed options
    ds_config: Optional[str] = None  # path to DeepSpeed JSON config
    # Megatron options (used by megatron_tp / megatron_pp)
    tp_size: int = 1  # tensor-model-parallel size (megatron_tp)
    pp_size: int = 1  # pipeline-model-parallel size (megatron_pp)
    sequence_parallel: bool = False  # enable SP on top of TP (megatron_tp)
    num_microbatches: int = 1  # microbatches per step (megatron_pp)
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

        # Megatron strategies build their own (sharded) model and need the
        # Megatron parallel state set up first; they also load the *same* full
        # MiniGPT weights (broadcast from rank 0) so every strategy starts from
        # an identical point and benchmark losses stay comparable.
        if cfg.strategy in ("megatron_tp", "megatron_pp"):
            if not is_dist_initialized():
                raise RuntimeError(
                    f"{cfg.strategy} requires an initialized process group"
                )
            self._build_megatron()
        else:
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
            elif cfg.strategy == "deepspeed":
                if not is_dist_initialized():
                    raise RuntimeError(
                        "DeepSpeed requires an initialized process group"
                    )
                if not cfg.ds_config:
                    raise ValueError(
                        "DeepSpeed strategy requires a --ds-config JSON file"
                    )
                # DeepSpeed engine replaces the model AND manages the optimizer.
                self.model, self.optimizer = wrap_deepspeed(
                    self.model,
                    cfg.ds_config,
                    device=self.device,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    beta1=cfg.beta1,
                    beta2=cfg.beta2,
                    mixed_precision=cfg.mixed_precision,
                    micro_batch_size=cfg.micro_batch_size,
                    grad_clip=cfg.grad_clip,
                    seed=cfg.seed,
                )
            else:
                raise ValueError(f"Unknown strategy '{cfg.strategy}'")

        # Optimizer. For FSDP, only the *leaf* FSDP modules own real sharded
        # parameters; inner FSDP modules' params are already flattened into the
        # leaf that wraps them. Collecting every FSDP module's parameters would
        # double-count nested ones, so we only take leaf FSDP modules.
        # DeepSpeed already created its own optimizer inside wrap_deepspeed, so
        # we skip this block for the "deepspeed" strategy.
        if cfg.strategy == "deepspeed":
            pass  # self.optimizer already set by wrap_deepspeed
        elif cfg.strategy == "fsdp":
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

        # AMP autocast context for single / ddp. DeepSpeed manages its own
        # precision internally, so we disable the outer autocast for it.
        self.autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16,
            enabled=(
                self.device.type == "cuda"
                and cfg.mixed_precision != "none"
                and cfg.strategy != "deepspeed"
            ),
        )

        self._step = 0
        self._tokens_seen = 0

    # ------------------------------------------------------------------ #
    # Megatron model construction
    # ------------------------------------------------------------------ #
    def _build_megatron(self) -> None:
        """Build the Megatron TP / PP model and load the shared MiniGPT weights.

        ``megatron_tp`` builds a ``MegatronMiniGPT`` sharded across the TP group;
        ``megatron_pp`` builds a ``PipelineStage`` per rank plus a 1F1B
        ``PipelineRunner``.  Both load the *same* full ``MiniGPT`` weights
        (broadcast from rank 0) so every strategy starts from an identical point.
        """
        cfg = self.cfg
        # Megatron params stay fp32; the outer autocast context handles bf16
        # compute (matching the single/ddp code path).
        params_dtype = torch.float32
        bf16 = False

        if cfg.strategy == "megatron_tp":
            from .distributed.megatron_model import (
                MegatronMiniGPT,
                init_megatron_parallel,
            )

            init_megatron_parallel(
                tensor_model_parallel_size=cfg.tp_size,
                pipeline_model_parallel_size=1,
                seed=cfg.seed,
            )
            self.model = MegatronMiniGPT(
                cfg.model,
                tp_size=cfg.tp_size,
                pp_size=1,
                sequence_parallel=cfg.sequence_parallel,
                params_dtype=params_dtype,
                bf16=bf16,
            ).to(self.device)
            full_sd = broadcast_minigpt_state_dict(
                cfg.model, self.device, seed=cfg.seed
            )
            tp_rank = self.rank % cfg.tp_size
            load_tp_weights(self.model, full_sd, cfg.tp_size, tp_rank)
            self._megatron_runner = None

        elif cfg.strategy == "megatron_pp":
            from .distributed.megatron_model import init_megatron_parallel
            from .distributed.megatron_pipeline import (
                PipelineRunner,
                PipelineStage,
            )

            init_megatron_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=cfg.pp_size,
                seed=cfg.seed,
            )
            pp_rank = self.rank % cfg.pp_size
            self.model = PipelineStage(
                cfg.model,
                tp_size=1,
                pp_size=cfg.pp_size,
                pp_rank=pp_rank,
                params_dtype=params_dtype,
                bf16=bf16,
            ).to(self.device)
            full_sd = broadcast_minigpt_state_dict(
                cfg.model, self.device, seed=cfg.seed
            )
            load_pp_weights(self.model, full_sd)
            self.model.tie_embedding_weight()
            self._megatron_runner = PipelineRunner(self.model)
            self._pp_rank = pp_rank
            self._pp_size = cfg.pp_size

        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unknown megatron strategy '{cfg.strategy}'")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one forward + backward + optimizer step. Returns metrics dict."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        target_ids = batch["target_ids"].to(self.device, non_blocking=True)

        if self.cfg.strategy == "deepspeed":
            # DeepSpeed engine drives the whole step: forward on the engine,
            # then engine.backward(loss) (which reduces gradients) and
            # engine.step() (which clips + applies the optimizer). DeepSpeed
            # manages its own autocast / precision internally.
            self.optimizer.zero_grad()
            loss = self.model(input_ids, targets=target_ids)
            self.model.backward(loss)
            self.model.step()
        elif self.cfg.strategy == "megatron_pp":
            # Pipeline parallelism: split the batch into microbatches and run the
            # 1F1B schedule.  Only the last stage computes a loss; we broadcast it
            # back to the first stage (rank 0) so the benchmark harness records a
            # meaningful value on the main process.
            self.optimizer.zero_grad(set_to_none=True)
            M = max(1, self.cfg.num_microbatches)
            mb = input_ids.shape[0] // M
            microbatches = [
                (input_ids[i * mb:(i + 1) * mb], target_ids[i * mb:(i + 1) * mb])
                for i in range(M)
            ]
            with self.autocast_ctx:
                local_loss = self._megatron_runner.step(microbatches)
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()
            loss = self._sync_pp_loss(local_loss)
        else:
            self.optimizer.zero_grad(set_to_none=True)

            with self.autocast_ctx:
                loss = self.model(input_ids, targets=target_ids)

            loss.backward()

            # Under sequence parallelism the replicated LayerNorm gradients must
            # be summed across the TP group before the optimizer step.
            if self.cfg.strategy == "megatron_tp":
                self.model.reduce_gradients()

            # Clip gradients (FSDP: clip on the flattened params).
            if self.cfg.grad_clip > 0:
                if self.cfg.strategy == "fsdp":
                    from torch.distributed.fsdp import (
                        FullyShardedDataParallel as FSDP,
                    )

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

    def _sync_pp_loss(self, local_loss) -> torch.Tensor:
        """Broadcast the last stage's loss to every pipeline rank.

        ``local_loss`` is a scalar tensor on the last stage and ``None`` on the
        others.  We broadcast it from the last stage over the whole world so that
        every rank (in particular rank 0, the benchmark's main process) ends up
        with the same loss value.
        """
        if self._megatron_runner.is_last:
            t = local_loss.detach().float().reshape(1)
        else:
            t = torch.zeros(1, device=self.device)
        dist.broadcast(t, src=self._pp_size - 1)
        return t[0]

    def evaluate(self, val_loader) -> float:
        """Compute mean validation loss over one pass (no grad)."""
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                if self.cfg.strategy == "megatron_pp":
                    M = max(1, self.cfg.num_microbatches)
                    mb = input_ids.shape[0] // M
                    microbatches = [
                        (input_ids[i * mb:(i + 1) * mb], target_ids[i * mb:(i + 1) * mb])
                        for i in range(M)
                    ]
                    local_loss = self._megatron_runner.evaluate(microbatches)
                    loss = self._sync_pp_loss(local_loss)
                elif self.cfg.strategy == "deepspeed":
                    # DeepSpeed manages precision internally.
                    loss = self.model(input_ids, targets=target_ids)
                else:
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
        if self.cfg.strategy == "deepspeed":
            # DeepSpeed engine exposes the wrapped model via ``.module``.
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if self.cfg.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            FSDP.load_state_dict(self.model, state, strict=True)
        elif self.cfg.strategy == "deepspeed":
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
