"""FSDP (Fully Sharded Data Parallel) wrapper.

FSDP shards model parameters, gradients and (optionally) optimizer states across
ranks. Before each forward/backward, the needed parameters are gathered via an
**AllGather**; after use they are freed. Gradients are reduced via a
**ReduceScatter** so each rank keeps only its shard of the gradient.

This trades extra communication (AllGather/ReduceScatter per layer) for a large
reduction in per-GPU memory, enabling models that do not fit on a single GPU.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

from ..model import TransformerBlock


def _build_auto_wrap_policy(min_num_params: int = 1):
    """Auto-wrap policy that shards at TransformerBlock boundaries.

    In PyTorch >= 2.0 the wrap policies in ``torch.distributed.fsdp.wrap`` are
    plain callables of the form ``(module, recurse, nonwrapped_numel) -> bool``.
    To bind their extra configuration argument (e.g. ``transformer_layer_cls``)
    we must use ``functools.partial`` rather than calling them directly.
    """
    from torch.distributed.fsdp.wrap import (
        _or_policy,
        lambda_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    # Wrap each TransformerBlock (the natural FSDP unit for LLMs).
    block_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    # Also wrap any leftover large leaf modules (e.g. embedding/LM head) that
    # exceed min_num_params, so they get sharded too.
    lambda_policy = partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda m: sum(p.numel() for p in m.parameters(recurse=False))
        >= min_num_params,
    )
    return partial(_or_policy, policies=(block_policy, lambda_policy))


def wrap_fsdp(
    model: nn.Module,
    device: torch.device,
    *,
    sharding_strategy: str = "full_shard",
    mixed_precision: str = "bf16",
    use_activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    backward_prefetch: str = "backward_pre",
    min_num_params: int = 1_000_000,
) -> nn.Module:
    """Wrap a model with FullyShardedDataParallel.

    Args:
        model: Model to wrap (should be on CPU or the target device).
        device: Target device (cuda:rank).
        sharding_strategy: "full_shard" (FSDP/ZeRO-3) or "shard_grad_op"
            (ZeRO-2 style, params replicated, grads/opt sharded).
        mixed_precision: "bf16", "fp16", or "none".
        use_activation_checkpointing: Wrap transformer blocks with
            activation checkpointing to trade compute for memory.
        cpu_offload: Offload params/grads/opt states to CPU.
        backward_prefetch: "backward_pre" / "backward_post" / "none".
        min_num_params: Min params for lambda auto-wrap of large leaves.

    Returns:
        The FSDP-wrapped model.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before wrapping with FSDP"
        )

    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if sharding_strategy not in strategy_map:
        raise ValueError(
            f"Unknown sharding_strategy '{sharding_strategy}'. "
            f"Choose from {list(strategy_map)}"
        )

    # Mixed precision policy.
    if mixed_precision == "bf16":
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif mixed_precision == "fp16":
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif mixed_precision == "none":
        mp = None
    else:
        raise ValueError(f"Unknown mixed_precision '{mixed_precision}'")

    prefetch_map = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": BackwardPrefetch.BACKWARD_PRE,  # disabled below
    }
    prefetch = prefetch_map[backward_prefetch]
    if backward_prefetch == "none":
        prefetch = None

    auto_wrap_policy = _build_auto_wrap_policy(min_num_params)

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strategy_map[sharding_strategy],
        mixed_precision=mp,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        backward_prefetch=prefetch,
        device_id=device,
    )

    if use_activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        non_reentrant_wrapper = lambda m: checkpoint_wrapper(
            m,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=False,
        )
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: isinstance(m, TransformerBlock),
        )

    return fsdp_model
