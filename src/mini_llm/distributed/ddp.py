"""DDP (DistributedDataParallel) wrapper.

DDP keeps a full replica of the model on every rank. During the backward pass,
gradients are averaged across ranks via an **AllReduce** collective over gradient
buckets. This trades memory (full model + full gradients per GPU) for high
throughput and near-linear scaling when the model fits on a single GPU.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def wrap_ddp(
    model: nn.Module,
    device: torch.device,
    *,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
    bucket_cap_mb: int = 25,
) -> nn.Module:
    """Wrap a model with DistributedDataParallel.

    Args:
        model: The model to wrap (must already be on ``device``).
        device: The target device (cuda:rank).
        find_unused_parameters: Set True if some params may not receive grads.
        gradient_as_bucket_view: Avoid extra gradient copies (memory saving).
        static_graph: Optimize for a fixed graph (faster, less flexible).
        bucket_cap_mb: Gradient bucket size in MB (trades comm vs. sync latency).

    Returns:
        The DDP-wrapped model.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before wrapping with DDP"
        )
    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        bucket_cap_mb=bucket_cap_mb,
    )
