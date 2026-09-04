"""DeepSpeed ZeRO wrapper.

DeepSpeed shards optimizer states (ZeRO-1), gradients (ZeRO-2) and model
parameters (ZeRO-3) across ranks. Unlike DDP/FSDP where we keep our own
optimizer and call ``loss.backward()`` / ``optimizer.step()``, DeepSpeed takes
over the whole training step through an *engine*:

    engine, optimizer, _, _ = deepspeed.initialize(...)
    loss = engine(batch)          # forward (engine is the wrapped model)
    engine.backward(loss)         # backward + gradient reduction
    engine.step()                 # optimizer step + gradient clipping

This module exposes a thin ``wrap_deepspeed`` helper that builds the engine from
a raw model and a DeepSpeed JSON config, so the unified Trainer can treat
"deepspeed" as just another strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def _load_ds_config(path: str) -> dict:
    """Load a DeepSpeed JSON config (path or inline dict)."""
    if isinstance(path, dict):
        return path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_deepspeed(
    model: nn.Module,
    ds_config_path: str,
    *,
    device: torch.device,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    mixed_precision: str = "bf16",
    micro_batch_size: int = 8,
    grad_clip: float = 1.0,
    seed: int = 0,
):
    """Build a DeepSpeed engine for the given model.

    Returns the ``engine`` (a wrapped nn.Module exposing ``backward``/``step``)
    and the ``optimizer``. The engine replaces the raw model in the Trainer.
    """
    import deepspeed
    from torch.distributed import get_world_size

    ds_cfg = _load_ds_config(ds_config_path)

    # Ensure the model is on the right device before handing it to DeepSpeed.
    model = model.to(device)

    # ---- Batch size ------------------------------------------------------ #
    # We do NOT hand a dataloader to deepspeed.initialize(), so DeepSpeed cannot
    # auto-resolve "auto" batch sizes. Resolve them to concrete numbers here:
    #   train_batch_size = micro_batch * grad_accum * world_size
    world_size = get_world_size()
    grad_accum = int(ds_cfg.get("gradient_accumulation_steps", 1))
    if ds_cfg.get("train_micro_batch_size_per_gpu") in (None, "auto"):
        ds_cfg["train_micro_batch_size_per_gpu"] = micro_batch_size
    if ds_cfg.get("train_batch_size") in (None, "auto"):
        ds_cfg["train_batch_size"] = (
            int(ds_cfg["train_micro_batch_size_per_gpu"]) * grad_accum * world_size
        )
    if ds_cfg.get("gradient_accumulation_steps") in (None, "auto"):
        ds_cfg["gradient_accumulation_steps"] = grad_accum

    # ---- Mixed precision ------------------------------------------------ #
    # DeepSpeed manages its own autocast via the config. Map our CLI precision
    # into the JSON if the user did not already set it there.
    if "fp16" not in ds_cfg and "bf16" not in ds_cfg:
        if mixed_precision == "fp16":
            ds_cfg["fp16"] = {"enabled": True}
        elif mixed_precision == "bf16":
            ds_cfg["bf16"] = {"enabled": True}
        else:
            ds_cfg["fp16"] = {"enabled": False}
            ds_cfg["bf16"] = {"enabled": False}

    # ---- Optimizer ------------------------------------------------------ #
    # If the config does not define an optimizer, let DeepSpeed build an AdamW
    # with our hyper-parameters.
    if "optimizer" not in ds_cfg:
        ds_cfg["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [beta1, beta2],
                "eps": 1e-8,
                "weight_decay": weight_decay,
            },
        }

    # ---- Gradient clipping --------------------------------------------- #
    if "gradient_clipping" not in ds_cfg:
        ds_cfg["gradient_clipping"] = grad_clip

    # ---- ZeRO stage sanity --------------------------------------------- #
    zero_opt = ds_cfg.get("zero_optimization", {})
    stage = zero_opt.get("stage", 0)
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported DeepSpeed ZeRO stage: {stage}")

    # ---- Build the engine ---------------------------------------------- #
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg,
        dist_init_required=False,  # we already initialized the process group
    )
    return engine, optimizer
