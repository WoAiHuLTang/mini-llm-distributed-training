"""Shared utilities: config loading, logging, memory stats, distributed init."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.distributed as dist
import yaml


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """Initialize the default process group (idempotent)."""
    if is_dist_initialized():
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    dist.init_process_group(backend=backend)


def destroy_distributed() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()


def synchronize() -> None:
    """Barrier across all ranks (no-op on single process)."""
    if is_dist_initialized():
        dist.barrier()


# --------------------------------------------------------------------------- #
# Memory helpers
# --------------------------------------------------------------------------- #
def get_peak_memory_gb() -> float:
    """Peak CUDA memory allocated by the current process, in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_device(rank: int | None = None) -> torch.device:
    """Return the device for this process (cuda:rank if available else cpu)."""
    if torch.cuda.is_available():
        local_rank = rank if rank is not None else get_rank()
        return torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a (possibly nested) dataclass to a plain dict."""
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
class Logger:
    """Rank-aware logger that only prints from the main process."""

    def __init__(self, rank: int = 0, main_only: bool = True):
        self.rank = rank
        self.main_only = main_only

    def info(self, msg: str) -> None:
        if self.main_only and self.rank != 0:
            return
        print(f"[rank{self.rank}] {msg}", flush=True)

    def warn(self, msg: str) -> None:
        if self.main_only and self.rank != 0:
            return
        print(f"[rank{self.rank}][WARN] {msg}", flush=True)


def format_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


class Timer:
    """A simple context-manager / manual timer."""

    def __init__(self) -> None:
        self.start_time: float | None = None

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self.start_time is not None
        return time.perf_counter() - self.start_time
