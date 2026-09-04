"""Distributed training strategy wrappers (single / DDP / FSDP)."""

from .ddp import wrap_ddp
from .fsdp import wrap_fsdp

__all__ = ["wrap_ddp", "wrap_fsdp"]
