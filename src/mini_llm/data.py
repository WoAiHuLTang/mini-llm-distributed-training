"""Synthetic data pipeline.

For a *systems* benchmark we deliberately use synthetic random tokens instead of
a real corpus. This makes runs:
  - deterministic and reproducible,
  - independent of disk I/O and tokenizer,
  - focused purely on compute / memory / communication behaviour.

The dataset yields (input_ids, target_ids) pairs of shape (seq_len,) where
target_ids = input_ids shifted by one (standard causal LM setup).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticLMDataset(Dataset):
    """In-memory synthetic causal-LM dataset of random token ids."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Pre-generate all tokens once (cheap for typical benchmark sizes).
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(
            0, vocab_size, (num_samples, seq_len + 1), generator=g
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.tokens[idx]
        return {
            "input_ids": seq[:-1],
            "target_ids": seq[1:],
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a list of samples into a batched dict of tensors."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    target_ids = torch.stack([b["target_ids"] for b in batch])
    return {"input_ids": input_ids, "target_ids": target_ids}


def build_dataloaders(
    *,
    seq_len: int,
    vocab_size: int,
    micro_batch_size: int,
    num_train_samples: int = 4096,
    num_val_samples: int = 256,
    num_workers: int = 2,
    seed: int = 0,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    DistributedSampler | None,
]:
    """Build train/val dataloaders.

    If ``distributed`` is True, a DistributedSampler is attached so that each
    rank sees a disjoint shard of the data (required for correct DDP/FSDP).
    """
    from torch.utils.data import DataLoader, DistributedSampler

    train_ds = SyntheticLMDataset(num_train_samples, seq_len, vocab_size, seed=seed)
    val_ds = SyntheticLMDataset(num_val_samples, seq_len, vocab_size, seed=seed + 1)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, train_sampler
