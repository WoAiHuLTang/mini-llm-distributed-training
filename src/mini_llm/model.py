"""A minimal GPT-style decoder-only transformer used as the unified workload.

The model is intentionally simple and configurable so that the *same* model can
be trained under different distributed strategies (single GPU / DDP / FSDP) and
compared fairly. We care about *system performance* (memory / throughput /
communication), not language-modeling quality, so we keep the architecture
standard and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration for the MiniGPT model."""

    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    seq_len: int = 512
    intermediate_size: Optional[int] = None  # default: 4 * hidden_size
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    # FSDP-friendly: allow wrapping at transformer-block granularity.
    block_names: tuple = field(default=("blocks",), init=False)

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")


class RMSNorm(nn.Module):
    """RMSNorm used by GPT-style models (no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return (self.weight * x.to(dtype))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (registered buffer, moved with the module).
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
            1, 1, config.seq_len, config.seq_len
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Position-wise feed-forward network (SwiGLU-style optional, GELU default)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attn + mlp with residual connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """A minimal GPT-style language model.

    Structure:
        token embedding
          -> N x TransformerBlock
          -> final RMSNorm
          -> LM head (tied or untied)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_f = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying between embedding and LM head (standard GPT practice).
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Apply special scaled init to residual projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits if targets is None, else the mean cross-entropy loss."""
        B, T = idx.shape
        assert T <= self.config.seq_len, (
            f"seq len {T} exceeds configured max {self.config.seq_len}"
        )

        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return loss

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if not trainable_only or p.requires_grad
        )


def build_model(config: GPTConfig) -> MiniGPT:
    """Factory helper."""
    return MiniGPT(config)
