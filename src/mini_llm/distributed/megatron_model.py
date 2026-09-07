"""Megatron-style tensor-parallel (TP) re-implementation of MiniGPT.

This module builds a model with the *same* architecture as ``MiniGPT`` (same
layer count / hidden size / heads / activations) but whose large linear layers
are replaced by Megatron's tensor-parallel primitives:

    - ``VocabParallelEmbedding``        for the token embedding (shards vocab)
    - ``ColumnParallelLinear``          for QKV / MLP-up (shards output columns)
    - ``RowParallelLinear``             for attention-proj / MLP-down
    - ``vocab_parallel_cross_entropy``  for the LM loss over the sharded vocab

Megatron's TP layers expect activations in ``[sequence, batch, hidden]`` layout
and internally issue the AllGather / ReduceScatter / AllReduce collectives over
the tensor-parallel process group (set up by ``parallel_state``).  We therefore
transpose to ``[S, B, C]`` at the start of each transformer block and back to
``[B, S, C]`` at the end, so the rest of the pipeline (data / trainer) can keep
its usual ``[B, S, C]`` convention.

Sequence parallelism (SP) is an option: when enabled, the LayerNorm / Dropout
(which are element-wise over the hidden dim) operate on a sequence-sharded
activation, saving the redundant activation memory.  The TP layers then expect
the sequence-parallel input directly (no gather before them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import GPTConfig, RMSNorm


# --------------------------------------------------------------------------- #
# Megatron parallel-state helpers
# --------------------------------------------------------------------------- #
def init_megatron_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    seed: int = 0,
) -> None:
    """Initialize Megatron's model-parallel process groups (idempotent).

    Must be called *after* ``torch.distributed.init_process_group``.  Splits the
    world into TP groups (and optionally PP groups) so that Megatron's layers can
    find their communication peers via the global ``parallel_state``.

    Also seeds Megatron's CUDA RNG tracker (required before building any TP layer
    that shards weights, e.g. ``VocabParallelEmbedding`` / ``ColumnParallelLinear``).
    """
    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        return
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    # Seed the model-parallel CUDA RNG tracker used for sharded weight init.
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(seed)


def destroy_megatron_parallel() -> None:
    """Tear down Megatron's model-parallel state (call before process-group end)."""
    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()


def _build_mp_config(
    *,
    tp_size: int,
    pp_size: int,
    sequence_parallel: bool,
    params_dtype: torch.dtype,
    bf16: bool,
) -> "ModelParallelConfig":
    """Build a Megatron ``ModelParallelConfig`` matching our runtime settings."""
    from megatron.core.model_parallel_config import ModelParallelConfig

    cfg = ModelParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        bf16=bf16,
        fp16=(not bf16 and params_dtype == torch.float16),
        perform_initialization=True,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        async_tensor_model_parallel_allreduce=False,
        enable_autocast=False,
    )
    return cfg


def _default_init_method(std: float):
    """Return a Megatron-compatible init callable (normal with given std)."""

    def init_method(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_method


# --------------------------------------------------------------------------- #
# Megatron TP attention / MLP / block
# --------------------------------------------------------------------------- #
class MegatronCausalSelfAttention(nn.Module):
    """Causal self-attention whose QKV/proj are tensor-parallel."""

    def __init__(self, config: GPTConfig, mp_cfg, sequence_parallel: bool):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.sequence_parallel = sequence_parallel

        from megatron.core.tensor_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        init_method = _default_init_method(0.02)
        # QKV: we use *three* independent ColumnParallelLinear layers (one per of
        # q / k / v) instead of a single fused ``ColumnParallelLinear(hidden,
        # 3*hidden)``.  A fused layer would shard the whole ``3*C`` output
        # dimension into *contiguous* blocks, so with TP>1 a rank would receive a
        # slice that straddles the q/k/v boundaries and ``chunk(3)`` could no
        # longer recover its local q/k/v.  Three separate layers shard each of
        # q/k/v independently (each rank keeps a full local q/k/v slice of size
        # ``C/tp``), which stays correct for any TP size and maps 1:1 onto the
        # MiniGPT fused ``qkv.weight`` (rows ``[0:C]``=q, ``[C:2C]``=k,
        # ``[2C:3C]``=v).
        self.q_lin = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        self.k_lin = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        self.v_lin = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        # proj: input is the sharded attention output; RowParallelLinear reduces
        # across TP ranks (AllReduce) so the residual stream is full again.
        self.proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
            1, 1, config.seq_len, config.seq_len
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S_local, B, C].  Under sequence parallelism S_local = S/tp (the
        # block scattered the sequence dim so LayerNorm/Dropout run on a slice);
        # otherwise S_local = S.
        #
        # IMPORTANT: each ColumnParallelLinear (q/k/v) internally all-gathers a
        # sequence-sharded input back to the FULL sequence before the matmul, so
        # q/k/v always come out at the full sequence length [S, B, C/tp].  The
        # attention therefore always runs over the full sequence; only the final
        # RowParallelLinear (proj) reduce-scatters back to [S/tp, B, C] under SP.
        from megatron.core import parallel_state

        S_local, B, C = x.shape
        q, _ = self.q_lin(x)  # [S, B, C/tp]  (full seq; SP all-gathers input)
        k, _ = self.k_lin(x)
        v, _ = self.v_lin(x)

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        local_heads = self.num_heads // tp_size
        head_dim = self.head_dim
        S_full = q.shape[0]  # full sequence length (== S)

        q = q.view(S_full, B, local_heads, head_dim).transpose(0, 2)  # [h, B, S, d]
        k = k.view(S_full, B, local_heads, head_dim).transpose(0, 2)
        v = v.view(S_full, B, local_heads, head_dim).transpose(0, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # [h, B, S, S]
        causal = self.causal_mask[:, :, :S_full, :S_full]
        att = att.masked_fill(causal == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # [h, B, S, d]
        y = y.transpose(0, 2).contiguous().view(S_full, B, local_heads * head_dim)
        # proj: RowParallelLinear.  Under SP it reduce-scatters to [S/tp, B, C];
        # otherwise it AllReduces to [S, B, C].
        y, _ = self.proj(y)
        return y


class MegatronMLP(nn.Module):
    """Feed-forward whose fc1/fc2 are tensor-parallel."""

    def __init__(self, config: GPTConfig, mp_cfg):
        super().__init__()
        from megatron.core.tensor_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        init_method = _default_init_method(0.02)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.fc1(x)  # [S, B, 4C/tp]
        y = F.gelu(y)
        y, _ = self.fc2(y)  # [S, B, C]  (AllReduce -> full)
        return self.dropout(y)


class MegatronTransformerBlock(nn.Module):
    """Pre-norm transformer block with tensor-parallel attn + mlp.

    Handles the [B, S, C] <-> [S, B, C] transpose and (optionally) the sequence
    scatter/gather for sequence parallelism.
    """

    def __init__(self, config: GPTConfig, mp_cfg, sequence_parallel: bool):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.ln1 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attn = MegatronCausalSelfAttention(config, mp_cfg, sequence_parallel)
        self.ln2 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.mlp = MegatronMLP(config, mp_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, B, C] in sequence-first layout.  Under sequence parallelism the
        # model has already scattered the sequence dimension once, so x is
        # [S/tp, B, C]; otherwise x is the full [S, B, C].  The block keeps the
        # sequence sharded end-to-end (no gather here) so the full [S, B, C]
        # activation is never materialised between blocks -- this is what makes
        # SP save activation memory.  All the Column/RowParallelLinear layers
        # inherit ``sequence_parallel`` from the shared ModelParallelConfig:
        # the q/k/v projections all-gather the sharded input back to the full
        # sequence internally, and the proj / fc2 RowParallelLinear
        # reduce-scatter their output back to [S/tp, B, C].
        residual = x
        h = self.ln1(x)  # RMSNorm is per-token -> shard-safe
        h = self.attn(h)  # [S, B, C] or [S/tp, B, C]
        x = residual + h

        residual = x
        h = self.ln2(x)
        h = self.mlp(h)  # [S, B, C] or [S/tp, B, C]
        x = residual + h
        return x


# --------------------------------------------------------------------------- #
# Full Megatron TP model
# --------------------------------------------------------------------------- #
class MegatronMiniGPT(nn.Module):
    """Tensor-parallel MiniGPT with the same architecture as ``MiniGPT``.

    The token embedding is sharded over the vocab (``VocabParallelEmbedding``);
    the LM head shares the embedding weight and the loss is computed with
    ``vocab_parallel_cross_entropy`` so no full-vocab logits are ever materialised.
    """

    def __init__(
        self,
        config: GPTConfig,
        *,
        tp_size: int = 1,
        pp_size: int = 1,
        sequence_parallel: bool = False,
        params_dtype: torch.dtype = torch.float32,
        bf16: bool = False,
    ):
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.sequence_parallel = sequence_parallel

        mp_cfg = _build_mp_config(
            tp_size=tp_size,
            pp_size=pp_size,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
            bf16=bf16,
        )
        self.mp_cfg = mp_cfg

        from megatron.core.tensor_parallel import (
            ColumnParallelLinear,
            VocabParallelEmbedding,
        )

        init_method = _default_init_method(0.02)
        # Token embedding sharded over vocab.
        self.tok_emb = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_method=init_method,
            config=mp_cfg,
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                MegatronTransformerBlock(config, mp_cfg, sequence_parallel)
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = RMSNorm(config.hidden_size, config.layer_norm_eps)

        # LM head: a column-parallel projection onto the (sharded) vocab.  Its
        # weight is tied to the token embedding (standard GPT weight tying), so
        # the embedding matrix is reused as the output projection.
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=mp_cfg,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        # Weight tying: reuse the (vocab-sharded) embedding weight.
        self.lm_head.weight = self.tok_emb.weight

        # Scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return loss if targets given, else the (sharded) logits.

        NOTE: when targets is None we return the *per-rank* logits which only
        cover the local vocab shard -- this is only meaningful for debugging.
        Training always passes targets so the loss is computed correctly.
        """
        B, T = idx.shape
        assert T <= self.config.seq_len

        # Embedding: idx [B, T] -> [B, T, C] (vocab-sharded internally).
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)

        # Switch to the sequence-first [S, B, C] layout used by all the
        # Column/RowParallelLinear layers.  Under sequence parallelism we
        # scatter the sequence dimension ONCE here so that every transformer
        # block (and the final LayerNorm) runs on a 1/tp slice of the sequence;
        # the full [S, B, C] activation is only reconstructed inside the LM-head
        # projection (a ColumnParallelLinear that all-gathers its input when
        # ``sequence_parallel`` is on).
        x = x.transpose(0, 1).contiguous()  # [T, B, C]
        if self.sequence_parallel:
            from megatron.core.tensor_parallel import (
                scatter_to_sequence_parallel_region,
            )

            x = scatter_to_sequence_parallel_region(x)  # [T/tp, B, C]

        for block in self.blocks:
            x = block(x)  # [T, B, C] or [T/tp, B, C]
        x = self.ln_f(x)  # [T, B, C] or [T/tp, B, C]

        # LM-head projection onto the (vocab-sharded) logits.
        # ColumnParallelLinear expects [S, B, C].  Under SP it all-gathers the
        # sharded sequence back to the full length internally, so logits always
        # come out at the full sequence length [T, B, V/tp].
        logits, _ = self.lm_head(x)  # [T, B, V/tp]
        logits = logits.transpose(0, 1).contiguous()  # [B, T, V/tp]

        if targets is None:
            # Return local logits (vocab shard only) -- for debugging.
            return logits

        from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

        # vocab_parallel_cross_entropy expects [S*B, V/tp] logits and [S*B] labels.
        logits_2d = logits.view(-1, logits.size(-1))
        labels = targets.view(-1)
        loss = vocab_parallel_cross_entropy(logits_2d, labels)
        # Mask out ignore_index (-1) positions.
        mask = (labels != -1).float()
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() * 0.0
        return loss

    def num_parameters(self, trainable_only: bool = True) -> int:
        # For a TP model each rank holds 1/tp of the big matrices; report the
        # *full* model size by scaling local params up by tp_size.
        local = sum(
            p.numel()
            for p in self.parameters()
            if not trainable_only or p.requires_grad
        )
        return local * self.tp_size

    def reduce_gradients(self) -> None:
        """All-reduce the gradients of replicated (non tensor-parallel) params.

        Under sequence parallelism the LayerNorm weights (``ln1``/``ln2``/
        ``ln_f``) are replicated across the tensor-parallel ranks but each rank
        only runs them on a 1/tp slice of the sequence, so every rank computes a
        *partial* gradient (a sum over its own sequence shard).  These partial
        gradients must be summed across the TP group before the optimizer step,
        otherwise the replicated LayerNorms would diverge between ranks.

        The tensor-parallel-sharded weights (q/k/v, proj, fc1/fc2, tok_emb,
        lm_head) are *not* touched: their gradients are already the correct
        local shards.  ``pos_emb`` is also left alone because it is added to the
        embedding *before* the sequence scatter, so each rank already holds the
        full gradient.  This is a no-op when sequence parallelism is disabled.
        """
        if not self.sequence_parallel or self.tp_size <= 1:
            return
        from megatron.core import parallel_state

        group = parallel_state.get_tensor_model_parallel_group()
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            # Replicated LayerNorm weights (they run on a sequence shard).
            if (
                name == "ln_f.weight"
                or name.endswith(".ln1.weight")
                or name.endswith(".ln2.weight")
            ):
                torch.distributed.all_reduce(p.grad, group=group)
