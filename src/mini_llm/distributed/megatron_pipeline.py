"""Megatron-style pipeline parallelism (PP) for MiniGPT.

This module splits the transformer across ``pipeline_model_parallel_size``
stages (one per rank) and runs a microbatch pipeline schedule so that the
stages overlap their forward/backward work.  It is self-contained: it uses the
raw ``torch.distributed`` point-to-point primitives over Megatron's pipeline
process group (set up by ``parallel_state``) rather than Megatron's full
schedule machinery, which keeps the integration with the rest of this repo
light.

Layout (PP=2, layers split evenly):
    stage 0 (rank 0): tok_emb + pos_emb + drop + blocks[0 : N/2]
    stage 1 (rank 1): blocks[N/2 : N] + ln_f + lm_head

The hidden state exchanged between stages is the sequence-first ``[S, B, C]``
activation (PP is benchmarked on its own, not combined with SP here), matching
the layout that ``MegatronTransformerBlock`` expects.  Each stage is a plain
``nn.Module``; the ``PipelineRunner`` drives the non-interleaved 1F1B schedule
and the send/recv of activations (forward) and gradients (backward).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..model import GPTConfig, RMSNorm
from .megatron_model import (
    MegatronCausalSelfAttention,
    MegatronMLP,
    MegatronTransformerBlock,
    _build_mp_config,
    _default_init_method,
)


def _split_layers(num_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    """Return the [start, end) layer range owned by ``pp_rank``."""
    layers_per_stage = num_layers // pp_size
    start = pp_rank * layers_per_stage
    end = start + layers_per_stage
    return start, end


class PipelineStage(nn.Module):
    """The slice of the model owned by one pipeline rank.

    ``is_first`` stages own the embedding; ``is_last`` stages own the final
    LayerNorm and the LM head.  Middle stages own only transformer blocks.

    All forward methods operate in the sequence-first ``[S, B, C]`` layout that
    ``MegatronTransformerBlock`` expects (the embedding output is transposed
    once inside ``forward_first``).  The hidden tensor handed between stages is
    therefore ``[S, B, C]``.
    """

    def __init__(
        self,
        config: GPTConfig,
        *,
        tp_size: int = 1,
        pp_size: int = 1,
        pp_rank: int = 0,
        params_dtype: torch.dtype = torch.float32,
        bf16: bool = False,
    ):
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.is_first = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1
        self.params_dtype = params_dtype

        mp_cfg = _build_mp_config(
            tp_size=tp_size,
            pp_size=pp_size,
            sequence_parallel=False,  # PP benchmarked standalone (no SP)
            params_dtype=params_dtype,
            bf16=bf16,
        )
        self.mp_cfg = mp_cfg
        init_method = _default_init_method(0.02)

        from megatron.core.tensor_parallel import (
            ColumnParallelLinear,
            VocabParallelEmbedding,
        )

        start, end = _split_layers(config.num_layers, pp_size, pp_rank)
        self.layer_start = start
        self.layer_end = end

        if self.is_first:
            self.tok_emb = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                init_method=init_method,
                config=mp_cfg,
            )
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.seq_len, config.hidden_size)
            )
            self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                MegatronTransformerBlock(config, mp_cfg, sequence_parallel=False)
                for _ in range(start, end)
            ]
        )

        if self.is_last:
            self.ln_f = RMSNorm(config.hidden_size, config.layer_norm_eps)
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                config=mp_cfg,
                init_method=init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=True,
            )
            # Weight tying: reuse the (vocab-sharded) embedding weight, which
            # lives on stage 0.  We copy it over after loading (see
            # ``tie_embedding_weight``) rather than sharing the Parameter
            # object across ranks.
            self._tied = False

        # Scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    # -- forward methods (all in the [S, B, C] sequence-first domain) --------- #
    def forward_first(self, idx: torch.Tensor) -> torch.Tensor:
        """Embedding + first blocks.  idx: [B, T] -> hidden [T, B, C]."""
        B, T = idx.shape
        assert T <= self.config.seq_len
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]  # [B, T, C]
        x = self.drop(x)
        x = x.transpose(0, 1).contiguous()  # [T, B, C]
        for block in self.blocks:
            x = block(x)
        return x

    def forward_middle(self, x: torch.Tensor) -> torch.Tensor:
        """Blocks only.  x: [T, B, C] -> [T, B, C]."""
        for block in self.blocks:
            x = block(x)
        return x

    def forward_last(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Final blocks + ln_f + lm_head + loss.  Returns a scalar loss."""
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # [T, B, C]
        # LM head: ColumnParallelLinear expects [S, B, C] (x already is).
        logits, _ = self.lm_head(x)  # [T, B, V/tp]
        logits = logits.transpose(0, 1).contiguous()  # [B, T, V/tp]

        from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

        logits_2d = logits.view(-1, logits.size(-1))
        labels = targets.view(-1)
        loss = vocab_parallel_cross_entropy(logits_2d, labels)
        mask = (labels != -1).float()
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() * 0.0
        return loss

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dispatch to the right sub-forward based on this stage's position."""
        if self.is_first:
            return self.forward_first(x)
        if self.is_last:
            return self.forward_last(x, targets)
        return self.forward_middle(x)

    # -- weight tying across pipeline ranks ----------------------------------- #
    def tie_embedding_weight(self) -> None:
        """Copy the (vocab-sharded) embedding weight from stage 0 to the LM head.

        Must be called on *every* pipeline rank after weights are loaded.  Uses a
        broadcast over the pipeline group so the last stage's ``lm_head.weight``
        mirrors the first stage's ``tok_emb.weight`` (standard GPT weight tying).
        """
        from megatron.core import parallel_state

        group = parallel_state.get_pipeline_model_parallel_group()
        src_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        if self.is_first:
            src = self.tok_emb.weight.detach()
            dist.broadcast(src, src=src_rank, group=group)
        if self.is_last:
            dist.broadcast(self.lm_head.weight, src=src_rank, group=group)
            self._tied = True

    def num_parameters(self, trainable_only: bool = True) -> int:
        local = sum(
            p.numel()
            for p in self.parameters()
            if not trainable_only or p.requires_grad
        )
        return local * self.tp_size


class PipelineRunner:
    """Drives the non-interleaved 1F1B schedule for one pipeline stage.

    Each rank owns one ``PipelineStage``.  ``step`` splits a batch into
    microbatches and runs the warmup / steady-state (1F1B) / cooldown schedule,
    overlapping forward and backward work across stages via point-to-point
    communication over the pipeline process group.

    Only the last stage returns a (mean) loss; other ranks return ``None``.
    """

    def __init__(self, stage: PipelineStage):
        self.stage = stage
        from megatron.core import parallel_state

        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.group = parallel_state.get_pipeline_model_parallel_group()
        self.is_first = stage.is_first
        self.is_last = stage.is_last
        self.dtype = stage.params_dtype
        self.device = stage.pos_emb.device if stage.is_first else None
        if self.device is None:
            # Fall back to the device of the first block parameter.
            self.device = next(stage.parameters()).device

    # -- low-level p2p helpers (batch_isend_irecv avoids deadlock) ------------- #
    def _batch(self, send_tensor, recv_shape, send_rank, recv_rank):
        ops = []
        if send_tensor is not None:
            ops.append(dist.P2POp(dist.isend, send_tensor, send_rank, self.group))
        recv_tensor = None
        if recv_shape is not None:
            recv_tensor = torch.empty(
                recv_shape, dtype=self.dtype, device=self.device
            )
            recv_tensor.requires_grad_(True)
            ops.append(dist.P2POp(dist.irecv, recv_tensor, recv_rank, self.group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
        return recv_tensor

    def _send_to_next(self, t):
        self._batch(t, None, self.pp_rank + 1, None)

    def _recv_from_prev(self, shape):
        return self._batch(None, shape, None, self.pp_rank - 1)

    def _send_to_prev(self, t):
        self._batch(t, None, self.pp_rank - 1, None)

    def _recv_from_next(self, shape):
        return self._batch(None, shape, None, self.pp_rank + 1)

    def _send_to_next_recv_from_next(self, t, shape):
        return self._batch(t, shape, self.pp_rank + 1, self.pp_rank + 1)

    def _send_to_prev_recv_from_prev(self, t, shape):
        return self._batch(t, shape, self.pp_rank - 1, self.pp_rank - 1)

    # -- schedule steps -------------------------------------------------------- #
    def _forward_step(self, input_tensor, idx, targets):
        if self.is_first:
            return self.stage.forward_first(idx)
        if self.is_last:
            return self.stage.forward_last(input_tensor, targets)
        return self.stage.forward_middle(input_tensor)

    def _backward_step(self, input_tensor, output_tensor, grad_output):
        """Backward through one microbatch.  Returns grad w.r.t. input_tensor."""
        if self.is_last:
            # output_tensor is the scalar loss -> implicit grad of 1.0.
            output_tensor.backward()
        else:
            output_tensor.backward(grad_output)
        if input_tensor is not None and input_tensor.grad is not None:
            return input_tensor.grad
        return None

    def step(self, microbatches: list) -> Optional[torch.Tensor]:
        """Run one 1F1B pipeline step over ``microbatches``.

        ``microbatches`` is a list of ``(idx, targets)`` pairs, each a
        ``[B_mb, T]`` tensor.  Every stage receives the same list.  Returns the
        mean loss on the last stage, else ``None``.
        """
        M = len(microbatches)
        rank, world = self.pp_rank, self.pp_size
        num_warmup = min(world - rank - 1, M)
        num_steady = M - num_warmup

        # Shape of the hidden tensor exchanged between stages.
        _, T = microbatches[0][0].shape
        hidden_shape = (T, microbatches[0][0].shape[0], self.stage.config.hidden_size)

        input_tensors: list = []  # saved inputs for backward
        output_tensors: list = []  # saved outputs (hidden) for backward
        losses: list = []
        mb = 0

        # ---- warmup forward passes ----
        for _ in range(num_warmup):
            if rank > 0:
                input_tensor = self._recv_from_prev(hidden_shape)
            else:
                input_tensor = None
            idx, targets = microbatches[mb]
            mb += 1
            output_tensor = self._forward_step(input_tensor, idx, targets)
            if rank < world - 1:
                self._send_to_next(output_tensor)
            if self.is_last:
                losses.append(output_tensor)
            else:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

        # Receive the first forward tensor before entering steady state.
        if num_steady > 0:
            if rank > 0:
                input_tensor = self._recv_from_prev(hidden_shape)
            else:
                input_tensor = None

        # ---- steady state: 1 forward + 1 backward per iteration ----
        for i in range(num_steady):
            last_iter = i == num_steady - 1
            idx, targets = microbatches[mb]
            mb += 1
            output_tensor = self._forward_step(input_tensor, idx, targets)

            if self.is_last:
                losses.append(output_tensor)
                # Scale the loss by 1/M so the accumulated gradient matches a
                # single forward over the whole (concatenated) batch: each
                # microbatch loss is already normalised by its own token count,
                # so summing M of them would otherwise over-count by a factor M.
                self._backward_step(input_tensor, output_tensor / M, None)
                input_grad = (
                    input_tensor.grad if input_tensor is not None else None
                )
                if last_iter:
                    input_tensor = None
                    if rank > 0:
                        self._send_to_prev(input_grad)
                else:
                    input_tensor = self._send_to_prev_recv_from_prev(
                        input_grad, hidden_shape
                    )
            else:
                output_grad = self._send_to_next_recv_from_next(
                    output_tensor, hidden_shape
                )
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                # Backward through the oldest outstanding microbatch.
                it = input_tensors.pop(0)
                ot = output_tensors.pop(0)
                input_grad = self._backward_step(it, ot, output_grad)
                if last_iter:
                    input_tensor = None
                    if rank > 0:
                        self._send_to_prev(input_grad)
                else:
                    if rank > 0:
                        input_tensor = self._send_to_prev_recv_from_prev(
                            input_grad, hidden_shape
                        )
                    else:
                        input_tensor = None

        # ---- cooldown backward passes (only non-last stages) ----
        for _ in range(num_warmup):
            if rank < world - 1:
                output_grad = self._recv_from_next(hidden_shape)
            else:
                output_grad = None
            it = input_tensors.pop(0)
            ot = output_tensors.pop(0)
            input_grad = self._backward_step(it, ot, output_grad)
            if rank > 0:
                self._send_to_prev(input_grad)

        if self.is_last:
            return sum(losses) / len(losses)
        return None

    def evaluate(self, microbatches: list) -> Optional[torch.Tensor]:
        """Forward-only pass over ``microbatches`` (no backward, no grads).

        Used for validation: every stage runs its forward on each microbatch and
        forwards the hidden state to the next stage; the last stage accumulates
        the per-microbatch losses and returns their mean.  Other ranks return
        ``None``.  Must be called under ``torch.no_grad()``.
        """
        M = len(microbatches)
        rank, world = self.pp_rank, self.pp_size
        _, T = microbatches[0][0].shape
        hidden_shape = (T, microbatches[0][0].shape[0], self.stage.config.hidden_size)

        losses: list = []
        for mb in range(M):
            if rank > 0:
                input_tensor = self._recv_from_prev(hidden_shape)
            else:
                input_tensor = None
            idx, targets = microbatches[mb]
            output_tensor = self._forward_step(input_tensor, idx, targets)
            if rank < world - 1:
                self._send_to_next(output_tensor)
            if self.is_last:
                losses.append(output_tensor)
        if self.is_last:
            return sum(losses) / len(losses)
        return None
