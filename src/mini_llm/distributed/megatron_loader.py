"""Load a full ``MiniGPT`` state dict into a Megatron TP / PP model.

The Megatron variants (``MegatronMiniGPT`` for tensor parallelism, and
``PipelineStage`` for pipeline parallelism) hold *shards* of the same weights
that a plain ``MiniGPT`` holds.  To keep every strategy on the *same* starting
weights (so benchmark losses are comparable), we build a full ``MiniGPT`` on
rank 0, broadcast its state dict to every rank, and then each rank copies the
slice it owns into its local Megatron model.

The broadcast is done key-by-key with an explicit shape exchange (the same
trick used by the equivalence tests) so that non-zero ranks know the tensor
shape before allocating the receive buffer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from ..model import GPTConfig, MiniGPT


def _all_state_keys(cfg: GPTConfig) -> list[str]:
    """Return every MiniGPT state-dict key for the given config."""
    keys = ["pos_emb", "tok_emb.weight", "ln_f.weight", "lm_head.weight"]
    for i in range(cfg.num_layers):
        keys += [
            f"blocks.{i}.ln1.weight",
            f"blocks.{i}.attn.qkv.weight",
            f"blocks.{i}.attn.proj.weight",
            f"blocks.{i}.ln2.weight",
            f"blocks.{i}.mlp.fc1.weight",
            f"blocks.{i}.mlp.fc2.weight",
        ]
    return keys


def broadcast_minigpt_state_dict(
    cfg: GPTConfig,
    device: torch.device,
    seed: int = 0,
    src: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a full ``MiniGPT`` on ``src`` and broadcast its state dict.

    Every rank receives an identical dict of full-size tensors (on ``device``).
    The reference model is seeded deterministically on the source rank so all
    strategies start from the same weights.
    """
    keys = _all_state_keys(cfg)
    full_sd: dict[str, torch.Tensor] = {}

    src_sd: Optional[dict] = None
    if dist.get_rank() == src:
        torch.manual_seed(seed)
        ref = MiniGPT(cfg).to(device)
        src_sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    for k in keys:
        if dist.get_rank() == src:
            s = list(src_sd[k].shape)  # type: ignore[union-attr]
            shp = torch.tensor(s + [0] * (4 - len(s)), dtype=torch.long, device=device)
        else:
            shp = torch.zeros(4, dtype=torch.long, device=device)
        dist.broadcast(shp, src=src)
        shape = tuple(int(x) for x in shp.tolist() if int(x) > 0)
        if dist.get_rank() == src:
            t = src_sd[k]  # type: ignore[index]
        else:
            t = torch.empty(shape, device=device)
        dist.broadcast(t, src=src)
        full_sd[k] = t

    return full_sd


def load_tp_weights(
    model,
    full_sd: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> None:
    """Copy the shards of ``full_sd`` owned by ``tp_rank`` into a MegatronMiniGPT.

    Shard dimensions (matching the TP layer layout):
        - tok_emb / lm_head / q_lin / k_lin / v_lin / fc1 : rows (dim 0)
        - proj / fc2                                      : columns (dim 1)
        - pos_emb / ln1 / ln2 / ln_f                      : replicated
    """
    cfg = model.config
    C = cfg.hidden_size
    V = cfg.vocab_size
    Vp = V // tp_size
    per = C // tp_size
    inter = cfg.intermediate_size or 4 * C
    inter_p = inter // tp_size

    with torch.no_grad():
        model.tok_emb.weight.copy_(full_sd["tok_emb.weight"][tp_rank * Vp:(tp_rank + 1) * Vp])
        model.lm_head.weight.copy_(full_sd["lm_head.weight"][tp_rank * Vp:(tp_rank + 1) * Vp])
        model.pos_emb.copy_(full_sd["pos_emb"])
        for i in range(cfg.num_layers):
            b = model.blocks[i]
            b.ln1.weight.copy_(full_sd[f"blocks.{i}.ln1.weight"])
            b.ln2.weight.copy_(full_sd[f"blocks.{i}.ln2.weight"])
            qkv = full_sd[f"blocks.{i}.attn.qkv.weight"]
            qw, kw, vw = qkv[:C], qkv[C:2 * C], qkv[2 * C:]
            b.attn.q_lin.weight.copy_(qw[tp_rank * per:(tp_rank + 1) * per])
            b.attn.k_lin.weight.copy_(kw[tp_rank * per:(tp_rank + 1) * per])
            b.attn.v_lin.weight.copy_(vw[tp_rank * per:(tp_rank + 1) * per])
            b.attn.proj.weight.copy_(
                full_sd[f"blocks.{i}.attn.proj.weight"][:, tp_rank * per:(tp_rank + 1) * per]
            )
            b.mlp.fc1.weight.copy_(
                full_sd[f"blocks.{i}.mlp.fc1.weight"][tp_rank * inter_p:(tp_rank + 1) * inter_p]
            )
            b.mlp.fc2.weight.copy_(
                full_sd[f"blocks.{i}.mlp.fc2.weight"][:, tp_rank * inter_p:(tp_rank + 1) * inter_p]
            )
        model.ln_f.weight.copy_(full_sd["ln_f.weight"])


def load_pp_weights(
    stage,
    full_sd: dict[str, torch.Tensor],
) -> None:
    """Copy the layer slice owned by a ``PipelineStage`` from ``full_sd``.

    Each stage owns ``blocks[layer_start:layer_end]``; the first stage also owns
    the embedding and the last stage owns ``ln_f`` + ``lm_head``.  The fused
    MiniGPT ``qkv.weight`` is split into the stage's q/k/v projections.
    """
    cfg = stage.config
    C = cfg.hidden_size
    with torch.no_grad():
        if stage.is_first:
            stage.tok_emb.weight.copy_(full_sd["tok_emb.weight"])
            stage.pos_emb.copy_(full_sd["pos_emb"])
        for j in range(len(stage.blocks)):
            gi = stage.layer_start + j  # global layer index
            b = stage.blocks[j]
            b.ln1.weight.copy_(full_sd[f"blocks.{gi}.ln1.weight"])
            b.ln2.weight.copy_(full_sd[f"blocks.{gi}.ln2.weight"])
            qkv = full_sd[f"blocks.{gi}.attn.qkv.weight"]
            b.attn.q_lin.weight.copy_(qkv[:C])
            b.attn.k_lin.weight.copy_(qkv[C:2 * C])
            b.attn.v_lin.weight.copy_(qkv[2 * C:])
            b.attn.proj.weight.copy_(full_sd[f"blocks.{gi}.attn.proj.weight"])
            b.mlp.fc1.weight.copy_(full_sd[f"blocks.{gi}.mlp.fc1.weight"])
            b.mlp.fc2.weight.copy_(full_sd[f"blocks.{gi}.mlp.fc2.weight"])
        if stage.is_last:
            stage.ln_f.weight.copy_(full_sd["ln_f.weight"])
            stage.lm_head.weight.copy_(full_sd["lm_head.weight"])
