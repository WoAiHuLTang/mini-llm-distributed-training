"""Unit tests for the MiniGPT model (CPU, no GPU required)."""

import torch

from mini_llm.model import GPTConfig, MiniGPT


def make_small_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        seq_len=32,
        intermediate_size=128,
    )


def test_forward_shape():
    cfg = make_small_config()
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_loss_backward():
    cfg = make_small_config()
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = model(x, targets=targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    # Every trainable parameter should receive a gradient.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"param {name} has no gradient"
        assert torch.isfinite(p.grad).all()


def test_causal_attention_no_future_leak():
    """Token at position t must not depend on tokens after t."""
    cfg = make_small_config()
    model = MiniGPT(cfg)
    model.eval()
    with torch.no_grad():
        # Two sequences identical up to position 5, differing after.
        x1 = torch.randint(0, cfg.vocab_size, (1, 16))
        x2 = x1.clone()
        x2[:, 6:] = torch.randint(0, cfg.vocab_size, (1, 10))
        out1 = model(x1)
        out2 = model(x2)
        # Outputs at positions <= 5 must match (causal).
        assert torch.allclose(out1[:, :6], out2[:, :6], atol=1e-6)


def test_weight_tying():
    cfg = make_small_config()
    model = MiniGPT(cfg)
    assert model.tok_emb.weight is model.lm_head.weight


def test_num_parameters_positive():
    cfg = make_small_config()
    model = MiniGPT(cfg)
    assert model.num_parameters() > 0


def test_seq_len_guard():
    cfg = make_small_config()
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len + 1))
    try:
        model(x)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError for over-length sequence"


def test_deterministic_forward():
    cfg = make_small_config()
    model = MiniGPT(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)
