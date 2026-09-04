"""Integration tests for the single-GPU trainer (CPU-safe)."""

import torch

from mini_llm.data import build_dataloaders
from mini_llm.model import GPTConfig
from mini_llm.trainer import Trainer, TrainerConfig


def make_tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=500,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        seq_len=16,
        intermediate_size=64,
    )


def test_single_trainer_loss_decreases():
    """Training on synthetic data should reduce loss over a few steps."""
    cfg = make_tiny_config()
    trainer_cfg = TrainerConfig(
        strategy="single",
        model=cfg,
        # A modest LR. 1e-2 is too large for this tiny model and makes the loss
        # oscillate; 5e-3 gives a clear, stable decrease.
        lr=5e-3,
        mixed_precision="none",
        seed=0,
    )
    trainer = Trainer(trainer_cfg)

    train_loader, _, _ = build_dataloaders(
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        micro_batch_size=4,
        num_train_samples=256,
        num_val_samples=16,
        num_workers=0,
        seed=0,
    )

    losses = []
    for i, batch in enumerate(train_loader):
        if i >= 30:
            break
        m = trainer.train_step(batch)
        losses.append(m["loss"])

    assert len(losses) >= 20
    # The per-step loss is noisy, so comparing the first and last points is
    # unreliable. Instead assert that training produced a clear decrease: the
    # minimum loss must drop well below the initial value.
    assert min(losses) < losses[0] - 0.02, f"loss did not decrease: {losses}"


def test_single_trainer_benchmark():
    """benchmark() should return sane metrics on CPU."""
    cfg = make_tiny_config()
    trainer_cfg = TrainerConfig(
        strategy="single",
        model=cfg,
        lr=1e-3,
        mixed_precision="none",
        seed=0,
    )
    trainer = Trainer(trainer_cfg)

    train_loader, _, _ = build_dataloaders(
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        micro_batch_size=4,
        num_train_samples=128,
        num_val_samples=16,
        num_workers=0,
        seed=0,
    )

    metrics = trainer.benchmark(
        train_loader, warmup_steps=2, measure_steps=5
    )
    assert metrics["step_ms"] > 0
    assert metrics["tokens_per_s"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_model_forward_backward_consistency():
    """A single train_step should not raise and should update params."""
    cfg = make_tiny_config()
    trainer_cfg = TrainerConfig(
        strategy="single",
        model=cfg,
        lr=1e-3,
        mixed_precision="none",
        seed=0,
    )
    trainer = Trainer(trainer_cfg)

    # Snapshot a parameter before.
    before = trainer.model.lm_head.weight.detach().clone()

    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, cfg.seq_len)),
        "target_ids": torch.randint(0, cfg.vocab_size, (2, cfg.seq_len)),
    }
    m = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(m["loss"]))

    after = trainer.model.lm_head.weight.detach()
    # Weight should have changed after an optimizer step.
    assert not torch.equal(before, after)
