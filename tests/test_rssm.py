"""
Tests for TrikaCoreLevel (pwm.world_model.rssm).

Philosophical grounding:
  Pratyabhijñā (ĪPK 1.3–1.4, Utpaladeva): recognition density q_φ(z_t|h_t,o_t).
  Spanda (SpandaK 1.1, Vasugupta): stochastic latent z_t ~ Cat(32×32).

All tests run on CPU (GRU fallback path). No GPU required.
"""

from __future__ import annotations

import pytest
import torch

from pwm.world_model.rssm import TrikaCoreLevel, straight_through_sample  # type: ignore[import]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 64
HIDDEN_DIM = 128
ACTION_DIM = 32
STOCH_DIM = 8
STOCH_CLASSES = 8
BATCH = 4
DEVICE = torch.device("cpu")


@pytest.fixture()
def level() -> TrikaCoreLevel:
    return TrikaCoreLevel(
        level=0,
        obs_dim=OBS_DIM,
        stoch_dim=STOCH_DIM,
        stoch_classes=STOCH_CLASSES,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        backbone="gru",
        free_bits=1.0,
    )


def _init(model: TrikaCoreLevel, B: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    return model.init_state(B, DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_observe_shapes(level: TrikaCoreLevel) -> None:
    """observe() returns tensors of correct shapes (pratyabhijñā recognition step)."""
    h, z = _init(level)
    obs = torch.randn(BATCH, OBS_DIM)
    action = torch.randn(BATCH, ACTION_DIM)

    h_t, z_t, logits_post, logits_prior = level.observe(obs, h, z, action)

    assert h_t.shape == (BATCH, HIDDEN_DIM), f"h_t shape {h_t.shape}"
    assert z_t.shape == (BATCH, STOCH_DIM, STOCH_CLASSES), f"z_t shape {z_t.shape}"
    assert logits_post.shape == (BATCH, STOCH_DIM, STOCH_CLASSES), f"logits_post shape {logits_post.shape}"
    assert logits_prior.shape == (BATCH, STOCH_DIM, STOCH_CLASSES), f"logits_prior shape {logits_prior.shape}"


def test_imagine_shapes(level: TrikaCoreLevel) -> None:
    """imagine() returns tensors of correct shapes (pure prior, no encoder)."""
    h, z = _init(level)
    action = torch.randn(BATCH, ACTION_DIM)

    h_t, z_t, logits_prior = level.imagine(h, z, action)

    assert h_t.shape == (BATCH, HIDDEN_DIM), f"h_t shape {h_t.shape}"
    assert z_t.shape == (BATCH, STOCH_DIM, STOCH_CLASSES), f"z_t shape {z_t.shape}"
    assert logits_prior.shape == (BATCH, STOCH_DIM, STOCH_CLASSES), f"logits_prior shape {logits_prior.shape}"


def test_world_model_loss_runs(level: TrikaCoreLevel) -> None:
    """world_model_loss() runs on a (B, T, D) sequence and returns all expected keys."""
    B, T = 2, 8
    obs_seq = torch.randn(B, T, OBS_DIM)
    action_seq = torch.randn(B, T, ACTION_DIM)
    reward_seq = torch.randn(B, T)
    done_seq = torch.zeros(B, T)

    h, z = level.init_state(B, DEVICE)
    losses = level.world_model_loss(obs_seq, action_seq, reward_seq, done_seq, h, z)

    expected_keys = {"total", "obs", "reward", "continue", "dyn", "rep", "vfe"}
    assert expected_keys <= losses.keys(), f"Missing keys: {expected_keys - losses.keys()}"

    total = losses["total"]
    assert isinstance(total, torch.Tensor), "total loss must be a Tensor"
    assert total.shape == (), f"total loss must be scalar, got {total.shape}"
    assert torch.isfinite(total), f"total loss is not finite: {total.item()}"


def test_init_state_zeros(level: TrikaCoreLevel) -> None:
    """init_state() returns zero tensors of correct shape."""
    h, z = level.init_state(BATCH, DEVICE)

    assert h.shape == (BATCH, HIDDEN_DIM)
    assert z.shape == (BATCH, STOCH_DIM, STOCH_CLASSES)
    assert h.abs().max().item() == 0.0, "h should be all-zeros"
    assert z.abs().max().item() == 0.0, "z should be all-zeros"
    assert h.device == DEVICE
    assert z.device == DEVICE


def test_kl_free_bits(level: TrikaCoreLevel) -> None:
    """KL loss with identical post and prior should be at free_bits floor, not zero."""
    from pwm.world_model.losses import kl_categorical_free_bits  # type: ignore[import]

    B = 4
    logits = torch.zeros(B, STOCH_DIM, STOCH_CLASSES)

    # Identical distributions → KL = 0 analytically, but free_bits clamps it up
    kl = kl_categorical_free_bits(logits, logits, free_bits=1.0)

    # Should be at the free_bits floor (1.0 nats), not zero
    assert kl.item() >= 1.0 - 1e-5, f"KL should be >= free_bits=1.0, got {kl.item()}"
    # Should not be astronomically large
    assert kl.item() < 10.0, f"KL unexpectedly large: {kl.item()}"


def test_straight_through_gradient() -> None:
    """Gradients flow through straight_through_sample via the soft path."""
    logits = torch.randn(BATCH, STOCH_DIM, STOCH_CLASSES, requires_grad=True)
    sample = straight_through_sample(logits)

    # Loss is sum of sample — gradient should propagate back to logits
    loss = sample.sum()
    loss.backward()

    assert logits.grad is not None, "No gradient flowed back through straight_through_sample"
    assert logits.grad.abs().max().item() > 0.0, "Gradient is all-zeros"
    assert torch.isfinite(logits.grad).all(), "Non-finite gradient"
