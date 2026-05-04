"""
Tests for VFE loss utilities (pwm.world_model.losses).

Philosophical grounding:
  Spanda (SpandaK 1.1, Vasugupta): stochastic latent dynamics shaped by these losses.
  The KL free-bits floor preserves genuine spanda — non-collapsed posterior.
"""

from __future__ import annotations

import pytest
import torch

from pwm.world_model.losses import (  # type: ignore[import]
    kl_categorical_free_bits,
    make_twohot_bins,
    symexp,
    symlog,
    twohot_encode,
    twohot_loss,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_symlog_symexp_inverse() -> None:
    """symexp(symlog(x)) ≈ x across wide value ranges."""
    test_values = [
        torch.tensor([-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]),
        torch.randn(50) * 500,   # large random values
        torch.randn(50) * 0.01,  # small random values near zero
    ]
    for x in test_values:
        reconstructed = symexp(symlog(x))
        assert torch.allclose(reconstructed, x, atol=1e-4, rtol=1e-4), (
            f"symexp(symlog(x)) != x. Max deviation: {(reconstructed - x).abs().max().item():.6f}"
        )


def test_twohot_encode_sums_to_one() -> None:
    """Two-hot encoding of any scalar value sums to 1.0 (valid probability distribution)."""
    bins = make_twohot_bins(n_bins=255)

    # Test at a variety of target values within and near the bin range
    for val in [-15.0, -5.0, 0.0, 3.7, 5.0, 15.0]:
        targets = torch.tensor([val, val * 0.5])
        encoded = twohot_encode(targets, bins)
        sums = encoded.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Two-hot encoding for val={val} sums to {sums.tolist()}, expected 1.0"
        )


def test_kl_free_bits_clamp() -> None:
    """Identical post and prior distributions yield KL at the free_bits floor."""
    B, stoch_dim, stoch_classes = 8, 32, 32
    logits = torch.zeros(B, stoch_dim, stoch_classes)

    free_bits = 1.0
    kl = kl_categorical_free_bits(logits, logits, free_bits=free_bits)

    # Analytically KL(uniform ‖ uniform) = 0, free bits clamps to free_bits
    assert abs(kl.item() - free_bits) < 1e-5, (
        f"Expected KL = free_bits = {free_bits}, got {kl.item()}"
    )

    # With free_bits=0, identical distributions should give KL ≈ 0
    kl_zero_fb = kl_categorical_free_bits(logits, logits, free_bits=0.0)
    assert kl_zero_fb.item() < 1e-5, (
        f"With free_bits=0, identical dist KL should be ~0, got {kl_zero_fb.item()}"
    )


def test_twohot_loss_shape() -> None:
    """twohot_loss returns a scalar tensor."""
    n_bins = 255
    B, T = 4, 8
    bins = make_twohot_bins(n_bins)

    # Simulate distributional head outputs and targets
    pred_logits = torch.randn(B, T, n_bins)
    target_scalars = torch.randn(B, T)
    target_hot = twohot_encode(target_scalars, bins)

    loss = twohot_loss(pred_logits, target_hot)

    assert loss.shape == (), f"twohot_loss must be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"twohot_loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"twohot_loss should be non-negative, got {loss.item()}"
