"""
Tests for TrikaWorldModel (pwm.world_model.trika).

Philosophical grounding:
  Trika ('triad') — three-level Aparā/Parāparā/Parā hierarchy (TĀ, Abhinavagupta).
  Top-down h conditioning: pure awareness (Parā Śakti) pervades lower-level dynamics.

All tests run on CPU only.
"""

from __future__ import annotations

import pytest
import torch

from pwm.world_model.trika import TrikaWorldModel  # type: ignore[import]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

OBS_DIM = 32
ACTION_DIM = 16
HIDDEN_DIM = 64
STOCH_DIM = 4
STOCH_CLASSES = 4
BATCH = 2
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(n_levels: int) -> TrikaWorldModel:
    return TrikaWorldModel(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        n_levels=n_levels,
        hidden_dim=HIDDEN_DIM,
        stoch_dim=STOCH_DIM,
        stoch_classes=STOCH_CLASSES,
        free_bits=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_level_forward() -> None:
    """n_levels=1: observe_step produces correct h and z shapes for Aparā level."""
    model = _make_model(n_levels=1)
    states = model.init_state(BATCH, DEVICE)
    obs = torch.randn(BATCH, OBS_DIM)
    action = torch.randn(BATCH, ACTION_DIM)

    new_states, logits_post, logits_prior = model.observe_step(obs, action, states, step=0)

    assert len(new_states) == 1, f"Expected 1 state tuple, got {len(new_states)}"
    h, z = new_states[0]
    assert h.shape == (BATCH, HIDDEN_DIM), f"h shape {h.shape}"
    assert z.shape == (BATCH, STOCH_DIM, STOCH_CLASSES), f"z shape {z.shape}"

    assert len(logits_post) == 1
    assert len(logits_prior) == 1
    assert logits_post[0].shape == (BATCH, STOCH_DIM, STOCH_CLASSES)
    assert logits_prior[0].shape == (BATCH, STOCH_DIM, STOCH_CLASSES)


def test_two_level_forward() -> None:
    """n_levels=2: at step=0 both levels are active; at step=1 only level 0 is active."""
    model = _make_model(n_levels=2)
    obs = torch.randn(BATCH, OBS_DIM)
    action = torch.randn(BATCH, ACTION_DIM)

    # --- step=0: both levels (stride 1 and 4 both divide 0) ---
    states = model.init_state(BATCH, DEVICE)
    new_states_0, logits_post_0, logits_prior_0 = model.observe_step(
        obs, action, states, step=0
    )

    assert len(new_states_0) == 2
    h0, z0 = new_states_0[0]
    h1, z1 = new_states_0[1]
    assert h0.shape == (BATCH, HIDDEN_DIM), f"Level-0 h shape {h0.shape}"
    assert h1.shape == (BATCH, HIDDEN_DIM), f"Level-1 h shape {h1.shape}"

    # Both logits_post entries should be proper tensors (not the scalar stub)
    assert logits_post_0[0].shape == (BATCH, STOCH_DIM, STOCH_CLASSES)
    assert logits_post_0[1].shape == (BATCH, STOCH_DIM, STOCH_CLASSES)

    # --- step=1: only level 0 active (1 % 1 == 0; 1 % 4 != 0) ---
    new_states_1, logits_post_1, logits_prior_1 = model.observe_step(
        obs, action, new_states_0, step=1
    )

    assert len(new_states_1) == 2
    h0_new, z0_new = new_states_1[0]
    h1_unchanged, z1_unchanged = new_states_1[1]

    # Level-0 state should have changed
    assert not torch.allclose(h0_new, h0), "Level-0 h should have updated at step=1"

    # Level-1 stub logits should be the scalar placeholder (shape == (1,))
    assert logits_post_1[1].shape == (1,), (
        f"Level-1 should return scalar stub at step=1, got {logits_post_1[1].shape}"
    )


def test_world_model_loss_single_level() -> None:
    """world_model_loss with n_levels=1 produces 'apara_total' key and finite total."""
    model = _make_model(n_levels=1)
    B, T = 2, 4
    obs_seq = torch.randn(B, T, OBS_DIM)
    action_seq = torch.randn(B, T, ACTION_DIM)
    reward_seq = torch.randn(B, T)
    done_seq = torch.zeros(B, T)

    init_states = model.init_state(B, DEVICE)
    losses = model.world_model_loss(
        obs_seq, action_seq, reward_seq, done_seq, init_states
    )

    assert "apara_total" in losses, f"Missing 'apara_total'. Keys: {list(losses.keys())}"
    assert "total" in losses, f"Missing 'total'. Keys: {list(losses.keys())}"

    total = losses["total"]
    assert isinstance(total, torch.Tensor)
    assert total.shape == ()
    assert torch.isfinite(total), f"total loss is not finite: {total.item()}"
