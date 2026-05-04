"""
Tests for CittaStore (pwm.memory.citta_store).

Philosophical grounding:
  Citta: dual-mode Hopfield associative memory.
  Smṛti (YS 1.11): episodic sharp recall (high β).
  Ālayavijñāna (Vasubandhu Triṃśikā 5): semantic blended recall (low β).

All tests run on CPU.
"""

from __future__ import annotations

import pytest
import torch

from pwm.memory.citta_store import CittaStore  # type: ignore[import]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 64
BATCH = 4
N_PATTERNS = 10
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> CittaStore:
    return CittaStore(
        hidden_dim=DIM,
        n_levels=1,
        beta_episodic=4.0,
        beta_semantic=0.25,
        max_episodic=512,
        max_semantic=256,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_store_and_recall_episodic(store: CittaStore) -> None:
    """Episodic store/recall: output shape is (B, dim) after storing patterns."""
    # Store N patterns (each batch element stored individually inside store_episode)
    for _ in range(N_PATTERNS):
        h = torch.randn(BATCH, DIM)
        store.store_episode(h, level=0)

    query = torch.randn(BATCH, DIM)
    recalled = store.recall(query, level=0, mode="episodic")

    assert recalled.shape == (BATCH, DIM), f"Episodic recall shape {recalled.shape}"
    assert torch.isfinite(recalled).all(), "Non-finite values in episodic recall"


def test_store_and_recall_semantic(store: CittaStore) -> None:
    """Semantic store/recall: output shape is (B, dim) after storing patterns."""
    for _ in range(N_PATTERNS):
        h = torch.randn(BATCH, DIM)
        store.store_semantic(h, level=0)

    query = torch.randn(BATCH, DIM)
    recalled = store.recall(query, level=0, mode="semantic")

    assert recalled.shape == (BATCH, DIM), f"Semantic recall shape {recalled.shape}"
    assert torch.isfinite(recalled).all(), "Non-finite values in semantic recall"


def test_hopfield_entropy_decreases(store: CittaStore) -> None:
    """Storing many identical patterns drives entropy toward 0 (concentrated memory)."""
    # Before storing: no patterns → get_entropy returns 0.0 baseline
    entropy_before = store.hopfield_entropy(level=0, mode="episodic")

    # Store many identical patterns
    fixed_pattern = torch.ones(1, DIM)
    for _ in range(50):
        store.store_episode(fixed_pattern, level=0)

    entropy_after = store.hopfield_entropy(level=0, mode="episodic")

    # Entropy should be lower (or at most equal) after identical-pattern concentration
    assert entropy_after <= entropy_before + 1e-6, (
        f"Entropy should not increase after identical-pattern storage: "
        f"before={entropy_before:.4f}, after={entropy_after:.4f}"
    )
    # Entropy should be very low (near 0) since all patterns are identical
    assert entropy_after < 0.1, (
        f"Entropy should be near 0 with identical patterns, got {entropy_after:.4f}"
    )


def test_capacities(store: CittaStore) -> None:
    """capacities() returns correct keys for all configured levels."""
    caps = store.capacities()

    # n_levels=1 → only 'apara' prefix
    assert "apara_episodic" in caps, f"Missing 'apara_episodic'. Keys: {list(caps.keys())}"
    assert "apara_semantic" in caps, f"Missing 'apara_semantic'. Keys: {list(caps.keys())}"

    # No patterns stored yet → capacity = 0
    assert caps["apara_episodic"] == 0
    assert caps["apara_semantic"] == 0

    # Store one batch of episodes and verify count increases
    h = torch.randn(BATCH, DIM)
    store.store_episode(h, level=0)
    caps_after = store.capacities()
    assert caps_after["apara_episodic"] == BATCH, (
        f"Expected {BATCH} stored episodic patterns, got {caps_after['apara_episodic']}"
    )
