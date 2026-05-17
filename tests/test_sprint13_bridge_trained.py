"""
Sprint 13 tests: VimarsaBridgeV2 with trained checkpoint.

Verifies:
  - checkpoint loads without error
  - KL-div between biased and unbiased distributions > 0.05
  - weights are non-zero (real training happened, not random init)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
from pwm.generation.engine import VIMARSA_BRIDGE_CHECKPOINT

CKPT = VIMARSA_BRIDGE_CHECKPOINT


def test_checkpoint_exists():
    assert CKPT.exists(), f"trained bridge checkpoint missing: {CKPT}"


def test_checkpoint_loads():
    bridge = VimarsaBridgeV2.load_or_init(
        hidden_dim=512, vocab_size=128256, ckpt_path=CKPT
    )
    assert bridge is not None
    assert bridge.hidden_dim == 512
    assert bridge.vocab_size == 128256


def test_weights_not_zero():
    """Trained weights must differ meaningfully from zero init."""
    bridge = VimarsaBridgeV2.load_or_init(
        hidden_dim=512, vocab_size=128256, ckpt_path=CKPT
    )
    w = bridge.proj.weight.detach()
    assert torch.any(w != 0), "all weights zero — bridge not trained"
    assert w.abs().mean().item() > 1e-4, (
        f"weights too small ({w.abs().mean().item():.2e}) — likely untrained"
    )


def test_kl_div_above_threshold():
    bridge = VimarsaBridgeV2.load_or_init(
        hidden_dim=512, vocab_size=128256, ckpt_path=CKPT
    )
    torch.manual_seed(0)
    h = torch.randn(1, 512)
    proc = bridge.as_logits_processor(h)

    rng = np.random.default_rng(0)
    base = rng.standard_normal(128256).astype(np.float32)
    biased = proc([0], base.copy())

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    p = softmax(base) + 1e-10
    q = softmax(biased) + 1e-10
    kl = float(np.sum(p * np.log(p / q)))
    assert kl > 0.05, f"KL-div too low: {kl:.5f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
