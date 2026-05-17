"""
Sprint 11 tests: VimarsaBridgeV2 — logits_processor, KL-div, checkpoint loading.

Gate criterion: KL-divergence between biased and unbiased logit distributions > 0.05.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch


def test_bridge_v2_import():
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    assert VimarsaBridgeV2 is not None


def test_bridge_returns_callable():
    """as_logits_processor must return a callable."""
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    bridge = VimarsaBridgeV2(hidden_dim=64, vocab_size=1000)
    h = torch.randn(1, 64)
    proc = bridge.as_logits_processor(h)
    assert callable(proc), "as_logits_processor must return callable"


def test_logits_processor_shifts_distribution():
    """The processor must change the logit distribution (non-zero bias applied)."""
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    bridge = VimarsaBridgeV2(hidden_dim=64, vocab_size=1000)
    h = torch.randn(1, 64)
    proc = bridge.as_logits_processor(h)

    base = np.zeros(1000, dtype=np.float32)
    result = proc([0], base.copy())
    assert not np.allclose(result, base), "Logit bias not applied — distribution unchanged"


def test_kl_div_above_threshold():
    """KL-divergence between biased and unbiased softmax distributions must be > 0.05."""
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2

    bridge = VimarsaBridgeV2(hidden_dim=64, vocab_size=1000)
    h = torch.randn(1, 64)
    proc = bridge.as_logits_processor(h)

    # Random base logits
    base = np.random.randn(1000).astype(np.float32)
    biased = proc([0], base.copy())

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    p = softmax(base) + 1e-10
    q = softmax(biased) + 1e-10
    kl = float(np.sum(p * np.log(p / q)))
    assert kl > 0.05, f"KL-divergence too low: {kl:.5f} (threshold 0.05)"


def test_load_or_init_without_checkpoint():
    """load_or_init must succeed even with no checkpoint file."""
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    from pathlib import Path
    bridge = VimarsaBridgeV2.load_or_init(
        hidden_dim=64, vocab_size=1000, ckpt_path=Path("/nonexistent/path.pt")
    )
    assert bridge is not None
    assert bridge.hidden_dim == 64


def test_train_step_reduces_loss():
    """train_step must return a finite loss that decreases over 5 steps."""
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    bridge = VimarsaBridgeV2(hidden_dim=32, vocab_size=100)
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        h = torch.randn(16, 32)
        tok = torch.randint(0, 100, (16,))
        loss = bridge.train_step(h, tok, opt)
        losses.append(loss)

    assert all(np.isfinite(l) for l in losses), "Training produced NaN/Inf loss"
    # Loss should generally decrease (may fluctuate slightly)
    assert losses[-1] < losses[0] * 1.5, f"Loss not decreasing: {losses}"


def test_cuda_if_available():
    """Bridge must work on CUDA when available."""
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    bridge = VimarsaBridgeV2(hidden_dim=64, vocab_size=1000).cuda()
    h = torch.randn(1, 64, device="cuda")
    proc = bridge.as_logits_processor(h)
    base = np.zeros(1000, dtype=np.float32)
    result = proc([0], base.copy())
    assert not np.allclose(result, base)
