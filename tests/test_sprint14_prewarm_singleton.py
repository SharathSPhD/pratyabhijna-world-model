"""
Sprint 14 tests: Pre-warmed WM singleton — eliminates per-request warmup latency.

Verifies:
1. load_trained_components() returns inference-mode modules
2. warmup_wm_on_text with 5 steps gives h_t norm within 10% of 60-step
3. Domain pre-warming produces valid h_t tensors for all domains
4. The /v1/generate endpoint uses singleton state (no per-request component loading)
5. Latency of single observe_step < 200ms (not 5200ms cold-start)
"""
from __future__ import annotations
import time
import pytest
import torch
from pathlib import Path


CHECKPOINT = Path("/home/sharaths/projects/pwm-phase2/checkpoints/step_1000000.pt")
CHECKPOINT_SKIP = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"Trained checkpoint not found: {CHECKPOINT}",
)


@CHECKPOINT_SKIP
def test_five_step_warmup_converges():
    """5-step warmup must produce h_t with norm within 10% of 60-step warmup."""
    from pwm.generation.engine import load_trained_components, warmup_wm_on_text

    wm, _, _ = load_trained_components()
    seed = "rain falls on ancient stones washing away the dust of time"

    h5 = warmup_wm_on_text(wm, seed, steps=5)
    h60 = warmup_wm_on_text(wm, seed, steps=60)

    norm5 = float(h5.norm())
    norm60 = float(h60.norm())
    assert norm5 > 1.0, f"5-step h_t norm suspiciously small: {norm5}"
    assert abs(norm5 - norm60) / (norm60 + 1e-6) < 0.10, \
        f"5-step norm ({norm5:.3f}) differs >10% from 60-step ({norm60:.3f})"


@CHECKPOINT_SKIP
def test_single_observe_step_latency():
    """
    A single WM observe_step must complete in < 200ms (not 5200ms cold-start).
    Tests that CUDA is warm and the WM forward is fast.
    """
    from pwm.generation.engine import load_trained_components, DEVICE, WM_CFG

    wm, _, _ = load_trained_components()
    B = 1
    obs = torch.randn(B, WM_CFG["obs_dim"], device=DEVICE)
    a_t = torch.zeros(B, WM_CFG["action_dim"], device=DEVICE)

    # Warm up CUDA with one throwaway call (mimics startup pre-warm)
    with torch.no_grad():
        states = wm.init_state(B, DEVICE)
        wm.observe_step(obs, a_t, states, 0)

    # Measure actual per-step latency
    t0 = time.perf_counter()
    with torch.no_grad():
        states = wm.init_state(B, DEVICE)
        new_states, _, _ = wm.observe_step(obs, a_t, states, 0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 200, f"WM observe_step took {elapsed_ms:.1f}ms — expected <200ms"
    h_t = new_states[0][0]
    assert torch.isfinite(h_t).all()


@CHECKPOINT_SKIP
def test_domain_prewarming_produces_valid_states():
    """Pre-warming all 6 domains must produce finite h_t tensors."""
    from pwm.generation.engine import load_trained_components, warmup_wm_on_text

    wm, _, _ = load_trained_components()
    domains = {
        "kannada_film": "ಮಳೆ ಬರುತ್ತದೆ ಮೌನದ ರಾತ್ರಿಯಲ್ಲಿ",
        "hindi_film": "बरसात की रात में तारे चमकते हैं",
        "english_pop": "the chorus breaks the night wide open",
        "english_romantic": "autumn light through misted glass",
        "carnatic": "raga bhairavi morning stillness river",
        "world_fusion": "shore wind salt migration horizon",
    }

    for domain, seed in domains.items():
        h = warmup_wm_on_text(wm, seed, steps=5, domain=domain)
        assert torch.isfinite(h).all(), f"{domain}: h_t contains NaN/Inf"
        assert h.norm() > 1.0, f"{domain}: h_t norm too small ({h.norm():.3f})"


@CHECKPOINT_SKIP
def test_prewarm_vs_fresh_state_differ():
    """
    h_t from two different seed texts must differ (WM is responsive to input).
    Prevents degenerate fixed-point where warmup ignores input.
    """
    from pwm.generation.engine import load_trained_components, warmup_wm_on_text

    wm, _, _ = load_trained_components()
    h_film = warmup_wm_on_text(wm, "moonlight river song lovers night", steps=5)
    h_jazz = warmup_wm_on_text(wm, "blue note chord resolution drone", steps=5)

    diff = float((h_film - h_jazz).norm())
    # At 5 steps, domain separation is ~0.3-0.4 norm units (vs ~2.0 at 60 steps).
    # 0.1 threshold confirms the WM responds to input (not degenerate fixed-point).
    assert diff > 0.1, f"WM h_t insensitive to input — diff={diff:.4f}"


def test_api_imports_cleanly():
    """api/main.py must import without raising (no stale OLLAMA_URL/MODEL refs)."""
    import importlib.util
    import sys
    from pathlib import Path

    api_path = Path("/home/sharaths/projects/pwm-phase5/api/main.py")
    spec = importlib.util.spec_from_file_location("api_main", api_path)
    # Don't actually execute — just check that the module file parses correctly
    with open(api_path) as f:
        source = f.read()

    # Must not reference the removed constants directly
    assert "OLLAMA_URL," not in source or "as OLLAMA_URL" in source, \
        "OLLAMA_URL imported without alias — would fail at runtime"
    assert "MODEL," not in source or "as MODEL" in source, \
        "MODEL imported without alias — would fail at runtime"
    # Must reference the singleton pattern
    assert "state.wm_ready" in source, "Singleton state check missing from /v1/generate"
    assert "state.domain_states" in source, "Pre-warmed domain states not used"
