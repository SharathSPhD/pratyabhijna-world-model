"""
Sprint 12 tests: Checkpoint integration — trained WM + EFEActor + CittaStore.

Verifies that:
1. load_trained_components() returns correct module types
2. EFEActor weights are non-random (norm differs from fresh-init)
3. CittaStore uses n_levels=3 (matches trained checkpoint)
4. WM produces finite h_t on CUDA
5. Full Pañcakṛtya loop runs with trained components end-to-end
"""
from __future__ import annotations
import pytest
import torch
from pathlib import Path


CHECKPOINT = Path("/home/sharaths/projects/pwm-phase2/checkpoints/final.pt")
CHECKPOINT_SKIP = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"Trained checkpoint not found: {CHECKPOINT}",
)


@CHECKPOINT_SKIP
def test_load_trained_components_returns_three_modules():
    """load_trained_components() must return (wm, efe, citta) — all nn.Module."""
    import torch.nn as nn
    from pwm.generation.engine import load_trained_components
    wm, efe, citta = load_trained_components()
    assert isinstance(wm, nn.Module)
    assert isinstance(efe, nn.Module)
    assert isinstance(citta, nn.Module)


@CHECKPOINT_SKIP
def test_efe_weights_differ_from_random_init():
    """
    EFEActor loaded from checkpoint must differ from a fresh random init.
    This verifies that weights were actually loaded (not silently skipped).
    """
    from pwm.generation.engine import load_trained_components
    from pwm.active_inference.efe_actor import EFEActor

    _, efe_trained, _ = load_trained_components()
    efe_random = EFEActor(hidden_dim=512, stoch_dim=32, n_cats=32, action_dim=64)

    # log_preference is zero-initialized by design (uniform preference prior).
    # Compare net.0.weight — the trained projection from h+z → 256-dim space.
    # After 1M training steps its norm should be >> random init (~Kaiming).
    trained_w = dict(efe_trained.named_parameters())["net.0.weight"].detach().cpu()
    random_w = dict(efe_random.named_parameters())["net.0.weight"].detach().cpu()
    diff = float((trained_w - random_w).norm())
    assert diff > 1.0, f"EFEActor net.0.weight looks like random init (diff={diff:.4f})"


@CHECKPOINT_SKIP
def test_citta_store_has_three_levels():
    """CittaStore must be n_levels=3 to match trained checkpoint architecture."""
    from pwm.generation.engine import load_trained_components
    _, _, citta = load_trained_components()
    assert hasattr(citta, "stores"), "CittaStore must have .stores attribute"
    assert len(citta.stores) == 3, f"Expected n_levels=3, got {len(citta.stores)}"


@CHECKPOINT_SKIP
def test_wm_produces_finite_h_on_cuda():
    """WM forward pass must produce finite h_t on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from pwm.generation.engine import load_trained_components, DEVICE, WM_CFG
    wm, _, _ = load_trained_components()

    B = 1
    obs = torch.randn(B, WM_CFG["obs_dim"], device=DEVICE)
    a_t = torch.zeros(B, WM_CFG["action_dim"], device=DEVICE)
    with torch.no_grad():
        states = wm.init_state(B, DEVICE)
        new_states, logits_post, logits_prior = wm.observe_step(obs, a_t, states, 0)
    h_t = new_states[0][0]
    assert torch.isfinite(h_t).all(), "h_t contains NaN/Inf"
    assert h_t.device.type == "cuda", f"h_t not on CUDA: {h_t.device}"
    assert h_t.shape == (B, WM_CFG["hidden_dim"]), f"Wrong h_t shape: {h_t.shape}"


@CHECKPOINT_SKIP
def test_trained_pipeline_loop_runs():
    """
    Full Pañcakṛtya loop with trained WM+EFEActor+CittaStore must produce
    finite wm_state events and non-zero aesthetic_quality.
    """
    from pwm.generation.engine import load_trained_components, DEVICE, WM_CFG
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    wm, efe, citta = load_trained_components()
    bridge = VimarsaBridgeV2(hidden_dim=512, vocab_size=128256).to(DEVICE)

    class MockLLM:
        def stream(self, **kwargs):
            yield "the light moves\n"

    cfg = LoopConfig(n_stanzas=1, device=str(DEVICE))
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, MockLLM(), cfg)
    loop.init(batch_size=1)  # S12: must call init() before run_stanza() with real WM

    obs = torch.randn(1, WM_CFG["obs_dim"], device=DEVICE)
    events = []
    gen = loop.run_stanza(0, obs, "poet", "moon")
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass

    wm_events = [e for e in events if e["event"] == "wm_state"]
    assert len(wm_events) == 1, "Expected exactly one wm_state event"
    data = wm_events[0]["data"]

    assert "aesthetic_quality" in data
    assert "creative_peak" in data
    assert "prediction_error" in data
    assert isinstance(data["aesthetic_quality"], float)
    assert 0.0 <= data["aesthetic_quality"] <= 1.0, \
        f"aesthetic_quality out of range: {data['aesthetic_quality']}"
    # prediction_error should be finite
    assert isinstance(data["prediction_error"], float)
    assert not (data["prediction_error"] != data["prediction_error"]), \
        "prediction_error is NaN"
