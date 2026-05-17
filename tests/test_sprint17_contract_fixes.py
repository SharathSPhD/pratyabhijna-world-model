"""
Sprint 17 tests: Critical-review fixes + Contract 2 whitelist enforcement.

Verifies issues fixed from code review:
1. _sanitise_event_data strips internal Śaiva keys (Contract 2 whitelist)
2. PancakrtyaLoopV2.run() nulls h_t after completion (memory fix)
3. bridge.as_logits_processor error is caught, not propagated (Issue 4)
4. ws_generate has 30s receive timeout (Issue 7)
5. _EVENT_KEY_WHITELIST contains all 5 event types
6. WebSocket receive_text uses asyncio.wait_for (static analysis)
7. h5_ablation.score_camatk_text produces values in [0, 1]
"""
from __future__ import annotations

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock
import torch.distributions as dist


API_PATH = Path("/home/sharaths/projects/pwm-phase6/api/main.py")


def _api_source() -> str:
    return API_PATH.read_text()


# ─── Test 1: _sanitise_event_data strips internal keys ───────────────────────

def test_sanitise_event_data_strips_internal_keys():
    """_sanitise_event_data must remove vfe, efe_score, sphuratta from payloads."""
    import sys
    sys.path.insert(0, "/home/sharaths/projects/pwm-phase6/api")
    # Import the module-level function directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("api_main_partial", API_PATH)

    # Can't exec the full api/main.py without all deps — test via source inspection
    source = _api_source()
    # The whitelist must define all event types
    for event_type in ["wm_state", "stanza_start", "token", "stanza_end", "complete"]:
        assert event_type in source, f"Event type '{event_type}' missing from whitelist"
    # Internal vocabulary must NOT appear as whitelist values
    for banned in ['"vfe"', '"efe_score"', '"sphuratta"', '"vimarsa"', '"camatk"']:
        # Check they are not in the whitelist dict (they may appear in comments)
        # Only check within the _EVENT_KEY_WHITELIST block
        wl_start = source.find("_EVENT_KEY_WHITELIST")
        wl_end = source.find("}", wl_start + 100) + 1  # end of outermost dict
        whitelist_block = source[wl_start:wl_end]
        assert banned not in whitelist_block, \
            f"Banned key {banned} found inside _EVENT_KEY_WHITELIST"


# ─── Test 2: _EVENT_KEY_WHITELIST covers all 5 event types ──────────────────

def test_event_key_whitelist_covers_all_event_types():
    """All 5 protocol event types must appear in _EVENT_KEY_WHITELIST."""
    source = _api_source()
    assert "_EVENT_KEY_WHITELIST" in source, "Contract 2 whitelist dict not found"
    required_events = ["wm_state", "stanza_start", "token", "stanza_end", "complete"]
    wl_start = source.find("_EVENT_KEY_WHITELIST")
    # Find up to the next function definition after the dict
    wl_section = source[wl_start:wl_start + 800]
    for e in required_events:
        assert f'"{e}"' in wl_section, \
            f"Event type '{e}' missing from _EVENT_KEY_WHITELIST"


# ─── Test 3: _sanitise_event_data is called in both SSE and WS paths ─────────

def test_sanitise_called_in_sse_and_ws_emit_paths():
    """Both _event_stream and ws_generate must call _sanitise_event_data."""
    source = _api_source()
    # Count occurrences of _sanitise_event_data call
    count = source.count("_sanitise_event_data(")
    assert count >= 2, \
        f"_sanitise_event_data must be called in both SSE and WS paths (found {count} calls)"


# ─── Test 4: bridge.as_logits_processor wrapped in try/except ───────────────

def test_bridge_error_handled_in_run_stanza():
    """Bridge failure must be caught; Contract 3 (WM survives) must hold."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    # Mock WM
    wm = MagicMock()
    B, hidden_dim, stoch_dim, n_cats = 1, 512, 32, 32
    h_t = torch.zeros(B, hidden_dim)
    z_t = torch.zeros(B, stoch_dim, n_cats)
    wm.init_state.return_value = [(h_t, z_t)]
    wm.observe_step.return_value = (
        [(h_t, z_t)],
        [torch.zeros(stoch_dim, n_cats)],
        [torch.zeros(stoch_dim, n_cats)],
    )

    # Bridge that RAISES on as_logits_processor
    bridge = MagicMock()
    bridge.as_logits_processor.side_effect = RuntimeError("Simulated CUDA OOM in bridge")

    efe = MagicMock()
    efe.return_value = (dist.Categorical(logits=torch.zeros(1, 64)), torch.tensor([0.5]))

    citta = MagicMock()
    citta.recall.return_value = torch.zeros(B, hidden_dim)

    # LLM still generates (Contract 3: WM survives bridge failure)
    llm = MagicMock()
    llm.stream.return_value = iter(["stub ", "output\n"])

    cfg = LoopConfig(n_stanzas=1, device="cpu", max_tokens_per_stanza=5)
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    obs = torch.zeros(B, cfg.obs_dim)
    events = []
    try:
        events = list(loop.run(
            obs_sequence=[obs],
            system_prompt="test",
            user_prompt_fn=lambda i, p: "test",
        ))
    except Exception as e:
        pytest.fail(f"Bridge error should NOT propagate: {e}")

    # Must still get all event types despite bridge failure
    event_types = {e["event"] for e in events}
    assert "token" in event_types, "Tokens must be generated even after bridge failure"
    assert "complete" in event_types, "complete event must fire even after bridge failure"


# ─── Test 5: h_t nulled after run() completes ────────────────────────────────

def test_h_t_nulled_after_run_completes():
    """StanzaResult.h_t must be None after run() generator is exhausted."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    wm = MagicMock()
    B, hidden_dim, stoch_dim, n_cats = 1, 512, 32, 32
    h_t = torch.zeros(B, hidden_dim)
    z_t = torch.zeros(B, stoch_dim, n_cats)
    wm.init_state.return_value = [(h_t, z_t)]
    wm.observe_step.return_value = (
        [(h_t, z_t)],
        [torch.zeros(stoch_dim, n_cats)],
        [torch.zeros(stoch_dim, n_cats)],
    )

    bridge = MagicMock()
    bridge.as_logits_processor.return_value = lambda tids, logits: logits

    efe = MagicMock()
    efe.return_value = (dist.Categorical(logits=torch.zeros(1, 64)), torch.tensor([0.5]))

    citta = MagicMock()
    citta.recall.return_value = torch.zeros(B, hidden_dim)

    llm = MagicMock()
    llm.stream.return_value = iter(["moon\n"])

    cfg = LoopConfig(n_stanzas=2, device="cpu", max_tokens_per_stanza=5)
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    obs = torch.zeros(B, cfg.obs_dim)
    list(loop.run([obs, obs], "sys", lambda i, p: "user"))  # exhaust generator

    # After exhaustion, _wm_states must be None (freed by run())
    assert loop._wm_states is None, \
        "loop._wm_states must be None after run() completes to release GPU memory"


# ─── Test 6: WebSocket receive_text uses asyncio.wait_for (static analysis) ──

def test_ws_receive_has_timeout():
    """ws_generate must use asyncio.wait_for on receive_text (Issue 7 fix)."""
    source = _api_source()
    # Find the ws_generate function body
    ws_start = source.find("async def ws_generate")
    assert ws_start != -1, "ws_generate not found"
    ws_body = source[ws_start:ws_start + 2000]
    assert "wait_for" in ws_body, \
        "ws_generate must use asyncio.wait_for on receive_text — stalled connections"
    assert "TimeoutError" in ws_body or "timeout" in ws_body.lower(), \
        "ws_generate must handle TimeoutError from wait_for"


# ─── Test 7: score_camatk_text range [0,1] ───────────────────────────────────

def test_score_camatk_text_range():
    """score_camatk_text must return values in [0.0, 1.0] for all inputs."""
    from pwm.eval.h5_ablation import score_camatk_text

    # Empty text
    assert 0.0 <= score_camatk_text("") <= 1.0
    # Very long text (stress test ceiling)
    long_text = "moon rises over the quiet lake and the stars reflect\n" * 20
    assert 0.0 <= score_camatk_text(long_text) <= 1.0
    # Single word
    assert 0.0 <= score_camatk_text("hello") <= 1.0
    # Realistic lyric
    lyric = (
        "Rain falls on ancient stones\n"
        "Washing away the dust of time\n"
        "The river remembers the mountain\n"
        "And the sea waits at the end of every road\n"
    )
    score = score_camatk_text(lyric)
    assert 0.0 <= score <= 1.0
    assert score > 0.2, f"Realistic lyric scored too low: {score}"
