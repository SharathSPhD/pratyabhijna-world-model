"""
Sprint 20 tests: End-to-end pipeline wiring — WMReasoningTrace through PancakrtyaLoopV2.

Verifies the full ADR-002 integration:

1. loop_passes_think_prefill_when_domain_set   — domain in LoopConfig → think_prefill flows to LLM
2. loop_no_think_prefill_without_domain        — empty domain → think_prefill=None
3. loop_think_prefill_is_assistant_role        — role==assistant in the prefill dict
4. loop_think_prefill_has_think_tags           — <think>…</think> present
5. loop_prefill_contains_domain_label          — domain label visible in trace
6. loop_prefill_camatk_score_present           — aesthetic_quality threaded in
7. loop_all_events_still_fire                  — wm_state/stanza_start/token/stanza_end/complete
8. loop_domain_in_config_propagates_api        — api/main.py passes req.domain to LoopConfig
9. loop_ws_domain_propagates_api               — ws_generate passes domain to LoopConfig
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributions as dist

sys.path.insert(0, "/home/sharaths/projects/pwm-phase7")

API_PATH = Path("/home/sharaths/projects/pwm-phase7/api/main.py")


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_loop(domain: str = "carnatic") -> tuple:
    """Return (loop, captured_dict, cfg) for inspection after run()."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    B, H, Sz, Nc = 1, 512, 32, 32
    h = torch.zeros(B, H)
    z = torch.zeros(B, Sz, Nc)

    wm = MagicMock()
    wm.init_state.return_value = [(h, z)]
    wm.observe_step.return_value = (
        [(h, z)],
        [torch.zeros(Sz, Nc)],
        [torch.zeros(Sz, Nc)],
    )

    efe = MagicMock()
    efe.return_value = (dist.Categorical(logits=torch.zeros(1, 64)), torch.tensor([0.5]))

    citta = MagicMock()
    citta.recall.return_value = torch.zeros(B, H)

    bridge = MagicMock()
    bridge.as_logits_processor.return_value = lambda ids, logits: logits

    captured: dict = {}

    def _stream_spy(**kw):
        captured["think_prefill"] = kw.get("think_prefill")
        yield "moon rises\n"

    llm = MagicMock()
    llm.stream.side_effect = _stream_spy

    cfg = LoopConfig(n_stanzas=1, device="cpu", domain=domain)
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)
    return loop, captured, cfg


def _run_loop(domain: str = "carnatic") -> tuple[list[dict], dict]:
    loop, captured, _ = _make_loop(domain=domain)
    obs = torch.zeros(1, 512)
    events = list(loop.run([obs], "system prompt", lambda i, p: "user prompt"))
    return events, captured


# ─── Test 1: domain set → think_prefill flows to LLM ─────────────────────────

def test_loop_passes_think_prefill_when_domain_set():
    """When cfg.domain is set, llm.stream() must receive a non-None think_prefill."""
    _, captured = _run_loop(domain="carnatic")
    assert captured.get("think_prefill") is not None, (
        "think_prefill must not be None when LoopConfig.domain is set"
    )


# ─── Test 2: empty domain → think_prefill=None ────────────────────────────────

def test_loop_no_think_prefill_without_domain():
    """When cfg.domain is empty, llm.stream() must receive think_prefill=None."""
    _, captured = _run_loop(domain="")
    assert captured.get("think_prefill") is None, (
        f"think_prefill must be None when domain is empty, got: {captured.get('think_prefill')}"
    )


# ─── Test 3: think_prefill role == "assistant" ────────────────────────────────

def test_loop_think_prefill_is_assistant_role():
    """think_prefill must have role=='assistant' (OpenAI assistant-prefill format)."""
    _, captured = _run_loop(domain="kannada_film")
    prefill = captured.get("think_prefill", {}) or {}
    assert prefill.get("role") == "assistant", (
        f"think_prefill role must be 'assistant', got: {prefill.get('role')!r}"
    )


# ─── Test 4: think_prefill contains <think>…</think> ─────────────────────────

def test_loop_think_prefill_has_think_tags():
    """think_prefill content must contain <think>…</think> tags."""
    _, captured = _run_loop(domain="hindustani")
    content = (captured.get("think_prefill") or {}).get("content", "")
    assert "<think>" in content, f"<think> tag missing from prefill content"
    assert "</think>" in content, f"</think> tag missing from prefill content"


# ─── Test 5: domain label visible in trace ────────────────────────────────────

def test_loop_prefill_contains_domain_label():
    """trace must contain human-readable domain label."""
    _, captured = _run_loop(domain="western_jazz")
    content = (captured.get("think_prefill") or {}).get("content", "")
    assert "jazz standard" in content, (
        f"'jazz standard' not found in think_prefill for domain='western_jazz': {content[:200]}"
    )


# ─── Test 6: aesthetic_quality threaded into trace ────────────────────────────

def test_loop_prefill_camatk_score_present():
    """trace must reference aesthetic / camatk reading."""
    _, captured = _run_loop(domain="carnatic")
    content = (captured.get("think_prefill") or {}).get("content", "")
    # aesthetic_quality from run_stanza feeds camatk_score — should appear as number
    assert "camatk" in content.lower() or "aesthetic" in content.lower(), (
        f"Aesthetic quality reading missing from trace: {content[:300]}"
    )


# ─── Test 7: all 5 SSE events still fire ─────────────────────────────────────

def test_loop_all_events_still_fire():
    """Introducing think_prefill must not break the 5-event protocol."""
    events, _ = _run_loop(domain="hindi_film")
    event_types = [e["event"] for e in events]
    for expected in ("wm_state", "stanza_start", "token", "stanza_end", "complete"):
        assert expected in event_types, (
            f"Event '{expected}' missing from loop output: {event_types}"
        )


# ─── Test 8: api/main.py SSE path passes req.domain to LoopConfig ────────────

def test_loop_domain_in_config_propagates_api():
    """api/main.py _event_stream must pass req.domain to LoopConfig."""
    source = API_PATH.read_text()
    # Find the _event_stream LoopConfig block
    stream_start = source.find("async def _event_stream")
    assert stream_start != -1, "_event_stream not found in api/main.py"
    stream_body = source[stream_start: stream_start + 1500]
    assert "domain=req.domain" in stream_body, (
        "SSE _event_stream must pass domain=req.domain to LoopConfig (ADR-002)"
    )


# ─── Test 9: api/main.py WS path passes domain to LoopConfig ─────────────────

def test_loop_ws_domain_propagates_api():
    """api/main.py ws_generate must pass domain to LoopConfig."""
    source = API_PATH.read_text()
    ws_start = source.find("async def ws_generate")
    assert ws_start != -1, "ws_generate not found in api/main.py"
    ws_body = source[ws_start: ws_start + 2500]
    assert "domain=domain" in ws_body, (
        "ws_generate must pass domain=domain to LoopConfig (ADR-002)"
    )
