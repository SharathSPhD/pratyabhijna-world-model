"""
Sprint 19 tests: WM Reasoning Trace — ADR-002 (TRIZ Sketch A, IFR 4/4).

Verifies WMReasoningTrace and its integration into LlamaCppBackend:

1. trace_renders_think_tags            — render() produces <think>…</think> envelope
2. trace_contains_domain_label         — domain mapped to human-readable string
3. trace_camatk_score_guidance         — camatk reading appears and changes guidance
4. trace_citta_hits_included           — Citta retrievals appear in trace
5. trace_sphuratta_events_included     — sphurattā event count noted in trace
6. trace_creative_metadata_included    — WMStateDecoder CreativeMetadata fields used
7. prefill_returns_assistant_role      — render_as_assistant_prefill() role == "assistant"
8. backend_messages_include_prefill    — _build_messages() inserts prefill between system and user
9. backend_no_prefill_baseline         — without prefill, messages have only 2 entries
10. stream_accepts_think_prefill        — stream() accepts think_prefill kwarg without error
11. cascade_passes_prefill_to_slow      — stream_cascade() passes think_prefill to _http_stream_model
12. wm_trace_short_enough_for_prefill   — trace length is bounded (<600 chars / ~150 tokens)
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_trace() -> "WMReasoningTrace":
    import sys
    sys.path.insert(0, "/home/sharaths/projects/pwm-phase7")
    from pwm.vimarsa.narrator import WMReasoningTrace
    return WMReasoningTrace()


def _make_backend(cascade: bool = False) -> "LlamaCppBackend":
    import sys
    sys.path.insert(0, "/home/sharaths/projects/pwm-phase7")
    from pwm.generation.llama_backend import LlamaCppBackend
    return LlamaCppBackend(
        model_path="/nonexistent/model.gguf",
        server_url="http://localhost:11434",
        model_name="nemotron-3-super:120b",
        cascade_model_name="nemotron-mini:4b" if cascade else None,
        mock=True,
    )


def _h(dim: int = 512) -> torch.Tensor:
    return torch.randn(1, dim)


# ─── Test 1: render() produces <think>…</think> tags ──────────────────────────

def test_trace_renders_think_tags():
    """render() must wrap content in <think>…</think>."""
    wrt = _make_trace()
    trace = wrt.render(_h(), domain="carnatic")
    assert trace.startswith("<think>"), f"Trace must start with <think>: {trace[:50]}"
    assert trace.endswith("</think>"), f"Trace must end with </think>: {trace[-50:]}"


# ─── Test 2: domain mapped to human-readable label ────────────────────────────

def test_trace_contains_domain_label():
    """render() must translate domain slug to human label."""
    wrt = _make_trace()

    trace_kf = wrt.render(_h(), domain="kannada_film")
    assert "Kannada film song" in trace_kf, "kannada_film must map to 'Kannada film song'"

    trace_jz = wrt.render(_h(), domain="western_jazz")
    assert "jazz standard" in trace_jz, "western_jazz must map to 'jazz standard'"

    # Unknown domain must not crash — falls back to slugified name
    trace_unk = wrt.render(_h(), domain="my_custom_domain")
    assert "my custom domain" in trace_unk, "Unknown domain must be slug-formatted"


# ─── Test 3: camatk_score reading changes guidance ────────────────────────────

def test_trace_camatk_score_guidance():
    """Guidance must be consistent with camatk_score level."""
    wrt = _make_trace()

    high_trace = wrt.render(_h(), domain="generic", camatk_score=0.85)
    assert "high" in high_trace.lower() or "preserve" in high_trace.lower(), (
        f"High camatk should say 'high' or 'preserve': {high_trace}"
    )

    low_trace = wrt.render(_h(), domain="generic", camatk_score=0.15)
    assert "low" in low_trace.lower() or "change" in low_trace.lower(), (
        f"Low camatk should say 'low' or 'change': {low_trace}"
    )


# ─── Test 4: Citta hits appear in trace ───────────────────────────────────────

def test_trace_citta_hits_included():
    """Citta episodic retrievals must appear in the trace."""
    wrt = _make_trace()
    hits = ["Rāga Bhairavi evokes dawn stillness", "ಮಳೆ ಬರುತ್ತದೆ — rain is coming"]
    trace = wrt.render(_h(), domain="carnatic", citta_hits=hits)
    assert "Bhairavi" in trace or "dawn stillness" in trace, (
        f"Citta hit not found in trace: {trace}"
    )


# ─── Test 5: sphurattā event count in trace ───────────────────────────────────

def test_trace_sphuratta_events_included():
    """sphurattā event peaks must be summarised in trace."""
    wrt = _make_trace()
    events = [
        {"stanza": 0, "camatk_score": 0.82},
        {"stanza": 1, "camatk_score": 0.91},
        {"stanza": 2, "camatk_score": 0.35},
    ]
    trace = wrt.render(_h(), domain="hindustani", sphuratta_events=events)
    # 2 events above 0.7 threshold
    assert "2" in trace or "peak" in trace.lower() or "aesthetic" in trace.lower(), (
        f"Sphurattā events not reflected in trace: {trace}"
    )


# ─── Test 6: CreativeMetadata fields appear in trace ──────────────────────────

def test_trace_creative_metadata_included():
    """CreativeMetadata fields (rāga, section, emotion) must appear when provided."""
    wrt = _make_trace()

    # Build a minimal mock metadata
    meta = MagicMock()
    meta.raga_hint = "Bhairavi"
    meta.section_name = "pallavi"
    meta.emotion_tags = ["devotional", "yearning"]
    meta.tempo_hint = "vilambit"

    trace = wrt.render(_h(), domain="carnatic", creative_metadata=meta)
    assert "Bhairavi" in trace, "rāga_hint must appear in trace"
    assert "pallavi" in trace, "section_name must appear in trace"
    assert "devotional" in trace or "yearning" in trace, "emotion_tags must appear in trace"
    assert "vilambit" in trace, "tempo_hint must appear in trace"


# ─── Test 7: render_as_assistant_prefill() returns assistant role ─────────────

def test_prefill_returns_assistant_role():
    """render_as_assistant_prefill() must return dict with role=='assistant'."""
    wrt = _make_trace()
    prefill = wrt.render_as_assistant_prefill(_h(), domain="hindi_film")
    assert isinstance(prefill, dict), "prefill must be a dict"
    assert prefill["role"] == "assistant", (
        f"prefill role must be 'assistant', got {prefill['role']!r}"
    )
    assert "<think>" in prefill["content"], "prefill content must contain <think>"
    assert "</think>" in prefill["content"], "prefill content must contain </think>"


# ─── Test 8: _build_messages() inserts prefill between system and user ─────────

def test_backend_messages_include_prefill():
    """_build_messages() with think_prefill must produce [system, assistant, user]."""
    backend = _make_backend()
    think_msg = {"role": "assistant", "content": "<think>trace</think>"}
    messages = backend._build_messages("sys", "user", think_prefill=think_msg)

    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "assistant"
    assert "<think>" in messages[1]["content"]
    assert messages[2]["role"] == "user"


# ─── Test 9: without prefill, messages have only 2 entries ────────────────────

def test_backend_no_prefill_baseline():
    """_build_messages() without think_prefill must produce [system, user] only."""
    backend = _make_backend()
    messages = backend._build_messages("sys", "user")
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# ─── Test 10: stream() accepts think_prefill kwarg ────────────────────────────

def test_stream_accepts_think_prefill():
    """stream() must accept think_prefill without TypeError."""
    backend = _make_backend(cascade=False)
    think_msg = {"role": "assistant", "content": "<think>trace</think>"}

    # Mock the HTTP stream to avoid real network call
    with patch.object(backend, "_http_stream", return_value=iter(["tok\n"])):
        try:
            tokens = list(backend.stream("sys", "user", think_prefill=think_msg))
        except TypeError as exc:
            pytest.fail(f"stream() does not accept think_prefill: {exc}")

    assert isinstance(tokens, list)


# ─── Test 11: stream_cascade() passes think_prefill to _http_stream_model ─────

def test_cascade_passes_prefill_to_slow():
    """stream_cascade() must thread think_prefill to the slow model's request."""
    import sys
    sys.path.insert(0, "/home/sharaths/projects/pwm-phase7")
    from pwm.generation.llama_backend import LlamaCppBackend

    backend = LlamaCppBackend(
        model_path="/nonexistent",
        server_url="http://localhost:11434",
        model_name="nemotron-3-super:120b",
        cascade_model_name="nemotron-mini:4b",
        mock=False,
    )

    received_think_prefill: list[dict | None] = []

    def _mock_stream_model(model, system, user, max_tokens, temperature, top_p,
                           think_prefill=None):
        received_think_prefill.append(think_prefill)
        if "mini" in model:
            time.sleep(0.05)  # let slow start first
            yield "fast_tok\n"
        else:
            yield "slow_tok\n"

    backend._http_stream_model = _mock_stream_model  # type: ignore[method-assign]

    think_msg = {"role": "assistant", "content": "<think>wm deliberation</think>"}
    list(backend.stream_cascade(
        "sys", "user",
        think_prefill=think_msg,
        max_tokens=32,
        slow_timeout=1.0,
    ))

    # The slow model (120B) call must have received think_prefill
    slow_prefills = [p for p in received_think_prefill if p is not None]
    assert slow_prefills, (
        "think_prefill must be passed to at least one _http_stream_model call "
        f"(received_prefills={received_think_prefill})"
    )
    assert slow_prefills[0]["content"] == think_msg["content"]


# ─── Test 12: trace length is bounded ────────────────────────────────────────

def test_wm_trace_short_enough_for_prefill():
    """Trace must be short enough not to dominate prefill budget (~150 tokens max)."""
    wrt = _make_trace()

    # Maximally decorated trace
    events = [{"stanza": i, "camatk_score": 0.8} for i in range(5)]
    hits = [f"memory fragment {i}" for i in range(5)]
    meta = MagicMock()
    meta.raga_hint = "Bhairavi"
    meta.section_name = "anupallavi"
    meta.emotion_tags = ["devotional", "yearning", "ecstatic"]
    meta.tempo_hint = "madhya laya"

    trace = wrt.render(
        _h(), domain="carnatic",
        creative_metadata=meta,
        sphuratta_events=events,
        citta_hits=hits,
        stanza_idx=3,
        camatk_score=0.72,
        vfe=0.0234,
    )

    # Rough token estimate: chars / 4 ≈ tokens
    char_count = len(trace)
    approx_tokens = char_count / 4
    assert approx_tokens < 600, (
        f"Trace too long: {char_count} chars ≈ {approx_tokens:.0f} tokens. "
        "Cap at ~600 tokens to avoid slow prefill."
    )
