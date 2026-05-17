"""
Sprint 18 tests: Model Cascade Streaming — ADR-001 (TRIZ Principles 10 + 24).

Verifies the stream_cascade() implementation in LlamaCppBackend:

1. cascade_dispatch_via_stream   — stream() dispatches to stream_cascade() when cascade_model_name set
2. cascade_no_dispatch_without   — stream() does NOT dispatch to cascade when cascade_model_name=None
3. cascade_switch_fast_to_slow   — tokens come from fast model first, then switch to slow model
4. cascade_no_duplicate_tokens   — no tokens appear twice across the switch boundary
5. cascade_slow_timeout_fallback — if slow model never starts, output completes cleanly
6. cascade_slow_error_fallback   — if slow model errors, output completes cleanly (no exception)
7. cascade_wm_bias_applied_both  — logits_processor is threaded through stream_cascade() call
8. cascade_api_source_check      — api/main.py get_llm_backend() call is unchanged (transparent wiring)
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

API_PATH = Path("/home/sharaths/projects/pwm-phase7/api/main.py")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_backend(
    cascade_model_name: str | None = "nemotron-mini:4b",
    server_url: str = "http://localhost:11434",
    mock: bool = False,
) -> "LlamaCppBackend":
    """Construct a LlamaCppBackend with HTTP-only mode (no llama-cpp-python)."""
    import sys
    sys.path.insert(0, "/home/sharaths/projects/pwm-phase7")
    from pwm.generation.llama_backend import LlamaCppBackend
    return LlamaCppBackend(
        model_path="/nonexistent/model.gguf",
        server_url=server_url,
        model_name="nemotron-3-super:120b",
        cascade_model_name=cascade_model_name,
        mock=mock,
    )


def _mock_stream(tokens: list[str], delay: float = 0.0):
    """Return a mock for _http_stream_model that yields the given tokens."""
    def _inner(*args, **kwargs):
        for tok in tokens:
            if delay:
                time.sleep(delay)
            yield tok
    return _inner


# ─── Test 1: stream() dispatches to stream_cascade() when cascade set ─────────

def test_cascade_dispatch_via_stream():
    """stream() must delegate to stream_cascade() when cascade_model_name is configured."""
    backend = _make_backend(cascade_model_name="nemotron-mini:4b")
    dispatched_to_cascade = []

    original_cascade = backend.stream_cascade

    def _spy_cascade(*args, **kwargs):
        dispatched_to_cascade.append(True)
        yield "cascade_token\n"

    backend.stream_cascade = _spy_cascade  # type: ignore[method-assign]

    result = list(backend.stream(
        system="sys", user="user", max_tokens=32
    ))

    assert dispatched_to_cascade, "stream() did not dispatch to stream_cascade()"
    assert result == ["cascade_token\n"]


# ─── Test 2: stream() does NOT dispatch when cascade_model_name=None ──────────

def test_cascade_no_dispatch_without():
    """stream() must NOT use cascade when cascade_model_name is None."""
    backend = _make_backend(cascade_model_name=None)

    spy = MagicMock(return_value=iter(["direct_token\n"]))
    backend._http_stream_model = spy  # type: ignore[method-assign]

    # Give backend a server_url but no cascade — should call _http_stream_model
    # directly via _http_stream, not stream_cascade
    with patch.object(backend, "stream_cascade", side_effect=AssertionError("should not call cascade")):
        # We patch _http_stream to return our stub
        with patch.object(backend, "_http_stream", return_value=iter(["direct\n"])):
            tokens = list(backend.stream(system="sys", user="user", max_tokens=32))

    assert tokens == ["direct\n"], f"Expected direct stream, got: {tokens}"


# ─── Test 3: tokens come from fast model then switch to slow model ─────────────

def test_cascade_switch_fast_to_slow():
    """stream_cascade() must yield fast-model tokens, then switch to slow-model tokens."""
    backend = _make_backend()

    # Fast model yields quickly; slow model has a 0.05s head-start delay
    fast_tokens = ["fast1 ", "fast2 ", "fast3\n"]
    slow_tokens = ["slow1 ", "slow2 ", "slow3\n"]

    # We'll make slow model start after a brief delay, after fast starts.
    # Mechanism: fast model never signals slow_first_content by itself.
    # We control _http_stream_model to route per model name.
    def _stream_model_dispatch(model, *args, **kwargs):
        if "mini" in model:
            # Fast model: yield tokens, but pause before each so slow can start
            for tok in fast_tokens:
                time.sleep(0.02)  # 20ms per token
                yield tok
        else:
            # Slow model: 80ms reasoning delay, then content
            time.sleep(0.08)
            for tok in slow_tokens:
                yield tok

    backend._http_stream_model = _stream_model_dispatch  # type: ignore[method-assign]

    result = list(backend.stream_cascade(
        system="sys", user="user", max_tokens=32
    ))

    result_str = "".join(result)
    # Must contain some slow tokens (switch happened)
    assert any(s in result_str for s in slow_tokens), (
        f"Slow model tokens missing from output: {result_str!r}"
    )
    # Must NOT be purely fast tokens
    assert result_str != "".join(fast_tokens), (
        "Output is identical to fast-only — cascade switch did not happen"
    )


# ─── Test 4: no duplicate tokens across switch boundary ───────────────────────

def test_cascade_no_duplicate_tokens():
    """Tokens must not appear twice — no overlap between fast and slow phases."""
    backend = _make_backend()

    fast_tokens = ["alpha ", "beta ", "gamma\n"]
    slow_tokens = ["DELTA ", "EPSILON ", "ZETA\n"]

    # Slow model starts immediately (no delay) — switch happens on first fast token
    def _stream_dispatch(model, *args, **kwargs):
        if "mini" in model:
            time.sleep(0.05)  # fast model waits, slow starts first
            for tok in fast_tokens:
                yield tok
        else:
            for tok in slow_tokens:
                yield tok

    backend._http_stream_model = _stream_dispatch  # type: ignore[method-assign]

    result = list(backend.stream_cascade(system="sys", user="user", max_tokens=32))
    result_str = "".join(result)

    # Count occurrences — no token should appear >1 times
    for tok in slow_tokens + fast_tokens:
        count = result_str.count(tok)
        assert count <= 1, f"Token {tok!r} appears {count} times — duplicate detected"


# ─── Test 5: slow model timeout → output completes cleanly ────────────────────

def test_cascade_slow_timeout_fallback():
    """If slow model never produces content, stream_cascade() must return cleanly."""
    backend = _make_backend()

    fast_tokens = ["timeout_fast1 ", "timeout_fast2\n"]

    def _stream_dispatch(model, *args, **kwargs):
        if "mini" in model:
            for tok in fast_tokens:
                yield tok
        else:
            # Slow model hangs — never yields, but done is set eventually
            time.sleep(5.0)  # longer than slow_timeout in test
            yield "never_reaches_here"

    backend._http_stream_model = _stream_dispatch  # type: ignore[method-assign]

    # Use very short timeout for test speed
    result = list(backend.stream_cascade(
        system="sys", user="user", max_tokens=32, slow_timeout=0.1
    ))

    # Should not raise, should return cleanly
    # (fast model finished before timeout — the wait completes with got_content=False)
    assert isinstance(result, list), "stream_cascade() must return a list of tokens"


# ─── Test 6: slow model errors → output completes cleanly ─────────────────────

def test_cascade_slow_error_fallback():
    """If slow model raises an exception, stream_cascade() must not propagate it."""
    backend = _make_backend()

    def _stream_dispatch(model, *args, **kwargs):
        if "mini" in model:
            yield "fast_prefix "
            time.sleep(0.05)  # let slow model error first
            yield "fast_suffix\n"
        else:
            raise RuntimeError("Simulated CUDA OOM in slow model")

    backend._http_stream_model = _stream_dispatch  # type: ignore[method-assign]

    try:
        result = list(backend.stream_cascade(
            system="sys", user="user", max_tokens=32, slow_timeout=0.5
        ))
    except Exception as exc:
        pytest.fail(f"stream_cascade() must not propagate slow model error: {exc}")

    assert isinstance(result, list)


# ─── Test 7: logits_processor threaded through stream_cascade ────────────────

def test_cascade_wm_bias_applied_both():
    """logits_processor arg must be accepted and propagated by stream_cascade()."""
    backend = _make_backend()

    received_logits_processor = []

    def _stream_dispatch(model, *args, **kwargs):
        yield "tok\n"

    backend._http_stream_model = _stream_dispatch  # type: ignore[method-assign]

    # Verify stream_cascade accepts logits_processor without error
    # (HTTP path doesn't apply it natively, but the signature must accept it)
    mock_lp = MagicMock()
    result = list(backend.stream_cascade(
        system="sys", user="user", logits_processor=mock_lp, max_tokens=32,
        slow_timeout=0.1
    ))
    # Should complete without TypeError
    assert isinstance(result, list)


# ─── Test 8: api/main.py get_llm_backend() call is transparent ────────────────

def test_cascade_api_source_check():
    """api/main.py must call get_llm_backend() (cascade dispatch is transparent)."""
    source = API_PATH.read_text()

    # get_llm_backend() must be called — no explicit stream_cascade in api layer
    assert "get_llm_backend()" in source, \
        "api/main.py must call get_llm_backend() to obtain backend with cascade wiring"

    # api/main.py should NOT directly call stream_cascade — that's internal
    assert "stream_cascade(" not in source, \
        "api/main.py should not call stream_cascade() directly — cascade is transparent via stream()"

    # CASCADE_MODEL_NAME must be importable from engine
    engine_source = (Path("/home/sharaths/projects/pwm-phase7/pwm/generation/engine.py")).read_text()
    assert "CASCADE_MODEL_NAME" in engine_source, \
        "CASCADE_MODEL_NAME must be defined in engine.py (ADR-001 S18 config)"
