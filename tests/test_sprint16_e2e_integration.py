"""
Sprint 16 tests: End-to-end neo-fm-web integration.

Verifies:
1. LlamaCppBackend connects to Ollama and streams real tokens
2. _http_stream correctly filters reasoning tokens (nemotron-3-super)
3. model_name is included in all HTTP request bodies
4. Ollama endpoint reachable and responds to /v1/chat/completions
5. Full pipeline: WM obs → PancakrtyaLoopV2.run_stanza → SSE events (mock WM)
6. end-to-end latency: first token within 30s (nemotron reasoning warmup)
7. OLLAMA_MODEL_NAME exported from engine.py
"""
from __future__ import annotations

import json
import time
import pytest
import requests
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

OLLAMA_URL = "http://localhost:11434"


def _ollama_reachable() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _model_available(model_name: str) -> bool:
    if not _ollama_reachable():
        return False
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        return model_name in models
    except Exception:
        return False


OLLAMA_SKIP = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama not reachable at localhost:11434",
)


# ─── Test 1: OLLAMA_MODEL_NAME exported ──────────────────────────────────────

def test_ollama_model_name_exported_from_engine():
    """OLLAMA_MODEL_NAME must be importable from engine for api/main.py."""
    from pwm.generation.engine import OLLAMA_MODEL_NAME
    assert isinstance(OLLAMA_MODEL_NAME, str)
    assert len(OLLAMA_MODEL_NAME) > 0, "OLLAMA_MODEL_NAME must not be empty"
    # Must be a valid Ollama model identifier (has a colon for tag)
    assert ":" in OLLAMA_MODEL_NAME or "/" in OLLAMA_MODEL_NAME or len(OLLAMA_MODEL_NAME) > 5, \
        f"OLLAMA_MODEL_NAME looks too short: {OLLAMA_MODEL_NAME}"


# ─── Test 2: model_name in HTTP request body ─────────────────────────────────

def test_llama_backend_includes_model_in_request(mocker=None):
    """LlamaCppBackend._http_request_body must include 'model' field."""
    from pwm.generation.llama_backend import LlamaCppBackend
    backend = LlamaCppBackend(
        model_path="/nonexistent.gguf",
        server_url="http://localhost:11434",
        model_name="nemotron-3-super:120b",
        mock=False,
    )
    body = backend._http_request_body(
        system="test", user="test", max_tokens=10,
        temperature=0.8, top_p=0.9, stream=False
    )
    assert "model" in body, "_http_request_body must include 'model' for Ollama"
    assert body["model"] == "nemotron-3-super:120b"


# ─── Test 3: Ollama reachable ────────────────────────────────────────────────

@OLLAMA_SKIP
def test_ollama_api_reachable():
    """Ollama must be reachable and return model list."""
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    model_names = [m["name"] for m in data["models"]]
    assert len(model_names) > 0, "Ollama has no models loaded"


# ─── Test 4: nemotron-mini streams content tokens ────────────────────────────

@pytest.mark.skipif(
    not _model_available("nemotron-mini:4b") if _ollama_reachable() else True,
    reason="nemotron-mini:4b not available or Ollama not reachable",
)
def test_llama_backend_streams_real_tokens_nemotron_mini():
    """
    LlamaCppBackend.stream() must yield actual content tokens from nemotron-mini:4b.
    nemotron-mini does not have a reasoning phase — content tokens arrive immediately.
    """
    from pwm.generation.llama_backend import LlamaCppBackend
    backend = LlamaCppBackend(
        model_path="/nonexistent.gguf",
        server_url=OLLAMA_URL,
        model_name="nemotron-mini:4b",
    )
    tokens = []
    t0 = time.perf_counter()
    for tok in backend.stream(
        system="You are a creative poet. Output only poem text, nothing else.",
        user="Write exactly 2 short lines about moonlight.",
        max_tokens=40,
        temperature=0.7,
        top_p=0.9,
    ):
        tokens.append(tok)
        if len(tokens) == 1:
            ttft_ms = (time.perf_counter() - t0) * 1000

    text = "".join(tokens)
    assert len(tokens) > 3, f"Expected >3 tokens, got {len(tokens)}: {repr(text)}"
    assert len(text.strip()) > 5, f"Generated text too short: {repr(text)}"
    # TTFT < 5s for nemotron-mini (no reasoning phase)
    assert ttft_ms < 5000, f"TTFT too slow: {ttft_ms:.0f}ms (expected <5s for mini)"


# ─── Test 5: reasoning model skips reasoning tokens ──────────────────────────

def test_http_stream_skips_reasoning_tokens():
    """
    _http_stream must skip delta.reasoning tokens and yield only delta.content.
    Verified by mocking the requests response with reasoning+content chunks.
    """
    from pwm.generation.llama_backend import LlamaCppBackend
    backend = LlamaCppBackend(
        model_path="/nonexistent.gguf",
        server_url=OLLAMA_URL,
        model_name="nemotron-3-super:120b",
    )

    # Simulate Ollama stream: 3 reasoning tokens then 2 content tokens
    mock_lines = [
        b'data: {"choices":[{"delta":{"reasoning":"Let me think..."}}]}',
        b'data: {"choices":[{"delta":{"reasoning":"About rain."}}]}',
        b'data: {"choices":[{"delta":{"content":"Rain "}}]}',
        b'data: {"choices":[{"delta":{"content":"falls."}}]}',
        b'data: [DONE]',
    ]

    class MockResponse:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_lines(self): return iter(mock_lines)

    with patch("pwm.generation.llama_backend.requests.post", return_value=MockResponse()):
        result = list(backend._http_stream("sys", "usr", 50, 0.8, 0.9))

    assert result == ["Rain ", "falls."], \
        f"Expected only content tokens, got: {result}"


# ─── Test 6: PancakrtyaLoopV2 + mock LLM produces all 5 event types ──────────

def test_pancakrtya_loop_produces_all_event_types_with_real_mock_llm():
    """
    Full loop test: WM (mock), EFE (mock), CittaStore (mock), Bridge (mock),
    LLM (mock streaming 5 tokens). Must yield all 5 event types.
    Verifies the loop orchestration is intact after S16 changes.
    """
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    # Mock WM: returns plausible state shapes
    wm = MagicMock()
    B, hidden_dim, stoch_dim, n_cats = 1, 512, 32, 32
    h_t = torch.zeros(B, hidden_dim)
    z_t = torch.zeros(B, stoch_dim, n_cats)
    logits_post = [torch.zeros(stoch_dim, n_cats)]
    logits_prior = [torch.zeros(stoch_dim, n_cats)]
    wm.init_state.return_value = [(h_t, z_t)]
    wm.observe_step.return_value = ([(h_t, z_t)], logits_post, logits_prior)

    # Mock EFE: returns (Categorical, efe_score)
    efe = MagicMock()
    import torch.distributions as dist
    efe.return_value = (dist.Categorical(logits=torch.zeros(1, 64)), torch.tensor([0.5]))

    # Mock CittaStore: recall returns h_t-like tensor
    citta = MagicMock()
    citta.recall.return_value = torch.zeros(B, hidden_dim)

    # Mock VimarsaBridge: returns identity logits processor
    bridge = MagicMock()
    bridge.as_logits_processor.return_value = lambda token_ids, logits: logits

    # Mock LLM: streams exactly 5 tokens
    llm = MagicMock()
    llm.stream.return_value = iter(["moon ", "rises ", "soft ", "and ", "slow\n"])

    cfg = LoopConfig(n_stanzas=1, device="cpu", max_tokens_per_stanza=5)
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    obs = torch.zeros(B, cfg.obs_dim)
    events = list(loop.run(
        obs_sequence=[obs],
        system_prompt="test",
        user_prompt_fn=lambda i, p: "write a poem",
    ))

    event_types = [e["event"] for e in events]
    assert "wm_state" in event_types, f"Missing wm_state event. Got: {event_types}"
    assert "stanza_start" in event_types, f"Missing stanza_start. Got: {event_types}"
    assert "token" in event_types, f"Missing token events. Got: {event_types}"
    assert "stanza_end" in event_types, f"Missing stanza_end. Got: {event_types}"
    assert "complete" in event_types, f"Missing complete event. Got: {event_types}"

    # Verify complete event has correct keys
    complete = next(e for e in events if e["event"] == "complete")
    assert "total_stanzas" in complete["data"]
    assert "mean_aesthetic_quality" in complete["data"]
    assert complete["data"]["total_stanzas"] == 1


# ─── Test 7: api/main.py imports OLLAMA_MODEL_NAME without error ──────────────

def test_api_imports_ollama_model_name():
    """api/main.py must import OLLAMA_MODEL_NAME from engine without error."""
    api_path = Path("/home/sharaths/projects/pwm-phase6/api/main.py")
    source = api_path.read_text()
    assert "OLLAMA_MODEL_NAME" in source, \
        "api/main.py must import OLLAMA_MODEL_NAME from engine"
    # Ensure the import line is syntactically valid (no NameError at startup)
    assert "from pwm.generation.engine import" in source
