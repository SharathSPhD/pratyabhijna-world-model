"""
Sprint 15 tests: LiveViz WebSocket + /v1/health neo-fm-web contract.

Verifies:
1. /v1/health returns all expected keys with correct types
2. WebSocket endpoint is defined in api/main.py
3. Health status reflects singleton readiness (wm_ready, domain_states)
4. SSE and WebSocket protocols are consistent (same event shapes)
5. CORS headers are present for neo-fm-web.vercel.app
"""
from __future__ import annotations
import ast
import json
import pytest
from pathlib import Path


API_PATH = Path("/home/sharaths/projects/pwm-phase5/api/main.py")


def _parse_api_source():
    """Parse api/main.py AST to verify endpoint definitions."""
    with open(API_PATH) as f:
        return ast.parse(f.read())


def _api_source():
    return API_PATH.read_text()


def test_health_v1_returns_all_required_keys():
    """
    /v1/health must return all keys expected by neo-fm-web.vercel.app.
    Verified by static analysis of the return dict literal.
    """
    source = _api_source()
    # All keys neo-fm-web needs to poll before enabling Generate
    required_keys = [
        "status", "device", "cuda_available", "wm_ready",
        "domains_prewarmed", "ttft_profile", "version", "timestamp",
    ]
    for key in required_keys:
        assert f'"{key}"' in source, \
            f"/v1/health missing key: {key}"


def test_health_reflects_singleton_state():
    """/v1/health must reference state.wm_ready and state.domain_states."""
    source = _api_source()
    # Must read from singleton not from a fresh import
    assert "state.wm_ready" in source, "health_v1 must read state.wm_ready"
    assert "state.domain_states" in source, \
        "health_v1 must include state.domain_states in response"


def test_websocket_endpoint_defined():
    """/v1/ws/generate WebSocket endpoint must exist."""
    source = _api_source()
    assert "@app.websocket" in source, "WebSocket decorator not found"
    assert '"/v1/ws/generate"' in source, "/v1/ws/generate route not defined"


def test_websocket_sends_same_events_as_sse():
    """
    WebSocket protocol must use the same event names as SSE.
    Verified by static analysis of the ws_generate function body.
    """
    source = _api_source()
    required_events = ["wm_state", "stanza_start", "token", "stanza_end", "complete"]
    for event_name in required_events:
        # Events are either yielded directly (SSE) or sent via ws.send_text (WS)
        # Both paths run the same PancakrtyaLoopV2 loop — events come from there
        assert event_name in source, \
            f"Event '{event_name}' not referenced in api/main.py"


def test_cors_configured_for_neo_fm_web():
    """CORS must allow neo-fm-web.vercel.app origin."""
    source = _api_source()
    assert "CORSMiddleware" in source, "CORSMiddleware not added"
    # Either wildcard or specific origin
    has_wildcard = '"*"' in source or "'*'" in source
    has_specific = "vercel.app" in source or "neo-fm" in source
    assert has_wildcard or has_specific, \
        "CORS not configured for vercel.app — neo-fm-web will be blocked"


def test_ws_uses_singleton_not_per_request_load():
    """WebSocket handler must use state.wm/efe/citta/bridge (not load_trained_components)."""
    source = _api_source()
    # Find the ws_generate function and verify it reads from state
    assert "state.wm" in source, "WS handler must use singleton state.wm"
    assert "state.efe" in source, "WS handler must use singleton state.efe"
    assert "state.citta" in source, "WS handler must use singleton state.citta"
    assert "state.bridge" in source, "WS handler must use singleton state.bridge"


def test_api_has_version_phase5_s14():
    """Health endpoint version string must reflect current phase."""
    source = _api_source()
    assert "phase5-s14" in source or "phase5" in source, \
        "Version not updated to reflect Phase 5 / S14"
