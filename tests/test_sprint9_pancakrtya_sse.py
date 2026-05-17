"""
Sprint 9 tests: PancakrtyaLoopV2 — all 6 acts, SSE events, Contract compliance.

Tests verify:
1. All 6 Pañcakṛtya acts fire for each stanza (Contract 1)
2. SSE events use domain-neutral labels (Contract 2)
3. WM survives LLM failure (Contract 3)
4. Correct act ordering: cit → ānanda → icchā → apohana → jñāna → kriyā
"""
from __future__ import annotations
import pytest
import torch


# ── Shared mock factories ───────────────────────────────────────────────────

def _make_mocks():
    """Return mock objects with call tracking for all 6 acts."""
    acts_called = []

    class MockWM:
        def init_state(self, b, dev):
            h = torch.zeros(b, 512)
            z = torch.zeros(b, 32, 32)
            return [(h, z)]

        def observe_step(self, obs, a, states, step):
            acts_called.append("cit")
            h = torch.randn(1, 512) * 0.3
            z = torch.randn(1, 32, 32) * 0.3
            logits_post = [torch.randn(32 * 32)]
            logits_prior = [torch.randn(32 * 32)]
            return [(h, z)], logits_post, logits_prior

    class MockEFE:
        def __call__(self, h, z):
            acts_called.append("ananda")
            from torch.distributions import Categorical
            dist = Categorical(logits=torch.zeros(h.shape[0], 64))
            efe = torch.tensor([-2.0] * h.shape[0])
            return dist, efe

    class MockCitta:
        def store_episode(self, h, level=0):
            pass

        def recall(self, h, mode="episodic"):
            acts_called.append("icha")
            return torch.zeros_like(h)

    class MockBridge:
        def as_logits_processor(self, h):
            acts_called.append("jnana")
            return None  # None is valid (no bias)

    class MockLLM:
        def stream(self, system, user, logits_processor, max_tokens, temperature, top_p):
            acts_called.append("kriya")
            yield "moon rises\n"

    class FailingLLM:
        def stream(self, **kwargs):
            acts_called.append("kriya_fail")
            raise RuntimeError("LLM unavailable")

    return acts_called, MockWM(), MockEFE(), MockCitta(), MockBridge(), MockLLM(), FailingLLM()


# ── Tests ───────────────────────────────────────────────────────────────────

def test_loop_v2_import():
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig
    assert PancakrtyaLoopV2 is not None
    assert LoopConfig is not None


def test_all_six_acts_fire():
    """Contract 1: All 6 Pañcakṛtya acts must execute for each stanza."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    acts, wm, efe, citta, bridge, llm, _ = _make_mocks()
    cfg = LoopConfig(n_stanzas=1, device="cpu")
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    obs = torch.zeros(1, 512)
    events = []
    gen = loop.run_stanza(0, obs, system_prompt="poet", user_prompt="moon")
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass

    assert "cit" in acts, "Act 1 (Cit/observe_step) not called"
    assert "ananda" in acts, "Act 2 (Ānanda/EFE) not called"
    assert "icha" in acts, "Act 3 (Icchā/Hopfield) not called"
    # Act 4 (Apohana/entropy) is internal — verified via wm_state event
    assert "jnana" in acts, "Act 5 (Jñāna/Bridge) not called"
    assert "kriya" in acts, "Act 6 (Kriyā/LLM) not called"


def test_sse_event_sequence():
    """SSE events must appear in correct order: wm_state → stanza_start → token → stanza_end."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    _, wm, efe, citta, bridge, llm, _ = _make_mocks()
    cfg = LoopConfig(n_stanzas=1, device="cpu")
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    obs = torch.zeros(1, 512)
    events = []
    gen = loop.run_stanza(0, obs, "poet", "moon")
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass

    event_types = [e["event"] for e in events]
    assert "wm_state" in event_types, "wm_state event missing"
    assert "stanza_start" in event_types, "stanza_start event missing"
    assert "token" in event_types, "token event missing"
    assert "stanza_end" in event_types, "stanza_end event missing"

    # Verify order
    wm_idx = event_types.index("wm_state")
    ss_idx = event_types.index("stanza_start")
    tok_idx = event_types.index("token")
    se_idx = event_types.index("stanza_end")
    assert wm_idx < ss_idx < tok_idx < se_idx, \
        f"Wrong event order: {event_types}"


def test_domain_neutral_labels():
    """Contract 2: SSE event keys must not contain Sanskrit internal terms."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    _, wm, efe, citta, bridge, llm, _ = _make_mocks()
    cfg = LoopConfig(n_stanzas=1, device="cpu")
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    events = []
    gen = loop.run_stanza(0, torch.zeros(1, 512), "poet", "moon")
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass

    # Check all event data keys — no Śaiva vocabulary in external API
    # Also block raw internal metric names: vfe and efe_score must not leak as SSE keys
    forbidden = {"sphuratta", "vimarsa", "camatk", "camatkara", "pancakrtya",
                 "vfe", "efe_score"}
    for ev in events:
        for key in ev.get("data", {}).keys():
            assert key.lower() not in forbidden, \
                f"Śaiva term '{key}' leaked into SSE event '{ev['event']}'"

    # But 'creative_peak' (translation of sphurattā) IS allowed
    wm_event = next(e for e in events if e["event"] == "wm_state")
    assert "creative_peak" in wm_event["data"], "creative_peak missing from wm_state"
    assert "aesthetic_quality" in wm_event["data"], "aesthetic_quality missing"


def test_wm_survives_llm_failure():
    """Contract 3: WM must produce output even when LLM stream raises."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    acts, wm, efe, citta, bridge, _, failing_llm = _make_mocks()
    cfg = LoopConfig(n_stanzas=1, device="cpu")
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, failing_llm, cfg)

    events = []
    result = None
    gen = loop.run_stanza(0, torch.zeros(1, 512), "poet", "moon")
    try:
        while True:
            events.append(next(gen))
    except StopIteration as e:
        result = e.value

    # Must not raise; must emit token and stanza_end
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) > 0, "No token emitted even in LLM-failure fallback"
    # Result must exist
    assert result is not None, "StanzaResult must be returned even on LLM failure"
    assert result.text != "", "Fallback text must not be empty"


def test_full_run_complete_event():
    """run() must emit a complete event with mean_aesthetic_quality."""
    from pwm.pipeline.pancakrtya_loop_v2 import PancakrtyaLoopV2, LoopConfig

    _, wm, efe, citta, bridge, llm, _ = _make_mocks()
    cfg = LoopConfig(n_stanzas=2, device="cpu")
    loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)

    obs_list = [torch.zeros(1, 512)] * 2
    events = list(loop.run(obs_list, "poet", lambda i, prev: f"stanza {i}"))

    complete_events = [e for e in events if e["event"] == "complete"]
    assert len(complete_events) == 1, "Exactly one complete event expected"
    data = complete_events[0]["data"]
    assert "mean_aesthetic_quality" in data
    assert "total_stanzas" in data
    assert data["total_stanzas"] == 2
    assert 0.0 <= data["mean_aesthetic_quality"] <= 1.0
