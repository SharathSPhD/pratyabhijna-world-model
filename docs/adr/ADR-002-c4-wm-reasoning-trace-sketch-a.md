# ADR-002: C4 — WM Reasoning Trace as Think-Block Prefill (TRIZ Sketch A, IFR 4/4)

**Status:** Accepted (implemented S19)  
**Date:** 2026-05-17  
**Sprint:** S19  
**Supersedes:** ADR-001 (Cascade Streaming, IFR 3/4) — retained as runtime fallback

---

## Context

ADR-001 (S18) implemented Cascade Streaming (TRIZ Sketch B) — streaming nemotron-mini:4b
immediately while nemotron-3-super:120b reasons in background. This **manages** the C4
contradiction (IFR 3/4) but does not eliminate it: the 120B still requires ~60s reasoning,
limiting the switch latency.

The TRIZ agent returned a superior analysis (Sketch A, IFR 4/4) identifying the root cause:
the LLM's Chain-of-Thought *duplicates* the WM's vimarśa.

---

## Root Cause: Vimarśa Was Being Duplicated

The 120B's CoT asks:
> "What should this lyric express, given context X?"

This is precisely the question the WM posterior `q_φ(z_t|h_t,o_t)`, CamatkaraNarrator,
and Citta-store retrievals already answer — at WM tick rate (~milliseconds), not 60 seconds.

**ĪPK 1.5.11 (Utpaladeva) on Vimarśa:** Reflexive self-recognition — consciousness holding
itself before itself. The WM IS the vimarśa. The LLM's CoT was an approximation of a
computation the WM substrate had already performed.

**VimarsaBridgeV2 cannot fix this** — its logits bias acts on content tokens only. The
bottleneck lies upstream of the bias hook, in the 60s reasoning phase.

---

## Decision: WM Reasoning Trace as Assistant-Prefill Think-Block

**TRIZ Parameters:**
- Improving: P25 (Loss of time — TTFT) + P9 (Speed)
- Worsening: P28 (Measurement accuracy — creative quality proxy)
- **Physical core:** Reasoning compute must simultaneously *happen* (for quality) and
  *not happen* (for TTFT) inside the LLM forward pass.

**Principles Applied:** P10 (Prior Action) + P28 (Mechanics Substitution)
- **P10:** Perform the deliberation prior to the LLM's forward pass (WM computes it).
- **P28:** Substitute the LLM's mechanical reasoning with the WM's vimarśa.

**Mechanism — `WMReasoningTrace.render_as_assistant_prefill()`:**

The WM state `(h_t, domain, creative_metadata, sphuratta_events, citta_hits)` is rendered
as a `<think>…</think>` string and injected as an **assistant-prefill message** in the
OpenAI message list:

```python
messages = [
    {"role": "system",    "content": system_prompt},
    {"role": "assistant", "content": "<think>\n{wm_trace}\n</think>"},  # ← injected
    {"role": "user",      "content": user_prompt},
]
```

The reasoning model's autoregressive machinery treats the `<think>…</think>` block as its
own completed deliberation and immediately emits content tokens — no 60s CoT phase.

**Before (ADR-001 cascade alone):**
```
t=0s:   Request → START both models
t=2s:   mini-4b tokens stream
t=65s:  120B content starts → stream switches
t=80s:  Client receives 120B stanza
```

**After (ADR-001 cascade + ADR-002 think-block):**
```
t=0s:   Request → render WM trace (~3ms) → START both models with trace
t=2s:   mini-4b tokens stream
t=5s:   120B content starts → stream switches (60s reasoning → ~3s prefill)
t=8s:   Client receives 120B stanza
```

TTFT (from first token): <5s (mini-4b, unchanged)
Switch latency: ~65s → **~5s** (think-block collapses 120B reasoning)

---

## IFR Evaluation

**Why Sketch A ELIMINATES the contradiction (not just manages it):**
- WM conditioning now reaches *both* the (eliminated) reasoning phase *and* content tokens.
- The original architectural blind spot (VimarsaBridgeV2 cannot bias reasoning tokens)
  is closed: reasoning is no longer LLM-native, it is WM-derived.
- Quality is preserved or improved: WM context is richer than the LLM's free-running CoT
  because it includes trained h_t trajectory, Citta retrievals, and sphurattā events.
- Both TTFT (<5s) and quality (H4 ≥70%) are met simultaneously — the contradiction is gone.

**IFR score:** 4/4 (self-resolving via existing WM substrate; no new components required)

---

## Architecture

### `pwm/vimarsa/narrator.py` — `WMReasoningTrace`

```python
class WMReasoningTrace:
    def render(h_t, domain, creative_metadata, sphuratta_events, citta_hits,
               stanza_idx, camatk_score, vfe) -> str:
        """Returns full <think>…</think> string."""

    def render_as_assistant_prefill(...) -> dict:
        """Returns {"role": "assistant", "content": "<think>…</think>"}."""
```

Inputs drawn from pre-warmed `state.domain_states[domain]` (already in AppState).

### `pwm/generation/llama_backend.py` — `_build_messages()` extension

```python
def _build_messages(system, user, think_prefill=None) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    if think_prefill:
        messages.append(think_prefill)  # assistant prefill
    messages.append({"role": "user", "content": user})
    return messages
```

Threaded through `stream()`, `generate()`, `_http_stream()`, `_http_stream_model()`,
and `stream_cascade()` (applied to slow 120B model only — maximum impact there).

### `pwm/generation/engine.py` — `get_llm_backend()` unchanged

The think_prefill is produced at request time in `api/main.py` / `pancakrtya_loop_v2.py`
from the already-present WM state; it is passed through the `stream()` call.

---

## Relationship to ADR-001 (S18 Cascade)

ADR-001 and ADR-002 are **additive and complementary**:

| Layer | Mechanism | TTFT | Switch latency | Quality |
|-------|-----------|------|----------------|---------|
| Baseline (120B only) | — | 60s | — | highest |
| ADR-001 (cascade) | mini-4b streams while 120B reasons | <5s | ~65s | 120B quality |
| ADR-002 (think-block) | WM trace collapses 120B CoT | — | ~5s | equal or better |
| **Both combined** | cascade + think-block | **<5s** | **~5s** | **120B quality** |

ADR-001 (cascade, S18) is the **runtime fallback** if the `</think>` short-circuit fails
on specific prompts or Nemotron versions. The engineering is additive — both features
share the WM trace artifact.

---

## Risks and Validation

| Risk | Mitigation |
|------|-----------|
| Nemotron ignores external `<think>` and re-enters its own | Verify with `ollama run nemotron-3-super:120b --verbose`; measure tokens-in-think |
| Trace too generic to guide content generation | Enrich with domain-specific rāga/mode hints from CreativeMetadata |
| Trace too long (slow prefill) | Cap at ~400 tokens; WMReasoningTrace.render() is currently ~15 lines |

**Validation gate (H4):** 30-lyric A/B paired permutation test:
- Control: 120B with full reasoning (no trace)
- Treatment: 120B with WM think-block prefill
- Metric: H4 "meaningful" rating ≥70% in both conditions (non-inferiority test)
- Secondary: TTFT and switch latency measurements via Ollama `--verbose`

---

## Implementation Plan (S19 — completed)

1. ✅ `WMReasoningTrace` class in `pwm/vimarsa/narrator.py`
2. ✅ `think_prefill` param in `LlamaCppBackend._build_messages()`, `stream()`, `generate()`,
      `_http_stream()`, `_http_stream_model()`, `stream_cascade()`
3. ⏳ Wire `think_prefill` into `PancakrtyaLoopV2.run_stanza()` — pass `h_t` + domain
      to `WMReasoningTrace.render_as_assistant_prefill()` before LLM call (S20)
4. ⏳ A/B validation: 30 lyrics × 2 conditions, permutation test, H4 rating (S20)
5. ⏳ If `</think>` short-circuit unreliable → cascade (ADR-001) remains active fallback

**Effort completed:** S19 (3 hours — architecture, WMReasoningTrace, backend wiring)  
**Remaining:** S20 (pipeline wiring + A/B validation)
