# ADR-001: C4 — Model Cascade Streaming (TRIZ Principle 10 + 24)

**Status:** Accepted (implementation planned for Phase 7)  
**Date:** 2026-05-17  
**Sprint:** S17 review output  

---

## Context: C4 Contradiction

**Improving:** TTFT (Time To First Content Token) — neo-fm-web UX target <5s  
**Worsening:** Creative quality — nemotron-mini:4b produces generic output; nemotron-3-super:120b requires ~60s reasoning phase before first content token

**TRIZ Parameters:**  
- Speed (Parameter 9) improving → Quality of manufactured object (Parameter 12) worsening  
- Matrix recommendation: Principles 10 (Prior Action), 24 (Intermediary), 35 (Parameter Changes)  

**Tried and failed:**
- `think:false` API param — ignored by nemotron-3-super:120b via Ollama; still emits reasoning tokens
- System prompt coaching ("answer directly without thinking") — reduces but doesn't eliminate reasoning

---

## Decision: Model Cascade Streaming

**TRIZ Principle 10 (Prior Action) + Principle 24 (Intermediary):**

Start streaming `nemotron-mini:4b` tokens immediately (TTFT <5s) while `nemotron-3-super:120b` runs its reasoning phase in the background. When the 120B model's content tokens begin flowing, switch the stream to the 120B output. Client gets immediate response AND high-quality final content.

```
Timeline:
  t=0s: Request arrives
  t=0s: START both models simultaneously
  t=2s: nemotron-mini tokens arrive → STREAM immediately
  t=5s: client sees first lyrics line (TTFT met)
  t=65s: nemotron-3-super content starts → SWITCH stream
  t=80s: client gets full high-quality stanza from 120B

vs today:
  t=0s: Request arrives
  t=0-60s: nemotron-3-super reasoning (client waits)
  t=60s: first content token → TTFT = 60s (too slow)
```

**Fallback logic:**
- If 120B reasoning takes >120s (timeout), yield rest of mini-4b output as final
- WM conditioning (VimarsaBridgeV2) applied to BOTH models — mini-4b benefits from WM bias too

---

## Architecture Impact

**LlamaCppBackend changes (S18):**
```python
class LlamaCppBackend:
    def __init__(self, ..., cascade_model_name: Optional[str] = None):
        self.cascade_model_name = cascade_model_name  # e.g. "nemotron-mini:4b"
    
    def stream_cascade(self, system, user, ...) -> Generator[str, None, None]:
        """Stream fast model immediately; switch to slow model when content arrives."""
        import threading
        slow_buf: list[str] = []
        slow_started = threading.Event()
        
        # Start slow model in background thread
        def _slow_thread():
            for tok in self._http_stream(system, user, ...):
                slow_buf.append(tok)
                slow_started.set()
        threading.Thread(target=_slow_thread, daemon=True).start()
        
        # Stream fast model until slow model starts
        for tok in self._fast_stream(system, user, ...):
            if slow_started.is_set():
                break  # switch to slow model
            yield tok
        
        # Flush accumulated slow tokens, then continue streaming slow
        yield from slow_buf
        while not slow_done.is_set():
            if slow_buf:
                yield slow_buf.pop(0)
```

**Config:** Add `cascade_model_name: "nemotron-mini:4b"` to `llm_backend.yaml`.

---

## Alternatives Considered

1. **nemotron-mini only** (current fast path): Simple, <5s TTFT, but 4B quality is noticeably lower. WM conditioning partially compensates.

2. **Prompt coaching** to suppress 120B reasoning: Tested — insufficient. Model continues reasoning with shorter chain.

3. **TRIZ Principle 35 (Parameter Changes)**: Increase VimarsaBridgeV2 bias scale on nemotron-mini:4b to compensate for quality gap. Orthogonal to latency fix — implement alongside cascade.

4. **Speculative decoding at token level**: Ollama doesn't expose draft-model speculative decoding for non-llama-family models. Nemotron is nemotron_h_moe family — not compatible.

---

## Implementation Plan (Phase 7, S18)

1. Add `stream_cascade()` to `LlamaCppBackend` — ~80 lines
2. Add `cascade_model_name` to `engine.py` config
3. Wire `ws_generate` and `_event_stream` to use `stream_cascade` when `cascade_model_name` set
4. Tests: cascade correctly switches from fast to slow stream; no duplicate tokens; WM bias applied to both
5. Gate metric: first token < 5s AND final stanza camatk_score ≥ nemotron-mini standalone

**Effort:** S18 (1 week)  
**Risk:** Threading complexity in asyncio context — use thread-safe queue, not list

---

## Status of C4 After This ADR

- Immediate term: nemotron-mini:4b as default (fast, WM-conditioned, <5s TTFT)
- Phase 7 S18: cascade streaming (fast TTFT + 120B quality at stanza end)
- `get_llm_backend()` in engine.py: `cascade_model_name="nemotron-mini:4b"` when TTFT <5s required, None for research/high-quality mode
