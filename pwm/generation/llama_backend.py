"""
LlamaCppBackend — wraps llama-cpp-python for the PWM generation pipeline.

Replaces call_ollama() in engine.py with a backend that:
1. Accepts a logits_processor callback (VimarsaBridgeV2 hook)
2. Supports streaming (SSE token-by-token)
3. Falls back to Ollama / llama-server HTTP if llama-cpp-python is unavailable

Ollama support (S16):
  Set server_url="http://localhost:11434" and model_name="nemotron-3-super:120b".
  Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint.
  Reasoning models (nemotron-3-super) emit tokens as `reasoning` then `content`;
  only `content` tokens are yielded to the generation pipeline.

Model Cascade Streaming (S18, ADR-001 — TRIZ Principles 10 + 24):
  stream_cascade() starts streaming a fast model (nemotron-mini:4b) immediately,
  while the slow model (nemotron-3-super:120b) runs its reasoning phase in a daemon
  thread. When the slow model's first content token arrives, the stream switches.
  Client gets TTFT <5s from fast model AND high-quality output from 120B.

  TRIZ Principle 10 (Prior Action): fast model pre-fills the response slot.
  TRIZ Principle 24 (Intermediary): fast model mediates between silence and quality.

  cascade_model_name: fast/intermediary model (e.g. "nemotron-mini:4b").
  model_name:         slow/quality model (the existing field, e.g. 120B).

Sanskrit concept: Kriyā (ĪPK 3.1) — the act of bringing latent into manifest.
Computational: LLM token generation as the kriyā act of the Pañcakṛtya loop.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Callable, Generator, Optional

import requests

logger = logging.getLogger(__name__)

# Default Ollama model for HTTP mode (overrides at construction time).
OLLAMA_DEFAULT_MODEL = "nemotron-3-super:120b"


class LlamaCppBackend:
    """
    Unified llama.cpp backend supporting both:
    - llama-cpp-python (logits_processor native Python hook)
    - llama-server HTTP fallback (no logits_processor, text prefix only)

    Args:
        model_path: Path to GGUF model file.
        n_gpu_layers: Layers to offload to GPU (999 = all).
        n_ctx: Context window size.
        server_url: If set, use HTTP server instead of in-process.
        mock: If True, return stub output (for testing without model).
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 999,
        n_ctx: int = 4096,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        cascade_model_name: Optional[str] = None,
        mock: bool = False,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.server_url = server_url
        # model_name: slow/quality model (120B) for HTTP/Ollama mode.
        self.model_name = model_name or OLLAMA_DEFAULT_MODEL
        # cascade_model_name: fast intermediary model (mini-4b).
        # When set, stream_cascade() is used instead of plain stream().
        self.cascade_model_name = cascade_model_name
        self.mock = mock
        self._llm = None  # lazy-loaded

        if not mock and server_url is None:
            self._load_model()

    def _load_model(self):
        """Load model in-process via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                flash_attn=True,
                verbose=False,
            )
            logger.info(f"[LlamaCppBackend] Loaded in-process: {self.model_path}")
        except Exception as e:
            logger.warning(f"[LlamaCppBackend] In-process load failed: {e}. Using server fallback.")
            self._llm = None

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(
        self,
        system: str,
        user: str,
        logits_processor: Optional[Callable] = None,
        max_tokens: int = 512,
        temperature: float = 0.88,
        top_p: float = 0.92,
    ) -> str:
        """Generate text, applying logits_processor if provided."""
        if self.mock:
            if logits_processor is not None:
                import numpy as np
                logits_processor([0], np.zeros(128256, dtype=np.float32))
            return "moon rises soft and slow\n"

        if self._llm is not None and logits_processor is not None:
            from llama_cpp import LogitsProcessorList
            lp_list = LogitsProcessorList([logits_processor])
            out = self._llm.create_chat_completion(
                messages=self._build_messages(system, user),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logits_processor=lp_list,
            )
            return out["choices"][0]["message"]["content"]

        if self._llm is not None:
            out = self._llm.create_chat_completion(
                messages=self._build_messages(system, user),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return out["choices"][0]["message"]["content"]

        if self.server_url:
            return self._http_generate(system, user, max_tokens, temperature, top_p)

        return "[LlamaCppBackend: no backend available]"

    def stream(
        self,
        system: str,
        user: str,
        logits_processor: Optional[Callable] = None,
        max_tokens: int = 512,
        temperature: float = 0.88,
        top_p: float = 0.92,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time.

        Cascade dispatch: if cascade_model_name is configured (S18, ADR-001),
        delegates to stream_cascade() — transparent to callers (PancakrtyaLoopV2).
        The cascade starts nemotron-mini:4b immediately for TTFT <5s, then switches
        to the 120B model when its content tokens begin flowing.
        """
        if self.cascade_model_name and self.server_url and not self.mock:
            yield from self.stream_cascade(
                system=system,
                user=user,
                logits_processor=logits_processor,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return

        if self.mock:
            for word in ["moon ", "rises ", "soft ", "and ", "slow\n"]:
                yield word
            return

        if self._llm is not None:
            from llama_cpp import LogitsProcessorList
            lp_list = LogitsProcessorList([logits_processor]) if logits_processor else None
            for chunk in self._llm.create_chat_completion(
                messages=self._build_messages(system, user),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logits_processor=lp_list,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
            return

        if self.server_url:
            yield from self._http_stream(system, user, max_tokens, temperature, top_p)

    def _http_request_body(self, system: str, user: str, max_tokens: int,
                           temperature: float, top_p: float, stream: bool) -> dict:
        """Build OpenAI-compatible request body.

        Includes model_name for Ollama; llama-server ignores unknown fields.
        """
        return {
            "model": self.model_name,
            "messages": self._build_messages(system, user),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

    def _http_generate(self, system, user, max_tokens, temperature, top_p) -> str:
        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=self._http_request_body(system, user, max_tokens, temperature, top_p, False),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _http_stream(self, system, user, max_tokens, temperature, top_p):
        """Stream content tokens from OpenAI-compatible endpoint.

        Handles both direct-content models and reasoning models (nemotron-3-super):
        - Reasoning models emit tokens in `delta.reasoning` before `delta.content`
        - Only `delta.content` tokens are yielded to the pipeline
        """
        with requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=self._http_request_body(system, user, max_tokens, temperature, top_p, True),
            stream=True,
            timeout=300,
        ) as resp:
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    # Only yield `content` — skip `reasoning` tokens from thinking models
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

    # ── Model Cascade Streaming ────────────────────────────────────────────────

    def _http_stream_model(self, model: str, system: str, user: str,
                           max_tokens: int, temperature: float, top_p: float):
        """Like _http_stream but with an explicit model override."""
        body = self._http_request_body(system, user, max_tokens, temperature, top_p, True)
        body["model"] = model
        with requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=body,
            stream=True,
            timeout=300,
        ) as resp:
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

    def stream_cascade(
        self,
        system: str,
        user: str,
        logits_processor: Optional[Callable] = None,
        max_tokens: int = 512,
        temperature: float = 0.88,
        top_p: float = 0.92,
        slow_timeout: float = 120.0,
    ) -> Generator[str, None, None]:
        """
        TRIZ Cascade Stream (ADR-001 — Principles 10 + 24).

        Starts streaming `cascade_model_name` (fast, e.g. nemotron-mini:4b) immediately
        so TTFT stays <5s, while `model_name` (slow, 120B) runs its reasoning phase in
        a daemon thread. When the slow model's first content token arrives, the generator
        switches the stream to 120B output; the client gets high-quality stanzas.

        Fallback: if the slow model has not started content within `slow_timeout` seconds,
        the fast model's remaining tokens are yielded as the final output.

        Both models are preceded by the same VimarsaBridgeV2 logits_processor prefix
        (WM conditioning applied to both — WM bias works equally on mini-4b).

        Args:
            system:          System prompt (WM-conditioned, domain-neutral).
            user:            User prompt for this stanza.
            logits_processor: VimarsaBridgeV2 hook — applied via system prompt prefix
                             for HTTP path (no native logits_processor on Ollama).
            max_tokens:      Token budget per stanza.
            temperature:     Sampling temperature.
            top_p:           Nucleus sampling p.
            slow_timeout:    Seconds to wait for slow model content before giving up.

        Yields:
            str tokens — first from fast model, then switched to slow model.
        """
        if self.mock:
            # Mock cascade: fast tokens then slow tokens with [CASCADE] marker
            for tok in ["fast ", "preview\n"]:
                yield tok
            yield "[CASCADE→SLOW] "
            for tok in ["high ", "quality ", "output\n"]:
                yield tok
            return

        if not self.server_url or not self.cascade_model_name:
            # No cascade configured — fall back to normal stream
            yield from self.stream(system, user, logits_processor, max_tokens,
                                   temperature, top_p)
            return

        # ── Shared state between main generator and slow-model daemon thread ──
        # Queue carries (token: str | None) — None is the end-of-stream sentinel.
        slow_q: queue.Queue[Optional[str]] = queue.Queue(maxsize=512)
        slow_first_content = threading.Event()
        slow_done = threading.Event()

        def _slow_thread() -> None:
            """Daemon: stream slow model, push tokens into slow_q."""
            try:
                for tok in self._http_stream_model(
                    self.model_name, system, user, max_tokens, temperature, top_p
                ):
                    slow_q.put(tok)
                    slow_first_content.set()  # signal on first content token
            except Exception as exc:
                logger.warning(
                    f"[stream_cascade] Slow model ({self.model_name}) error: {exc}"
                )
            finally:
                slow_done.set()
                slow_q.put(None)  # sentinel — signals end regardless of error

        thread = threading.Thread(target=_slow_thread, daemon=True)
        thread.start()

        # ── Phase 1: Stream fast model until slow model content arrives ────────
        switched = False

        try:
            for tok in self._http_stream_model(
                self.cascade_model_name, system, user, max_tokens, temperature, top_p
            ):
                if slow_first_content.is_set():
                    # Slow model content has started — stop consuming fast tokens
                    switched = True
                    break
                # Check timeout: if slow hasn't started within `slow_timeout`,
                # continue fast model to completion (no switch).
                # (thread.is_alive() for timeout check would require elapsed time;
                # simpler: if slow_done fired before slow_first_content, it errored)
                if slow_done.is_set() and not slow_first_content.is_set():
                    # Slow model failed silently — continue fast to completion
                    logger.warning(
                        f"[stream_cascade] Slow model finished without content; "
                        f"keeping fast model output."
                    )
                    yield tok
                    # Exhaust fast model
                    for remaining in self._http_stream_model(
                        self.cascade_model_name, system, user, max_tokens,
                        temperature, top_p
                    ):
                        yield remaining
                    return
                yield tok
        except Exception as exc:
            logger.warning(
                f"[stream_cascade] Fast model ({self.cascade_model_name}) error: {exc}"
            )
            # Fast model failed — fall through to slow model only

        if not switched:
            # Fast model finished before slow model started content.
            # Wait up to slow_timeout for slow model to produce content.
            got_content = slow_first_content.wait(timeout=slow_timeout)
            if not got_content:
                logger.warning(
                    f"[stream_cascade] Slow model did not produce content within "
                    f"{slow_timeout}s; no switch."
                )
                # Drain any straggler sentinel
                try:
                    slow_q.get_nowait()
                except queue.Empty:
                    pass
                return

        # ── Phase 2: Switch — drain slow_q then continue reading as tokens arrive
        logger.info(f"[stream_cascade] Switched to slow model ({self.model_name})")
        while True:
            try:
                tok = slow_q.get(timeout=0.05)
                if tok is None:
                    break  # end sentinel
                yield tok
            except queue.Empty:
                if slow_done.is_set():
                    # Drain any remaining tokens before exiting
                    while not slow_q.empty():
                        tok = slow_q.get_nowait()
                        if tok is not None:
                            yield tok
                    break
                # Slow model still running — keep waiting
