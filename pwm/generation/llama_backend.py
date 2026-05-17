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

Sanskrit concept: Kriyā (ĪPK 3.1) — the act of bringing latent into manifest.
Computational: LLM token generation as the kriyā act of the Pañcakṛtya loop.
"""
from __future__ import annotations

import json
import logging
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
        mock: bool = False,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.server_url = server_url
        # model_name for HTTP/Ollama mode (required by Ollama, optional for llama-server)
        self.model_name = model_name or OLLAMA_DEFAULT_MODEL
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
        """Stream tokens one at a time."""
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
