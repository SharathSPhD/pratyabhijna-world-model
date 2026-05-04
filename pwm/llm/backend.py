"""
LLMBackend: Provider-agnostic LLM interface via LiteLLM.

Philosophical grounding:
  Āgama (PHṛ sūtra 2, Kṣemarāja): 'received scriptural knowledge' — the LLM is the
  modern āgama layer, encyclopaedic but static. This backend routes calls to whichever
  āgama instance is configured, without coupling the WM code to any specific provider.

  The LLM is called ONLY at: jñāna slow path, kriyā fluency pass, vimarśa deliberation,
  sphurattā narration. Never per-step. The WM computes; the LLM narrates.
"""

from __future__ import annotations
import os
from typing import Any

try:
    import litellm  # type: ignore[import]
    litellm.drop_params = True
    HAS_LITELLM = True
    _litellm = litellm
except ImportError:
    HAS_LITELLM = False
    _litellm = None  # type: ignore[assignment]


ROLE_TIER: dict[str, str] = {
    "vimarsha": "primary",   # deep reasoning — 120B or equivalent
    "memory":   "primary",
    "sleep":    "primary",
    "jnana":    "fast",      # knowledge call — 49B or smaller
    "kriya":    "fast",
    "ananda":   "fast",
    "icha":     "fast",
}


class LLMBackend:
    """
    Unified LLM call interface via LiteLLM.
    Provider is set by config key llm.provider (or LLM_PROVIDER env var).
    Switching providers requires only a config change — no code changes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if not HAS_LITELLM:
            raise ImportError("litellm not installed. Run: pip install litellm")
        self.cfg = config
        self.provider = config.get("provider", os.getenv("LLM_PROVIDER", "nemotron-local"))
        self._provider_cfg = config.get(self.provider, {})

    def call(
        self,
        role: str,
        system: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Make an LLM call routed by role → primary/fast tier.

        Args:
            role: one of vimarsha/memory/sleep/jnana/kriya/ananda/icha
            system: system prompt (should include Sākṣī witness invariant)
            prompt: user prompt
            temperature: override; uses config default if None
            max_tokens: override; uses config default if None
        Returns:
            Generated text string
        """
        tier = ROLE_TIER.get(role, "fast")
        model_cfg = self._provider_cfg.get(tier, self._provider_cfg.get("fast", {}))

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        response = _litellm.completion(  # type: ignore[union-attr]
            model=model_cfg.get("model", "gpt-4o-mini"),
            messages=messages,
            api_base=model_cfg.get("api_base"),
            api_key=model_cfg.get("api_key"),
            temperature=temperature or model_cfg.get("temperature", 0.7),
            max_tokens=max_tokens or model_cfg.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    def encode(self, text: str) -> list[float]:
        """Encode text to embedding (for goal specification). Uses fast tier."""
        model_cfg = self._provider_cfg.get("fast", {})
        response = _litellm.embedding(  # type: ignore[union-attr]
            model=model_cfg.get("embedding_model", model_cfg.get("model", "text-embedding-3-small")),
            input=[text],
            api_base=model_cfg.get("api_base"),
            api_key=model_cfg.get("api_key"),
        )
        return response.data[0].embedding

    @classmethod
    def from_env(cls) -> "LLMBackend":
        """Construct from environment variables (for scripts)."""
        from dotenv import load_dotenv
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "nemotron-local")
        config = {
            "provider": provider,
            provider: {
                "primary": {
                    "model": os.getenv("LLM_PRIMARY_MODEL", "openai/nemotron-120b"),
                    "api_base": os.getenv("LLM_PRIMARY_API_BASE", "http://localhost:8000/v1"),
                    "api_key": os.getenv("LLM_PRIMARY_API_KEY", "local"),
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                "fast": {
                    "model": os.getenv("LLM_FAST_MODEL", "openai/nemotron-30b"),
                    "api_base": os.getenv("LLM_FAST_API_BASE", "http://localhost:8001/v1"),
                    "api_key": os.getenv("LLM_FAST_API_KEY", "local"),
                    "temperature": 0.9,
                    "max_tokens": 512,
                },
            },
        }
        return cls(config)
