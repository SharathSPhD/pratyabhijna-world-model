"""
VimarsaBridge: Cross-attention bridge between WM hidden state and LLM token space.

Philosophical grounding:
  Vimarśa (ĪPK 1.5.11, Utpaladeva; TĀ 1.24, Abhinavagupta):
  'Self-reflexive luminosity' — the act by which Śiva recognises his own
  manifestation in the creative object. The bridge is the computational locus
  where the world model's continuous latent h_t becomes the LLM's context,
  enabling pratyabhijñā (recognition) of the sphurattā event.

  The bridge implements āgama (scriptural testimony, TĀ 1.18): the LLM is the
  frozen āgama text; cross-attention translates WM features into LLM prefix
  tokens. Only fires at sphurattā events (cf. PañcakṛtyaLoop jñāna step).

Architecture:
  h_t (hidden_dim) → Linear → k cross-attention keys (k=4, key_dim=64)
  LLM prefix: k soft-prompt tokens of dim=llm_embed_dim
  Phase 5+: replaces zero-context LLM calls in PañcakṛtyaLoop.

  Cross-attention is read-only: WM→LLM; no gradient flows back to frozen LLM.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class VimarsaBridge(nn.Module):
    """
    Maps WM hidden state h_t to soft-prompt prefix tokens for LLM conditioning.

    Creates k learnable prefix tokens whose values are modulated by h_t via
    cross-attention. The LLM prepends these as its context window prefix.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        llm_embed_dim: int = 4096,
        n_prefix_tokens: int = 4,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.llm_embed_dim = llm_embed_dim

        # Learnable query tokens (what the LLM "asks" the WM)
        self.query_tokens = nn.Parameter(torch.randn(n_prefix_tokens, llm_embed_dim) * 0.02)

        # WM → cross-attention key/value projection
        self.wm_key_proj = nn.Linear(hidden_dim, llm_embed_dim)
        self.wm_val_proj = nn.Linear(hidden_dim, llm_embed_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(llm_embed_dim)

    def forward(self, h: Tensor) -> Tensor:
        """
        Generate soft-prompt prefix conditioned on WM hidden state h.

        Args:
            h: (B, hidden_dim) WM hidden state at sphurattā event
        Returns:
            prefix: (B, n_prefix_tokens, llm_embed_dim) — prepend to LLM input
        """
        B = h.shape[0]

        # Queries: expand learnable tokens for batch
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, k, llm_dim)

        # Keys/values from WM hidden state (single token attending from WM)
        k = self.wm_key_proj(h).unsqueeze(1)   # (B, 1, llm_dim)
        v = self.wm_val_proj(h).unsqueeze(1)   # (B, 1, llm_dim)

        # Cross-attend: prefix tokens query WM state
        attn_out, _ = self.cross_attn(q, k, v)  # (B, k, llm_dim)
        prefix = self.norm(q + attn_out)          # residual + norm

        return prefix

    def format_prefix_text(self, h: Tensor, tokenizer: object | None = None) -> str:  # noqa: ARG002
        """
        Format WM state as text prefix for text-based LLMs (no tokenizer required).

        Returns a condensed description of the WM state to prepend as system context.
        Used when LLM backend doesn't support soft prompts (text-only mode).
        """
        with torch.no_grad():
            h_mean = h.mean(0)  # collapse batch
            energy = float(h_mean.norm().item())
            top_dims = h_mean.abs().topk(3).indices.tolist()

        return (
            f"[WM state: energy={energy:.3f}, active_dims={top_dims}] "
            f"A creative sphurattā event has been detected. "
        )
