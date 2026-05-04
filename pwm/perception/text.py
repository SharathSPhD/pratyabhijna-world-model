"""
TextEncoder: BPE tokeniser + sentence embedding for creative text observations.

Philosophical grounding:
  Śabda (Bhartṛhari, Vākyapadīya 1.1): 'Word as Brahman' — language is not merely
  symbolic but the direct expression of Cit. The text encoder translates śabda
  into the latent continuum (obs_dim=512) that the RSSM can recognise (pratyabhijñā).

  Sentence-transformers provide the ālayavijñāna embedding space — a pre-trained
  semantic manifold that grounds creative text before the WM fine-tunes on top.

Architecture:
  Phase 0–2: sentence-transformers `all-MiniLM-L6-v2` (dim=384) → Linear(384, obs_dim)
  Phase 5+: replace with domain-adapted encoder from corpus fine-tuning

Observation tensor shape: (B, obs_dim=512) — matches TrikaCoreLevel obs_dim.
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor


class TextEncoder(nn.Module):
    """
    Tokenise and embed creative text into WM observation space.

    Wraps sentence-transformers for sentence-level embeddings.
    Lazy-loads model on first call to avoid startup cost.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, CPU+GPU

    def __init__(self, obs_dim: int = 512, model_name: str | None = None) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.model_name = model_name or self.DEFAULT_MODEL
        self._st_model: Any = None  # sentence_transformers.SentenceTransformer

        # Projection from embedding dim → RSSM obs_dim
        # Embedding dim known at first forward call; set lazily
        self.proj: nn.Linear | None = None

    def _load_model(self) -> None:
        """Lazy-load sentence-transformer (avoids import cost at startup)."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._st_model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )

    def encode_text(self, texts: list[str], device: torch.device) -> Tensor:
        """
        Encode a list of text strings into observation tensors.

        Args:
            texts: list of B strings
            device: target device
        Returns:
            (B, obs_dim) float tensor
        """
        if self._st_model is None:
            self._load_model()

        # sentence-transformers encode: returns numpy (B, embed_dim)
        embeddings = self._st_model.encode(  # type: ignore[union-attr]
            texts, convert_to_tensor=True, device=device, show_progress_bar=False
        )  # (B, embed_dim)

        embed_dim = embeddings.shape[-1]
        if self.proj is None:
            self.proj = nn.Linear(embed_dim, self.obs_dim, bias=False).to(device)
            nn.init.orthogonal_(self.proj.weight)

        return self.proj(embeddings.float())

    def forward(self, texts: list[str], device: torch.device | None = None) -> Tensor:
        dev = device or torch.device("cpu")
        return self.encode_text(texts, dev)


class TokenSequenceEncoder(nn.Module):
    """
    Token-level encoder for sequence modelling (Phase 3+).

    Uses a learnable embedding table + positional encoding.
    Produces per-token embeddings for RNN/Mamba input.
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        embed_dim: int = 512,
        max_seq_len: int = 512,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        token_ids: (B, T) long tensor
        Returns: (B, T, embed_dim)
        """
        _, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        return self.norm(self.embed(token_ids) + self.pos_embed(positions))
