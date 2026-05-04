"""
MemoryAgent: Post-commit consolidation agent.

Philosophical grounding:
  Smṛti (PHṛ sūtra 9, Kṣemarāja): Retentive memory — the ālayavijñāna's capacity
  to hold latent impressions (saṃskāras) and make them available for future
  recognition. MemoryAgent fires after VimarshaAgent commits a narration to
  consolidate the event into the CittaStore semantic bank.

  This is NOT a full smolagents agent (those are reserved for vimarśa deliberation).
  It is a lightweight post-commit hook operating on the shared CittaStore.

Architecture:
  Triggers: VimarshaAgent.run() returns committed=True
  Actions:
    1. Store h_t into CittaStore episodic bank (smṛti)
    2. Retrieve similar past episodes (top-k similarity)
    3. If camatk_score > semantic_threshold: also store in semantic bank (ālayavijñāna)
    4. Update ReplayBuffer priority for this transition
"""

from __future__ import annotations
from typing import Any
import torch
from torch import Tensor


class MemoryAgent:
    """
    Post-commit memory consolidation: episodic + conditional semantic storage.

    Called by PañcakṛtyaLoop after a successful vimarśa commit.
    Not a smolagents agent — operates synchronously on the shared call stack.
    """

    SEMANTIC_THRESHOLD = 0.75  # camatk_score threshold for semantic consolidation

    def __init__(
        self,
        citta_store: Any,          # CittaStore
        replay_buffer: Any = None, # ReplayBuffer
        n_levels: int = 3,
    ) -> None:
        self.citta = citta_store
        self.buf = replay_buffer
        self.n_levels = n_levels
        self._commit_count = 0

    def consolidate(
        self,
        h: Tensor,
        z: Tensor,  # reserved for level-specific latent storage (Phase 5+)  # noqa: ARG002
        camatk_score: float,
        step: int,
        transition_idx: int | None = None,
    ) -> dict[str, Any]:
        """
        Consolidate a sphurattā event into CittaStore.

        Args:
            h: (B, hidden_dim) WM hidden state at sphurattā
            z: (B, stoch_dim, n_cats) latent sample
            camatk_score: committed event's camatkāra score
            step: current training step
            transition_idx: ReplayBuffer index to update priority (if available)
        Returns:
            dict with stored_episodic, stored_semantic, retrieved_similar
        """
        # Store in episodic memory for ALL levels (using level 0 representative)
        h_mean = h.mean(0, keepdim=True)  # (1, hidden_dim) — batch mean
        self.citta.store_episodic(h_mean, level=0)

        # Check if semantic consolidation is warranted
        stored_semantic = False
        if camatk_score >= self.SEMANTIC_THRESHOLD:
            self.citta.store_semantic(h_mean, level=0)
            stored_semantic = True

        # Retrieve top-3 similar past episodes for context coherence
        retrieved = self.citta.retrieve(h_mean, level=0, top_k=3)
        similarity = float(
            torch.nn.functional.cosine_similarity(h_mean, retrieved).mean().item()
        )

        # Update replay priority if buffer provided
        if self.buf is not None and transition_idx is not None:
            import numpy as np
            self.buf.update_priorities(
                [transition_idx], np.array([camatk_score + 1e-6])
            )

        self._commit_count += 1

        return {
            "stored_episodic": True,
            "stored_semantic": stored_semantic,
            "similarity_to_past": similarity,
            "total_commits": self._commit_count,
            "step": step,
        }

    def retrieve_context(self, h: Tensor, level: int = 0, top_k: int = 5) -> Tensor:
        """Retrieve top-k similar memories for context conditioning."""
        return self.citta.retrieve(h.mean(0, keepdim=True), level=level, top_k=top_k)
