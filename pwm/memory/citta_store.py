"""
CittaStore: Dual-mode Hopfield associative memory.

Philosophical grounding:
  Citta (Buddhist/Śaiva synthesis): 'Mind-stream' — the continuous accumulation of
  mental impressions (saṃskāras). The Hopfield network implements two modes:

  Smṛti (Episodic, high β): 'Memory as recognition' — sharp recall of specific
    episodes. High inverse temperature → single attractor → specific past pattern.
    Reference: YS 1.11 (smṛtiḥ paribhraṣṭa-viṣayā), Abhinavagupta's ViSp 3.1.

  Ālayavijñāna (Semantic, low β): 'Store-consciousness' — blended, diffuse
    representations. Low β → many attractors averaging → concept generalisation.
    Reference: Yogācāra vijñāna-mātra, Vasubandhu's Triṃśikā 5.

  Modern Hopfield networks (Ramsauer et al. 2020) with energy:
    E = -β * log ∑_i exp(β * q·kᵢ - β²/2)
  achieve exponential storage capacity M ~ exp(d/2) vs classical O(d²/log d).

Architecture:
  - One CittaStoreLevel per Trika level
  - Each level has two Hopfield banks: episodic + semantic
  - Storage: `store(pattern)` appends to a rolling buffer
  - Retrieval: `recall(query, mode)` returns attractor state
  - Sleep consolidation reads/writes both banks
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import deque


class HopfieldBank(nn.Module):
    """
    Modern continuous Hopfield network (Ramsauer et al. 2020).

    Stores patterns as key matrix; retrieval is one step of Hopfield energy descent.
    """

    def __init__(
        self,
        dim: int,
        max_patterns: int = 1024,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_patterns = max_patterns
        self.beta = beta

        # Pattern store — not a trainable parameter (runtime buffer)
        self._patterns: deque[Tensor] = deque(maxlen=max_patterns)

    def store(self, pattern: Tensor) -> None:
        """Store a new pattern (h_t or z_flat)."""
        self._patterns.append(pattern.detach().cpu())

    def recall(self, query: Tensor) -> Tensor:
        """
        Retrieve attractor for query via one-step modern Hopfield update.

        Args:
            query: (B, dim) query vector (current h or z)
        Returns:
            retrieved: (B, dim) attractor state
        """
        if not self._patterns:
            return query  # nothing stored yet — identity recall

        # Stack patterns into key matrix: (M, dim)
        keys = torch.stack(list(self._patterns), dim=0).to(query.device)  # (M, dim)

        # Attention: softmax(β * q·K^T) K  [Ramsauer et al. eq. 2]
        # query: (B, dim), keys: (M, dim) → scores: (B, M)
        scores = self.beta * (query @ keys.T)  # (B, M)
        attn = F.softmax(scores, dim=-1)        # (B, M)
        retrieved = attn @ keys                  # (B, dim)
        return retrieved

    def capacity(self) -> int:
        return len(self._patterns)

    def clear(self) -> None:
        self._patterns.clear()

    def get_entropy(self) -> float:
        """Hopfield retrieval entropy — used in sphurattā detection."""
        if not self._patterns:
            return 0.0
        # Estimate entropy from pairwise similarity of stored patterns
        keys = torch.stack(list(self._patterns), dim=0).float()
        norms = F.normalize(keys, dim=-1)
        sim = (norms @ norms.T).abs().mean().item()
        # High similarity → low entropy (concentrated memory)
        return float(1.0 - sim)


class CittaStoreLevel(nn.Module):
    """
    Dual-mode Hopfield store for one Trika level.

    Episodic bank (high β): smṛti — sharp recall.
    Semantic bank (low β): ālayavijñāna — blended concepts.
    """

    def __init__(
        self,
        dim: int,
        max_episodic: int = 512,
        max_semantic: int = 256,
        beta_episodic: float = 4.0,
        beta_semantic: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.episodic = HopfieldBank(dim, max_episodic, beta_episodic)
        self.semantic = HopfieldBank(dim, max_semantic, beta_semantic)

        # Projection to blend recall with current h (trainable gate)
        self.blend_gate = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.blend_gate.weight)
        nn.init.zeros_(self.blend_gate.bias)

    def store_episode(self, h: Tensor) -> None:
        """Store current h in episodic bank (called every step)."""
        for b in range(h.shape[0]):
            self.episodic.store(h[b])

    def store_semantic(self, h: Tensor) -> None:
        """Store in semantic bank (called at sleep consolidation)."""
        for b in range(h.shape[0]):
            self.semantic.store(h[b])

    def recall(self, query: Tensor, mode: str = "episodic") -> Tensor:
        """
        Retrieve from episodic or semantic bank.

        Returns a blended result: gate * recalled + (1 - gate) * query.
        This ensures the WM doesn't blindly follow memory; blend is learned.
        """
        bank = self.episodic if mode == "episodic" else self.semantic
        recalled = bank.recall(query)                          # (B, dim)
        gate_in = torch.cat([query, recalled], dim=-1)         # (B, 2*dim)
        gate = torch.sigmoid(self.blend_gate(gate_in))         # (B, dim)
        return gate * recalled + (1.0 - gate) * query

    def hopfield_entropy(self, mode: str = "episodic") -> float:
        bank = self.episodic if mode == "episodic" else self.semantic
        return bank.get_entropy()


class CittaStore(nn.Module):
    """
    Multi-level Citta associative memory — one CittaStoreLevel per Trika level.
    Integrates with TrikaWorldModel by accepting level index.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_levels: int = 1,
        beta_episodic: float = 4.0,
        beta_semantic: float = 0.25,
        max_episodic: int = 512,
        max_semantic: int = 256,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        store_list: list[CittaStoreLevel] = [
            CittaStoreLevel(
                dim=hidden_dim,
                max_episodic=max_episodic,
                max_semantic=max_semantic,
                beta_episodic=beta_episodic,
                beta_semantic=beta_semantic,
            )
            for _ in range(n_levels)
        ]
        self._store_list = store_list          # typed for internal use
        self.stores = nn.ModuleList(store_list)  # registered for parameter tracking

    def store_episode(self, h: Tensor, level: int = 0) -> None:
        self._store_list[level].store_episode(h)

    def store_semantic(self, h: Tensor, level: int = 0) -> None:
        self._store_list[level].store_semantic(h)

    def recall(self, query: Tensor, level: int = 0, mode: str = "episodic") -> Tensor:
        return self._store_list[level].recall(query, mode)

    def hopfield_entropy(self, level: int = 0, mode: str = "episodic") -> float:
        return self._store_list[level].hopfield_entropy(mode)

    def capacities(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for i, store in enumerate(self._store_list):
            lname = ["apara", "aparapara", "para"][i]
            result[f"{lname}_episodic"] = store.episodic.capacity()
            result[f"{lname}_semantic"] = store.semantic.capacity()
        return result
