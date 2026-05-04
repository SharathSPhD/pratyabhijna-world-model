"""
ReplayBuffer: Prioritised experience replay for WM training.

Philosophical grounding:
  Saṃskāra (YS 1.18, Vyāsa's Bhāṣya): 'impression' — every experience leaves a trace
  that shapes future perception. Prioritised replay biases training toward high-surprise
  transitions (high TD-error ↔ high VFE), i.e., those most informationally generative.

  In the sleep consolidation context (Phase 4), the replay buffer is the source of
  NREM replay — the WM revisits high-priority transitions to consolidate Hopfield memory.

Implementation:
  Standard sum-tree prioritised replay (Schaul et al. 2016, PER).
  Priority = VFE loss at time of insertion (high VFE = high surprise = high informational value).
  α=0.6, β_is starts 0.4 → 1.0 over training (importance sampling correction).
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    obs: Any        # numpy array or tensor (CPU)
    action: Any
    reward: float
    done: bool
    next_obs: Any
    vfe: float = 0.0  # priority signal (VFE at insertion time)


class SumTree:
    """Binary sum-tree for O(log N) priority updates and sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[Transition | None] = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: Transition) -> None:
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> tuple[int, float, Transition | None]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def __len__(self) -> int:
        return self.n_entries


class ReplayBuffer:
    """
    Prioritised experience replay buffer.

    Stores Transition objects; priority = VFE loss at insertion.
    Sequences are reconstructed by sampling adjacent transitions.
    """

    ALPHA = 0.6   # priority exponent
    BETA_START = 0.4
    BETA_FRAMES = 100_000

    def __init__(self, capacity: int = 100_000) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self._max_priority: float = 1.0
        self._frame = 0

    def add(self, transition: Transition) -> None:
        priority = (transition.vfe + 1e-6) ** self.ALPHA
        self._max_priority = max(self._max_priority, priority)
        self.tree.add(priority, transition)

    def sample(
        self, batch_size: int
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions using prioritised sampling.

        Returns:
            transitions: list of Transition
            indices: sum-tree indices (for priority updates)
            weights: importance sampling weights
        """
        self._frame += 1
        beta = min(
            1.0,
            self.BETA_START + self._frame * (1.0 - self.BETA_START) / self.BETA_FRAMES,
        )

        transitions: list[Transition] = []
        indices: list[int] = []
        priorities: list[float] = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            transitions.append(data)
            indices.append(idx)
            priorities.append(priority)

        n = len(self.tree)
        probs = np.array(priorities) / self.tree.total
        weights = (n * probs) ** (-beta)
        weights /= weights.max()

        return transitions, np.array(indices), weights

    def update_priorities(self, indices: np.ndarray, vfe_values: np.ndarray) -> None:
        """Update priorities after WM loss computation on replayed batch."""
        for idx, vfe in zip(indices, vfe_values):
            priority = (float(vfe) + 1e-6) ** self.ALPHA
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(int(idx), priority)

    def sample_sequence(
        self, seq_len: int, batch_size: int
    ) -> list[list[Transition]]:
        """
        Sample sequences of length seq_len for WM sequence training.
        Simple random contiguous-slice sampling (not PER for sequences — PER is per-step).
        """
        seqs: list[list[Transition]] = []
        data_list = [d for d in self.tree.data if d is not None]
        if len(data_list) < seq_len:
            return []
        for _ in range(batch_size):
            start = random.randint(0, len(data_list) - seq_len)
            seqs.append(data_list[start : start + seq_len])
        return seqs

    def __len__(self) -> int:
        return len(self.tree)
