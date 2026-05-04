"""
CorpusDataset: Streaming text corpus loader for PWM Phase 1 training.

Philosophical grounding:
  Śabda (Vākyapadīya 1.1, Bhartṛhari): Word as Brahman — the text corpus is
  the ālaya of all śabda, from which the WM learns to recognise (pratyabhijñā)
  creative patterns. Each text sequence is a partial revelation of the
  underlying spanda (creative vibration) that the WM must learn to model.

Architecture:
  Phase 1: Load pre-tokenised text from corpus/data, embed via TextEncoder,
           return (B, T, obs_dim=512) batches for WM sequence training.

  CorpusDataset:       torch.utils.data.Dataset over .txt files
  CorpusDataLoader:    DataLoader with sentence-transformer embedding
  PhaseOneEnv:         Drop-in replacement for TextEnv stub (train.py)
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class CorpusDataset(Dataset[str]):
    """
    Dataset over all .txt files in a corpus directory.

    Each item is a text chunk of approximately chunk_chars characters.
    Files are loaded lazily — only the file list is preloaded.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        chunk_chars: int = 1024,
        min_chars: int = 50,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.chunk_chars = chunk_chars
        self.min_chars = min_chars

        # Collect all .txt files recursively
        all_files = sorted(self.corpus_dir.rglob("*.txt"))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(all_files)
        self._files = all_files

        # Build flat index of (file_idx, char_start) for each chunk
        self._chunks: list[tuple[int, int]] = []
        for f_idx, f in enumerate(self._files):
            try:
                size = f.stat().st_size
                for start in range(0, max(1, size - min_chars), chunk_chars):
                    self._chunks.append((f_idx, start))
            except OSError:
                pass

        if not self._chunks:
            raise RuntimeError(f"No text chunks found in {corpus_dir}")

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> str:
        f_idx, char_start = self._chunks[idx]
        try:
            text = self._files[f_idx].read_text(encoding="utf-8", errors="ignore")
            chunk = text[char_start:char_start + self.chunk_chars]
            return chunk.strip() or "."
        except OSError:
            return "."


class PhaseOneEnv:
    """
    Real corpus environment for Phase 1 — replaces TextEnv stub.

    Feeds sentence-embedded text sequences (B, T, obs_dim=512) to the WM.
    The "action" is meaningless in Phase 1 (no actual action space yet);
    we use zero actions to train the WM on the prediction task only.

    Usage in train.py: replace TextEnv(…) with PhaseOneEnv(…).
    """

    def __init__(
        self,
        corpus_dir: str | Path = "data/corpus",
        batch_size: int = 32,
        seq_len: int = 16,          # sentence-level sequences
        obs_dim: int = 512,
        action_dim: int = 64,
        device: torch.device | None = None,
        num_workers: int = 2,
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dataset = CorpusDataset(corpus_dir)
        self._loader = DataLoader(
            self._dataset,
            batch_size=batch_size * seq_len,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
        self._iter = iter(self._loader)
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from pwm.perception.text import TextEncoder  # type: ignore[import]
            self._encoder = TextEncoder(obs_dim=self.obs_dim).to(self.device)
        return self._encoder

    def _next_text_batch(self) -> list[str]:
        """Get next B*T text chunks from corpus, refilling iterator as needed."""
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            batch = next(self._iter)
        return list(batch)

    def sample_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample (B, T, obs_dim), (B, T, action_dim), (B, T), (B, T) tensors.

        Text chunks → sentence embeddings → reshape to (B, T, obs_dim).
        Returns: obs_seq, action_seq (zeros), reward_seq (zeros), done_seq (zeros)
        """
        texts = self._next_text_batch()  # B*T strings
        enc = self._get_encoder()

        with torch.no_grad():
            embs = enc(texts, device=self.device)  # (B*T, obs_dim)

        obs_seq = embs.reshape(self.batch_size, self.seq_len, self.obs_dim)

        action_seq = torch.zeros(
            self.batch_size, self.seq_len, self.action_dim, device=self.device
        )
        reward_seq = torch.zeros(self.batch_size, self.seq_len, device=self.device)
        done_seq = torch.zeros(
            self.batch_size, self.seq_len, dtype=torch.bool, device=self.device
        )
        return obs_seq, action_seq, reward_seq, done_seq

    def reset(self) -> Tensor:
        texts = self._next_text_batch()[:self.batch_size]
        enc = self._get_encoder()
        with torch.no_grad():
            embs = enc(texts, device=self.device)
        return embs[:self.batch_size]

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        del action
        obs, _, rew, done = self.sample_batch()
        return obs[:, 0], rew[:, 0], done[:, 0]
