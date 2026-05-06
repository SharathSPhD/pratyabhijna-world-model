"""
Corpus pre-embedding cache for fast Phase 1 training.

Encodes all corpus .txt files via sentence-transformer once,
stores as (N, obs_dim) float16 memmap on disk.
Training then reads from cache at ~1M embeddings/sec vs ~1K/sec live.

Usage:
  python -m pwm.data.embed_cache \\
    --corpus-dir /path/to/corpus \\
    --cache-dir data/embed_cache \\
    --obs-dim 512

Output:
  data/embed_cache/embeddings.npy   — (N, obs_dim) float16 memmap
  data/embed_cache/labels.json      — {"n": N, "obs_dim": D, "domain_offsets": {...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def build_embed_cache(
    corpus_dir: str | Path,
    cache_dir: str | Path,
    obs_dim: int = 512,
    chunk_chars: int = 512,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 42,
) -> Path:
    """
    Pre-embed entire corpus into a numpy memmap.

    Returns path to the saved embeddings.npy file.
    """
    from pwm.perception.text import TextEncoder  # type: ignore[import]

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_dir = Path(corpus_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Gather all chunks
    log.info("Scanning corpus: %s", corpus_dir)
    rng = random.Random(seed)

    chunks: list[tuple[str, str]] = []  # (text_chunk, domain_label)
    domain_offsets: dict[str, int] = {}

    for domain_dir in sorted(corpus_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        txt_files = list(domain_dir.rglob("*.txt"))
        if not txt_files:
            continue

        domain_offsets[domain_dir.name] = len(chunks)
        rng.shuffle(txt_files)

        for f in txt_files:
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                for start in range(0, max(1, len(txt) - 50), chunk_chars):
                    chunk = txt[start : start + chunk_chars].strip()
                    if chunk:
                        chunks.append((chunk, domain_dir.name))
            except OSError:
                pass

    # Also handle flat corpus layout (no domain subdirs)
    if not chunks:
        log.info("No domain subdirs — scanning flat layout")
        txt_files = list(corpus_dir.rglob("*.txt"))
        domain_offsets["corpus"] = 0
        for f in txt_files:
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                for start in range(0, max(1, len(txt) - 50), chunk_chars):
                    chunk = txt[start : start + chunk_chars].strip()
                    if chunk:
                        chunks.append((chunk, "corpus"))
            except OSError:
                pass

    N = len(chunks)
    log.info("Total chunks: %d", N)

    # Create memmap
    emb_path = cache_dir / "embeddings.npy"
    emb_map = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(N, obs_dim))

    enc = TextEncoder(obs_dim=obs_dim).to(dev)
    enc.train(False)

    log.info("Embedding %d chunks (batch_size=%d)...", N, batch_size)
    for i in range(0, N, batch_size):
        batch_texts = [c[0] for c in chunks[i : i + batch_size]]
        with torch.no_grad():
            embs = enc(batch_texts, device=dev)   # (B, obs_dim)
        emb_map[i : i + len(batch_texts)] = embs.float().cpu().numpy().astype(np.float16)
        if i % (batch_size * 50) == 0:
            log.info("  %d / %d  (%.1f%%)", i, N, 100 * i / N)
        emb_map.flush()

    # Save metadata
    meta = {
        "n": N,
        "obs_dim": obs_dim,
        "chunk_chars": chunk_chars,
        "domain_offsets": domain_offsets,
        "domain_labels": [c[1] for c in chunks],
        "seed": seed,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log.info("Cache saved: %s (%d embeddings, %.1f MB)", emb_path, N, emb_map.nbytes / 1e6)
    return emb_path


class CachedCorpusEnv:
    """
    Drop-in replacement for PhaseOneEnv using pre-embedded cache.

    Reads embeddings from memmap — no GPU required for corpus loading.
    Delivers (B, T, obs_dim) batches at ~100K step/sec.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        batch_size: int = 32,
        seq_len: int = 64,
        obs_dim: int = 512,
        action_dim: int = 64,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        import json

        cache_dir = Path(cache_dir)
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cache meta not found: {meta_path}. Run pwm.data.embed_cache first.")

        self.meta = json.loads(meta_path.read_text())
        self.embeddings = np.memmap(
            cache_dir / "embeddings.npy",
            dtype=np.float16,
            mode="r",
            shape=(self.meta["n"], self.meta["obs_dim"]),
        )
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = np.random.default_rng(seed)

    def _sample_obs_seq(self) -> torch.Tensor:
        """Sample (B, T, obs_dim) tensor from cache (zero-copy)."""
        N = self.meta["n"]
        idx = self._rng.integers(0, N - self.seq_len, size=self.batch_size)
        seqs = np.stack([self.embeddings[i : i + self.seq_len] for i in idx], axis=0)
        return torch.tensor(seqs, dtype=torch.float32, device=self.device)

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq = self._sample_obs_seq()
        action_seq = torch.zeros(self.batch_size, self.seq_len, self.action_dim, device=self.device)
        reward_seq = torch.zeros(self.batch_size, self.seq_len, device=self.device)
        done_seq = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        return obs_seq, action_seq, reward_seq, done_seq

    def reset(self) -> torch.Tensor:
        return self._sample_obs_seq()[:, 0]

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del action
        obs_seq = self._sample_obs_seq()
        rew = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return obs_seq[:, 0], rew, done


class DomainSelectiveCachedCorpusEnv(CachedCorpusEnv):
    """
    Domain-selective corpus environment for Phase 2 v5.

    Philosophical grounding (svātantrya, ĪPK 2.1):
      Svātantrya = autonomous self-determination. An agent that CAN choose
      which domain of knowledge to attend to exercises genuine creative
      autonomy. This environment makes that choice consequential: the action
      determines which corpus domain the next observations come from.

    Fix for H1 failure chain Layer 3+4:
      CachedCorpusEnv is passive (obs_{t+1} independent of a_t), which
      (a) starves W_a of VFE gradients and (b) keeps the prior near-uniform.
      DomainSelectiveCachedCorpusEnv breaks passivity: the action index
      selects a corpus domain, so obs_{t+1} IS conditioned on a_t.
      This creates genuine VFE gradients through W_a and allows the prior
      to learn domain-appropriate distributions → prior entropy becomes
      action-dependent → EFE epistemic signal is non-zero.

    Domain mapping (64 actions, 2 domains):
      Actions 0 ..  N_DOMAIN_ACTIONS-1   → domain_0 (gutenberg, literary prose)
      Actions N_DOMAIN_ACTIONS .. 63     → domain_1 (hf_wiki_philosophy)

    Usage:
      env = DomainSelectiveCachedCorpusEnv(cache_dir=..., ...)
      obs_seq, action_seq, rew, done = env.sample_batch(action_batch=actions)
    """

    def __init__(self, *args, n_domain_actions: int = 32, **kwargs) -> None:
        """
        Args:
            n_domain_actions: number of action indices mapped to domain_0.
                Actions 0..n_domain_actions-1 → domain_0 (gutenberg).
                Actions n_domain_actions..63  → domain_1 (philosophy).
        """
        super().__init__(*args, **kwargs)
        self.n_domain_actions = n_domain_actions
        offsets = self.meta.get("domain_offsets", {})
        domain_names = sorted(offsets.keys(), key=lambda k: offsets[k])
        if len(domain_names) < 2:
            raise ValueError(
                f"DomainSelectiveCachedCorpusEnv requires ≥2 domains; found: {domain_names}"
            )
        self._domain_names = domain_names
        N = self.meta["n"]
        # Build domain index ranges
        self._domain_ranges: list[tuple[int, int]] = []
        for i, name in enumerate(domain_names):
            start = offsets[name]
            end = offsets[domain_names[i + 1]] if i + 1 < len(domain_names) else N
            self._domain_ranges.append((start, end))
        import logging
        logging.getLogger(__name__).info(
            "DomainSelectiveCachedCorpusEnv: domains=%s ranges=%s actions_per_domain=[%d,%d]",
            domain_names,
            self._domain_ranges,
            n_domain_actions,
            64 - n_domain_actions,
        )

    def _action_to_domain(self, action_idx: int) -> int:
        """Map a scalar action index (0-63) to domain index (0 or 1)."""
        return 0 if action_idx < self.n_domain_actions else 1

    def _sample_obs_seq_domain(self, domain_idx: int) -> torch.Tensor:
        """Sample (B, T, obs_dim) from a specific domain slice."""
        start, end = self._domain_ranges[domain_idx]
        max_start = end - self.seq_len
        if max_start <= start:
            return self._sample_obs_seq()  # domain too small → fall back to uniform
        idx = self._rng.integers(start, max_start, size=self.batch_size)
        seqs = np.stack([self.embeddings[i : i + self.seq_len] for i in idx], axis=0)
        return torch.tensor(seqs, dtype=torch.float32, device=self.device)

    def step(  # type: ignore[override]
        self,
        action: "torch.Tensor",
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Domain-selective step: action index determines which corpus domain to sample.

        Action encoding:
          - Integer / long tensor  (B,)         → direct indices
          - One-hot float          (B, action_dim) → argmax indices
          - Zero / scalar shape    (B, 1)         → fall back to uniform (passive)

        The modal action across the batch selects the domain. This creates genuine
        p(o_{t+1} | a_t) conditioning, supplying VFE gradients through W_a.
        """
        domain_idx = self._decode_action_domain(action)
        obs_seq = self._sample_obs_seq_domain(domain_idx)
        rew = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return obs_seq[:, 0], rew, done

    def _decode_action_domain(self, action: "torch.Tensor") -> int:
        """Extract domain index from an action tensor of arbitrary encoding."""
        a = action.detach().cpu()
        if a.dtype in (torch.int32, torch.int64):
            # Integer indices (B,) — take mode
            modal = int(torch.mode(a.flatten()).values.item())
        elif a.shape[-1] > 1:
            # One-hot (B, action_dim) — argmax then mode
            indices = a.float().argmax(dim=-1).flatten()
            modal = int(torch.mode(indices).values.item())
        else:
            # (B, 1) zeros — passive fallback
            modal = 0
        return self._action_to_domain(modal)

    def sample_batch(  # type: ignore[override]
        self,
        action_batch: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Sample a batch, using action_batch to select corpus domain.

        Args:
            action_batch: (B,) integer action indices. The MODAL action
                (most frequent in the batch) determines which domain to
                sample from. None → uniform sampling (passive fallback).

        Returns:
            obs_seq, action_seq, reward_seq, done_seq — same shape as
            CachedCorpusEnv.sample_batch().
        """
        if action_batch is not None:
            # Compute modal action for this batch
            modal_action = int(torch.mode(action_batch.flatten().cpu()).values.item())
            domain_idx = self._action_to_domain(modal_action)
            obs_seq = self._sample_obs_seq_domain(domain_idx)
        else:
            obs_seq = self._sample_obs_seq()

        action_seq = torch.zeros(self.batch_size, self.seq_len, self.action_dim, device=self.device)
        reward_seq = torch.zeros(self.batch_size, self.seq_len, device=self.device)
        done_seq = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        return obs_seq, action_seq, reward_seq, done_seq


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pre-embed corpus for fast training")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--cache-dir", default="data/embed_cache")
    parser.add_argument("--obs-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    build_embed_cache(
        corpus_dir=args.corpus_dir,
        cache_dir=args.cache_dir,
        obs_dim=args.obs_dim,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
