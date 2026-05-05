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
