"""
Svātantrya score: latent novelty and creative freedom measurement.

Philosophical grounding:
  Svātantrya (IPK 2.1, Utpaladeva): The absolute freedom of consciousness to
  recognise and manifest any potential. In the WM, svātantrya is operationalised
  as the entropy of the discrete stochastic latent z_t ~ Categorical(32*32).

  High svātantrya = WM explores a wide swath of latent space = creative freedom.
  Low svātantrya = WM is stuck in a collapsed prior = creative death.

Measurement:
  Collect z_t samples from a validation run, compute:
  - Marginal entropy H(z) — how uniformly z uses its 32x32=1024 categories.
  - Coverage: fraction of categories visited (should be > 0.3 for Phase 1).
  - Sample diversity: mean pairwise Hamming distance between z samples.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


def collect_latents(
    world_model: Any,
    loader: Any,
    n_batches: int = 100,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run WM on held-out data, collect (h_t, z_t) for each step.

    Returns:
        h_latents: (N, hidden_dim) float32 array
        z_indices:  (N, stoch_dim) int32 array of argmax category per z_t dim
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model.train(False)

    h_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            obs_seq = batch[0].to(dev) if isinstance(batch, (list, tuple)) else batch.to(dev)
            if obs_seq.dim() == 2:
                obs_seq = obs_seq.unsqueeze(0)

            B, T, _ = obs_seq.shape
            action_seq = torch.zeros(B, T, 64, device=dev)
            states = world_model.init_state(B, dev)

            # Step through sequence, collect latents at each step
            for t in range(T):
                obs_t = obs_seq[:, t]
                act_t = action_seq[:, t]
                # observe_step returns (new_states, logits_post, logits_prior)
                states, logits_post, logits_prior = world_model.observe_step(obs_t, act_t, states, step=t)
                h_t, z_t = states[0]  # level 0 (Apara)
                h_list.append(h_t.float().cpu().numpy())         # (B, hidden_dim)
                z_list.append(z_t.float().cpu().numpy())         # (B, stoch_dim, stoch_classes)

    world_model.train()

    if not h_list:
        log.warning("No latents collected from observe_step.")
        return np.zeros((0, 512)), np.zeros((0, 1), dtype=np.int32)

    h_latents = np.concatenate(h_list, axis=0)            # (N*T, hidden_dim)
    z_arrays = np.concatenate(z_list, axis=0)              # (N*T, stoch_dim, stoch_classes)
    z_indices = z_arrays.argmax(axis=-1).astype(np.int32)  # (N*T, stoch_dim)

    log.info("Collected %d (h, z) pairs from %d batches.", h_latents.shape[0], i + 1)
    return h_latents, z_indices


def compute_sva_score(z_indices: np.ndarray, n_cats: int = 32) -> dict[str, float]:
    """
    Compute Svātantrya (creative freedom) metrics from z_t categorical indices.

    z_indices: (N, stoch_dim) int32 array, values in [0, n_cats)

    Returns:
        sva_entropy: mean marginal entropy per stoch_dim (bits, max = log2(n_cats))
        sva_coverage: fraction of categories visited across all dims
        sva_diversity: mean pairwise Hamming distance between z samples (sampled)
        sva_score:     composite score in [0, 1] (normalised by max entropy)
    """
    N, D = z_indices.shape
    max_entropy = np.log2(n_cats)

    # Marginal entropy per dimension
    entropies: list[float] = []
    for d in range(D):
        counts = np.bincount(z_indices[:, d], minlength=n_cats).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))

    sva_entropy = float(np.mean(entropies))

    # Coverage: fraction of (dim, category) combinations visited
    visited: set[tuple[int, int]] = set()
    for d in range(D):
        for c in np.unique(z_indices[:, d]):
            visited.add((d, int(c)))
    sva_coverage = len(visited) / (D * n_cats)

    # Sample diversity: mean Hamming on up to 1000 random pairs
    rng = np.random.default_rng(42)
    n_pairs = min(1000, N)
    idx = rng.choice(N, size=(n_pairs, 2), replace=True)
    hamming = (z_indices[idx[:, 0]] != z_indices[idx[:, 1]]).mean(axis=1).mean()

    sva_score = sva_entropy / max_entropy

    return {
        "sva_entropy_bits": sva_entropy,
        "sva_entropy_max": max_entropy,
        "sva_coverage": sva_coverage,
        "sva_diversity_hamming": float(hamming),
        "sva_score": sva_score,
        "sva_gate_pass": sva_coverage > 0.15 and sva_entropy > 1.0,
    }


def run_svat_report(
    world_model: Any,
    loader: Any,
    device: torch.device | None = None,
    n_batches: int = 100,
    stoch_classes: int = 32,
) -> dict[str, float]:
    """Full Svātantrya evaluation report. Returns dict with sva_* metrics."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_latents, z_indices = collect_latents(
        world_model=world_model, loader=loader, n_batches=n_batches, device=dev
    )

    if z_indices.shape[0] == 0:
        log.error("No latents collected — check WM forward interface.")
        return {"sva_score": 0.0, "sva_gate_pass": False, "error": "no_latents"}

    metrics = compute_sva_score(z_indices, n_cats=stoch_classes)
    log.info(
        "Sva: entropy=%.3f/%.3f  coverage=%.3f  hamming=%.3f  pass=%s",
        metrics["sva_entropy_bits"], metrics["sva_entropy_max"],
        metrics["sva_coverage"], metrics["sva_diversity_hamming"],
        metrics["sva_gate_pass"],
    )
    metrics["n_samples"] = z_indices.shape[0]
    return metrics
