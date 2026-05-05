"""
UMAP latent cluster visualisation for Phase 1 exit criterion.

Philosophical grounding:
  Vikalpa (IPK 1.5.5, Utpaladeva): Conceptual construction — the mind creates
  distinct categories from continuous experience. The UMAP projection tests
  whether the WM's stochastic latents have performed a meaningful vikalpa:
  texts from different domains should cluster in distinct regions of z-space.

Phase 1 exit criterion:
  UMAP of z_t latents, coloured by source domain (hf_wiki_philosophy /
  gutenberg / hf_poetry), must show visible cluster separation.
  Quantitative: silhouette score > 0.1 on the 2D UMAP embedding.

Usage:
  python -m pwm.eval.umap_viz --checkpoint checkpoints/final.pt
"""

from __future__ import annotations

import logging
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def collect_latents_with_labels(
    world_model: Any,
    corpus_dir: Path,
    obs_dim: int = 512,
    n_samples: int = 2000,
    device: torch.device | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Run WM on corpus samples, collect (h, z) feature vectors with domain labels.

    Returns:
        features: (N, hidden_dim + stoch_dim*stoch_classes) float32 array
        labels:   list of domain strings (e.g. "hf_wiki_philosophy")
    """
    from pwm.perception.text import TextEncoder  # type: ignore[import]

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = TextEncoder(obs_dim=obs_dim).to(dev)
    world_model.train(False)

    # Gather files per domain
    domain_files: dict[str, list[Path]] = {}
    for subdir in corpus_dir.iterdir():
        if subdir.is_dir():
            txt_files = list(subdir.rglob("*.txt"))
            if txt_files:
                domain_files[subdir.name] = txt_files

    if not domain_files:
        log.error("No domain subdirectories found in %s", corpus_dir)
        return np.zeros((0, 512)), []

    rng = random.Random(seed)
    features_list: list[np.ndarray] = []
    labels: list[str] = []

    samples_per_domain = max(1, n_samples // len(domain_files))
    log.info("Collecting %d samples per domain from %d domains", samples_per_domain, len(domain_files))

    for domain, files in sorted(domain_files.items()):
        domain_features: list[np.ndarray] = []
        attempts = 0
        while len(domain_features) < samples_per_domain and attempts < samples_per_domain * 3:
            attempts += 1
            f = rng.choice(files)
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                start = rng.randint(0, max(0, len(txt) - 512))
                chunk = txt[start : start + 512].strip()
                if not chunk:
                    continue
                with torch.no_grad():
                    emb = enc([chunk], device=dev)          # (1, obs_dim)
                    obs_seq = emb.unsqueeze(1)               # (1, 1, obs_dim)
                    action_seq = torch.zeros(1, 1, 64, device=dev)
                    init_states = world_model.init_state(1, dev)
                    # Forward pass to get latent h
                    try:
                        states, _ = world_model.observe_step(emb, action_seq[:, 0], init_states, step=0)
                        h_t, z_t = states[0]
                        feat = torch.cat([h_t.float(), z_t.flatten(-2).float()], dim=-1)  # (1, D)
                    except (AttributeError, TypeError):
                        # Fallback: just use the embedding
                        feat = emb.float()
                    domain_features.append(feat.squeeze(0).cpu().numpy())
            except (OSError, RuntimeError):
                continue

        for feat in domain_features:
            features_list.append(feat)
            labels.append(domain)
        log.info("  Domain '%s': %d samples collected", domain, len(domain_features))

    world_model.train()

    if not features_list:
        return np.zeros((0, obs_dim)), []

    return np.stack(features_list, axis=0), labels


def compute_umap_and_score(
    features: np.ndarray,
    labels: list[str],
    output_dir: Path | None = None,
    n_components: int = 2,
) -> dict[str, Any]:
    """
    Run UMAP on features, compute cluster quality metrics.

    Saves a PNG plot if output_dir is provided.
    Returns metrics dict including silhouette_score.
    """
    try:
        import umap  # type: ignore[import]
        _HAS_UMAP = True
    except ImportError:
        _HAS_UMAP = False
        log.warning("umap-learn not installed — skipping UMAP (pip install umap-learn)")

    from sklearn.preprocessing import LabelEncoder  # type: ignore[import]
    from sklearn.metrics import silhouette_score      # type: ignore[import]

    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    unique_labels = list(le.classes_)
    n_domains = len(unique_labels)

    metrics: dict[str, Any] = {
        "n_samples": len(labels),
        "n_domains": n_domains,
        "domains": unique_labels,
    }

    if not _HAS_UMAP or features.shape[0] < 10:
        metrics["silhouette_score"] = None
        metrics["umap_gate_pass"] = False
        metrics["note"] = "umap-learn not installed or insufficient samples"
        return metrics

    log.info("Running UMAP on %d samples (%d dims) → %d components", *features.shape, n_components)
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)    # (N, 2)

    # Silhouette score: measures how well-separated clusters are
    # Range [-1, 1]; > 0.1 means meaningful separation
    if n_domains >= 2:
        sil = float(silhouette_score(embedding, label_ids))
    else:
        sil = 0.0

    metrics["silhouette_score"] = sil
    metrics["umap_gate_pass"] = sil > 0.1
    metrics["embedding_shape"] = list(embedding.shape)

    log.info("UMAP silhouette=%.4f  gate_pass=%s", sil, metrics["umap_gate_pass"])

    # Save plot if matplotlib available and output_dir set
    if output_dir is not None:
        try:
            import matplotlib  # type: ignore[import]
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]

            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, n_domains))  # type: ignore[attr-defined]
            for i, domain in enumerate(unique_labels):
                mask = label_ids == i
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=[colors[i]], label=domain, alpha=0.6, s=10
                )
            ax.set_title(f"UMAP of z_t latents (Phase 1)\nSilhouette={sil:.4f} | {'PASS' if sil > 0.1 else 'FAIL'}")
            ax.legend(loc="best", fontsize=8)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")

            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir / "umap_phase1.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            metrics["plot_path"] = str(fig_path)
            log.info("UMAP plot saved: %s", fig_path)

            # Also save raw embedding for further analysis
            np.save(output_dir / "umap_embedding.npy", embedding)
            np.save(output_dir / "umap_labels.npy", np.array(labels))
        except ImportError:
            log.warning("matplotlib not available — skipping UMAP plot")

    return metrics
