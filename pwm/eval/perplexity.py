"""
Held-out perplexity evaluation for Phase 1 exit criterion.

Philosophical grounding:
  Pratyabhijñā (IPK 1.3, Utpaladeva): Recognition is measured by how well the
  WM predicts unseen text embeddings — a low reconstruction error means the
  model has "recognised" the generative structure of the corpus.

Protocol:
  1. Load checkpoint.
  2. Run WM on 20% held-out split (deterministic, no grad).
  3. Compute reconstruction MSE (embedding prediction error).
  4. Compare against LSTM baseline (same hidden_dim=512, trained from scratch).
  5. Report ratio WM_MSE / LSTM_MSE — should be < 1 for Phase 1 exit.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


# ── LSTM baseline ─────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    Single-layer LSTM with linear projection head.
    Same hidden_dim as Phase 1 WM for fair capacity comparison.
    """

    def __init__(self, obs_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, obs_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, obs_dim) → pred: (B, T, obs_dim). One-step prediction."""
        out, _ = self.lstm(x)
        return self.proj(out)

    def prediction_mse(self, obs_seq: Tensor) -> float:
        """Compute one-step prediction MSE on a (B, T, D) sequence."""
        with torch.no_grad():
            pred = self.forward(obs_seq[:, :-1])
            target = obs_seq[:, 1:]
            mse = (pred - target).pow(2).mean().item()
        return float(mse)


def train_lstm_baseline(
    loader: Any,
    obs_dim: int = 512,
    hidden_dim: int = 512,
    n_steps: int = 2000,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> LSTMBaseline:
    """Train LSTM baseline on corpus for fair comparison with WM."""
    dev = device or torch.device("cpu")
    model = LSTMBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            obs_seq: Tensor = batch[0].to(dev) if isinstance(batch, (list, tuple)) else batch.to(dev)
            if obs_seq.dim() == 2:
                obs_seq = obs_seq.unsqueeze(0)

            pred = model(obs_seq[:, :-1])
            loss = (pred - obs_seq[:, 1:]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

    model.to("cpu")
    return model


# ── WM reconstruction eval ────────────────────────────────────────────────────

def compute_wm_reconstruction(
    world_model: Any,
    loader: Any,
    n_batches: int = 50,
    device: torch.device | None = None,
    mp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
) -> dict[str, float]:
    """
    Compute reconstruction MSE for the WM on held-out data.

    Returns dict with wm_recon_mse, wm_vfe_mean, wm_vfe_std, n_batches.
    """
    dev = device or torch.device("cpu")
    world_model.train(False)

    recon_mses: list[float] = []
    vfes: list[float] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            obs_seq = batch[0].to(dev) if isinstance(batch, (list, tuple)) else batch.to(dev)
            if obs_seq.dim() == 2:
                obs_seq = obs_seq.unsqueeze(0)

            B, T, D = obs_seq.shape
            action_seq = torch.zeros(B, T, 64, device=dev)
            reward_seq = torch.zeros(B, T, device=dev)
            done_seq = torch.zeros(B, T, device=dev)
            init_states = world_model.init_state(B, dev)

            with torch.autocast(device_type=dev.type, dtype=mp_dtype, enabled=use_amp):
                loss_dict = world_model.world_model_loss(
                    obs_seq, action_seq, reward_seq, done_seq, init_states
                )

            vfe = float(loss_dict["total"].item())
            recon = float(loss_dict.get("obs_recon_loss", loss_dict["total"]).item())
            recon_mses.append(recon)
            vfes.append(vfe)

    world_model.train()
    return {
        "wm_recon_mse": float(np.mean(recon_mses)),
        "wm_vfe_mean": float(np.mean(vfes)),
        "wm_vfe_std": float(np.std(vfes)),
        "n_batches": len(recon_mses),
    }


def build_held_out_loader(
    corpus_dir: str | Path,
    obs_dim: int = 512,
    batch_size: int = 16,
    seq_len: int = 32,
    held_out_frac: float = 0.2,
    device: torch.device | None = None,
    seed: int = 42,
) -> tuple[Any, Any]:
    """
    Build (train_iter, held_out_iter) yielding (B, T, obs_dim) tensors.

    Splits by file to avoid data leakage between train and held-out.
    Returns iterables (not DataLoader objects — they are generator classes).
    """
    from pwm.perception.text import TextEncoder  # type: ignore[import]

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_dir = Path(corpus_dir)

    all_files = sorted(corpus_dir.rglob("*.txt"))
    rng = random.Random(seed)
    rng.shuffle(all_files)

    n_held = max(1, int(len(all_files) * held_out_frac))
    held_files = all_files[:n_held]
    train_files = all_files[n_held:]

    log.info("Corpus split: %d train / %d held-out files", len(train_files), len(held_files))
    enc = TextEncoder(obs_dim=obs_dim).to(dev)

    class EmbeddedSequenceIter:
        """Yields (batch_size, seq_len, obs_dim) tensors from text files."""

        def __init__(self, files: list[Path], n_batches: int) -> None:
            self._files = files
            self._n_batches = n_batches
            self._rng = random.Random(seed + 1)

        def __iter__(self):  # type: ignore[return]
            for _ in range(self._n_batches):
                texts = []
                for _ in range(batch_size * seq_len):
                    f = self._rng.choice(self._files)
                    try:
                        txt = f.read_text(encoding="utf-8", errors="ignore")
                        start = self._rng.randint(0, max(0, len(txt) - 512))
                        texts.append(txt[start : start + 512].strip() or ".")
                    except OSError:
                        texts.append(".")
                with torch.no_grad():
                    embs = enc(texts, device=dev)
                yield embs.reshape(batch_size, seq_len, obs_dim)

    return EmbeddedSequenceIter(train_files, 500), EmbeddedSequenceIter(held_files, 100)


# ── Full Phase 1 perplexity report ────────────────────────────────────────────

def run_perplexity_report(
    world_model: Any,
    corpus_dir: str | Path,
    obs_dim: int = 512,
    device: torch.device | None = None,
    lstm_train_steps: int = 2000,
    n_batches: int = 50,
) -> dict[str, float]:
    """
    Full Phase 1 perplexity gate computation.

    Trains LSTM baseline, compares with WM on held-out split.
    Phase 1 exit: wm_vs_lstm_ratio < 1.0 (WM beats LSTM).

    Returns gate metrics dict.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, held_loader = build_held_out_loader(corpus_dir=corpus_dir, obs_dim=obs_dim, device=dev)

    log.info("Training LSTM baseline for %d steps...", lstm_train_steps)
    lstm = train_lstm_baseline(loader=train_loader, obs_dim=obs_dim, n_steps=lstm_train_steps, device=dev)
    lstm = lstm.to(dev)

    log.info("Computing WM reconstruction on held-out set...")
    wm_metrics = compute_wm_reconstruction(world_model=world_model, loader=held_loader, n_batches=n_batches, device=dev)

    log.info("Computing LSTM baseline on held-out set...")
    lstm_mses: list[float] = []
    for i, batch in enumerate(held_loader):
        if i >= n_batches:
            break
        obs_seq = batch[0].to(dev) if isinstance(batch, (list, tuple)) else batch.to(dev)
        lstm_mses.append(lstm.prediction_mse(obs_seq))

    lstm_mse = float(np.mean(lstm_mses))
    wm_mse = wm_metrics["wm_recon_mse"]
    ratio = wm_mse / max(lstm_mse, 1e-9)

    result = {
        **wm_metrics,
        "lstm_recon_mse": lstm_mse,
        "wm_vs_lstm_ratio": ratio,
        "perplexity_gate_pass": ratio < 1.0,
    }
    log.info(
        "Perplexity: WM_MSE=%.4f  LSTM_MSE=%.4f  ratio=%.3f  pass=%s",
        wm_mse, lstm_mse, ratio, result["perplexity_gate_pass"],
    )
    return result
