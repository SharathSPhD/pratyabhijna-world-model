"""
VFE Loss utilities for the Trika World Model.
Implements symlog, twohot encoding, and KL divergence with free-bits regularisation.

Philosophical grounding:
  Spanda (SpandaK 1.1, Vasugupta): the stochastic latent transition z_t ~ Cat(32×32)
  is realised through these loss functions that shape the recognition density q_φ(z|h,o).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Symlog transform: sign(x) * log(|x| + 1). Scale-invariant for wide reward ranges."""
    return torch.sign(x) * torch.log(x.abs() + 1)


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (x.abs().exp() - 1)


def twohot_encode(x: Tensor, bins: Tensor) -> Tensor:
    """
    Two-hot encoding for distributional value/reward heads.
    Interpolates between the two adjacent bins for each scalar value.

    Args:
        x: scalar targets, shape (*)
        bins: bin edges, shape (B,)
    Returns:
        two-hot distribution, shape (*, B)
    """
    x = symlog(x)
    x = x.unsqueeze(-1)
    below = (bins <= x).sum(-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    # Interpolation weight
    lo = bins[below]
    hi = bins[above]
    weight_hi = (x.squeeze(-1) - lo) / (hi - lo + 1e-8)
    weight_lo = 1.0 - weight_hi
    target = torch.zeros(*x.shape[:-1], len(bins), device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), weight_lo.unsqueeze(-1))
    target.scatter_add_(-1, above.unsqueeze(-1), weight_hi.unsqueeze(-1))
    return target


def twohot_loss(pred_logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy loss for distributional head output against two-hot targets."""
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return -(target * log_probs).sum(-1).mean()


def kl_categorical_free_bits(
    logits_post: Tensor,
    logits_prior: Tensor,
    free_bits: float = 1.0,
) -> Tensor:
    """
    KL divergence between posterior q_φ(z|h,o) and prior p_θ(z|h) for categorical
    distributions, with free bits regularisation to prevent posterior collapse.

    Free bits (Kingma et al.): each categorical variable must contribute at least
    `free_bits` nats of KL; variables below threshold are not penalised.
    This is essential for maintaining spanda — genuine stochastic latent transitions.

    Args:
        logits_post: posterior logits, shape (B, stoch_dim, stoch_classes)
        logits_prior: prior logits, shape (B, stoch_dim, stoch_classes)
        free_bits: minimum KL per categorical variable (nats)
    Returns:
        scalar KL loss (averaged over batch and categorical variables)
    """
    post = F.softmax(logits_post, dim=-1)
    prior = F.softmax(logits_prior, dim=-1)
    kl = (post * (post.clamp(min=1e-8).log() - prior.clamp(min=1e-8).log())).sum(-1)
    # Free bits: clamp per variable, then average
    kl = kl.clamp(min=free_bits)
    return kl.mean()


def symlog_mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE loss in symlog space — scale-agnostic reconstruction loss."""
    return F.mse_loss(symlog(pred), symlog(target))


def make_twohot_bins(n_bins: int = 255, lo: float = -20.0, hi: float = 20.0) -> Tensor:
    """Create bin edges for distributional heads (symlog-spaced)."""
    return torch.linspace(lo, hi, n_bins)
