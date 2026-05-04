"""
MālaRegularisers: Three impurity regularisers from Śaiva Siddhānta.

Philosophical grounding:
  Mala (TS 10.4, Śiva Purāṇa; Kṣemarāja, PHṛ commentary on sūtra 2):
  The three malas are the fundamental impurities binding the individual soul (paśu)
  from recognising its identity with Śiva:

    1. Āṇavamala (atomic impurity): sense of limited selfhood — computational
       analogue: latent collapse (all z_t identical → no creative diversity)
    2. Māyīyamala (māyā impurity): sense of difference from others — computational
       analogue: mode collapse (generator ignores input distribution)
    3. Kārmamala (karma impurity): bondage to past actions — computational
       analogue: reward hacking (policy exploits the reward function)

  The regularisers prevent these three pathologies during training.

Architecture:
  AnavaRegulariser:  entropy penalty to prevent z_t collapse (H[q(z)] > threshold)
  MayiyaRegulariser: diversity loss to prevent mode collapse (batch z variance)
  KarmaRegulariser:  entropy bonus on action distribution (anti-exploitation)
  MalaRegulariser:   combines all three with Hydra-configurable weights
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MalaWeights:
    """Hydra-configurable mala regulariser weights."""
    anava: float = 0.01     # āṇavamala (latent entropy)
    mayiya: float = 0.005   # māyīyamala (batch diversity)
    karma: float = 0.003    # kārmamala (action entropy)
    free_nats: float = 1.0  # minimum entropy threshold for āṇava


class AnavaRegulariser:
    """
    Āṇavamala regulariser: penalises latent collapse.

    Forces the posterior q(z|o) to maintain minimum entropy (diversity of
    discrete latent codes). Prevents the model from ignoring the stochastic
    component (collapsing to deterministic representations).

    Loss: max(0, free_nats - H[q(z)]) per sample.
    """

    def __init__(self, free_nats: float = 1.0) -> None:
        self.free_nats = free_nats

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Args:
            logits: (B, stoch_dim, n_cats) posterior logits
        Returns:
            scalar loss (lower = more diverse latents)
        """
        log_probs = F.log_softmax(logits, dim=-1)  # (B, D, K)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1)      # (B, D) — entropy per variable
        mean_entropy = entropy.mean(-1)              # (B,) — avg over D variables

        # Penalise when entropy falls below free_nats threshold
        collapse_penalty = F.relu(self.free_nats - mean_entropy).mean()
        return collapse_penalty


class MayiyaRegulariser:
    """
    Māyīyamala regulariser: penalises mode collapse in batch.

    Encourages diversity across the batch by penalising low variance in the
    mean latent code. A batch of identical latents → variance=0 → maximum penalty.
    """

    def __call__(self, z_sample: Tensor) -> Tensor:
        """
        Args:
            z_sample: (B, stoch_dim, n_cats) or (B, D) sampled latents
        Returns:
            scalar loss
        """
        z_flat = z_sample.reshape(z_sample.shape[0], -1).float()  # (B, D*K)
        # Variance across batch dimension; penalise low variance
        batch_var = z_flat.var(dim=0).mean()  # scalar
        # Negative variance = penalty (we want high variance)
        return F.softplus(-batch_var)


class KarmaRegulariser:
    """
    Kārmamala regulariser: penalises reward hacking / action entropy collapse.

    Prevents the policy from collapsing to a single action (exploitation loop).
    Adds entropy bonus to ensure the policy maintains creative diversity.
    """

    def __call__(self, action_logits: Tensor) -> Tensor:
        """
        Args:
            action_logits: (B, action_dim) unnormalised action scores
        Returns:
            scalar loss (lower = more diverse actions)
        """
        log_probs = F.log_softmax(action_logits, dim=-1)  # (B, A)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1).mean()     # scalar
        # Penalise low entropy (exploitation)
        return F.softplus(-entropy)


class MalaRegulariser:
    """
    Combined māla regulariser: āṇava + māyīya + kārma.

    Call signature: losses = mala(logits=..., z_sample=..., action_logits=...)
    Returns dict of individual losses + combined weighted total.
    """

    def __init__(self, weights: MalaWeights | None = None) -> None:
        w = weights or MalaWeights()
        self.w = w
        self.anava = AnavaRegulariser(free_nats=w.free_nats)
        self.mayiya = MayiyaRegulariser()
        self.karma = KarmaRegulariser()

    def __call__(
        self,
        logits: Tensor | None = None,
        z_sample: Tensor | None = None,
        action_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute all active mala losses.

        Any component can be None to skip (e.g. during WM-only Phase A training).
        Returns dict with 'anava', 'mayiya', 'karma', 'mala_total'.
        """
        losses: dict[str, Tensor] = {}
        device = torch.device("cpu")

        if logits is not None:
            device = logits.device
            losses["anava"] = self.anava(logits) * self.w.anava

        if z_sample is not None:
            device = z_sample.device
            losses["mayiya"] = self.mayiya(z_sample) * self.w.mayiya

        if action_logits is not None:
            device = action_logits.device
            losses["karma"] = self.karma(action_logits) * self.w.karma

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=device)
        losses["mala_total"] = total  # type: ignore[assignment]
        return losses
