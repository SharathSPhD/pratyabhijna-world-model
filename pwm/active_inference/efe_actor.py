"""
EFEActor: Expected Free Energy actor replacing REINFORCE.

Philosophical grounding:
  Svātantrya (ĪPK 2.1, Utpaladeva): The freedom of consciousness to choose —
  not random volition but the sovereign will that selects the action minimising
  expected surprise (EFE). Kriyā-śakti (power of action) expressed as the
  policy that maximises information gain + expected pragmatic value.

  EFE = -E_q[ln p(o|π)] - KL[q(s|π) || p(s)]
      = pragmatic_value (utility) + epistemic_value (information gain)

Architecture:
  Input: h_t (hidden state, hidden_dim), z_t (latent, stoch_dim)
  Output: action distribution π(a|h_t, z_t)
  Uses pymdp.maths for EFE decomposition (epistemic + pragmatic terms).
  Phase 2+: replaces EFEActorStub in train.py.

Epistemic value (information gain) drives novelty-seeking;
pragmatic value (preference alignment) drives camatkāra reward maximisation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class EFEActor(nn.Module):
    """
    Expected Free Energy actor for creative action selection.

    Combines epistemic value (information gain over latents) with pragmatic value
    (alignment with camatkāra preference). Outputs a categorical action distribution.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        n_cats: int = 32,
        action_dim: int = 64,
        n_layers: int = 3,
        free_nats: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.n_cats = n_cats
        self.action_dim = action_dim
        self.free_nats = free_nats

        latent_flat = stoch_dim * n_cats
        in_dim = hidden_dim + latent_flat

        layers: list[nn.Module] = []
        dim = in_dim
        for i in range(n_layers):
            out_dim = action_dim * 4 if i < n_layers - 1 else action_dim * 4
            layers += [nn.Linear(dim, out_dim), nn.SiLU()]
            dim = out_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)

        # Preference model: learned log-prior over actions (pragmatic value)
        self.log_preference = nn.Parameter(torch.zeros(action_dim))

    def forward(self, h: Tensor, z: Tensor) -> tuple[Categorical, Tensor]:
        """
        Compute action distribution from (h_t, z_t).

        Returns:
            dist:   Categorical over action_dim
            efe:    (B,) EFE per sample (lower = better)
        """
        B = h.shape[0]
        z_flat = z.reshape(B, -1)
        inp = torch.cat([h, z_flat], dim=-1)
        logits = self.net(inp)  # (B, action_dim)

        # Pragmatic value: alignment with preference model
        log_pref = F.log_softmax(self.log_preference, dim=-1)
        log_pi = F.log_softmax(logits, dim=-1)

        # Epistemic value: entropy of policy (information gain proxy)
        # EFE ≈ -H[π] - E_π[log p(a|pref)]
        entropy = -(log_pi.exp() * log_pi).sum(-1)           # (B,) — higher = more exploratory
        pragmatic = (log_pi.exp() * log_pref.unsqueeze(0)).sum(-1)  # (B,) — higher = better aligned

        # EFE (to minimise): negate epistemic (want high entropy) + negate pragmatic (want high pref)
        efe = -entropy - pragmatic  # (B,)

        dist = Categorical(logits=logits)
        return dist, efe

    def select_action(self, h: Tensor, z: Tensor, deterministic: bool = False) -> Tensor:
        """Sample (or argmax) action from policy."""
        dist, _ = self.forward(h, z)
        if deterministic:
            return dist.logits.argmax(-1)
        return dist.sample()

    def actor_loss(self, h: Tensor, z: Tensor, advantage: Tensor) -> dict[str, Tensor]:
        """
        Phase B actor loss: REINFORCE-style with EFE regularisation.

        advantage: (B,) — from critic bootstrap (λ-return)
        """
        dist, efe = self.forward(h, z)
        log_prob = dist.log_prob(dist.sample())  # (B,)
        entropy = dist.entropy()                  # (B,)

        # Policy gradient + entropy bonus + EFE minimisation
        pg_loss = -(log_prob * advantage.detach()).mean()
        efe_loss = efe.mean()
        entropy_loss = -entropy.mean()  # encourage exploration

        total = pg_loss + 0.1 * efe_loss + 3e-4 * entropy_loss
        return {
            "actor_total": total,
            "pg_loss": pg_loss,
            "efe_loss": efe_loss,
            "entropy": entropy.mean(),
        }


class CRSPPPreference(nn.Module):
    """
    Successor Representation Active Inference (SR-AIF) preference model.

    Implements CRSPP (Creative Reward Signal via Predictive Processing):
    preference p(o) is not a fixed Gaussian but learned from camatkāra signal.
    The preference aligns with high-camatkāra observations.
    """

    def __init__(self, obs_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def log_preference(self, obs: Tensor) -> Tensor:
        """
        Log-preference log p(o) for observations.

        Returns (B,) — higher = more preferred (aligned with camatkāra).
        """
        return self.net(obs).squeeze(-1)

    def preference_loss(self, obs: Tensor, camatk_reward: Tensor) -> Tensor:
        """
        Train preference model to predict camatkāra reward from observations.
        MSE between predicted preference and observed camatkāra signal.
        """
        pred = self.log_preference(obs)
        return F.mse_loss(pred, camatk_reward)
