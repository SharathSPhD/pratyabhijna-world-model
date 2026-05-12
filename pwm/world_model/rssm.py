"""
TrikaCoreLevel: One level of the three-level Trika RSSM hierarchy.

Philosophical grounding:
  Pratyabhijñā (ĪPK 1.3–1.4, Utpaladeva): 'Recognition' — every perception is already
  a re-cognition of the Self in the object. The recognition density q_φ(z_t|h_t,o_t)
  is the technical realisation: every observation collapses the posterior onto a latent
  that is 'recognised' as continuous with the prior history h_t.

  Spanda (SpandaK 1.1, Vasugupta): The categorical sampling z_t ~ Cat(32×32) is the
  stochastic latent 'pulsation' of the generative process — genuine internal events that
  the system carries forward through h_t.

Levels:
  Level 0 (Aparā): stride=1, GRU backbone, fast embodied dynamics
  Level 1 (Parāparā): stride=4, GRU backbone, mid-level coupling
  Level 2 (Para): stride=16, S4 backbone, slow global dynamics (added Phase 5)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import deque

from pwm.world_model.losses import (
    kl_categorical_free_bits,
    symlog_mse_loss,
    twohot_encode,
    twohot_loss,
    make_twohot_bins,
)
from pwm.world_model.mamba_backbone import MambaBackbone, GRUFallback  # type: ignore[import]


def straight_through_sample(logits: Tensor) -> Tensor:
    """
    Straight-through categorical sample.
    Forward: one-hot argmax (deterministic for the gradient). Backward: soft gradient.
    Returns a one-hot tensor of shape (..., stoch_dim, stoch_classes).
    """
    probs = F.softmax(logits, dim=-1)
    sample = torch.zeros_like(probs).scatter_(-1, probs.argmax(-1, keepdim=True), 1.0)
    # Straight-through: gradient passes through as if we used probs
    return (sample - probs).detach() + probs


class RollingWindowStats:
    """Running percentile tracker for sphurattā threshold calibration."""

    def __init__(self, window: int = 200) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, val: float) -> None:
        self._buf.append(val)

    def percentile(self, p: float) -> float:
        if not self._buf:
            return float("inf")
        import numpy as np
        return float(np.percentile(list(self._buf), p))

    @property
    def mean(self) -> float:
        return sum(self._buf) / len(self._buf) if self._buf else 0.0


class SymlogTwohotHead(nn.Module):
    """Distributional reward/value head with symlog-spaced bins (DreamerV3 style)."""

    def __init__(self, in_dim: int, n_bins: int = 255) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, n_bins)
        self.bins: Tensor
        self.register_buffer("bins", make_twohot_bins(n_bins))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        target_hot = twohot_encode(targets, self.bins)
        return twohot_loss(logits, target_hot)

    def mean(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bins).sum(-1)


class TrikaCoreLevel(nn.Module):
    """
    One level of the Trika RSSM hierarchy.

    Implements: observe() for the recognition step (pratyabhijñā)
                imagine() for pure prior imagination (sṛṣṭi)
                world_model_loss() for VFE training objective
    """

    def __init__(
        self,
        level: int,
        obs_dim: int,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
        action_dim: int = 64,
        backbone: str = "gru",
        free_bits: float = 1.0,
        kl_balance_dyn: float = 0.5,
        kl_balance_rep: float = 0.1,
        decoder_z_only: bool = False,
    ) -> None:
        super().__init__()
        self.level = level
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.hidden_dim = hidden_dim
        self.free_bits = free_bits
        self.kl_balance_dyn = kl_balance_dyn
        self.kl_balance_rep = kl_balance_rep
        # Layer 6 fix (v7+): decoder_z_only=True prevents GRU posterior bypass.
        # When True, decoder input is z_t only (latent_dim), not (h_t, z_t).
        # This forces encoder to carry o_t information → non-zero reconstruction gradient.
        self.decoder_z_only = decoder_z_only

        latent_dim = stoch_dim * stoch_classes

        # Recognition density q_φ(z_t | h_t, o_t) — pratyabhijñā
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Input projection: cat(z_flat, action) → hidden_dim (before backbone)
        # Validated pattern from DreamPrice / dreamprice mamba_backbone.py
        self.input_proj = nn.Linear(latent_dim + action_dim, hidden_dim)

        # Recurrent backbone h_t = f(z_t, a_t) — spanda dynamics
        if backbone == "gru":
            self.sequence_model: nn.Module = GRUFallback(d_model=hidden_dim)
            self.backbone_type = "gru"
        elif backbone == "mamba":
            # Mamba-2 SSM; GRU fallback on CPU. Validated on DGX Spark GB10.
            self.sequence_model = MambaBackbone(d_model=hidden_dim)
            self.backbone_type = "mamba"
        else:
            raise ValueError(f"Unknown backbone: {backbone!r} — use 'gru' or 'mamba'")

        # Prior p_θ(z_t | h_t) — the cit level (pure awareness before observation)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder p_θ(o_t | z_t) when decoder_z_only=True (v7+)
        # or p_θ(o_t | h_t, z_t) otherwise (legacy).
        # z-only: architecturally prevents GRU from bypassing encoder.
        decoder_in = latent_dim if decoder_z_only else hidden_dim + latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward head p_θ(r_t | h_t, z_t) — always uses both (not the bottleneck)
        self.reward_head = SymlogTwohotHead(hidden_dim + latent_dim)

        # Continue head p_θ(c_t | h_t, z_t)
        self.continue_head = nn.Linear(hidden_dim + latent_dim, 1)

        # VFE tracking for sphurattā detection
        self.vfe_tracker = RollingWindowStats(window=200)
        self.last_vfe: float = float("inf")

    def _recurrent_step(self, z_flat: Tensor, action: Tensor, _h: Tensor) -> Tensor:
        """One step of the recurrent backbone (GRU or Mamba-2).

        _h is unused: both GRUFallback and MambaBackbone carry their own state.
        Kept in signature for call-site uniformity during observe/imagine.
        """
        inp = self.input_proj(torch.cat([z_flat, action], dim=-1))
        return self.sequence_model.step(inp)  # type: ignore[no-any-return,union-attr]

    def observe(
        self,
        obs: Tensor,
        prev_h: Tensor,
        prev_z: Tensor,
        prev_a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Recognition step: q_φ(z_t | h_t, o_t) — pratyabhijñā.

        Args:
            obs: current observation embedding, shape (B, obs_dim)
            prev_h: previous deterministic state, shape (B, hidden_dim)
            prev_z: previous stochastic state one-hot, shape (B, stoch_dim, stoch_classes)
            prev_a: previous action embedding, shape (B, action_dim)
        Returns:
            h_t: deterministic state
            z_t: sampled stochastic state (straight-through)
            logits_post: posterior logits for KL computation
            logits_prior: prior logits for KL computation
        """
        prev_z_flat = prev_z.flatten(-2)  # (B, stoch_dim * stoch_classes)
        h_t = self._recurrent_step(prev_z_flat, prev_a, prev_h)

        # Posterior (recognition density)
        logits_post = self.encoder(torch.cat([obs, h_t], dim=-1))
        logits_post = logits_post.reshape(-1, self.stoch_dim, self.stoch_classes)
        logits_post = torch.nan_to_num(logits_post, nan=0.0, posinf=20.0, neginf=-20.0)
        z_t = straight_through_sample(logits_post)

        # Prior (for KL computation, not used in z_t selection here)
        logits_prior = self.prior(h_t)
        logits_prior = logits_prior.reshape(-1, self.stoch_dim, self.stoch_classes)
        logits_prior = torch.nan_to_num(logits_prior, nan=0.0, posinf=20.0, neginf=-20.0)

        return h_t, z_t, logits_post, logits_prior

    def imagine(
        self,
        prev_h: Tensor,
        prev_z: Tensor,
        prev_a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Imagination step: ẑ_t ~ p_θ(z_t | h_t) — pure prior (no encoder).
        Used for EFE planning rollouts (sṛṣṭi — creation).
        """
        prev_z_flat = prev_z.flatten(-2)
        h_t = self._recurrent_step(prev_z_flat, prev_a, prev_h)
        logits_prior = self.prior(h_t).reshape(-1, self.stoch_dim, self.stoch_classes)
        logits_prior = torch.nan_to_num(logits_prior, nan=0.0, posinf=20.0, neginf=-20.0)
        z_t = straight_through_sample(logits_prior)
        return h_t, z_t, logits_prior

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """Decode latent state to observation space.

        v7+: when decoder_z_only=True, only z_t is used — prevents GRU posterior bypass.
        Legacy: both h_t and z_t (decoder can learn to ignore z_t → encoder collapse).
        """
        if self.decoder_z_only:
            return self.decoder(z.flatten(-2))
        return self.decoder(torch.cat([h, z.flatten(-2)], dim=-1))

    def compute_vfe(self, logits_post: Tensor, logits_prior: Tensor) -> Tensor:
        """
        Variational Free Energy: complexity term D_KL[q_φ ‖ p_θ] with free bits.
        Isomorphic to the complexity term in Friston's VFE (Friston 2010).
        """
        return kl_categorical_free_bits(logits_post, logits_prior, self.free_bits)

    def world_model_loss(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        reward_seq: Tensor,
        done_seq: Tensor,
        init_h: Tensor,
        init_z: Tensor,
    ) -> dict[str, Tensor | float]:
        """
        Full VFE training loss for one sequence batch.
        L = L_pred (accuracy) + β_dyn * L_dyn + β_rep * L_rep (complexity)

        This is isomorphic to Friston's VFE decomposition:
        F = complexity - accuracy = KL[Q(s)||P(s)] - E_Q[log P(o|s)]

        Args:
            obs_seq: (B, T, obs_dim)
            action_seq: (B, T, action_dim)
            reward_seq: (B, T)
            done_seq: (B, T)
            init_h, init_z: initial states
        Returns:
            dict with 'total', 'obs', 'reward', 'continue', 'dyn', 'rep', 'vfe'
        """
        _, T, _ = obs_seq.shape
        h, z = init_h, init_z

        h_seq, z_seq, post_seq, prior_seq = [], [], [], []

        for t in range(T):
            h, z, logits_post, logits_prior = self.observe(
                obs_seq[:, t], h, z, action_seq[:, t]
            )
            h_seq.append(h)
            z_seq.append(z)
            post_seq.append(logits_post)
            prior_seq.append(logits_prior)

        h_seq = torch.stack(h_seq, dim=1)  # (B, T, hidden_dim)
        z_seq = torch.stack(z_seq, dim=1)  # (B, T, stoch_dim, stoch_classes)
        post_seq = torch.stack(post_seq, dim=1)
        prior_seq = torch.stack(prior_seq, dim=1)

        feat = torch.cat([h_seq, z_seq.flatten(-2)], dim=-1)  # (B, T, hidden+latent)

        # Prediction losses (accuracy term)
        # v7+: decoder uses only z_t (decoder_z_only=True) — prevents GRU posterior bypass
        obs_pred = (self.decoder(z_seq.flatten(-2)) if self.decoder_z_only
                    else self.decoder(feat))
        l_obs = symlog_mse_loss(obs_pred, obs_seq)

        reward_logits = self.reward_head(feat)
        l_reward = self.reward_head.loss(reward_logits, reward_seq)

        continue_logits = self.continue_head(feat).squeeze(-1)
        l_continue = F.binary_cross_entropy_with_logits(
            continue_logits, (1.0 - done_seq).float()
        )
        l_pred = l_obs + l_reward + l_continue

        # KL losses (complexity term) — KL balancing (DreamerV3 ADR-001)
        l_dyn = kl_categorical_free_bits(
            post_seq.detach(), prior_seq, self.free_bits
        ) * self.kl_balance_dyn
        l_rep = kl_categorical_free_bits(
            post_seq, prior_seq.detach(), self.free_bits
        ) * self.kl_balance_rep

        l_total = l_pred + l_dyn + l_rep

        # Track VFE for sphurattā detection
        vfe = (l_dyn + l_rep).item()
        self.vfe_tracker.update(vfe)
        self.last_vfe = vfe

        return {
            "total": l_total,
            "obs": l_obs,
            "reward": l_reward,
            "continue": l_continue,
            "dyn": l_dyn,
            "rep": l_rep,
            "vfe": vfe,
        }

    def init_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Initialise h and z to zeros and reset backbone recurrent state."""
        self.sequence_model.reset_state()  # type: ignore[union-attr]
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        return h, z
