"""
CRSPP: Creative Reward Signal via Predictive Processing.

Philosophical grounding:
  Spanda (SpandaK 1.1, Vasugupta): The divine vibration — consciousness pulsing
  between expansion (sphurattā) and rest. CRSPP formalises camatkāra as the
  Successor Representation of high-spanda states: the predicted cumulative
  surprise under the creative policy.

  SR-AIF (Lefrançois et al. 2024): Successor representation + active inference.
  The SR M(s,s') = E[Σ_t γ^t 1[s_t=s'] | π] encodes how often future states
  are visited, enabling decomposition of value into preference and SR term.

Architecture:
  SRMatrix: (hidden_dim → hidden_dim) linear SR estimator per state pair
  CRSPPModel: SR matrix + preference model + composite reward signal
  Phase 2+: provides the preference gradient to EFEActor
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SRMatrix(nn.Module):
    """
    Successor Representation matrix M(s, ·): s → expected future state occupancy.

    Trained to satisfy Bellman-like SR update:
      M(s) = φ(s) + γ * M(s')
    where φ(s) is the feature map (identity on latent h_t).
    """

    def __init__(self, hidden_dim: int = 512, gamma: float = 0.99) -> None:
        super().__init__()
        self.gamma = gamma
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Target network for stable SR updates (Mnih et al. style)
        self.target = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self._update_target(tau=1.0)

    def forward(self, h: Tensor) -> Tensor:
        """Map state h → successor representation (B, hidden_dim)."""
        return self.net(h)

    def sr_loss(self, h_t: Tensor, h_tp1: Tensor, done: Tensor) -> Tensor:
        """
        TD-style SR loss: M(s_t) ≈ φ(s_t) + γ * M(s_{t+1})
        done mask zeroes the bootstrap on terminal states.
        """
        m_t = self.forward(h_t)
        with torch.no_grad():
            m_tp1 = self.target(h_tp1)
            target = h_t + self.gamma * (1.0 - done.unsqueeze(-1)) * m_tp1
        return F.mse_loss(m_t, target)

    def _update_target(self, tau: float = 0.005) -> None:
        """Soft target update."""
        for p, pt in zip(self.net.parameters(), self.target.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)


class CRSPPModel(nn.Module):
    """
    Full CRSPP preference model combining SR + learned camatkāra preference.

    Provides:
      - value(h): V(s) = w · M(s)  (preference-weighted SR = creative value)
      - reward_signal: combines extrinsic R_camatk + SR-derived intrinsic value
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        gamma: float = 0.99,
        sr_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.sr = SRMatrix(hidden_dim, gamma)
        self.sr_weight = sr_weight

        # Preference weights: w in V(s) = w · M(s)
        self.preference_w = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # Intrinsic creativity preference: maps state → creative preference score
        self.creativity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def value(self, h: Tensor) -> Tensor:
        """Creative value: V(s) = w · M(s) + creativity_head(h). Shape: (B,)."""
        m = self.sr(h)  # (B, hidden_dim)
        sr_value = (self.preference_w.unsqueeze(0) * m).sum(-1)   # (B,)
        creativity = self.creativity_head(h).squeeze(-1)            # (B,)
        return sr_value + self.sr_weight * creativity

    def composite_reward(
        self,
        h: Tensor,
        camatk_reward: Tensor,
        alpha_extrinsic: float = 0.7,
    ) -> Tensor:
        """
        Composite reward signal: α * R_camatk + (1-α) * V_SR(s).

        Blends camatkāra extrinsic reward with SR-derived intrinsic creative value.
        """
        intrinsic = self.value(h)
        return alpha_extrinsic * camatk_reward + (1.0 - alpha_extrinsic) * intrinsic

    def update(
        self,
        h_t: Tensor,
        h_tp1: Tensor,
        done: Tensor,
        camatk_t: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Combined SR + creativity-head update step."""
        optimizer.zero_grad()

        sr_loss = self.sr.sr_loss(h_t, h_tp1, done)
        creativity_pred = self.creativity_head(h_t).squeeze(-1)
        creativity_loss = F.mse_loss(creativity_pred, camatk_t)

        total = sr_loss + creativity_loss
        total.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()

        self.sr._update_target()

        return {
            "sr_loss": float(sr_loss.item()),
            "creativity_loss": float(creativity_loss.item()),
        }
