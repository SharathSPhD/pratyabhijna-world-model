"""
TrikaWorldModel: Three-level Trika RSSM hierarchy.

Philosophical grounding:
  The Trika ('triad') refers to the three śakti levels of Kashmir Śaivism:
    Aparā (Level 0): The 'lower' energy — embodied, fast, stride=1, GRU.
                     Immediate sensory-motor dynamics. Spanda at finest granularity.
    Parāparā (Level 1): The 'middle' energy — stride=4, GRU.
                        Integrates Aparā patterns into mid-level temporal abstractions.
    Parā (Level 2): The 'supreme' energy — stride=16, Mamba backbone (Phase 5+).
                    Slow global context. Top-down conditioning mirrors Cit pervading Aparā.

  Cross-level conditioning: h_para → h_aparapara → h_apara implements the Śaiva insight
  that pure awareness (Parā Śakti) pervades and informs lower-level activity,
  not the reverse.

Architecture:
  - Each level is a TrikaCoreLevel (rssm.py)
  - Top-down h projection: h_{l+1} linearly projects onto obs_dim of level l
  - Stride skipping: only levels that process the current timestep are active
  - Phase guard: Para (level 2) is inactive unless cfg.trika.n_levels == 3
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from pwm.world_model.rssm import TrikaCoreLevel  # type: ignore[import]


class TrikaWorldModel(nn.Module):
    """
    Three-level (or one/two-level) Trika RSSM with top-down h conditioning.

    Level activations by stride:
      t=0,1,2,...  → Level 0 (Aparā) always active
      t=0,4,8,...  → Level 1 (Parāparā) active every 4 steps
      t=0,16,32,.. → Level 2 (Parā) active every 16 steps (n_levels==3 only)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_levels: int = 1,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        free_bits: float = 1.0,
        kl_balance_dyn: float = 0.5,
        kl_balance_rep: float = 0.1,
        decoder_z_only: bool = False,
    ) -> None:
        super().__init__()
        assert 1 <= n_levels <= 3, f"n_levels must be 1, 2, or 3; got {n_levels}"
        self.n_levels = n_levels
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.strides = [1, 4, 16][:n_levels]

        level_cfgs = [
            {"level": 0, "backbone": "gru"},
            {"level": 1, "backbone": "gru"},
            {"level": 2, "backbone": "mamba"},
        ]

        level_list: list[TrikaCoreLevel] = []
        for i in range(n_levels):
            cfg = level_cfgs[i]
            level_obs_dim = obs_dim if i == 0 else hidden_dim
            level_list.append(
                TrikaCoreLevel(
                    level=cfg["level"],
                    obs_dim=level_obs_dim,
                    stoch_dim=stoch_dim,
                    stoch_classes=stoch_classes,
                    hidden_dim=hidden_dim,
                    action_dim=action_dim,
                    backbone=cfg["backbone"],
                    free_bits=free_bits,
                    kl_balance_dyn=kl_balance_dyn,
                    kl_balance_rep=kl_balance_rep,
                    decoder_z_only=decoder_z_only,
                )
            )
        self._level_list = level_list           # typed for internal use
        self.levels = nn.ModuleList(level_list)  # registered for param tracking

        # Top-down conditioning: project h_{l+1} → obs_dim of level l
        td_list: list[nn.Linear] = []
        for i in range(n_levels - 1):
            proj = nn.Linear(hidden_dim, obs_dim if i == 0 else hidden_dim, bias=False)
            nn.init.zeros_(proj.weight)
            td_list.append(proj)
        self._td_list = td_list
        self.top_down = nn.ModuleList(td_list)   # registered for param tracking

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[Tensor, Tensor]]:
        """Initialise all level states to zeros."""
        return [lv.init_state(batch_size, device) for lv in self._level_list]

    def observe_step(
        self,
        obs: Tensor,
        action: Tensor,
        states: list[tuple[Tensor, Tensor]],
        step: int,
    ) -> tuple[list[tuple[Tensor, Tensor]], list[Tensor], list[Tensor]]:
        """
        Single-step observe across active levels.

        Bottom-up: each level's h feeds the next higher level as its "observation".
        Top-down: h of higher level is added to lower level obs (zero-initialised gate).
        """
        new_states = list(states)
        all_logits_post: list[Tensor] = []
        all_logits_prior: list[Tensor] = []
        h_list = [s[0] for s in states]

        for i, (level, stride) in enumerate(zip(self._level_list, self.strides)):
            if step % stride != 0:
                all_logits_post.append(torch.zeros(1))
                all_logits_prior.append(torch.zeros(1))
                continue

            level_obs = obs if i == 0 else h_list[i - 1]

            # Top-down conditioning from level above (zero-init → harmless at start)
            if i < len(self._td_list):
                above_h = h_list[i + 1] if i + 1 < len(h_list) else torch.zeros_like(level_obs)
                level_obs = level_obs + self._td_list[i](above_h)

            prev_h, prev_z = states[i]
            h_t, z_t, logits_post, logits_prior = level.observe(
                level_obs, prev_h, prev_z, action
            )
            new_states[i] = (h_t, z_t)
            h_list[i] = h_t
            all_logits_post.append(logits_post)
            all_logits_prior.append(logits_prior)

        return new_states, all_logits_post, all_logits_prior

    def imagine_step(
        self,
        action: Tensor,
        states: list[tuple[Tensor, Tensor]],
        step: int,
    ) -> tuple[list[tuple[Tensor, Tensor]], list[Tensor]]:
        """Pure imagination step — no encoder, prior only."""
        new_states = list(states)
        all_logits_prior: list[Tensor] = []

        for i, (level, stride) in enumerate(zip(self._level_list, self.strides)):
            if step % stride != 0:
                all_logits_prior.append(torch.zeros(1))
                continue
            prev_h, prev_z = states[i]
            h_t, z_t, logits_prior = level.imagine(prev_h, prev_z, action)
            new_states[i] = (h_t, z_t)
            all_logits_prior.append(logits_prior)

        return new_states, all_logits_prior

    def decode(self, states: list[tuple[Tensor, Tensor]], level: int = 0) -> Tensor:
        """Decode from a specific level's latent state."""
        h, z = states[level]
        return self._level_list[level].decode(h, z)

    def get_features(
        self, states: list[tuple[Tensor, Tensor]], level: int = 0
    ) -> Tensor:
        """Concatenate h and z_flat for a given level (actor/critic input)."""
        h, z = states[level]
        return torch.cat([h, z.flatten(-2)], dim=-1)

    def world_model_loss(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        reward_seq: Tensor,
        done_seq: Tensor,
        init_states: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor | float]:
        """
        Multi-level VFE loss. Only Level 0 (Aparā) in Phase 1.
        Higher levels activated when n_levels > 1.
        """
        total_loss = torch.tensor(0.0, device=obs_seq.device)
        loss_dict: dict[str, Tensor | float] = {}

        for i, (level, stride) in enumerate(zip(self._level_list, self.strides)):
            if i == 0:
                level_obs = obs_seq
                level_actions = action_seq
                level_rewards = reward_seq
                level_done = done_seq
                init_h, init_z = init_states[0]
            else:
                level_obs = obs_seq[:, ::stride]
                level_actions = action_seq[:, ::stride]
                level_rewards = reward_seq[:, ::stride]
                level_done = done_seq[:, ::stride]
                init_h, init_z = init_states[i]

            lvl_losses = level.world_model_loss(
                level_obs, level_actions, level_rewards, level_done, init_h, init_z
            )
            prefix = ["apara", "aparapara", "para"][i]
            for k, v in lvl_losses.items():
                loss_dict[f"{prefix}_{k}"] = v
            if isinstance(lvl_losses["total"], Tensor):
                total_loss = total_loss + lvl_losses["total"]

        loss_dict["total"] = total_loss
        return loss_dict
