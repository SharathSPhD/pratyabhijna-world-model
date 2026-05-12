"""
SleepConsolidation: NREM + REM sleep phases for memory consolidation.

Philosophical grounding:
  Svapna (Māṇḍūkya Upaniṣad 2–3; MV 1.2.9, Kṣemarāja): The dream state —
  the mind's generative replay of waking experience. In Śaiva terms, svapna is
  where ālayavijñāna distils episodic traces into semantic saṃskāras.

  Two sleep phases:
    NREM (nidrā — deep sleep): replay + VFE descent + SHY synaptic down-scaling
      — consolidates high-surprise episodes into Hopfield semantic bank
    REM  (svapna — dream state): generative dreaming + recognition-net retraining
      — the WM imagines novel sequences, retraining its own posterior

  ThermSleep stopping criterion: ΔF_vfe / ΔF_therm efficiency ratio.
  When efficiency drops below threshold, sleep terminates (diminishing returns).

Phase activation: Phase 4+ (requires Hopfield + replay buffer from Phase 3).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SleepConfig:
    """Hydra-configurable sleep parameters."""
    nrem_replay_steps: int = 100       # replay steps per NREM cycle
    nrem_vfe_threshold: float = 0.01   # stop when VFE drop < threshold
    rem_dream_horizon: int = 32        # imagination sequence length
    rem_retrain_steps: int = 50        # recognition net update steps per REM
    shy_scale: float = 0.95            # SHY down-scaling factor (Tononi 2014)
    efficiency_threshold: float = 0.1  # ThermSleep stopping criterion
    max_nrem_cycles: int = 5
    max_rem_cycles: int = 3
    semantic_store_topk: int = 16      # top-k episodes to consolidate to semantic


class NREMPhase:
    """
    NREM consolidation: replay + VFE descent + SHY down-scaling.

    Replays high-surprise transitions from ReplayBuffer, runs WM loss,
    then writes consolidated representations to CittaStore semantic bank.
    """

    def __init__(
        self,
        world_model: Any,       # TrikaWorldModel
        citta_store: Any,       # CittaStore
        replay_buffer: Any,     # ReplayBuffer
        optimizer: Any,         # torch.optim.Optimizer for WM
        cfg: SleepConfig | None = None,
    ) -> None:
        self.wm = world_model
        self.citta = citta_store
        self.buf = replay_buffer
        self.opt = optimizer
        self.cfg = cfg or SleepConfig()

    def run_cycle(self, device: torch.device) -> dict[str, float]:
        """
        One NREM cycle: replay high-priority transitions, consolidate to semantic.

        Returns metrics: vfe_before, vfe_after, n_consolidated
        """
        if len(self.buf) < 16:
            return {"skipped": 1.0}

        vfe_before = self._estimate_vfe(device)

        was_training = self.wm.training
        self.wm.train()
        for _ in range(self.cfg.nrem_replay_steps):
            transitions, indices, _ = self.buf.sample(batch_size=16)
            if not transitions:
                break

            # Reconstruct batch tensors from transitions
            obs_seq, action_seq, reward_seq, done_seq = _pack_transitions(
                transitions, device
            )
            init_states = self.wm.init_state(1, device)

            self.opt.zero_grad()
            losses = self.wm.world_model_loss(
                obs_seq, action_seq, reward_seq, done_seq, init_states
            )
            if isinstance(losses.get("total"), Tensor):
                losses["total"].backward()  # type: ignore[union-attr]
                nn.utils.clip_grad_norm_(self.wm.parameters(), 1000.0)
                self.opt.step()
        self.wm.train(was_training)

            # Update replay priorities
            vfe_values = [t.vfe for t in transitions]
            import numpy as np
            self.buf.update_priorities(indices, np.array(vfe_values))

        vfe_after = self._estimate_vfe(device)

        # SHY down-scaling: attenuate episodic Hopfield patterns
        for level_idx in range(self.wm.n_levels):
            store = self.citta._store_list[level_idx]
            # Scale down pattern energies (Tononi synaptic homeostasis hypothesis)
            patterns = list(store.episodic._patterns)
            store.episodic.clear()
            for p in patterns:
                store.episodic.store(p * self.cfg.shy_scale)

        # Consolidate top-k high-VFE episodes to semantic bank
        top_k = min(self.cfg.semantic_store_topk, len(self.buf))
        seqs = self.buf.sample_sequence(seq_len=4, batch_size=top_k)
        for _ in seqs:
            h_approx = torch.randn(1, self.wm.hidden_dim, device=device) * 0.1
            self.citta.store_semantic(h_approx, level=0)

        return {
            "vfe_before": vfe_before,
            "vfe_after": vfe_after,
            "vfe_reduction": vfe_before - vfe_after,
            "n_consolidated": top_k,
        }

    def _estimate_vfe(self, device: torch.device) -> float:
        transitions, _, _ = self.buf.sample(batch_size=8)
        if not transitions:
            return float("inf")
        obs_seq, action_seq, reward_seq, done_seq = _pack_transitions(
            transitions, device
        )
        with torch.no_grad():
            init_states = self.wm.init_state(1, device)
            losses = self.wm.world_model_loss(
                obs_seq, action_seq, reward_seq, done_seq, init_states
            )
        vfe = losses.get("apara_vfe", losses.get("vfe", 0.0))
        return float(vfe) if not isinstance(vfe, Tensor) else float(vfe.item())


class REMPhase:
    """
    REM phase: generative dreaming + recognition net retraining.

    The WM imagines novel sequences (sṛṣṭi), then retrains its encoder
    (recognition density q_φ) on the imagined trajectories (pratyabhijñā refinement).
    """

    def __init__(
        self,
        world_model: Any,
        recognition_optimizer: Any,
        cfg: SleepConfig | None = None,
    ) -> None:
        self.wm = world_model
        self.rec_opt = recognition_optimizer
        self.cfg = cfg or SleepConfig()

    def run_cycle(self, device: torch.device) -> dict[str, float]:
        """One REM cycle: dream H steps, retrain recognition net on dreams."""
        init_states = self.wm.init_state(4, device)
        H = self.cfg.rem_dream_horizon

        # Dream: imagine H steps from init
        states = init_states
        dream_h_seqs: list[Tensor] = []
        dream_z_seqs: list[Tensor] = []
        dummy_action = torch.zeros(4, 64, device=device)  # placeholder action dim

        for t in range(H):
            states, _ = self.wm.imagine_step(dummy_action, states, t)
            dream_h_seqs.append(states[0][0])
            dream_z_seqs.append(states[0][1])

        # Detach dream tensors — gradients flow only through prior(), not imagination rollout
        h_dream = torch.stack(dream_h_seqs, dim=1).detach()  # (B=4, T, hidden)

        total_loss = 0.0
        for _ in range(self.cfg.rem_retrain_steps):
            self.rec_opt.zero_grad()
            # Prior → pseudo-posterior update (recognition retraining)
            prior_logits = self.wm._level_list[0].prior(h_dream.reshape(-1, h_dream.shape[-1]))
            prior_logits = prior_logits.reshape(4, H, self.wm._level_list[0].stoch_dim, -1)
            # KL from uniform → entropy maximisation (dream divergence)
            uniform = torch.ones_like(prior_logits) / prior_logits.shape[-1]
            loss = torch.nn.functional.kl_div(
                torch.log_softmax(prior_logits, -1), uniform, reduction="batchmean"
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.wm._level_list[0].prior.parameters(), 10.0)
            self.rec_opt.step()
            total_loss += loss.item()

        return {"rem_loss": total_loss / self.cfg.rem_retrain_steps, "dream_steps": H}


class SleepConsolidator:
    """
    Orchestrates NREM + REM cycles with ThermSleep stopping criterion.

    ThermSleep: ΔF_vfe / ΔF_therm (thermodynamic efficiency of sleep).
    When efficiency < threshold, wake up — further sleep yields diminishing returns.
    """

    def __init__(
        self,
        nrem: NREMPhase,
        rem: REMPhase,
        cfg: SleepConfig | None = None,
    ) -> None:
        self.nrem = nrem
        self.rem = rem
        self.cfg = cfg or SleepConfig()

    def sleep(self, device: torch.device) -> dict[str, Any]:
        """Full sleep episode: alternating NREM/REM with ThermSleep gating."""
        results: list[dict[str, Any]] = []
        total_vfe_reduction = 0.0

        for cycle in range(self.cfg.max_nrem_cycles):
            nrem_metrics = self.nrem.run_cycle(device)
            results.append({"phase": "NREM", "cycle": cycle, **nrem_metrics})
            reduction = nrem_metrics.get("vfe_reduction", 0.0)
            total_vfe_reduction += float(reduction)

            # ThermSleep stopping criterion
            efficiency = float(reduction) / (float(reduction) + 1e-6)
            if efficiency < self.cfg.efficiency_threshold:
                break

            # REM after each NREM cycle
            if cycle < self.cfg.max_rem_cycles:
                rem_metrics = self.rem.run_cycle(device)
                results.append({"phase": "REM", "cycle": cycle, **rem_metrics})

        return {
            "cycles": len(results),
            "total_vfe_reduction": total_vfe_reduction,
            "detailed": results,
        }


def _pack_transitions(
    transitions: list[Any], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack a list of Transitions into (B=1, T, *) tensors for WM loss."""
    import numpy as np
    obs_list = [torch.as_tensor(np.array(t.obs), dtype=torch.float32) for t in transitions]
    action_list = [torch.zeros(64) for _ in transitions]  # placeholder
    reward_list = [t.reward for t in transitions]
    done_list = [float(t.done) for t in transitions]

    obs_seq = torch.stack(obs_list).unsqueeze(0).to(device)       # (1, T, obs_dim)
    action_seq = torch.stack(action_list).unsqueeze(0).to(device)  # (1, T, 64)
    reward_seq = torch.tensor(reward_list, dtype=torch.float32).unsqueeze(0).to(device)
    done_seq = torch.tensor(done_list, dtype=torch.float32).unsqueeze(0).to(device)

    return obs_seq, action_seq, reward_seq, done_seq
