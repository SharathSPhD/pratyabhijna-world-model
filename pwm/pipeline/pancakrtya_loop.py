"""
PancakrtyaLoop: The five divine acts as a single Python call stack.

Philosophical grounding:
  Pañcakṛtya (MV 1.4, Kṣemarāja; ĪPK 3.1–3.2, Utpaladeva):
  The five acts of Śiva — ābhāsana (manifestation), rakti (colouring/pleasure),
  vimarśa (self-reflexive cognition), bīja-sthāpana (seed-planting), vilāpana (dissolution)
  — mapped to the computational sequence:

    cit        →  World model step (RSSM observe): ābhāsana — the world manifests
    ānanda     →  Camatkāra reward: rakti — the pleasure/surprise signal arises
    icchā      →  EFE actor: will selects the next action
    apohana    →  Memory store (smṛti read/write): context is refined
    jñāna      →  LLM knowledge call (fast path): patterns are named
    kriyā      →  Action commit + narration emit: the act is performed

  Architecture constraint: these six functions share a SINGLE continuous WM state
  (h_t, z_t). They must NOT be separate agents — that would serialise tensors at
  each boundary and destroy temporal continuity. Only VimarshaAgent (the deliberative
  gate, invoked on sphurattā events) is a true smolagents agent.

Sphurattā gating:
  When VFE < 5th-percentile threshold AND Hopfield entropy drops,
  the loop escalates to VimarshaAgent for deliberative commit/revise/reject.
  Without sphurattā, the loop completes in under 30 seconds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time

import torch
from torch import Tensor

# Sprint 5: ISO 15919 transliteration — available as a post-processing step
# on any narration string produced by the kriyā act.
_translit_fn = None
_TRANSLIT_AVAILABLE = False
try:
    from pwm.generation.transliterate import transliterate_text as _translit_fn  # type: ignore[assignment]
    _TRANSLIT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LoopState:
    """Live state of the Pañcakṛtya loop at a given timestep."""
    h: Tensor                           # RSSM deterministic state
    z: Tensor                           # RSSM stochastic state
    action: Tensor                      # Current action (for next step)
    step: int = 0
    vfe: float = float("inf")
    camatk_reward: float = 0.0
    sphuratta_fired: bool = False
    narration: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopConfig:
    """Configuration for PancakrtyaLoop (loaded from Hydra config)."""
    # Sphurattā thresholds
    sphuratta_vfe_percentile: float = 5.0
    sphuratta_hopfield_threshold: float = 0.3
    sphuratta_min_gap: int = 50
    # Camatkāra weights
    camatk_alpha_vfe: float = 0.4
    camatk_alpha_hopfield: float = 0.3
    camatk_alpha_empowerment: float = 0.3
    # Memory
    memory_level: int = 0
    # Rollout
    imagine_horizon: int = 13
    gamma: float = 0.99
    lambda_: float = 0.95
    # LLM (jñāna path)
    llm_enabled: bool = False
    llm_max_tokens: int = 128
    # EFE
    efe_enabled: bool = False   # Phase 2+
    # Debug
    log_every: int = 100


class PancakrtyaLoop:
    """
    The five divine acts as a coherent Python call-stack sharing (h_t, z_t).

    Usage:
        loop = PancakrtyaLoop(world_model, citta_store, camatk, cfg)
        state = loop.init(batch_size=1, device=device)
        for obs_t, done_t in environment:
            state, output = loop.step(obs_t, done_t, state)
            env.send_action(output["action"])
    """

    def __init__(
        self,
        world_model: Any,                # TrikaWorldModel
        citta_store: Any,                # CittaStore
        camatk: Any,                     # CamatkaraReward
        cfg: LoopConfig | None = None,
        llm_backend: Any = None,         # LLMBackend | None
        efe_actor: Any = None,           # EFEActor | nn.Module | None
        vimarsha_agent: Any = None,      # VimarshaAgent | None (smolagents)
        context_store: Any = None,       # AvacchedakaStore | None
        vimarsa_bridge: Any = None,      # VimarsaBridge | None (Phase 5+)
    ) -> None:
        self.wm = world_model
        self.citta = citta_store
        self.camatk = camatk
        self.cfg = cfg or LoopConfig()
        self.llm = llm_backend
        self.efe_actor = efe_actor
        self.vimarsha = vimarsha_agent
        self.ctx = context_store
        self.vimarsa_bridge = vimarsa_bridge

        self._last_sphuratta_step: int = -9999
        self._step_count: int = 0

    def init(self, batch_size: int, device: torch.device) -> LoopState:
        """Initialise loop state (call once per episode)."""
        states = self.wm.init_state(batch_size, device)
        h, z = states[0]
        action = torch.zeros(batch_size, self.wm.strides[0], device=device)
        return LoopState(h=h, z=z, action=action, step=0)

    def step(
        self,
        obs_t: Tensor,
        done_t: Tensor,
        state: LoopState,
        trika_states: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[LoopState, dict[str, Any]]:
        """
        Execute one full Pañcakṛtya cycle.

        Returns updated state and output dict with:
          action, vfe, camatk_reward, sphuratta_fired, narration, metrics
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        device = obs_t.device

        # Reset backbone at episode boundaries
        if bool(done_t.any()):
            self.wm._level_list[0].sequence_model.reset_state()  # type: ignore[union-attr]

        # ── Cit (ābhāsana): World model observation step ─────────────────────
        if trika_states is None:
            # Single-level (Phase 1): use Level 0 only
            h_t, z_t, logits_post, logits_prior = self.wm._level_list[0].observe(
                obs_t, state.h, state.z, state.action
            )
            vfe_val = float(self.wm._level_list[0].compute_vfe(logits_post, logits_prior).item())
            self.wm._level_list[0].vfe_tracker.update(vfe_val)
        else:
            # Multi-level (Phase 5): delegate to trika observe_step
            new_states, all_post, all_prior = self.wm.observe_step(
                obs_t, state.action, trika_states, state.step
            )
            h_t, z_t = new_states[0]
            vfe_val = float(self.wm._level_list[0].compute_vfe(all_post[0], all_prior[0]).item())

        # ── Ānanda (rakti): Camatkāra reward ──────────────────────────────────
        hopfield_entropy = self.citta.hopfield_entropy(level=cfg.memory_level)
        # Empowerment placeholder (Phase 4+: learned empowerment head)
        empowerment_val = torch.zeros(1, device=device)
        camatk_tensor, camatk_log = self.camatk.compute(
            curr_vfe=torch.tensor(vfe_val, device=device),
            hopfield_entropy_delta=torch.tensor(hopfield_entropy, device=device),
            empowerment=empowerment_val,
        )
        camatk_val = float(camatk_tensor.item())

        # ── Sphurattā detection ───────────────────────────────────────────────
        vfe_threshold = self.wm._level_list[0].vfe_tracker.percentile(
            cfg.sphuratta_vfe_percentile
        )
        sphuratta = self.camatk.sphuratta_score(
            vfe=vfe_val,
            vfe_percentile=vfe_threshold,
            hopfield_entropy=hopfield_entropy,
            hopfield_threshold=cfg.sphuratta_hopfield_threshold,
            last_sphuratta_step=self._last_sphuratta_step,
            current_step=state.step,
            min_gap=cfg.sphuratta_min_gap,
        )
        if sphuratta:
            self._last_sphuratta_step = state.step

        # ── Apohana (smṛti): Episodic memory store ───────────────────────────
        self.citta.store_episode(h_t, level=cfg.memory_level)
        # Optionally condition on episodic recall (Phase 3+)
        h_conditioned = h_t  # plain h_t until CittaStore is wired for recall

        # ── Jñāna (fast path): LLM knowledge call ────────────────────────────
        narration = ""
        if cfg.llm_enabled and self.llm is not None and sphuratta:
            # Only call LLM on sphurattā event (≪ 1% of steps)
            try:
                sakshi = self.ctx.get_sakshi() if self.ctx else "PWM: creative world model"
                # Phase 5+: VimarsaBridge enriches the prompt with WM hidden state summary
                wm_prefix = ""
                if self.vimarsa_bridge is not None:
                    with torch.no_grad():
                        wm_prefix = self.vimarsa_bridge.format_prefix_text(h_t)
                narration = self.llm.call(
                    role="jnana",
                    system=sakshi,
                    prompt=wm_prefix + f"VFE={vfe_val:.4f}, camatkāra={camatk_val:.4f}. Describe the creative moment.",
                    max_tokens=cfg.llm_max_tokens,
                )
            except Exception:
                narration = ""

        # ── Icchā (will): EFE actor selects action ───────────────────────────
        feat = torch.cat([h_conditioned, z_t.flatten(-2)], dim=-1)
        if cfg.efe_enabled and self.efe_actor is not None:
            action_new = self.efe_actor(feat)
        else:
            # Phase 1: random action (REINFORCE actor wired in Phase 2)
            action_new = torch.zeros_like(state.action)

        # ── Vimarśa (deliberative gate): only on sphurattā ───────────────────
        vimarsha_output: dict[str, Any] = {}
        if sphuratta and self.vimarsha is not None:
            # VimarshaAgent is the only true smolagents agent in the pipeline
            try:
                vimarsha_output = self.vimarsha.run(
                    h=h_t, z=z_t, vfe=vfe_val, narration=narration
                )
            except Exception:
                vimarsha_output = {}

        # ── Kriyā (action commit) ────────────────────────────────────────────
        dt = time.perf_counter() - t0
        self._step_count += 1

        metrics: dict[str, Any] = {
            "vfe": vfe_val,
            "camatk_reward": camatk_val,
            "sphuratta": sphuratta,
            "loop_ms": dt * 1000,
            "step": state.step,
            **{f"camatk_{k}": v for k, v in camatk_log.items()},
            **vimarsha_output,
        }

        new_state = LoopState(
            h=h_t,
            z=z_t,
            action=action_new,
            step=state.step + 1,
            vfe=vfe_val,
            camatk_reward=camatk_val,
            sphuratta_fired=sphuratta,
            narration=narration,
            metrics=metrics,
        )

        return new_state, {
            "action": action_new,
            "narration": narration,
            **metrics,
        }

    def reset(self, state: LoopState) -> LoopState:
        """Reset backbone state at episode boundary (done=True)."""
        self.wm._level_list[0].sequence_model.reset_state()  # type: ignore[union-attr]
        device = state.h.device
        batch_size = state.h.shape[0]
        return self.init(batch_size, device)

    @staticmethod
    def transliterate_narration(narration: str) -> dict[str, Any]:
        """
        Sprint 5 — kriyā post-processing: ISO 15919 romanisation of narration.

        Sanskrit concept (CLAUDE.md §9):
          Vimarśa (ĪPK 1.5.11) — the reflexive cognition that makes the inner
          act legible. Transliteration renders Indic-script narrations into
          IAST so they can be included in LaTeX paper figures directly.

        Usage:
            result = loop.step(state, obs, action)
            translit = PancakrtyaLoop.transliterate_narration(result[1]["narration"])
            # translit["iast"] — romanised text for the paper

        Args:
            narration: The narration string emitted by the kriyā act (may be
                       Devanāgarī, Kannada, Tamil, Telugu, Bengali, or Latin).

        Returns:
            dict with keys: iast, script, has_indic, mixed, latex_annotation.
            If transliteration is unavailable, returns {"iast": narration,
            "script": "unknown", "has_indic": False, ...}.
        """
        if not _TRANSLIT_AVAILABLE or not narration.strip():
            return {
                "iast": narration,
                "script": "unknown",
                "has_indic": False,
                "mixed": False,
                "latex_annotation": narration,
            }

        result = _translit_fn(narration)  # type: ignore[misc]
        return {
            "iast": result.iast,
            "script": result.script,
            "has_indic": result.has_indic,
            "mixed": result.mixed_language,
            "latex_annotation": result.latex_annotation(),
        }
