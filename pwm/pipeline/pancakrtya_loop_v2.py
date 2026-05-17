"""
PancakrtyaLoopV2 — production rewrite wiring all 6 Pañcakṛtya acts.

Philosophical grounding:
  Pañcakṛtya (ĪPK 3.1–3.2, Utpaladeva; MV 1.4, Kṣemarāja):
  The five divine acts of Śiva mapped to computation — each stanza is one
  complete cascade sharing a single (h_t, z_t) WM state (Contract 1).

  Act 1  cit / sṛṣṭi     — WM observe_step → (h_t, z_t)       [GPU stream 0]
  Act 2  ānanda / sthiti  — EFE actor → efe_score              [GPU stream 0]
  Act 3  icchā / saṃhāra  — Hopfield recall → mem_t            [GPU stream 0]
  Act 4  apohana / tirodhāna — Entropy gate → sphurattā         [CPU scalar]
  Act 5  jñāna / anugraha  — VimarsaBridge → logits_processor  [GPU stream 0]
  Act 6  kriyā             — LLM stream → tokens               [GPU stream 1]

Layer boundary (Contract 2):
  Internal Śaiva vocabulary (efe, vfe, sphurattā) stays inside this module.
  SSE event dicts use domain-neutral keys (aesthetic_quality, creative_peak).
  CamatkaraNarrator translates at the API boundary.

Contract 3: WM drives generation. LLM renders. If LLM is unavailable,
  WM still runs and returns a degraded-quality stub via _stub_generate().
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    """Configuration for PancakrtyaLoopV2."""
    n_stanzas: int = 4
    device: str = "cuda"
    tau_sphuratta: float = 0.65       # normalised entropy threshold for sphurattā
    max_tokens_per_stanza: int = 256
    temperature: float = 0.88
    top_p: float = 0.92
    obs_dim: int = 512
    action_dim: int = 64
    hidden_dim: int = 512


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class StanzaResult:
    """Internal result of one stanza — stays in the internal layer."""
    stanza_idx: int
    text: str
    efe_score: float
    vfe: float
    sphuratta: bool
    mem_resonance: float
    camatk_score: float
    h_t: Optional[Tensor] = None


# ─── Main loop ───────────────────────────────────────────────────────────────

class PancakrtyaLoopV2:
    """
    Production Pañcakṛtya loop — all 6 acts share one (h_t, z_t) call stack.

    The WM state is NOT serialised between acts (Contract 1: no JSON crossing).
    SSE events use domain-neutral keys (Contract 2: layer boundary).

    Usage:
        loop = PancakrtyaLoopV2(wm, efe, citta, bridge, llm, cfg)
        loop.init()
        for event in loop.run(obs_list, system_prompt, user_prompt_fn):
            sse_stream.send(event)
    """

    def __init__(
        self,
        world_model: Any,       # TrikaWorldModel (CUDA)
        efe_actor: Any,         # EFEActor — forward(h,z) → (Categorical, efe[B])
        citta_store: Any,       # CittaStore — recall(h[B,d]) → h[B,d]
        vimarsa_bridge: Any,    # VimarsaBridgeV2 — as_logits_processor(h) → Callable
        llm_backend: Any,       # LlamaCppBackend — stream(system,user,lp,...) → Iterator
        cfg: Optional[LoopConfig] = None,
    ):
        self.wm = world_model
        self.efe = efe_actor
        self.citta = citta_store
        self.bridge = vimarsa_bridge
        self.llm = llm_backend
        self.cfg = cfg or LoopConfig()
        self._device = torch.device(self.cfg.device)
        self._wm_states: Optional[list] = None

    def init(self, batch_size: int = 1):
        """Initialise WM hidden state for a new generation request."""
        self._wm_states = self.wm.init_state(batch_size, self._device)

    def run_stanza(
        self,
        stanza_idx: int,
        obs: Tensor,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[dict, None, StanzaResult]:
        """
        Execute all 6 Pañcakṛtya acts for one stanza.

        Yields SSE event dicts (domain-neutral). Returns StanzaResult (internal).
        """
        cfg = self.cfg
        dev = self._device

        # Auto-initialise WM state if run_stanza called without prior init()
        if self._wm_states is None:
            self.init(batch_size=obs.shape[0] if obs.dim() > 1 else 1)

        # ── Act 1: Cit (sṛṣṭi) — WM observe step ────────────────────────
        # observe_step(obs, action, states, step) → (new_states, logits_post, logits_prior)
        # h_t = new_states[0][0] : (B, hidden_dim)
        # z_t = new_states[0][1] : (B, stoch_dim, stoch_classes)
        a_t = torch.zeros(1, cfg.action_dim, device=dev)
        self._wm_states, logits_post, logits_prior = self.wm.observe_step(
            obs, a_t, self._wm_states, stanza_idx
        )
        h_t = self._wm_states[0][0]   # (1, hidden_dim)
        z_t = self._wm_states[0][1]   # (1, stoch_dim, stoch_classes)

        # VFE proxy: KL(posterior || prior) from level-0 logits
        vfe = _compute_vfe_proxy(logits_post[0], logits_prior[0])

        # ── Act 2: Ānanda — EFE actor ─────────────────────────────────────
        # EFEActor.forward(h, z) → (Categorical, efe: Tensor[B])
        _, efe_batch = self.efe(h_t, z_t)
        efe_score = float(efe_batch.mean())

        # ── Act 3: Icchā — Hopfield recall ───────────────────────────────
        # CittaStore.recall(query: Tensor[B, hidden_dim], mode) → Tensor[B, hidden_dim]
        mem_t = self.citta.recall(h_t, mode="episodic")   # (1, hidden_dim)
        mem_resonance = float(
            F.cosine_similarity(h_t, mem_t, dim=-1).mean()
        )

        # ── Act 4: Apohana — entropy gate (sphurattā detection) ──────────
        # Compute normalised entropy over z distribution
        z_probs = F.softmax(z_t.flatten().float(), dim=0)
        entropy = float(-torch.sum(z_probs * torch.log(z_probs + 1e-8)))
        max_entropy = math.log(z_probs.numel())
        norm_entropy = entropy / (max_entropy + 1e-8)
        sphuratta = norm_entropy > cfg.tau_sphuratta

        # ── Emit wm_state SSE event (domain-neutral labels — Contract 2) ─
        # aesthetic_quality: proxy from efe_score (negative EFE → better)
        aesthetic_quality = max(0.0, min(1.0, 1.0 - abs(efe_score) / 10.0))
        yield {
            "event": "wm_state",
            "data": {
                "energy": round(float(h_t.norm()), 3),
                "aesthetic_quality": round(aesthetic_quality, 3),
                "creative_peak": sphuratta,       # sphurattā → creative_peak
                "entropy": round(norm_entropy, 3),
                "prediction_error": round(vfe, 3),
                "stanza": stanza_idx,
            },
        }

        # ── Act 5: Jñāna — VimarsaBridge → logits_processor ─────────────
        # VimarsaBridgeV2.as_logits_processor(h_t) → Callable[[list,np.array], np.array]
        bias_fn = self.bridge.as_logits_processor(h_t)

        # ── Emit stanza_start ─────────────────────────────────────────────
        yield {"event": "stanza_start", "data": {"stanza": stanza_idx}}

        # ── Act 6: Kriyā — LLM stream ────────────────────────────────────
        generated_tokens: list[str] = []
        llm_ok = False
        try:
            for token_text in self.llm.stream(
                system=system_prompt,
                user=user_prompt,
                logits_processor=bias_fn,
                max_tokens=cfg.max_tokens_per_stanza,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            ):
                generated_tokens.append(token_text)
                yield {"event": "token", "data": {"text": token_text}}
                llm_ok = True
        except Exception as e:
            logger.warning(f"[PancakrtyaLoopV2] LLM stream error (s{stanza_idx}): {e}")
            # Contract 3: WM must survive LLM failure → stub output
            stub = _stub_generate(h_t, stanza_idx)
            generated_tokens = [stub]
            yield {"event": "token", "data": {"text": stub}}

        stanza_text = "".join(generated_tokens)

        # ── Post-kriyā: store h_t in episodic Hopfield memory ────────────
        # CittaStore.store_episode(h: Tensor[B, hidden_dim], level=0)
        self.citta.store_episode(h_t, level=0)

        # ── Camatkāra: weighted composite score ──────────────────────────
        # 0.30·(1-VFE/20) + 0.35·mem_resonance + 0.35·norm_entropy
        vfe_term = max(0.0, 1.0 - vfe / 20.0)
        camatk = max(0.0, min(1.0,
            0.30 * vfe_term
            + 0.35 * max(0.0, mem_resonance)
            + 0.35 * norm_entropy
        ))

        # ── Emit stanza_end ───────────────────────────────────────────────
        yield {
            "event": "stanza_end",
            "data": {
                "stanza": stanza_idx,
                "aesthetic_quality": round(camatk, 3),
                "memory_resonance": round(mem_resonance, 3),
                "selection_score": round(efe_score, 3),
                "prediction_error": round(vfe, 3),
            },
        }

        return StanzaResult(
            stanza_idx=stanza_idx,
            text=stanza_text,
            efe_score=efe_score,
            vfe=vfe,
            sphuratta=sphuratta,
            mem_resonance=mem_resonance,
            camatk_score=camatk,
            h_t=h_t.detach(),
        )

    def run(
        self,
        obs_sequence: list[Tensor],
        system_prompt: str,
        user_prompt_fn: Callable[[int, str], str],
    ) -> Generator[dict, None, None]:
        """
        Run full generation (n_stanzas). Yields all SSE events.

        Args:
            obs_sequence: list of obs tensors, one per stanza.
            system_prompt: LLM system message.
            user_prompt_fn: (stanza_idx, prev_stanza_text) → user message.
        """
        self.init()
        all_results: list[StanzaResult] = []
        prev_text = ""

        for i, obs in enumerate(obs_sequence):
            user_prompt = user_prompt_fn(i, prev_text)
            # Generator protocol: send exhausts the generator, gets return value
            gen = self.run_stanza(i, obs, system_prompt, user_prompt)
            result = None
            try:
                while True:
                    event = next(gen)
                    yield event
            except StopIteration as exc:
                result = exc.value

            if result is not None:
                prev_text = result.text
                all_results.append(result)

        mean_camatk = (
            sum(r.camatk_score for r in all_results) / len(all_results)
            if all_results else 0.0
        )
        yield {
            "event": "complete",
            "data": {
                "total_stanzas": len(all_results),
                "mean_aesthetic_quality": round(mean_camatk, 3),
                "creative_peaks": sum(1 for r in all_results if r.sphuratta),
                "generation_complete": True,
            },
        }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_vfe_proxy(logits_post: Tensor, logits_prior: Tensor) -> float:
    """KL(posterior || prior) as VFE proxy. Returns 0.0 on scalar/empty logits."""
    if logits_post.numel() <= 1:
        return 0.0
    try:
        lp = F.log_softmax(logits_post.reshape(1, -1).float(), dim=-1)
        pr = F.softmax(logits_prior.reshape(1, -1).float(), dim=-1)
        return float(F.kl_div(lp, pr, reduction="batchmean").clamp(0.0, 100.0).detach())
    except Exception:
        return 0.0


def _stub_generate(h_t: Tensor, stanza_idx: int) -> str:
    """
    Contract 3 fallback: generate a placeholder stanza when LLM is unavailable.
    Uses h_t norm and index to produce varied (but low quality) output.
    """
    energy = float(h_t.norm())
    templates = [
        "the light moves through still water\nmoment opens to moment",
        "sound carries across the valley\ntime folds into memory",
        "breath meets breath in the dark\nthe pattern holds its shape",
        "stone remembers the river's path\ncolor returns at dusk",
    ]
    return templates[stanza_idx % len(templates)] + f"\n[energy: {energy:.1f}]\n"
