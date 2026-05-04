"""
CamatkaraNarrator: Converts sphurattā events into skill library entries.

Philosophical grounding:
  Camatkāra (Abhinavagupta, Locana ad DhvA 1.1): Aesthetic wonder — the sudden
  flash of recognition when dhvani (resonance) strikes the sahṛdaya (connoisseur).
  The narrator is the computational sahṛdaya: it receives the WM's camatkāra signal
  and translates latent wonder into language the LLM can reason about.

  Abhijñā-yukti (ĪPK 2.4.20): The skill of recognition — knowing what to save
  and how. The narrator applies abhijñā-yukti to decide which sphurattā events
  become permanent skill entries versus ephemeral narrations.

Architecture:
  Input: (h_t, z_t, camatk_score, vfe, context_str)
  Output: narration string + skill_entry dict (for SkillLibrary)
  Uses LLM jñāna path for narration; stores to SkillLibrary at high quality.
  Threshold: camatk_score > 0.7 AND quality_score > 0.6 → permanent skill entry.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import Tensor


@dataclass
class NarrationResult:
    """Output of the CamatkaraNarrator for one sphurattā event."""
    narration: str
    camatk_score: float
    quality_score: float
    step: int
    committed_to_skill: bool = False
    skill_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CamatkaraNarrator:
    """
    Translates sphurattā events into narrations and skill library entries.

    Fires only on genuine sphurattā (camatkāra events). Generates a narration via
    the LLM āgama, then conditionally commits to the SkillLibrary for reuse.
    """

    SKILL_COMMIT_THRESHOLD = 0.7   # camatk_score above which to commit as skill
    QUALITY_COMMIT_THRESHOLD = 0.6  # LLM quality score to also pass

    def __init__(
        self,
        llm_backend: Any,
        skill_library: Any = None,
        bridge: Any = None,
    ) -> None:
        self._llm = llm_backend
        self._skills = skill_library
        self._bridge = bridge
        self._step_counter = 0

    def narrate(
        self,
        h: Tensor,
        z: Tensor,  # reserved for future latent-conditioned prompting  # noqa: ARG002
        camatk_score: float,
        vfe: float,
        context: str = "",
        step: int = 0,
    ) -> NarrationResult:
        """
        Generate a narration for a sphurattā event.

        Args:
            h: (B, hidden_dim) WM hidden state
            z: (B, stoch_dim, n_cats) latent sample
            camatk_score: camatkāra reward signal [0, 1]
            vfe: current variational free energy
            context: optional prior narration context
            step: current training step
        Returns:
            NarrationResult with narration + quality + commit status
        """
        prefix = ""
        if self._bridge is not None:
            prefix = self._bridge.format_prefix_text(h)

        prompt = self._build_prompt(prefix, camatk_score, vfe, context, step)
        narration = self._call_llm(prompt)
        quality = self._estimate_quality(narration, camatk_score)

        # Commit to skill library if high quality
        committed = False
        skill_key = ""
        if (
            self._skills is not None
            and camatk_score >= self.SKILL_COMMIT_THRESHOLD
            and quality >= self.QUALITY_COMMIT_THRESHOLD
        ):
            skill_key = f"camatk_{step:07d}"
            self._skills.add(
                skill_id=skill_key,
                description=narration[:256],
                embedding=self._narration_embedding(h),
                metadata={
                    "camatk_score": camatk_score,
                    "vfe": vfe,
                    "step": step,
                    "quality": quality,
                },
            )
            committed = True

        return NarrationResult(
            narration=narration,
            camatk_score=camatk_score,
            quality_score=quality,
            step=step,
            committed_to_skill=committed,
            skill_key=skill_key,
            metadata={"vfe": vfe, "prefix": prefix[:64]},
        )

    def _build_prompt(
        self,
        prefix: str,
        camatk_score: float,
        vfe: float,
        context: str,
        step: int,
    ) -> str:
        return (
            f"{prefix}"
            f"Step {step}: Creative intensity (camatkāra) = {camatk_score:.3f}, "
            f"surprise (VFE) = {vfe:.4f}.\n"
            f"{'Prior context: ' + context[:200] if context else ''}\n\n"
            "Compose a single evocative sentence (in English, with a Sanskrit aesthetic term) "
            "capturing this moment of creative recognition (pratyabhijñā). "
            "Convey wonder (camatkāra) without cliché."
        )

    def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            return "Consciousness recognises itself in this moment of creative emergence."
        try:
            return str(self._llm.call(
                role="jnana",
                system=(
                    "You are a Kashmirian aesthetician expressing creative wonder. "
                    "Reply with a single evocative sentence only."
                ),
                prompt=prompt,
                max_tokens=128,
            )).strip()
        except Exception:
            return "A flash of sphurattā illuminates the latent creative space."

    def _estimate_quality(self, narration: str, camatk_score: float) -> float:
        """Heuristic quality score: length + Sanskrit term presence + camatk alignment."""
        has_sanskrit = any(
            term in narration.lower()
            for term in ["sphuratṭā", "sphurattā", "camat", "pratyabhijñā", "vimarśa",
                         "spanda", "camatkāra", "ānanda", "śiva", "śakti"]
        )
        length_score = min(1.0, len(narration.split()) / 20.0)
        return float(0.4 * camatk_score + 0.4 * length_score + 0.2 * has_sanskrit)

    def _narration_embedding(self, h: Tensor) -> list[float]:
        """Use WM hidden state mean as embedding proxy (no extra model needed)."""
        with torch.no_grad():
            return h.mean(0)[:64].cpu().float().tolist()
