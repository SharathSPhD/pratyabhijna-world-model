"""
DECKARD: LLM-assisted world model (AWM) proposals for creative planning.

Philosophical grounding:
  Icchā-śakti (ĪPK 2.3, Utpaladeva): The power of will — the intentional
  creative impulse that precedes action. In PWM, icchā manifests as the LLM's
  generation of high-level creative intentions that the WM then executes
  through the EFE actor.

  DECKARD (Language Model Assisted World Models, Zhu et al. 2023): Uses frozen
  LLM to propose abstract world model transitions, enriching the RSSM imagination
  with semantic creativity priors. The LLM proposes; the WM executes.

  The name honours the tension in Blade Runner: DECKARD is the test for authentic
  experience — so too, icchā-śakti tests whether the WM's creative will is genuine.

Architecture:
  Input: current narration + skill context + step
  Output: list[dict] of proposed creative intentions (AWM transitions)
  LLM role: "icccha" (will/intention), temperature=0.9 for diversity
  Phase 5+: activates when vimarśa bridge is live.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import re


@dataclass
class AWMProposal:
    """A single abstract world model transition proposed by the LLM."""
    intention: str               # high-level creative intention (text)
    target_rasa: str             # target aesthetic emotion (rasa)
    suggested_action_bias: list[float] = field(default_factory=list)  # action logit bias
    confidence: float = 0.5
    raw: str = ""


class DECKARDPlanner:
    """
    DECKARD-style LLM creative planner.

    Generates abstract creative intentions from the current WM context, which
    the EFEActor can use as soft action priors during imagination rollouts.
    """

    NINE_RASAS = [
        "śṛṅgāra",   # love/beauty
        "hāsya",     # humour
        "karuṇā",    # compassion
        "raudra",    # fury
        "vīra",      # heroism
        "bhayānaka", # fear
        "bībhatsa",  # disgust
        "adbhuta",   # wonder
        "śānta",     # tranquility
    ]

    def __init__(self, llm_backend: Any, n_proposals: int = 3) -> None:
        self._llm = llm_backend
        self.n_proposals = n_proposals

    def propose(
        self,
        narration: str,
        step: int,
        skill_context: str = "",
        current_rasa: str = "adbhuta",
    ) -> list[AWMProposal]:
        """
        Generate AWM creative intention proposals from current WM context.

        Args:
            narration: current narration from CamatkaraNarrator
            step: current training step
            skill_context: retrieved similar skills from SkillLibrary
            current_rasa: current dominant aesthetic mode
        Returns:
            List of AWMProposals for the EFEActor to bias toward
        """
        if self._llm is None:
            return self._default_proposals()

        prompt = self._build_prompt(narration, step, skill_context, current_rasa)
        try:
            response = self._llm.call(
                role="icccha",
                system=(
                    "You are a Kashmirian aesthetician planning creative trajectories. "
                    "Respond ONLY with valid JSON: a list of objects, each with keys "
                    "'intention' (str), 'target_rasa' (str), 'confidence' (float 0-1)."
                ),
                prompt=prompt,
                max_tokens=512,
            )
            return self._parse_proposals(str(response))
        except Exception:
            return self._default_proposals()

    def _build_prompt(
        self,
        narration: str,
        step: int,
        skill_context: str,
        current_rasa: str,
    ) -> str:
        rasas_str = ", ".join(self.NINE_RASAS)
        return (
            f"Step {step}. Current creative state:\n"
            f"Narration: {narration[:300]}\n"
            f"Current rasa: {current_rasa}\n"
            f"{'Relevant past skills: ' + skill_context[:200] if skill_context else ''}\n\n"
            f"Propose {self.n_proposals} creative intentions for the next creative arc. "
            f"Each should specify a target rasa from: {rasas_str}. "
            "Aim for aesthetic progression, contrast, or deepening. "
            "Return JSON list only."
        )

    def _parse_proposals(self, response: str) -> list[AWMProposal]:
        # Extract JSON array from response
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if not match:
            return self._default_proposals()
        try:
            items = json.loads(match.group())
            proposals = []
            for item in items[:self.n_proposals]:
                proposals.append(AWMProposal(
                    intention=str(item.get("intention", "explore")),
                    target_rasa=str(item.get("target_rasa", "adbhuta")),
                    confidence=float(item.get("confidence", 0.5)),
                    raw=response[:128],
                ))
            return proposals
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._default_proposals()

    def _default_proposals(self) -> list[AWMProposal]:
        """Fallback proposals when LLM unavailable."""
        return [
            AWMProposal("deepen aesthetic wonder", "adbhuta", confidence=0.5),
            AWMProposal("introduce harmonic contrast", "śṛṅgāra", confidence=0.4),
            AWMProposal("resolve into tranquility", "śānta", confidence=0.3),
        ]
