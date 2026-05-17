"""
CamatkaraNarrator: Converts sphurattā events into skill library entries.
WMReasoningTrace: Pre-computes WM deliberation as LLM think-block (S19, ADR-002).

Philosophical grounding:
  Camatkāra (Abhinavagupta, Locana ad DhvA 1.1): Aesthetic wonder — the sudden
  flash of recognition when dhvani (resonance) strikes the sahṛdaya (connoisseur).
  The narrator is the computational sahṛdaya: it receives the WM's camatkāra signal
  and translates latent wonder into language the LLM can reason about.

  Abhijñā-yukti (ĪPK 2.4.20): The skill of recognition — knowing what to save
  and how. The narrator applies abhijñā-yukti to decide which sphurattā events
  become permanent skill entries versus ephemeral narrations.

  Vimarśa (ĪPK 1.5.11, Utpaladeva): Reflexive self-recognition — the cognitive act
  in which consciousness holds itself before itself. WMReasoningTrace renders the
  WM's vimarśa as a `<think>…</think>` block so the 120B LLM receives the deliberation
  that vimarśa has already performed, collapsing the 60s reasoning phase to ~3s prefill.
  The WM is not an approximation of the LLM's CoT — the CoT was an approximation of
  the WM's vimarśa. (ADR-002, TRIZ Sketch A, IFR 4/4.)

Architecture:
  Input: (h_t, z_t, camatk_score, vfe, context_str)
  Output: narration string + skill_entry dict (for SkillLibrary)
  Uses LLM jñāna path for narration; stores to SkillLibrary at high quality.
  Threshold: camatk_score > 0.7 AND quality_score > 0.6 → permanent skill entry.

WMReasoningTrace:
  Input: (h_t, z_t, domain, creative_metadata, sphuratta_events, citta_hits)
  Output: str — ready to inject as assistant-prefill think-block
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
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


# ──────────────────────────────────────────────────────────────────────────────
# WMReasoningTrace — TRIZ Sketch A (ADR-002, IFR 4/4)
# ──────────────────────────────────────────────────────────────────────────────

class WMReasoningTrace:
    """
    Renders the WM's vimarśa as a <think>…</think> block.

    The 120B reasoning model's CoT asks: "what should this lyric express, given
    context X?" — precisely the question the WM posterior q_φ(z_t|h_t,o_t),
    CamatkaraNarrator, and Citta-store retrievals already answer at WM tick rate.

    WMReasoningTrace.render() materialises that answer as structured text so the
    LLM receives deliberation-complete context. When injected as an assistant-prefill
    message (`{"role":"assistant","content":"<think>…</think>"}`), the model's
    autoregressive machinery treats the reasoning phase as complete and emits content
    tokens directly — collapsing TTFT from ~60s to ~3s of prefill.

    Philosophical source: Vimarśa (ĪPK 1.5.11, Utpaladeva) — reflexive self-recognition.
    The WM IS the vimarśa; the LLM's CoT was duplicating it.
    """

    def render(
        self,
        h_t: Tensor,
        domain: str,
        creative_metadata: Optional[Any] = None,
        sphuratta_events: Optional[list[dict]] = None,
        citta_hits: Optional[list[str]] = None,
        stanza_idx: int = 0,
        camatk_score: Optional[float] = None,
        vfe: Optional[float] = None,
    ) -> str:
        """
        Render the WM's deliberation as a <think>…</think> string.

        Args:
            h_t:              WM hidden state (B, hidden_dim) or (hidden_dim,).
            domain:           Creative domain (e.g. "kannada_film").
            creative_metadata: Optional CreativeMetadata from WMStateDecoder.
            sphuratta_events: List of sphurattā event dicts from the current run.
            citta_hits:       Episodic/semantic Hopfield retrievals (list of strs).
            stanza_idx:       Current stanza number (0-indexed).
            camatk_score:     Camatkāra aesthetic reward [0, 1] if available.
            vfe:              Variational free energy at current step if available.

        Returns:
            str — the full `<think>…</think>` block, including tags.
            Inject as assistant-prefill: {"role":"assistant","content":"<think>…</think>"}
        """
        with torch.no_grad():
            h = h_t.detach().float().flatten()
            energy = float(h.norm().item())
            entropy = float(h.softmax(0).mul(h.softmax(0).clamp(min=1e-8).log()).sum().neg().item())

        # Domain label for the trace (human-readable, not śaiva vocabulary)
        domain_labels = {
            "kannada_film": "Kannada film song",
            "carnatic": "Carnatic classical rāga composition",
            "hindustani": "Hindustani classical composition",
            "hindi_film": "Hindi film song",
            "western_pop": "English pop song",
            "western_jazz": "jazz standard",
            "world_fusion": "world fusion lyric",
            "tamil_classical": "Tamil classical lyric",
            "telugu_padyam": "Telugu lyric poem",
            "bengali_lyric": "Bengali lyric poem",
            "english_romantic": "English Romantic lyric",
            "english_modernist": "English modernist poem",
            "english_beat": "Beat poetry",
            "sanskrit_classical": "Sanskrit kāvya verse",
        }
        domain_label = domain_labels.get(domain, domain.replace("_", " "))

        # Build the deliberation trace line by line
        lines: list[str] = [
            f"Context: stanza {stanza_idx + 1}, domain={domain_label}.",
        ]

        # WM energy / register interpretation
        if energy < 2.0:
            register_note = "WM state: low energy — suggest restrained, introspective register."
        elif energy < 5.0:
            register_note = "WM state: moderate energy — balanced lyric register."
        elif energy < 10.0:
            register_note = "WM state: elevated energy — expressive, forward-moving register."
        else:
            register_note = "WM state: high energy — peak emotional intensity."
        lines.append(register_note)

        # Posterior entropy interpretation
        if entropy < 2.0:
            lines.append("Posterior: low entropy — WM is confident about next creative direction.")
        elif entropy < 4.0:
            lines.append("Posterior: moderate entropy — some creative ambiguity; favour surprising imagery.")
        else:
            lines.append("Posterior: high entropy — open creative space; any direction is viable.")

        # Creative metadata from WMStateDecoder (if available)
        if creative_metadata is not None:
            meta = creative_metadata
            if hasattr(meta, "raga_hint") and meta.raga_hint:
                lines.append(f"Rāga/mode hint: {meta.raga_hint}.")
            if hasattr(meta, "section_name") and meta.section_name:
                lines.append(f"Section: {meta.section_name}.")
            if hasattr(meta, "emotion_tags") and meta.emotion_tags:
                lines.append(f"Emotion register: {', '.join(meta.emotion_tags[:3])}.")
            if hasattr(meta, "tempo_hint") and meta.tempo_hint:
                lines.append(f"Tempo feel: {meta.tempo_hint}.")

        # Sphurattā events (aesthetic peaks from prior stanzas)
        if sphuratta_events:
            peak_count = len([e for e in sphuratta_events if e.get("camatk_score", 0) > 0.7])
            if peak_count > 0:
                lines.append(
                    f"Aesthetic peaks observed: {peak_count} high-quality moment(s) in prior stanzas. "
                    "Maintain or escalate creative intensity."
                )
            else:
                lines.append("No strong aesthetic peaks yet — build toward emotional climax.")

        # Citta episodic/semantic retrievals
        if citta_hits:
            # Surface up to 2 retrievals to avoid bloating the trace
            for hit in citta_hits[:2]:
                hit_short = str(hit)[:100].replace("\n", " ")
                lines.append(f"Episodic memory resonance: \"{hit_short}\".")

        # Camatkāra and VFE readings
        if camatk_score is not None:
            if camatk_score > 0.7:
                lines.append(f"Aesthetic reward high (camatkāra={camatk_score:.2f}) — preserve stylistic approach.")
            elif camatk_score > 0.4:
                lines.append(f"Aesthetic reward moderate (camatkāra={camatk_score:.2f}) — room to deepen imagery.")
            else:
                lines.append(f"Aesthetic reward low (camatkāra={camatk_score:.2f}) — change approach; try unexpected metaphor.")
        if vfe is not None:
            lines.append(f"Predictive surprise (VFE={vfe:.4f}) — higher surprise correlates with novel output.")

        # Conclusion: what the WM recommends to the LLM
        lines.append(
            "Recommendation: write one stanza consistent with the above creative state. "
            "Prioritise evocative imagery over narrative explanation. "
            "Match the register and emotional arc implied by the WM reading."
        )

        body = "\n".join(lines)
        return f"<think>\n{body}\n</think>"

    def render_as_assistant_prefill(
        self,
        h_t: Tensor,
        domain: str,
        creative_metadata: Optional[Any] = None,
        sphuratta_events: Optional[list[dict]] = None,
        citta_hits: Optional[list[str]] = None,
        stanza_idx: int = 0,
        camatk_score: Optional[float] = None,
        vfe: Optional[float] = None,
    ) -> dict:
        """
        Return an OpenAI-format assistant message dict containing the think-block.

        Inject this as the last message before the final user prompt:
          messages = [system_msg, *history, think_msg, user_msg]

        The model sees the think block as its own completed reasoning and emits
        content tokens immediately (no 60s reasoning phase).
        """
        trace = self.render(
            h_t=h_t,
            domain=domain,
            creative_metadata=creative_metadata,
            sphuratta_events=sphuratta_events,
            citta_hits=citta_hits,
            stanza_idx=stanza_idx,
            camatk_score=camatk_score,
            vfe=vfe,
        )
        return {"role": "assistant", "content": trace}
