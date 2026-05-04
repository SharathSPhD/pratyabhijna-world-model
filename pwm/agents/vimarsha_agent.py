"""
VimarshaAgent: The deliberative commit/revise/reject gate.

Philosophical grounding:
  Vimarśa (ĪPK 1.5.11, 2.3.11; Abhinavagupta, Tantrāloka 1.24–1.33):
  'Self-reflexive cognition' — Śiva's act of recognising Himself in His own
  manifestation. Vimarśa is not mere reflection but the evaluative creative act
  that distinguishes genuine recognition (sphurattā) from mechanical reaction.

  In PWM: VimarshaAgent fires ONLY on sphurattā events. It receives the WM's
  camatkāra signal and LLM narration, then deliberates (commit/revise/reject)
  using tool-calling. This is the ONLY true smolagents agent in the pipeline —
  all other śakti cascade steps share the continuous WM state (h_t, z_t) in
  a single Python call stack.

  The commit/revise/reject gate mirrors Abhinavagupta's three epistemic
  operations: abhāva (absence detection → reject), ākhyāti (mis-recognition →
  revise), and pratyabhijñā (recognition → commit).

Architecture:
  Uses smolagents ToolCallingAgent with three tools:
    EvaluateCreativityTool: checks camatkāra score against threshold
    ReviseNarrationTool:    rewrites poor narrations via LLM (fast path)
    CommitNarrationTool:    writes to AvacchedakaStore under 'vimarsha' qualificand
  max_revisions=3 (from LoopConfig)

Import guard: if smolagents not installed, VimarshaStub provides identity fallback.
"""

from __future__ import annotations
from typing import Any


try:
    from smolagents import ToolCallingAgent, Tool  # type: ignore[import]
    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False


# ── Tools ─────────────────────────────────────────────────────────────────────

if _HAS_SMOLAGENTS:
    class EvaluateCreativityTool(Tool):  # type: ignore[misc]
        """Score the camatkāra signal against its running threshold."""
        name = "evaluate_creativity"
        description = (
            "Evaluate whether the current camatkāra signal constitutes a genuine"
            " sphurattā event. Input: {'vfe': float, 'camatk': float, 'threshold': float}."
            " Returns: {'passes': bool, 'quality': float}."
        )
        inputs = {
            "vfe": {"type": "number", "description": "Current VFE value"},
            "camatk": {"type": "number", "description": "Current camatkāra reward"},
            "threshold": {"type": "number", "description": "Threshold for sphurattā"},
        }
        output_type = "object"

        def forward(self, vfe: float, camatk: float, threshold: float) -> dict:  # type: ignore[override]
            quality = float(camatk) / (float(threshold) + 1e-8)
            passes = quality > 0.5 and float(vfe) < float(threshold)
            return {"passes": passes, "quality": min(1.0, quality)}

    class ReviseNarrationTool(Tool):  # type: ignore[misc]
        """Rewrite a low-quality narration using the LLM fast path."""
        name = "revise_narration"
        description = (
            "Rewrite a narration that failed the quality threshold."
            " Input: {'narration': str, 'feedback': str}."
            " Returns: {'revised': str}."
        )
        inputs = {
            "narration": {"type": "string", "description": "Original narration"},
            "feedback": {"type": "string", "description": "Quality feedback"},
        }
        output_type = "object"

        def __init__(self, llm_backend: Any) -> None:
            super().__init__()
            self._llm = llm_backend

        def forward(self, narration: str, feedback: str) -> dict:  # type: ignore[override]
            if self._llm is None:
                return {"revised": narration}
            prompt = (
                f"Revise this creative narration to better express camatkāra"
                f" (aesthetic wonder).\n\nFeedback: {feedback}\n\n"
                f"Original: {narration}\n\nRevised:"
            )
            try:
                revised = self._llm.call(
                    role="jnana",
                    system="You are a Kashmirian aesthetician. Revise narrations for camatkāra quality.",
                    prompt=prompt,
                    max_tokens=256,
                )
                return {"revised": revised.strip()}
            except Exception:
                return {"revised": narration}

    class CommitNarrationTool(Tool):  # type: ignore[misc]
        """Commit a narration to the AvacchedakaStore."""
        name = "commit_narration"
        description = (
            "Write a committed narration to the context store under 'vimarsha' qualificand."
            " Input: {'narration': str, 'quality': float, 'step': int}."
            " Returns: {'committed': bool}."
        )
        inputs = {
            "narration": {"type": "string", "description": "Narration to commit"},
            "quality": {"type": "number", "description": "Quality score [0,1]"},
            "step": {"type": "integer", "description": "Current WM step"},
        }
        output_type = "object"

        def __init__(self, context_store: Any) -> None:
            super().__init__()
            self._ctx = context_store

        def forward(self, narration: str, quality: float, step: int) -> dict:  # type: ignore[override]
            if self._ctx is not None:
                self._ctx.context_insert(
                    qualificand="vimarsha",
                    key=f"narration_{step}",
                    value=narration,
                    precision=float(quality),
                    pramana="anumana",
                )
            return {"committed": True}


# ── Agent ─────────────────────────────────────────────────────────────────────

class VimarshaAgent:
    """
    Deliberative gate firing on sphurattā events.

    Encapsulates a smolagents ToolCallingAgent with evaluate/revise/commit tools.
    Falls back to VimarshaStub if smolagents not installed.
    """

    def __init__(
        self,
        llm_backend: Any,
        context_store: Any = None,
        max_revisions: int = 3,
        camatk_threshold: float = 0.5,
        model_id: str = "openai/nemotron-30b",
    ) -> None:
        self.llm = llm_backend
        self.ctx = context_store
        self.max_revisions = max_revisions
        self.camatk_threshold = camatk_threshold
        self._agent: Any = None

        if _HAS_SMOLAGENTS:
            try:
                from smolagents import OpenAIServerModel  # type: ignore[import]
                tools = [
                    EvaluateCreativityTool(),
                    ReviseNarrationTool(llm_backend),
                    CommitNarrationTool(context_store),
                ]
                sm_model = OpenAIServerModel(model_id=model_id)
                self._agent = ToolCallingAgent(  # type: ignore[possibly-unbound]
                    tools=tools,
                    model=sm_model,
                    max_steps=max_revisions + 2,
                )
            except Exception:
                self._agent = None

    def run(
        self,
        _h: Any,           # Tensor — reserved for future latent-conditioned prompting
        _z: Any,           # Tensor
        vfe: float,
        narration: str,
        step: int = 0,
    ) -> dict[str, Any]:
        """
        Gate: evaluate → revise up to max_revisions → commit or reject.

        Returns dict with committed, final_narration, revision_count, quality_score.
        """
        if self._agent is None or not _HAS_SMOLAGENTS:
            return _stub_result(narration)

        task = (
            f"Step {step}: VFE={vfe:.4f}, threshold={self.camatk_threshold:.4f}.\n"
            f"Narration: '{narration[:200]}'\n\n"
            f"1. evaluate_creativity(vfe={vfe:.4f}, camatk={abs(vfe):.4f},"
            f" threshold={self.camatk_threshold:.4f})\n"
            f"2. If passes: commit_narration(narration='...', quality=..., step={step})\n"
            f"3. If not passes: revise_narration(narration='...', feedback='too weak'),"
            f" then evaluate again, up to {self.max_revisions} attempts.\n"
            f"4. If still failing after revisions: do NOT commit."
        )

        try:
            result = self._agent.run(task)
            # Parse agent result
            committed = isinstance(result, dict) and result.get("committed", False)
            quality = isinstance(result, dict) and float(result.get("quality", 0.0))
            final_narration = (
                result.get("revised", narration)
                if isinstance(result, dict)
                else narration
            )
        except Exception:
            committed = False
            quality = 0.0
            final_narration = narration

        return {
            "committed": committed,
            "final_narration": final_narration,
            "revision_count": 0,
            "quality_score": quality,
        }


def _stub_result(narration: str) -> dict[str, Any]:
    """Pass-through when smolagents not available."""
    return {
        "committed": bool(narration),
        "final_narration": narration,
        "revision_count": 0,
        "quality_score": 0.5,
    }


class VimarshaStub:
    """Identity fallback when smolagents is not installed."""

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return _stub_result(kwargs.get("narration", ""))
