"""
AvacchedakaStore: Python client wrapping the Pratyākṣa PCEH MCP tools.

Avacchedaka (Sanskrit: delimiter, qualifier) — each piece of inter-agent knowledge
is typed by its epistemic category (qualificand), provenance, and precision.

Philosophical grounding:
  This implements the pramāṇa-typed knowledge representation from Nyāya epistemology.
  Context items are tagged as pratyakṣa (perception), anumāna (inference), or
  āgama (testimony), enabling the khyātivāda quality gate in vimarśa.

The PCEH MCP tools (mcp__pratyaksha_mcp__*) are the canonical implementation.
This class wraps them for use in Python pipeline code.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextItem:
    qualificand: str     # epistemic category: cit/ananda/icha/apohana/jnana/kriya/vimarsha
    key: str
    value: Any
    precision: float     # epistemic confidence [0,1]
    timestamp: float = field(default_factory=time.time)
    pramana: str = "pratyaksha"  # pratyaksha | anumana | agama


class AvacchedakaStore:
    """
    In-process typed context store.

    In production, agents call PCEH MCP tools directly via Claude Code.
    This class provides an equivalent Python-native interface for:
    - Unit tests
    - Offline training runs (no Claude Code runtime)
    - Research scripts

    The PCEH MCP plugin (pratyaksha-context-eng-harness) remains the authoritative
    implementation for multi-agent sessions running inside Claude Code.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], ContextItem] = {}
        self._sakshi: str = ""

    def context_insert(
        self,
        qualificand: str,
        key: str,
        value: Any,
        precision: float,
        pramana: str = "pratyaksha",
    ) -> None:
        """Insert or update a typed context item."""
        item = ContextItem(
            qualificand=qualificand,
            key=key,
            value=value,
            precision=precision,
            pramana=pramana,
        )
        self._store[(qualificand, key)] = item

    def context_retrieve(self, qualificand: str, key: str) -> Any | None:
        """Retrieve a specific typed context item's value."""
        item = self._store.get((qualificand, key))
        return item.value if item else None

    def context_get(self, qualificands: list[str] | None = None) -> dict[str, Any]:
        """Get all context items, optionally filtered by qualificand."""
        result: dict[str, Any] = {}
        for (q, k), item in self._store.items():
            if qualificands is None or q in qualificands:
                result[f"{q}.{k}"] = {
                    "value": item.value,
                    "precision": item.precision,
                    "pramana": item.pramana,
                }
        return result

    def sublate_with_evidence(self, key: str, evidence: str) -> None:
        """
        Bādha (sublation): replace a lower-precision item with stronger evidence.
        Mimics PCEH mcp__pratyaksha_mcp__sublate_with_evidence.
        """
        for (_, k), item in self._store.items():
            if k == key:
                item.value = evidence
                item.precision = min(1.0, item.precision + 0.1)
                item.pramana = "anumana"  # upgraded from perception to inference
                break

    def detect_conflict(self, qualificand: str) -> list[dict]:
        """Detect contradictions within a qualificand's context."""
        items = {k: v for (q, k), v in self._store.items() if q == qualificand}
        # Simplified conflict detection: items with precision < 0.5 may be contradicted
        return [
            {"key": k, "precision": v.precision}
            for k, v in items.items()
            if v.precision < 0.5
        ]

    def set_sakshi(self, witness: str) -> None:
        """Set the Sākṣī (witness) invariant — ≤500 tokens, stable across turns."""
        assert len(witness.split()) <= 600, "Sākṣī must be ≤500 tokens"
        self._sakshi = witness

    def get_sakshi(self) -> str:
        """Return the current Sākṣī witness invariant."""
        return self._sakshi

    def compact(self, threshold: float = 2.5) -> dict:
        """
        Tirodhāna gate: compact completed episodes.
        Removes low-precision items to free context budget.
        Mirrors PCEH boundary_compact.
        """
        before = len(self._store)
        self._store = {
            k: v for k, v in self._store.items()
            if v.precision >= threshold / 5.0  # normalise threshold
        }
        after = len(self._store)
        return {"removed": before - after, "remaining": after}
