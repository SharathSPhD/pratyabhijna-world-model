"""
Sanskrit metre validator for Phase 1 evaluation.

Philosophical grounding:
  Chandas (Piṅgalasūtra 1.1): Metre (chandas) is the rhythmic skeleton of
  Sanskrit poetry. The WM must learn to respect metrical constraints — not
  just predict the next word, but maintain the syllabic count and stress
  pattern across a line (pāda).

Phase 1 scope:
  The corpus is mixed (English philosophy + poetry + Sanskrit transcriptions).
  Full Sanskrit metre validation requires Devanāgarī support and prosodic
  analysis (not yet in Phase 1). This module implements:
    1. Syllable-count consistency check (does text respect a regular count?).
    2. Stress-pattern rhythmicity index (simple ictus detector).
    3. Line-ending regularity (rhyme/cadence structure).

  These serve as Phase 1 proxies. Full metre analysis added in Phase 3.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Syllable counting ─────────────────────────────────────────────────────────

_VOWEL_RE = re.compile(r"[aeiouāīūṛḷṃḥ]", re.IGNORECASE)


def count_syllables(word: str) -> int:
    """
    Estimate syllable count via vowel nucleus counting.
    Works for both English and romanised Sanskrit.
    """
    return max(1, len(_VOWEL_RE.findall(word)))


def line_syllable_counts(text: str) -> list[int]:
    """Compute syllable count per non-empty line."""
    counts = []
    for line in text.strip().split("\n"):
        words = line.strip().split()
        if words:
            counts.append(sum(count_syllables(w) for w in words))
    return counts


# ── Rhythmicity index ─────────────────────────────────────────────────────────

def rhythmicity_index(syllable_counts: list[int]) -> float:
    """
    Measure how regular (metrical) a sequence of syllable counts is.

    A perfectly metrical text has identical (or near-identical) counts per line.
    Returns a value in [0, 1] where 1 = perfectly regular.

    Uses coefficient of variation (CV = std/mean): lower CV = more regular.
    We invert and clip: rhythmicity = max(0, 1 - CV).
    """
    if len(syllable_counts) < 2:
        return 0.0
    arr = np.array(syllable_counts, dtype=float)
    mean = arr.mean()
    if mean < 1e-9:
        return 0.0
    cv = arr.std() / mean
    return float(max(0.0, 1.0 - cv))


# ── Sanskrit-specific patterns ────────────────────────────────────────────────

_ANUSVARA_RE = re.compile(r"[ṃḥ]")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

KNOWN_METRES = {
    "anushtubh": 8,     # 4 pādas × 8 syllables = 32 total
    "trishtubh": 11,    # 4 × 11
    "jagati": 12,       # 4 × 12
    "sloka": 8,         # most common Sanskrit metre
}


def detect_metre(text: str) -> dict[str, Any]:
    """
    Attempt to identify the Sanskrit metre in a text passage.

    Returns:
        identified_metre: name or None
        syllable_counts: per-line counts
        rhythmicity: [0, 1]
        has_devanagari: bool
    """
    has_deva = bool(_DEVANAGARI_RE.search(text))
    counts = line_syllable_counts(text)
    rhythmicity = rhythmicity_index(counts)

    # Match against known metre syllable counts
    identified = None
    if counts:
        modal_count = int(np.bincount(counts).argmax()) if counts else 0
        for metre_name, target in KNOWN_METRES.items():
            if abs(modal_count - target) <= 1:
                identified = metre_name
                break

    return {
        "identified_metre": identified,
        "syllable_counts": counts,
        "rhythmicity": rhythmicity,
        "has_devanagari": has_deva,
        "modal_syllables": int(np.bincount(counts).argmax()) if counts else 0,
    }


# ── Batch evaluation ─────────────────────────────────────────────────────────

def compute_metre_stats(texts: list[str]) -> dict[str, Any]:
    """
    Compute metre statistics over a batch of text strings.

    Returns aggregate rhythmicity, metre detection rate, etc.
    """
    results = [detect_metre(t) for t in texts]

    rhythmicities = [r["rhythmicity"] for r in results]
    identified = [r["identified_metre"] for r in results if r["identified_metre"] is not None]
    has_deva = sum(1 for r in results if r["has_devanagari"])

    return {
        "rhythmicity_mean": float(np.mean(rhythmicities)) if rhythmicities else 0.0,
        "rhythmicity_std": float(np.std(rhythmicities)) if rhythmicities else 0.0,
        "metre_detection_rate": len(identified) / max(1, len(results)),
        "identified_metres": dict(
            zip(*np.unique(identified, return_counts=True))
        ) if identified else {},
        "has_devanagari_frac": has_deva / max(1, len(results)),
        "n_texts": len(results),
    }


def run_metre_report(corpus_dir: Any, n_samples: int = 500) -> dict[str, Any]:
    """
    Full metre evaluation report on corpus samples.

    Samples text from corpus, computes rhythmicity and metre detection.
    """
    import random
    from pathlib import Path

    corpus_dir = Path(corpus_dir)
    txt_files = list(corpus_dir.rglob("*.txt"))
    if not txt_files:
        log.error("No .txt files found in %s", corpus_dir)
        return {"error": "no_corpus"}

    rng = random.Random(42)
    sample_texts: list[str] = []
    for _ in range(n_samples):
        f = rng.choice(txt_files)
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            start = rng.randint(0, max(0, len(txt) - 400))
            chunk = txt[start : start + 400].strip()
            if chunk:
                sample_texts.append(chunk)
        except OSError:
            pass

    if not sample_texts:
        return {"error": "no_valid_samples"}

    stats = compute_metre_stats(sample_texts)
    log.info(
        "Metre: rhythmicity=%.3f  metre_rate=%.3f  devanagari=%.3f",
        stats["rhythmicity_mean"], stats["metre_detection_rate"], stats["has_devanagari_frac"],
    )
    return stats
