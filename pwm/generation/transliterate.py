"""
transliterate.py — ISO 15919 / IAST transliteration for multilingual creative output.

Philosophical grounding (CLAUDE.md §9):
  Vimarśa (ĪPK 1.5.11, Utpaladeva): reflexive cognition that renders the inner
  creative act into expressible form. Transliteration is the kriyā step that makes
  non-Latin scripts legible to international readers without losing phonetic fidelity.

Computational realisation:
  Script auto-detection via Unicode character names → ISO 15919 / IAST romanisation
  using indic_transliteration (supports Devanāgarī, Tamil, Telugu, Kannada, Bengali).

Sprint 5 (TRIZ Principle 10 — Prior Action):
  Transliteration is pre-computed alongside generation so the paper LaTeX can use
  \\textit{...} for IAST annotations without a separate manual pass.

Usage:
  from pwm.generation.transliterate import transliterate_text, TranslitResult

  result = transliterate_text("ಮಳೆ ಬಂದೆ ಮಣ್ಣು")
  print(result.iast)      # "maḻè baṃdè maṇṇu"
  print(result.script)    # "kannada"
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

Script = Literal["devanagari", "tamil", "telugu", "kannada", "bengali", "latin", "mixed", "unknown"]

# Map detected script → indic_transliteration constant
_SCRIPT_MAP: dict[str, str] = {
    "devanagari": "DEVANAGARI",
    "tamil":      "TAMIL",
    "telugu":     "TELUGU",
    "kannada":    "KANNADA",
    "bengali":    "BENGALI",
}

# Unicode block ranges for fast script detection (complement to unicodedata)
_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "devanagari"),   # Devanāgarī (Hindi, Sanskrit, Marathi)
    (0x0B80, 0x0BFF, "tamil"),         # Tamil
    (0x0C00, 0x0C7F, "telugu"),        # Telugu
    (0x0C80, 0x0CFF, "kannada"),       # Kannada
    (0x0980, 0x09FF, "bengali"),       # Bengali
]


@dataclass
class TranslitResult:
    """Result of transliterating a creative text fragment.

    Attributes:
        original: The raw text as produced by the LLM.
        iast: ISO 15919 / IAST romanisation of the Indic portions.
              Latin-script segments are passed through unchanged.
        script: Dominant script detected in the text.
        has_indic: True when at least one Indic character was found.
        mixed_language: True when text contains both Latin and Indic segments.
    """
    original: str
    iast: str
    script: Script
    has_indic: bool
    mixed_language: bool

    def latex_annotation(self) -> str:
        """Return LaTeX-safe IAST annotation suitable for paper figures.

        Uses \\textit{} for IAST romanisation (IEEE LaTeX convention).
        """
        if not self.has_indic:
            return self.original
        return f"\\textit{{{self.iast}}}"


def _detect_char_script(ch: str) -> Script | None:
    """Return script name for a single Unicode character, or None if Latin/other."""
    cp = ord(ch)
    for lo, hi, name in _RANGES:
        if lo <= cp <= hi:
            return name  # type: ignore[return-value]
    # Fallback: unicodedata name check
    name = unicodedata.name(ch, "")
    for script in ("KANNADA", "TAMIL", "TELUGU", "DEVANAGARI", "BENGALI"):
        if script in name:
            return script.lower()  # type: ignore[return-value]
    return None


def detect_dominant_script(text: str) -> Script:
    """Return the most frequent Indic script in text, or 'latin' / 'unknown'."""
    counts: dict[str, int] = {}
    has_latin = False
    for ch in text:
        if ch.isalpha():
            script = _detect_char_script(ch)
            if script:
                counts[script] = counts.get(script, 0) + 1
            elif ord(ch) < 128:
                has_latin = True

    if not counts:
        return "latin" if has_latin else "unknown"
    dominant = max(counts, key=counts.__getitem__)
    return dominant  # type: ignore[return-value]


def _transliterate_segment(segment: str, script: str) -> str:
    """Transliterate a single-script segment to IAST."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _translit

        src_scheme = getattr(sanscript, _SCRIPT_MAP[script], None)
        if src_scheme is None:
            return segment
        return _translit(segment, src_scheme, sanscript.IAST)
    except Exception:
        return segment  # fallback: return unchanged


def transliterate_text(text: str) -> TranslitResult:
    """
    Auto-detect script and transliterate Indic portions to ISO 15919 / IAST.

    Strategy: split text into LINES, detect each line's dominant script, and
    transliterate whole lines as units. This preserves Indic script sequences
    (matras, viramas, anusvaras) that must stay adjacent to their base consonants.
    Splitting character-by-character breaks these sequences.

    Latin and punctuation lines are passed through unchanged. Mixed-language
    texts (e.g. World Fusion poems with [Tamil] [Bengali] [English] sections)
    are handled line-by-line.

    Args:
        text: Raw creative output from LLM (may contain any Unicode).

    Returns:
        TranslitResult with .iast romanisation and script metadata.
    """
    if not text.strip():
        return TranslitResult(text, text, "unknown", False, False)

    dominant = detect_dominant_script(text)
    has_indic = dominant not in ("latin", "unknown")

    # Fast path: purely Latin
    if not has_indic:
        return TranslitResult(text, text, dominant, False, False)  # type: ignore[arg-type]

    # Line-by-line transliteration (preserves Indic combining character sequences)
    lines = text.split("\n")
    iast_lines: list[str] = []
    line_scripts: set[str] = set()
    has_latin_line = False

    for line in lines:
        line_script = detect_dominant_script(line)
        if line_script in ("latin", "unknown") or line_script not in _SCRIPT_MAP:
            iast_lines.append(line)
            has_latin_line = True
        else:
            line_scripts.add(line_script)
            iast_lines.append(_transliterate_segment(line, line_script))

    iast = "\n".join(iast_lines)
    mixed = has_latin_line and bool(line_scripts)

    return TranslitResult(
        original=text,
        iast=iast,
        script=dominant,  # type: ignore[arg-type]
        has_indic=True,
        mixed_language=mixed,
    )


def annotate_output(output: dict) -> dict:
    """
    Post-process a generation output record to add ISO 15919 transliteration.

    Called by the generation engine after text is produced. Adds:
        output["transliteration"]["iast"]       — romanised text
        output["transliteration"]["script"]     — dominant script
        output["transliteration"]["has_indic"]  — bool
        output["transliteration"]["mixed"]      — bool

    Args:
        output: dict returned by generate_one() or the API job store.

    Returns:
        The same dict, mutated in place, with "transliteration" key added.
    """
    text = output.get("text", "")
    result = transliterate_text(text)
    output["transliteration"] = {
        "iast": result.iast,
        "script": result.script,
        "has_indic": result.has_indic,
        "mixed": result.mixed_language,
        "latex_annotation": result.latex_annotation() if result.has_indic else "",
    }
    return output
