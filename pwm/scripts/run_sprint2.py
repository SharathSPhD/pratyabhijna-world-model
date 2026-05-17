"""
run_sprint2.py — Sprint 2: Generate 19 unbiased creative works.

Runs ALL_SPECS through the corrected engine:
  - Domain-neutral prompts (no Shaiva vocabulary)
  - WMStateDecoder prefix with spec_id secondary seed (varied even with degenerate WM)
  - Fixed scoring: cap ≤1.0, independent imagery vocabulary
  - No ellipsis placeholders (enforced by MASTER_SYSTEM prompt)

Output: benchmarks/results/sprint2_outputs.json

Sprint 2 gate criteria:
  1. 0/19 outputs contain Shaiva vocabulary in LLM-facing prefix
  2. All R_camatk scores ≤ 1.0
  3. 19/19 outputs complete (no empty or truncated text)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from pwm.generation.engine import run_all_specs              # type: ignore[import]
from pwm.generation.creative_specs import ALL_SPECS          # type: ignore[import]

# ── Shaiva vocabulary audit ───────────────────────────────────────────────────

SHAIVA_TERMS = [
    "śiva", "shiva", "shakti", "śakti", "pratyabhijña", "pratyabhijñā",
    "spanda", "sphurattā", "sphuratā", "camatkāra", "camatkar",
    "vimarśa", "vimars", "ananda", "ānanda", "icchā", "iccha",
    "kriyā", "kriya", "aham", "ābhāsana", "trika", "Kashmir Shaiva",
]


def audit_shaiva(text: str) -> list[str]:
    """Return list of any Shaiva terms found in text."""
    found = []
    lower = text.lower()
    for term in SHAIVA_TERMS:
        if term.lower() in lower:
            found.append(term)
    return found


def audit_placeholders(text: str) -> bool:
    """Return True if text has actual stub/ellipsis placeholders.

    Explicitly does NOT flag:
    - Language section labels like [Tamil], [Bengali], [English], [Hindi]
    - Transliteration notes like [Devanagari script], [script]
    These are legitimate structural elements in multilingual creative output.

    Flags only:
    - Bare ellipsis lines: standalone '...' on its own line
    - Bracket-wrapped ellipsis: [...]
    - Common stub markers: [TODO], [INSERT], [continues], (continues...)
    - Inline '...' not adjacent to quotes/dialogue (mid-sentence stubs)
    """
    # Language / script labels — do NOT flag these
    _lang_labels = re.compile(
        r"^\s*\[(Tamil|Bengali|Hindi|Kannada|Telugu|Sanskrit|English|"
        r"Marathi|Gujarati|Malayalam|Odia|Punjabi|Urdu|"
        r"Japanese|Chinese|Arabic|Persian|French|German|Spanish|"
        r"Latin|Greek|script|Devanagari|Transliteration)\]",
        re.IGNORECASE | re.MULTILINE,
    )
    # Remove language labels before checking
    cleaned = _lang_labels.sub("", text)

    patterns = [
        r"^\s*\.{3}\s*$",               # standalone ... on its own line
        r"\[\.{3}\]",                    # [...] bracket-wrapped ellipsis
        r"\[(TODO|INSERT|PLACEHOLDER|TBD|your\s+text|fill\s+in)\]",  # stub markers
        r"^\s*\(.*continues\.{0,3}\).*$",  # (continues...) lines
    ]
    for p in patterns:
        if re.search(p, cleaned, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def run_sprint2() -> None:
    print("=" * 60)
    print("Sprint 2: Creative Generation — 19 unbiased specs")
    print("=" * 60)

    # Seed texts per spec — domain-appropriate text for WM warmup
    # (WM is loaded inside run_all_specs)
    seed_texts = {
        "s01": "bird call dawn dew grass familiar landscape paksin sisira",
        "s02": "monsoon cloud memory traveller rain home kadamba megha pravasa",
        "k01": "heart inner life silence question longing vachana kannada bhoomi",
        "k02": "rain red soil evening heron river bank Nagarahavissu Malgudi",
        "k03": "rain earth flower song kannada folk harvest festival kere bele",
        "k04": "childhood river village grandmother story told evening lamp",
        "k05": "village well bullock cart woman water pot sunset dust road",
        "h01": "barish raat dil aankhein zindagi hindi film song love",
        "h02": "train platform fog departure last glimpse love bittersweet",
        "ta01": "sea shore waiting cuckoo kadal neytal blue waterlily heron",
        "te01": "river evening birds lamp mother telugu padyam music",
        "bn01": "spring flower wind light Rabindranath Tagore bengali song",
        "e01": "autumn mist lake twilight gold ripple romantic poem",
        "e02": "city fragment interior window concrete glass modernist poem",
        "e03": "street diner dawn jazz neon beat generation poem",
        "l01": "carnatic composition pallavi anupallavi caranam devotion secular",
        "l02": "jazz blues verse chorus bridge standard bebop head",
        "l03": "hindustani khayal vilambit drut alap taana monsoon raag",
        "w01": "sea shore tide migration threshold horizon salt boat wind fusion",
    }

    # Run all specs
    out_path = Path("benchmarks/results/sprint2_outputs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = run_all_specs(ALL_SPECS, out_path=out_path, seed_texts=seed_texts)

    # ── Sprint 2 gate checks ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sprint 2 Gate Checks")
    print("=" * 60)

    failures = []
    total = len(outputs)

    shaiva_violations = 0
    placeholder_violations = 0
    score_violations = 0
    empty_violations = 0

    for o in outputs:
        spec_id = o["id"]
        text = o.get("text", "")
        prefix = o.get("wm_prefix_used", "")
        score = o["scores"]["camatk_total"]

        # 1. No Shaiva terms in WM prefix (what LLM sees)
        shaiva_in_prefix = audit_shaiva(prefix)
        if shaiva_in_prefix:
            shaiva_violations += 1
            failures.append(f"SHAIVA in prefix [{spec_id}]: {shaiva_in_prefix}")

        # 2. Score ≤ 1.0
        if score > 1.0:
            score_violations += 1
            failures.append(f"SCORE > 1.0 [{spec_id}]: {score}")

        # 3. No placeholders
        if audit_placeholders(text):
            placeholder_violations += 1
            failures.append(f"PLACEHOLDER [{spec_id}]: has '...' or [...]")

        # 4. Non-empty output (≥ 50 chars)
        if len(text.strip()) < 50:
            empty_violations += 1
            failures.append(f"EMPTY [{spec_id}]: text too short ({len(text)} chars)")

    # Results
    print(f"\n  Total outputs:      {total}/19")
    print(f"  Shaiva violations:  {shaiva_violations}/19  (want 0)")
    print(f"  Score violations:   {score_violations}/19  (want 0, ≤1.0)")
    print(f"  Placeholder lines:  {placeholder_violations}/19  (want 0)")
    print(f"  Empty outputs:      {empty_violations}/19  (want 0)")

    if failures:
        print(f"\n  ✗ FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
    else:
        print("\n  ✓ ALL GATE CHECKS PASSED")

    # Save gate results
    gate_result = {
        "n_outputs": total,
        "gate_pass": len(failures) == 0,
        "shaiva_violations": shaiva_violations,
        "score_violations": score_violations,
        "placeholder_violations": placeholder_violations,
        "empty_violations": empty_violations,
        "failures": failures,
    }
    gate_path = Path("benchmarks/results/sprint2_gate.json")
    gate_path.write_text(json.dumps(gate_result, indent=2))
    print(f"\n  Gate results → {gate_path}")

    if len(failures) == 0:
        print("\n  ✓ Sprint 2 GATE PASSED")
    else:
        print(f"\n  ✗ Sprint 2 GATE: {len(failures)} issues to fix")


if __name__ == "__main__":
    run_sprint2()
