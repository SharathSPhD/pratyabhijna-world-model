"""
GRETIL Sanskrit corpus ingestion.

Source: Göttingen Register of Electronic Texts in Indian Languages (GRETIL)
        http://gretil.sub.uni-goettingen.de/gretil.html
Focus: Śaiva philosophical texts + kāvya poetry.

Philosophical grounding:
  Āgama (TĀ 1.18, Abhinavagupta): Scriptural testimony — the primary pramāṇa
  for Śaiva epistemology. GRETIL holds the authoritative digital editions of
  the Kashmir Śaiva canon (Tantrāloka, ĪPK, Spanda texts).
"""

from __future__ import annotations
import re
import time
from pathlib import Path
from typing import Any

import requests


# Priority Śaiva texts with approximate GRETIL URL patterns
_SAIVA_TEXTS = {
    "tantraloka": "https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa/saiva/kashmiri/abhinav_tantralokatrans.htm",
    "spandakarika": "https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa/saiva/kashmiri/vasugupta_spandakarika.htm",
    "pratyabhijna": "https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa/saiva/kashmiri/utpala_ipk.htm",
    "vijnana_bhairava": "https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa/saiva/kashmiri/vijnana_bhairava.htm",
}

# Phase 5+: crawl kāvya index for broader poetic corpus
_KAVYA_INDEX = "https://gretil.sub.uni-goettingen.de/gretil/corpustei/sa/kavya/"  # noqa: F841


def _clean_devanagari(text: str) -> str:
    """Strip HTML tags and normalise whitespace from GRETIL HTML files."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_url(url: str, timeout: int = 30, retries: int = 3) -> str | None:
    """Fetch URL with retries; return text or None on failure."""
    headers = {"User-Agent": "PWM-corpus-builder/0.1 (research; qbz506@york.ac.uk)"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def ingest_gretil(output_dir: Path, min_tokens: int = 500) -> list[dict[str, Any]]:
    """
    Ingest priority Śaiva texts from GRETIL.

    Writes text + meta sidecar to output_dir/gretil/.
    Returns list of metadata dicts for manifested documents.
    """
    out = output_dir / "gretil"
    out.mkdir(parents=True, exist_ok=True)
    manifested: list[dict[str, Any]] = []

    for name, url in _SAIVA_TEXTS.items():
        raw = _fetch_url(url)
        if raw is None:
            continue
        text = _clean_devanagari(raw)
        tokens = text.split()
        if len(tokens) < min_tokens:
            continue

        doc_path = out / f"{name}.txt"
        doc_path.write_text(text, encoding="utf-8")

        meta: dict[str, Any] = {
            "source": "gretil",
            "name": name,
            "url": url,
            "language": "sa",
            "tradition": "kashmir_shaiva",
            "token_count": len(tokens),
        }
        (out / f"{name}.meta.json").write_text(
            __import__("json").dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifested.append(meta)
        time.sleep(1.5)  # polite delay

    return manifested
