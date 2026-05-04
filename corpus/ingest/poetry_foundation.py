"""
Poetry Foundation corpus ingestion.

Source: https://www.poetryfoundation.org
Target: English-language poetry for cross-lingual camatkāra training.

Uses the public poem listing pages with polite crawling (1.5s delay).
Only fetches poem text and metadata from publicly accessible pages.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


_BASE_URL = "https://www.poetryfoundation.org"
_POEMS_API = f"{_BASE_URL}/api/poems?page={{page}}&pageSize=20"


def _fetch_json(url: str, timeout: int = 15) -> dict[str, Any] | None:
    headers = {
        "User-Agent": "PWM-corpus-builder/0.1 (research; qbz506@york.ac.uk)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return dict(resp.json())
    except (requests.RequestException, ValueError):
        return None


def _clean_poem_text(html: str) -> str:
    """Strip HTML from poem body."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def ingest_poetry_foundation(
    output_dir: Path,
    max_pages: int = 50,
    min_tokens: int = 30,
) -> list[dict[str, Any]]:
    """
    Crawl Poetry Foundation poems API and write to output_dir/poetry_foundation/.

    Returns list of metadata dicts for all collected poems.
    """
    out = output_dir / "poetry_foundation"
    out.mkdir(parents=True, exist_ok=True)
    manifested: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        url = _POEMS_API.format(page=page)
        data = _fetch_json(url)
        if data is None:
            break

        poems = data.get("poems", data.get("results", []))
        if not poems:
            break

        for poem in poems:
            poem_id = str(poem.get("id", "unknown"))
            title = str(poem.get("title", "untitled"))
            author = str(poem.get("poet", {}).get("name", "unknown") if isinstance(poem.get("poet"), dict) else "unknown")
            body_html = str(poem.get("poem", poem.get("body", "")))
            text = _clean_poem_text(body_html)
            tokens = text.split()

            if len(tokens) < min_tokens:
                continue

            slug = re.sub(r"[^\w-]", "_", title.lower())[:60]
            fname = f"{poem_id}_{slug}"
            (out / f"{fname}.txt").write_text(text, encoding="utf-8")

            meta: dict[str, Any] = {
                "source": "poetry_foundation",
                "id": poem_id,
                "title": title,
                "author": author,
                "language": "en",
                "token_count": len(tokens),
            }
            (out / f"{fname}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifested.append(meta)

        time.sleep(1.5)

    return manifested
