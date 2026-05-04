"""
Project Gutenberg poetry ingestion.

Source: https://www.gutenberg.org
Target: Public domain English poetry (pre-1928); prioritises Romantic and Victorian.

Uses the Gutenberg catalog API and plain-text download endpoint.
All works are public domain — no scraping restrictions.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


_CATALOG_API = "https://gutendex.com/books/?topic=poetry&languages=en&page={page}"
_TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def _fetch(url: str, timeout: int = 30) -> str | None:
    headers = {"User-Agent": "PWM-corpus-builder/0.1 (research; qbz506@york.ac.uk)"}
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def _strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg boilerplate from plain-text files."""
    start_pattern = re.compile(
        r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
    )
    end_pattern = re.compile(
        r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
    )
    start_m = start_pattern.search(text)
    end_m = end_pattern.search(text)

    if start_m:
        text = text[start_m.end():]
    if end_m:
        text = text[:end_m.start()]
    return text.strip()


def ingest_gutenberg(
    output_dir: Path,
    max_pages: int = 10,
    min_tokens: int = 200,
    max_per_page: int = 5,
) -> list[dict[str, Any]]:
    """
    Download public-domain poetry from Project Gutenberg via gutendex API.

    Writes to output_dir/gutenberg/. Returns metadata list.
    """
    out = output_dir / "gutenberg"
    out.mkdir(parents=True, exist_ok=True)
    manifested: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        catalog_raw = _fetch(_CATALOG_API.format(page=page))
        if catalog_raw is None:
            break
        try:
            catalog = json.loads(catalog_raw)
        except (json.JSONDecodeError, ValueError):
            break

        results = catalog.get("results", [])
        if not results:
            break

        for book in results[:max_per_page]:
            book_id = str(book.get("id", ""))
            if not book_id:
                continue

            title = str(book.get("title", "untitled"))
            authors = [a.get("name", "") for a in book.get("authors", [])]
            author = authors[0] if authors else "unknown"

            text_raw = _fetch(_TEXT_URL.format(id=book_id))
            if text_raw is None:
                continue

            text = _strip_gutenberg_header_footer(text_raw)
            tokens = text.split()
            if len(tokens) < min_tokens:
                continue

            slug = re.sub(r"[^\w-]", "_", title.lower())[:60]
            fname = f"gut{book_id}_{slug}"
            (out / f"{fname}.txt").write_text(text[:500_000], encoding="utf-8")  # cap 500K chars

            meta: dict[str, Any] = {
                "source": "gutenberg",
                "id": book_id,
                "title": title,
                "author": author,
                "language": "en",
                "token_count": len(tokens),
                "license": "public_domain",
            }
            (out / f"{fname}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifested.append(meta)
            time.sleep(1.0)

        time.sleep(2.0)

    return manifested
