"""
arXiv scientific creativity papers ingestion.

Source: https://export.arxiv.org/api
Target: Papers on creativity, aesthetics, computational creativity, active inference.

Fetches abstracts + (optionally) full PDFs via arXiv API.
Search queries tuned for the PWM research context.
"""

from __future__ import annotations
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


_API_URL = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}

_SEARCH_QUERIES = [
    "ti:creativity AND abs:active+inference",
    "ti:camatkara OR ti:aesthetic+wonder",
    "abs:dreamer+world+model+creativity",
    "ti:active+inference+creative",
    "abs:variational+free+energy+art",
    "ti:computational+creativity+language+model",
]


def _query_arxiv(
    query: str,
    max_results: int = 20,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """Query arXiv API; return list of paper metadata dicts."""
    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        resp = requests.get(_API_URL, params=params, timeout=timeout,
                            headers={"User-Agent": "PWM-corpus-builder/0.1"})
        resp.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _NS):
        arxiv_id_elem = entry.find("atom:id", _NS)
        title_elem = entry.find("atom:title", _NS)
        summary_elem = entry.find("atom:summary", _NS)
        if arxiv_id_elem is None or title_elem is None or summary_elem is None:
            continue
        arxiv_id = str(arxiv_id_elem.text or "").split("/")[-1]
        title = re.sub(r"\s+", " ", str(title_elem.text or "")).strip()
        abstract = re.sub(r"\s+", " ", str(summary_elem.text or "")).strip()
        authors: list[str] = []
        for a in entry.findall("atom:author", _NS):
            name_elem = a.find("atom:name", _NS)
            if name_elem is not None and name_elem.text is not None:
                authors.append(str(name_elem.text))
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
        })
    return papers


def ingest_arxiv_creativity(
    output_dir: Path,
    max_per_query: int = 20,
    min_tokens: int = 50,
) -> list[dict[str, Any]]:
    """
    Fetch arXiv papers on computational creativity and write to output_dir/arxiv/.

    Writes abstract text + metadata. Returns metadata list.
    """
    out = output_dir / "arxiv"
    out.mkdir(parents=True, exist_ok=True)
    manifested: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query in _SEARCH_QUERIES:
        papers = _query_arxiv(query, max_results=max_per_query)
        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            text = f"{paper['title']}\n\n{paper['abstract']}"
            tokens = text.split()
            if len(tokens) < min_tokens:
                continue

            fname = f"arxiv_{arxiv_id.replace('/', '_')}"
            (out / f"{fname}.txt").write_text(text, encoding="utf-8")

            meta: dict[str, Any] = {
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "authors": paper["authors"],
                "query": query,
                "language": "en",
                "token_count": len(tokens),
            }
            (out / f"{fname}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifested.append(meta)

        time.sleep(3.0)  # arXiv API rate limit: 1 req/3s

    return manifested
