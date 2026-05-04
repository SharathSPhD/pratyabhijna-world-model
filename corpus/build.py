"""
corpus/build.py — Pratyabhijñā World Model corpus ingestion pipeline.

Sanskrit concept: Āgama (आगम) — valid testimony / received knowledge
Source: Īśvarapratyabhijñākārikā 1.1.5 (Utpaladeva)
Computational realisation: ingestion of public creative text corpora for
    world-model pre-training; each source is a stream of pramāṇa (epistemic
    evidence) that seeds the model's prior p_θ(z).

Usage:
    python -m corpus.build --sources hf_poetry,hf_wiki_philosophy,gutenberg
    python -m corpus.build --sources all --min-tokens 100000 --output-dir data/corpus
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urljoin, urlparse

import click
import requests  # type: ignore[import]
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm  # type: ignore[import]

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully if not installed
# ---------------------------------------------------------------------------
from typing import Any as _Any

try:
    from bs4 import BeautifulSoup  # type: ignore[import]
    _BS4_AVAILABLE = True
except ImportError:
    class BeautifulSoup:  # type: ignore[no-redef]
        def __init__(self, *a: _Any, **kw: _Any) -> None: ...
        def find_all(self, *a: _Any, **kw: _Any) -> list: return []
    _BS4_AVAILABLE = False

try:
    from datasets import load_dataset  # type: ignore[import]
    _DATASETS_AVAILABLE = True
except ImportError:
    def load_dataset(*args: _Any, **kwargs: _Any) -> _Any:  # type: ignore[misc]
        raise ImportError("datasets not installed: pip install datasets")
    _DATASETS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
console = Console()

# GRETIL Śaiva corpus base URL
GRETIL_SAIVA_BASE = "http://gretil.uni-goettingen.de/gretil/1_sanskr/4_rellit/saiva/"

# Project Gutenberg full-text search endpoint
GUTENBERG_SEARCH = "https://gutendex.com/books/"

# Wikipedia philosophy/arts category keywords (case-insensitive substring match)
WIKI_CATEGORIES = [
    "aesthetics", "consciousness", "poetry", "creativity", "art", "music",
    "literature", "philosophy", "epistemology", "metaphysics", "ontology",
    "phenomenology", "hermeneutics", "semiotics", "rhetoric", "narrative",
]

# Gutenberg topic queries
GUTENBERG_TOPICS = [
    "philosophy", "poetry", "aesthetics", "consciousness", "literary theory",
    "mind", "beauty", "imagination",
]

# HuggingFace model hub home
HF_HOME = os.environ.get("HF_HOME", "/home/sharaths/models")

# Minimum characters to consider a text meaningful
MIN_CHAR_LENGTH = 200

# Max docs per HF-streaming source (prevent runaway downloads)
HF_MAX_DOCS = 5_000

# Max Gutenberg books per topic query
GUTENBERG_MAX_BOOKS = 50

# HTTP request defaults
HTTP_TIMEOUT = 30
HTTP_HEADERS = {
    "User-Agent": (
        "PWM-CorpusBuilder/0.1 (Pratyabhijna World Model research; "
        "contact: qbz506@york.ac.uk)"
    )
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Count whitespace-split tokens (fast approximation of BPE count)."""
    return len(text.split())


def _sanitise_filename(name: str, max_len: int = 120) -> str:
    """Convert an arbitrary string into a safe filename stem."""
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\w\s\-]", "_", name, flags=re.ASCII)
    name = re.sub(r"[\s]+", "_", name)
    return name[:max_len].strip("_") or "document"


def _write_document(
    output_dir: Path,
    stem: str,
    text: str,
    meta: dict,
) -> int:
    """Write text + sidecar JSON metadata; return token count."""
    text = text.strip()
    if len(text) < MIN_CHAR_LENGTH:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitise_filename(stem)

    txt_path = output_dir / f"{stem}.txt"
    meta_path = output_dir / f"{stem}.meta.json"

    # Avoid overwriting existing files — append numeric suffix
    counter = 0
    while txt_path.exists():
        counter += 1
        txt_path = output_dir / f"{stem}_{counter}.txt"
        meta_path = output_dir / f"{stem}_{counter}.meta.json"

    txt_path.write_text(text, encoding="utf-8")

    tokens = _count_tokens(text)
    meta["word_count"] = tokens
    meta["char_count"] = len(text)
    meta["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return tokens


def _get_request(url: str, **kwargs) -> requests.Response | None:
    """GET with timeout + error logging; returns None on failure."""
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        console.log(f"[yellow]HTTP error[/yellow] {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# HuggingFace: poem_sentiment / poetry corpus
# ---------------------------------------------------------------------------

def ingest_hf_poetry(output_dir: Path) -> int:
    """
    Ingest the `poem_sentiment` HuggingFace dataset.

    Sanskrit concept: Rasa (रस) — aesthetic emotional essence
    Source: Nāṭyaśāstra 6.15 (Bharatamuni)
    These are short lyric poems annotated with sentiment — direct rasa data.
    """
    if not _DATASETS_AVAILABLE:
        console.log("[red]datasets library not available — skipping hf_poetry[/red]")
        return 0

    source_dir = output_dir / "hf_poetry"
    total_tokens = 0

    console.log("[cyan]Streaming merve/poetry dataset…[/cyan]")
    try:
        ds = load_dataset("merve/poetry", split="train", streaming=False)
    except Exception as exc:
        console.log(f"[red]Poetry load failed: {exc}[/red]")
        return 0

    count = 0
    for i, example in enumerate(tqdm(ds, desc="hf_poetry", unit="poems")):
        if i >= HF_MAX_DOCS:
            break
        # merve/poetry uses 'content' field
        text = (
            example.get("content")
            or example.get("verse_text")
            or example.get("poem")
            or example.get("text")
            or ""
        )
        if not text:
            continue
        label = str(example.get("label", example.get("sentiment", "unknown")))
        author = str(example.get("author", "unknown"))
        title = str(example.get("title", f"poem_{i:06d}"))
        stem = f"{i:06d}_{_sanitise_filename(title)[:60]}"
        meta = {
            "source": "hf_poetry",
            "dataset": "merve/poetry",
            "lang": "en",
            "label": label,
            "author": author,
            "title": title,
            "url": "",
        }
        tokens = _write_document(source_dir, stem, text, meta)
        total_tokens += tokens
        count += 1

    console.log(f"[green]hf_poetry:[/green] {count} poems, {total_tokens:,} tokens")
    return total_tokens


# ---------------------------------------------------------------------------
# HuggingFace: Wikipedia philosophy/arts/literature articles
# ---------------------------------------------------------------------------

def _wiki_article_matches(article: dict) -> bool:
    """Return True if a Wikipedia article belongs to a target category."""
    # wikimedia/wikipedia stores text in 'text' field, title in 'title'
    title = (article.get("title") or "").lower()
    url = (article.get("url") or "").lower()
    combined = title + " " + url
    return any(kw in combined for kw in WIKI_CATEGORIES)


def ingest_hf_wiki_philosophy(output_dir: Path) -> int:
    """
    Stream wikipedia 20231101.en, filter for philosophy/arts/literature articles.

    Sanskrit concept: Śabdapramāṇa (शब्दप्रमाण) — verbal testimony as valid knowledge
    Source: Nyāyasūtra 1.1.7 (Gautama)
    Wikipedia encodes the śabdapramāṇa of humanity's conceptual knowledge.
    """
    if not _DATASETS_AVAILABLE:
        console.log("[red]datasets library not available — skipping hf_wiki_philosophy[/red]")
        return 0

    source_dir = output_dir / "hf_wiki_philosophy"
    total_tokens = 0
    count = 0

    console.log("[cyan]Streaming wikimedia/wikipedia 20231101.en (filtered)…[/cyan]")
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        console.log(f"[red]Wikipedia dataset load failed: {exc}[/red]")
        return 0

    pbar = tqdm(desc="hf_wiki_philosophy", unit="articles")
    for example in ds:
        if count >= HF_MAX_DOCS:
            break
        if not _wiki_article_matches(example):
            pbar.update(1)
            continue
        text = example.get("text") or ""
        title = example.get("title") or f"article_{count:06d}"
        url = example.get("url") or ""
        stem = f"{count:06d}_{_sanitise_filename(title)[:80]}"
        meta = {
            "source": "hf_wiki_philosophy",
            "dataset": "wikimedia/wikipedia",
            "lang": "en",
            "title": title,
            "url": url,
        }
        tokens = _write_document(source_dir, stem, text, meta)
        total_tokens += tokens
        count += 1
        pbar.update(1)

    pbar.close()
    console.log(f"[green]hf_wiki_philosophy:[/green] {count} articles, {total_tokens:,} tokens")
    return total_tokens


# ---------------------------------------------------------------------------
# HuggingFace: allenai/c4 (creative/literary slice)
# ---------------------------------------------------------------------------

def ingest_hf_c4(output_dir: Path) -> int:
    """
    Stream allenai/c4 English; retain documents with creative/literary keywords.

    Sanskrit concept: Pratibhā (प्रतिभा) — creative flash / inspired insight
    Source: Abhinavagupta, Locana ad Dhvanyāloka 1.1
    C4 is a broad web corpus; filtering extracts the pratibhā-dense subset.
    """
    if not _DATASETS_AVAILABLE:
        console.log("[red]datasets library not available — skipping hf_c4[/red]")
        return 0

    source_dir = output_dir / "hf_c4"
    total_tokens = 0
    count = 0

    CREATIVE_KEYWORDS = [
        "poem", "poetry", "verse", "metaphor", "narrative", "novel", "fiction",
        "literary", "aesthetic", "consciousness", "imagination", "creative",
        "philosophy", "aesthetics", "mythology", "allegory",
    ]

    console.log("[cyan]Streaming allenai/c4 en (creative filter)…[/cyan]")
    try:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    except Exception as exc:
        console.log(f"[red]C4 dataset load failed: {exc}[/red]")
        return 0

    max_docs = HF_MAX_DOCS // 2  # C4 is very large — keep moderate
    pbar = tqdm(desc="hf_c4", unit="docs")
    checked = 0
    for example in ds:
        if count >= max_docs:
            break
        checked += 1
        if checked > max_docs * 40:  # scan at most 40x to avoid infinite streaming
            break
        text = example.get("text") or ""
        text_lower = text.lower()
        if not any(kw in text_lower for kw in CREATIVE_KEYWORDS):
            pbar.update(1)
            continue
        url = example.get("url") or ""
        stem = f"{count:06d}_{_sanitise_filename(urlparse(url).path)[:60]}"
        meta = {
            "source": "hf_c4",
            "dataset": "allenai/c4",
            "lang": "en",
            "url": url,
            "timestamp_hf": example.get("timestamp", ""),
        }
        tokens = _write_document(source_dir, stem, text, meta)
        total_tokens += tokens
        count += 1
        pbar.update(1)

    pbar.close()
    console.log(f"[green]hf_c4:[/green] {count} docs, {total_tokens:,} tokens")
    return total_tokens


# ---------------------------------------------------------------------------
# HuggingFace: EleutherAI/pile subsets
# ---------------------------------------------------------------------------

def ingest_hf_pile(output_dir: Path) -> int:
    """
    Pull EleutherAI/pile subsets: Books3, BookCorpus2 (creative text).

    Sanskrit concept: Smṛti (स्मृति) — remembered / traditional knowledge corpus
    Source: Pāṇini, Aṣṭādhyāyī (on śabdānuśāsana — grammatical tradition)
    Books are the densest store of smṛti in the digital world.
    """
    if not _DATASETS_AVAILABLE:
        console.log("[red]datasets library not available — skipping hf_pile[/red]")
        return 0

    source_dir = output_dir / "hf_pile"
    total_tokens = 0

    # Try each subset — some may require agreement or may be unavailable
    subsets_to_try = [
        ("EleutherAI/pile", "default", "Books3"),
        ("EleutherAI/pile", "default", "BookCorpus2"),
        ("EleutherAI/pile", "default", "FreeLaw"),
    ]

    for dataset_id, config, subset_name in subsets_to_try:
        console.log(f"[cyan]Trying {dataset_id} subset {subset_name}…[/cyan]")
        count = 0
        sub_tokens = 0
        try:
            ds = load_dataset(dataset_id, config, split="train", streaming=True)
        except Exception as exc:
            console.log(f"[yellow]{dataset_id}/{subset_name} unavailable: {exc}[/yellow]")
            # Try alternative Books corpus
            try:
                ds = load_dataset(
                    "bookcorpus",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                subset_name = "BookCorpus"
            except Exception as exc2:
                console.log(f"[yellow]BookCorpus fallback failed: {exc2}[/yellow]")
                continue

        pbar = tqdm(desc=f"hf_pile/{subset_name}", unit="docs")
        for i, example in enumerate(ds):
            if count >= HF_MAX_DOCS // 3:
                break
            text = example.get("text") or example.get("content") or ""
            # Filter to target subset if 'meta' field present
            meta_field = example.get("meta", {})
            if isinstance(meta_field, dict):
                pile_set = meta_field.get("pile_set_name", "")
                if pile_set and pile_set != subset_name:
                    pbar.update(1)
                    continue
            stem = f"{subset_name}_{count:06d}"
            meta = {
                "source": "hf_pile",
                "dataset": dataset_id,
                "subset": subset_name,
                "lang": "en",
                "url": meta_field.get("url", "") if isinstance(meta_field, dict) else "",
            }
            tokens = _write_document(source_dir, stem, text, meta)
            sub_tokens += tokens
            count += 1
            pbar.update(1)

        pbar.close()
        total_tokens += sub_tokens
        console.log(f"[green]hf_pile/{subset_name}:[/green] {count} docs, {sub_tokens:,} tokens")
        # Stop if we already have enough from Pile
        if total_tokens > 50_000:
            break

    console.log(f"[green]hf_pile total:[/green] {total_tokens:,} tokens")
    return total_tokens


# ---------------------------------------------------------------------------
# HuggingFace: Sanskrit datasets
# ---------------------------------------------------------------------------

def ingest_hf_sanskrit(output_dir: Path) -> int:
    """
    Search HuggingFace Hub for Sanskrit text datasets and ingest.

    Sanskrit concept: Vāk (वाक्) — primordial speech / logos
    Source: Ṛgveda 1.164.45 (the four levels of speech: parā, paśyantī, madhyamā, vaikharī)
    Sanskrit corpora instantiate vaikharī — the manifested, phonetic level of speech.
    """
    if not _DATASETS_AVAILABLE:
        console.log("[red]datasets library not available — skipping hf_sanskrit[/red]")
        return 0

    source_dir = output_dir / "hf_sanskrit"
    total_tokens = 0

    # Known Sanskrit-relevant HF datasets to try
    sanskrit_datasets = [
        ("Rajan0012/SanskritPoetry", None, "train"),
        ("ai4bharat/sangraha", "sanitized_level0_sa", "train"),
        ("rahular/itihasa", None, "train"),
        ("mrinalraj/sanskrit_verses", None, "train"),
        ("Cogito-ai/ManuscriptQA", None, "train"),
    ]

    for ds_id, config, split in sanskrit_datasets:
        count = 0
        sub_tokens = 0
        console.log(f"[cyan]Trying Sanskrit dataset: {ds_id}…[/cyan]")
        try:
            if config:
                ds = load_dataset(ds_id, config, split=split, streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(ds_id, split=split, streaming=True, trust_remote_code=True)
        except Exception as exc:
            console.log(f"[yellow]{ds_id} unavailable: {exc}[/yellow]")
            continue

        safe_ds_name = _sanitise_filename(ds_id.split("/")[-1])
        pbar = tqdm(desc=f"hf_sanskrit/{safe_ds_name}", unit="docs")
        for i, example in enumerate(ds):
            if count >= HF_MAX_DOCS // len(sanskrit_datasets):
                break
            # Try common field names
            text = (
                example.get("text")
                or example.get("content")
                or example.get("verse")
                or example.get("sloka")
                or example.get("sanskrit")
                or example.get("Translation")
                or ""
            )
            if not text:
                pbar.update(1)
                continue
            stem = f"{safe_ds_name}_{count:06d}"
            meta = {
                "source": "hf_sanskrit",
                "dataset": ds_id,
                "lang": "sa",
                "url": "",
            }
            tokens = _write_document(source_dir, stem, str(text), meta)
            sub_tokens += tokens
            count += 1
            pbar.update(1)

        pbar.close()
        total_tokens += sub_tokens
        console.log(f"[green]hf_sanskrit/{safe_ds_name}:[/green] {count} docs, {sub_tokens:,} tokens")

    console.log(f"[green]hf_sanskrit total:[/green] {total_tokens:,} tokens")
    return total_tokens


# ---------------------------------------------------------------------------
# GRETIL Sanskrit downloader
# ---------------------------------------------------------------------------

def _list_gretil_saiva_urls() -> list[str]:
    """
    Discover .htm file URLs under the GRETIL Śaiva section.
    Returns list of absolute URLs to parse.
    """
    if not _BS4_AVAILABLE:
        console.log("[yellow]beautifulsoup4 not available — using known GRETIL URL stubs[/yellow]")
        # Fallback: hardcoded known texts (Śaivāgama core)
        return [
            urljoin(GRETIL_SAIVA_BASE, "sivsu_u.htm"),   # Śivasūtra (Utpaladeva comm.)
            urljoin(GRETIL_SAIVA_BASE, "spandaku.htm"),   # Spandakārikā
            urljoin(GRETIL_SAIVA_BASE, "abhiipku.htm"),   # Abhinavagupta ĪPK
            urljoin(GRETIL_SAIVA_BASE, "isipkavu.htm"),   # Utpaladeva ĪPK verses
            urljoin(GRETIL_SAIVA_BASE, "vakyratu.htm"),   # Vakrokti-jīvita
            urljoin(GRETIL_SAIVA_BASE, "paramstu.htm"),   # Paramārthasāra
        ]

    console.log(f"[cyan]Fetching GRETIL Śaiva index: {GRETIL_SAIVA_BASE}[/cyan]")
    resp = _get_request(GRETIL_SAIVA_BASE)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if href.endswith(".htm") or href.endswith(".html"):
            full_url = urljoin(GRETIL_SAIVA_BASE, href)
            urls.append(full_url)
    console.log(f"[cyan]Found {len(urls)} GRETIL Śaiva texts[/cyan]")
    return urls


def _parse_gretil_page(html: str) -> str:
    """Extract IAST transliterated text from a GRETIL HTML page."""
    if not _BS4_AVAILABLE:
        # Minimal extraction: strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&#\d+;", "", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    soup = BeautifulSoup(html, "html.parser")
    # Remove navigation, scripts, styles
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    # GRETIL pages use <pre> or <p> for transliterated text
    blocks: list[str] = []
    for tag in soup.find_all(["pre", "p", "div"]):
        text = tag.get_text(separator="\n")  # type: ignore[union-attr]
        if len(text.strip()) > 50:
            blocks.append(text.strip())
    if blocks:
        return "\n\n".join(blocks)
    # Fallback: all text
    return soup.get_text(separator="\n").strip()  # type: ignore[union-attr]


def ingest_gretil(output_dir: Path) -> int:
    """
    Download and parse GRETIL Sanskrit Śaiva texts (IAST transliteration).

    Sanskrit concept: Āgama (आगम) — revealed / received textual tradition
    Source: Mālinīvijayottaratantra 1.4 (on the authority of āgamic texts)
    GRETIL provides the Āgamic foundation of PWM's philosophical substrate.
    """
    source_dir = output_dir / "gretil"
    total_tokens = 0

    urls = _list_gretil_saiva_urls()
    if not urls:
        console.log("[yellow]No GRETIL URLs to fetch[/yellow]")
        return 0

    for url in tqdm(urls, desc="gretil", unit="text"):
        resp = _get_request(url)
        if resp is None:
            continue

        # Detect encoding — GRETIL pages are often ISO-8859-1
        try:
            resp.encoding = resp.apparent_encoding or "utf-8"
            html = resp.text
        except Exception:
            html = resp.content.decode("utf-8", errors="replace")

        text = _parse_gretil_page(html)
        if len(text) < MIN_CHAR_LENGTH:
            continue

        path = urlparse(url).path
        stem = Path(path).stem or "gretil_text"
        meta = {
            "source": "gretil",
            "url": url,
            "lang": "sa",
            "encoding": "IAST",
        }
        tokens = _write_document(source_dir, stem, text, meta)
        total_tokens += tokens
        time.sleep(1.0)  # polite crawl delay for GRETIL server

    console.log(f"[green]gretil:[/green] {len(urls)} texts attempted, {total_tokens:,} tokens")
    return total_tokens


# ---------------------------------------------------------------------------
# Project Gutenberg downloader (via Gutendex API)
# ---------------------------------------------------------------------------

def _gutenberg_search(topic: str, page: int = 1) -> list[dict]:
    """Query Gutendex API for books matching topic; return list of book dicts."""
    params = {
        "search": topic,
        "languages": "en",
        "page": page,
    }
    resp = _get_request(GUTENBERG_SEARCH, params=params)
    if resp is None:
        return []
    try:
        data = resp.json()
        return data.get("results", [])
    except ValueError:
        return []


def _gutenberg_text_url(book: dict) -> str | None:
    """Extract the plain-text download URL from a Gutendex book dict."""
    formats: dict = book.get("formats", {})
    # Prefer UTF-8 plain text
    for mime in [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    ]:
        if mime in formats:
            return formats[mime]
    # Fallback: any text/plain key
    for key, url in formats.items():
        if key.startswith("text/plain"):
            return url
    return None


def ingest_gutenberg(output_dir: Path) -> int:
    """
    Download Project Gutenberg books on philosophy, poetry, aesthetics.

    Sanskrit concept: Vyutpatti (व्युत्पत्ति) — learning from tradition/books
    Source: Sāhityadarpaṇa 1.2 (Viśvanātha) — one of the three sources of poetic power
    Gutenberg provides the vyutpatti substrate for the model's creative grammar.
    """
    source_dir = output_dir / "gutenberg"
    total_tokens = 0
    seen_ids: set[int] = set()
    count = 0

    for topic in GUTENBERG_TOPICS:
        if count >= GUTENBERG_MAX_BOOKS:
            break
        console.log(f"[cyan]Gutenberg search: '{topic}'…[/cyan]")
        books = _gutenberg_search(topic)

        for book in tqdm(books, desc=f"gutenberg/{topic}", unit="book"):
            if count >= GUTENBERG_MAX_BOOKS:
                break
            book_id = book.get("id")
            if book_id in seen_ids:
                continue
            seen_ids.add(book_id)  # type: ignore[arg-type]

            text_url = _gutenberg_text_url(book)
            if text_url is None:
                continue

            resp = _get_request(text_url)
            if resp is None:
                continue

            try:
                text = resp.content.decode("utf-8", errors="replace")
            except Exception:
                continue

            # Strip Gutenberg header/footer boilerplate
            text = _strip_gutenberg_boilerplate(text)
            if len(text) < MIN_CHAR_LENGTH:
                continue

            title = book.get("title", f"book_{book_id}")
            authors = ", ".join(
                a.get("name", "Unknown") for a in book.get("authors", [])
            )
            stem = f"{book_id:06d}_{_sanitise_filename(title)[:70]}"
            meta = {
                "source": "gutenberg",
                "url": text_url,
                "book_id": book_id,
                "title": title,
                "authors": authors,
                "lang": "en",
                "subjects": book.get("subjects", []),
            }
            tokens = _write_document(source_dir, stem, text, meta)
            total_tokens += tokens
            count += 1
            time.sleep(0.5)  # polite crawl delay

    console.log(f"[green]gutenberg:[/green] {count} books, {total_tokens:,} tokens")
    return total_tokens


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Header: everything before "*** START OF"
    start_marker = re.search(
        r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG", text, re.IGNORECASE
    )
    if start_marker:
        text = text[start_marker.end():]
        # Skip the rest of the start line
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    # Footer: everything after "*** END OF"
    end_marker = re.search(
        r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG", text, re.IGNORECASE
    )
    if end_marker:
        text = text[: end_marker.start()]

    return text.strip()


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

# Maps source name → ingestion callable (output_dir: Path) -> int (tokens)
SOURCE_REGISTRY: dict[str, Callable[[Path], int]] = {
    "hf_poetry": ingest_hf_poetry,
    "hf_wiki_philosophy": ingest_hf_wiki_philosophy,
    "hf_c4": ingest_hf_c4,
    "hf_pile": ingest_hf_pile,
    "hf_sanskrit": ingest_hf_sanskrit,
    "gretil": ingest_gretil,
    "gutenberg": ingest_gutenberg,
}

ALL_SOURCES = list(SOURCE_REGISTRY.keys())

# Default sources for Phase 0 (lightweight, fast)
DEFAULT_SOURCES = ["hf_poetry", "hf_wiki_philosophy", "gutenberg"]


# ---------------------------------------------------------------------------
# CorpusBuilder class
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """
    Builds the PWM creative text corpus from public sources.

    Sanskrit concept: Pañcakṛtya (पञ्चकृत्य) — the five divine acts
    Source: Tantrasāra 1.2 (Abhinavagupta)
    CorpusBuilder performs sṛṣṭi (emission / creation) of the training corpus,
    which is the first of the five acts that constitute consciousness's
    self-revelation through the world model.

    Usage:
        builder = CorpusBuilder()
        total = builder.build(["hf_poetry", "gutenberg"], Path("data/corpus"))
    """

    def __init__(self) -> None:
        self.console = console

    def build(
        self,
        sources: list[str],
        output_dir: Path,
        min_tokens: int = 100_000,
    ) -> int:
        """
        Run the corpus build pipeline.

        Parameters
        ----------
        sources:
            List of source names from SOURCE_REGISTRY (or ["all"]).
        output_dir:
            Root directory for corpus output. Each source writes to a subdirectory.
        min_tokens:
            Phase 0 exit criterion — warn if corpus is below this threshold.

        Returns
        -------
        int
            Total token count across all sources.
        """
        if "all" in sources:
            sources = ALL_SOURCES

        # Validate sources
        unknown = [s for s in sources if s not in SOURCE_REGISTRY]
        if unknown:
            self.console.print(
                f"[red]Unknown sources: {unknown}. "
                f"Available: {list(SOURCE_REGISTRY.keys())}[/red]"
            )
            sources = [s for s in sources if s in SOURCE_REGISTRY]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(
            Panel(
                f"[bold cyan]PWM Corpus Builder[/bold cyan]\n"
                f"Sources: {sources}\n"
                f"Output:  {output_dir}\n"
                f"Target:  {min_tokens:,} tokens",
                title="āgama ingestion",
                expand=False,
            )
        )

        results: dict[str, int] = {}
        grand_total = 0

        for source_name in sources:
            ingestor = SOURCE_REGISTRY[source_name]
            self.console.rule(f"[bold]{source_name}[/bold]")
            try:
                tokens = ingestor(output_dir)
            except Exception as exc:
                self.console.log(
                    f"[red]Source {source_name} failed with unhandled exception:[/red]\n"
                    f"{traceback.format_exc()}"
                )
                tokens = 0
            results[source_name] = tokens
            grand_total += tokens
            self.console.log(
                f"[bold]{source_name}[/bold] → {tokens:,} tokens "
                f"(running total: {grand_total:,})"
            )

        # Summary table
        self._print_summary(results, grand_total, min_tokens)

        # Write corpus manifest
        manifest_path = output_dir / "corpus_manifest.json"
        manifest = {
            "build_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "sources": results,
            "total_tokens": grand_total,
            "min_tokens_target": min_tokens,
            "phase0_gate_passed": grand_total >= min_tokens,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.console.log(f"[green]Manifest written:[/green] {manifest_path}")

        if grand_total < min_tokens:
            self.console.print(
                f"[bold yellow]WARNING:[/bold yellow] corpus has {grand_total:,} tokens "
                f"< target {min_tokens:,}. Phase 0 gate NOT passed. "
                f"Run with --sources all to collect more."
            )
        else:
            self.console.print(
                f"[bold green]Phase 0 exit criterion met:[/bold green] "
                f"{grand_total:,} >= {min_tokens:,} tokens."
            )

        return grand_total

    def _print_summary(
        self, results: dict[str, int], grand_total: int, target: int
    ) -> None:
        table = Table(title="Corpus Build Summary", show_lines=True)
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Tokens", justify="right", style="green")
        table.add_column("% of total", justify="right")
        for source, tokens in results.items():
            pct = f"{100 * tokens / max(grand_total, 1):.1f}%"
            table.add_row(source, f"{tokens:,}", pct)
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{grand_total:,}[/bold]",
            f"[bold]{'✓' if grand_total >= target else '✗'} {grand_total >= target and 'PASS' or 'FAIL'}[/bold]",
        )
        self.console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command(name="pwm-corpus")
@click.option(
    "--sources",
    default=",".join(DEFAULT_SOURCES),
    show_default=True,
    help=(
        "Comma-separated source names to ingest. "
        f"Available: {', '.join(ALL_SOURCES)}, all. "
        "Example: --sources hf_poetry,gutenberg,gretil"
    ),
)
@click.option(
    "--output-dir",
    default="data/corpus",
    show_default=True,
    type=click.Path(),
    help="Root output directory for corpus files.",
)
@click.option(
    "--min-tokens",
    default=100_000,
    show_default=True,
    type=int,
    help="Minimum token count for Phase 0 gate.",
)
@click.option(
    "--list-sources",
    is_flag=True,
    default=False,
    help="Print available source names and exit.",
)
def main(
    sources: str,
    output_dir: str,
    min_tokens: int,
    list_sources: bool,
) -> None:
    """
    PWM corpus ingestion pipeline (Phase 0).

    Builds a creative text corpus from public sources (HuggingFace, GRETIL,
    Project Gutenberg) for world-model pre-training.

    Sanskrit concept: Āgama — valid received knowledge (cf. ĪPK 1.1.5).
    """
    if list_sources:
        console.print("[bold cyan]Available sources:[/bold cyan]")
        for name in ALL_SOURCES:
            console.print(f"  {name}")
        return

    # Set HF_HOME so datasets caches to the right place
    os.environ.setdefault("HF_HOME", HF_HOME)

    source_list = [s.strip() for s in sources.split(",") if s.strip()]

    builder = CorpusBuilder()
    builder.build(source_list, Path(output_dir), min_tokens=min_tokens)


if __name__ == "__main__":
    main()
