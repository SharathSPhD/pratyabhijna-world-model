"""
HuggingFace dataset downloaders for PWM corpus.

Sources:
  - poem_sentiment (surrey-nlp): English poetry with sentiment labels
  - mbien/poetry: Large English poetry dataset
  - ai4bharat/sangraha: High-quality Sanskrit + Indic text
  - wikipedia (20231101.en): Philosophy subset for grounding
  - allenai/c4: Creative writing subset (small sample)

All datasets downloaded via HuggingFace Hub API (uses cached tokens from
`huggingface-cli login` or HF_TOKEN environment variable).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


_HF_CONFIGS: list[dict[str, Any]] = [
    {
        "dataset_id": "surrey-nlp/PLAYSENT",
        "split": "train",
        "text_field": "text",
        "name_prefix": "playsent_poetry",
        "language": "en",
        "max_samples": 5000,
    },
    {
        "dataset_id": "merve/poetry",
        "split": "train",
        "text_field": "content",
        "name_prefix": "merve_poetry",
        "language": "en",
        "max_samples": 10000,
    },
    {
        "dataset_id": "ai4bharat/sangraha",
        "split": "train",
        "text_field": "text",
        "name_prefix": "sangraha_sanskrit",
        "language": "sa",
        "subset": "sa",
        "max_samples": 2000,
    },
]


def _safe_text(row: dict[str, Any], field: str) -> str:
    val = row.get(field, "")
    return str(val) if val is not None else ""


def ingest_hf_datasets(
    output_dir: Path,
    min_tokens: int = 20,
    configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Download and write HuggingFace datasets to output_dir/hf/.

    Returns metadata list. Skips configs that fail (network/auth errors).
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise ImportError("datasets package required: pip install datasets") from e

    out = output_dir / "hf"
    out.mkdir(parents=True, exist_ok=True)
    manifested: list[dict[str, Any]] = []
    cfg_list = configs or _HF_CONFIGS

    for cfg in cfg_list:
        dataset_id = cfg["dataset_id"]
        split = cfg.get("split", "train")
        text_field = cfg["text_field"]
        prefix = cfg["name_prefix"]
        language = cfg.get("language", "en")
        max_samples = cfg.get("max_samples", 5000)
        subset = cfg.get("subset")

        try:
            ds_kwargs: dict[str, Any] = {"split": split, "trust_remote_code": True}
            if subset:
                ds_kwargs["name"] = subset
            dataset = load_dataset(dataset_id, **ds_kwargs)
            # Limit samples
            if hasattr(dataset, "select"):
                n = min(max_samples, len(dataset))
                dataset = dataset.select(range(n))
        except Exception as exc:
            print(f"[hf_datasets] Skipping {dataset_id}: {exc}")
            continue

        doc_texts: list[str] = []
        for row in dataset:
            text = _safe_text(dict(row), text_field)
            if len(text.split()) >= min_tokens:
                doc_texts.append(text)

        if not doc_texts:
            continue

        # Write as one file per 500 samples to keep files manageable
        for batch_idx, start in enumerate(range(0, len(doc_texts), 500)):
            batch = doc_texts[start:start + 500]
            combined = "\n\n---\n\n".join(batch)
            fname = f"{prefix}_b{batch_idx:04d}"
            (out / f"{fname}.txt").write_text(combined, encoding="utf-8")

            meta: dict[str, Any] = {
                "source": "huggingface",
                "dataset_id": dataset_id,
                "split": split,
                "batch": batch_idx,
                "language": language,
                "n_docs": len(batch),
                "token_count": len(combined.split()),
            }
            (out / f"{fname}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifested.append(meta)

    return manifested
