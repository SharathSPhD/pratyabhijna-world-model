#!/usr/bin/env python3
"""
Prepare HuggingFace dataset from PWM benchmark results.

Creates hf_dataset/ with:
  - creative_outputs.jsonl  (13 PWM-generated creative works)
  - hypothesis_results.jsonl (H1-H9 phase gate results)
  - README.md               (dataset card, already written)

Usage:
    python scripts/prepare_hf_dataset.py [--upload]

HuggingFace repo: SharathSPhD/pratyabhijna-world-model-outputs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
HF_DIR = PROJECT_ROOT / "hf_dataset"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("hf_dataset")

# Phase gate → hypothesis mapping
PHASE_GATES = {
    "H1": ("phase_2_gate.json", "h1_pass"),
    "H2": ("phase3_gate.json", "h2_pass"),
    "H3": ("phase4_gate.json", "h3_pass"),
    "H4": ("phase5_gate.json", "h4_pass"),
    "H5": ("phase_5_gate_step0500000.json", "h5_pass"),
    "H6": ("phase_6_gate_step1000000.json", "h6_pass"),
    "H7": ("ablation_a6_1level_wm.json", "h7_ablation_gate"),
    "H8": ("phase_6_gate_step1000000.json", "h8_pass"),
    "H9": ("phase_6_gate_step1000000.json", "h9_pass"),
}

HYPOTHESIS_DESCRIPTIONS = {
    "H1": "EFE actor > REINFORCE on sparse creative reward (episodes to first sphurattā)",
    "H2": "Hopfield memory improves pattern completion accuracy",
    "H3": "Sleep consolidation reduces catastrophic forgetting in sequential domain training",
    "H4": "VimarsaBridge improves narration quality (human 'meaningful' rate ≥70%)",
    "H5": "PWM > PCE v0.4 on creative quality (R_camatk density + S_svātantrya)",
    "H6": "Camatkāra correlates with human aesthetic judgment (DTW distance)",
    "H7": "3-level Trika hierarchy outperforms 1-level Aparā-only on long-horizon creativity",
    "H8": "Mala regularisers prevent latent collapse (metre satisfaction rate)",
    "H9": "S_svātantrya correlates with human novelty ratings (Spearman ρ)",
}


def prepare_creative_outputs() -> list[dict]:
    """Extract and flatten creative outputs from benchmarks/results/creative_outputs.json."""
    src = RESULTS_DIR / "creative_outputs.json"
    if not src.exists():
        log.warning("creative_outputs.json not found at %s", src)
        return []

    with open(src) as f:
        raw = json.load(f)

    outputs = raw.get("outputs", [])
    flat = []
    for o in outputs:
        scores = o.get("scores", {})
        flat.append({
            "id": o.get("id", ""),
            "title": o.get("title", ""),
            "language": o.get("language", ""),
            "style": o.get("style", ""),
            "text": o.get("text", ""),
            "wm_seed": int(o.get("wm_seed", 0)),
            "warmup_steps": int(o.get("warmup_steps", 0)),
            "wm_vfe": float(o.get("wm_vfe", 0.0)),
            "wm_prefix": o.get("wm_prefix", ""),
            "camatk_total": float(scores.get("camatk_total", 0.0)),
            "r_camatk": float(scores.get("r_camatk", 0.0)),
            "vfe_score": float(scores.get("vfe_score", 0.0)),
            "term_score": float(scores.get("term_score", 0.0)),
            "structure_score": float(scores.get("structure_score", 0.0)),
            "word_count": int(scores.get("word_count", 0)),
            "generated_at": o.get("generated_at", ""),
            "model": o.get("model", ""),
            "checkpoint": o.get("checkpoint", ""),
        })

    log.info("Prepared %d creative outputs", len(flat))
    return flat


def prepare_hypothesis_results() -> list[dict]:
    """Extract hypothesis pass/fail status from phase gate JSONs."""
    rows = []
    for hyp_id, (gate_file, pass_key) in PHASE_GATES.items():
        gate_path = RESULTS_DIR / gate_file
        row = {
            "hypothesis_id": hyp_id,
            "description": HYPOTHESIS_DESCRIPTIONS[hyp_id],
            "gate_file": gate_file,
        }
        if gate_path.exists():
            with open(gate_path) as f:
                gate = json.load(f)
            # Try to find the pass key at top level or nested
            gate_pass = gate.get(pass_key, gate.get("gate_pass", gate.get("h7_ablation_gate")))
            if isinstance(gate_pass, str):
                gate_pass = gate_pass.startswith("PASS")
            row["gate_pass"] = bool(gate_pass) if gate_pass is not None else None
            # Extract key metrics
            for k in ["hedges_g", "p_value", "ratio", "vfe_ratio_1level",
                       "mean_camatk", "accuracy", "correlation"]:
                v = gate.get(k)
                if v is None:
                    # Search one level deep
                    for sub in gate.values():
                        if isinstance(sub, dict):
                            v = sub.get(k)
                            if v is not None:
                                break
                if v is not None:
                    row[k] = float(v) if isinstance(v, (int, float)) else v
        else:
            log.warning("Gate file not found: %s", gate_path)
            row["gate_pass"] = None

        rows.append(row)
        log.info("%s → %s (file: %s)", hyp_id, row.get("gate_pass"), gate_file)

    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows → %s", len(rows), path)


def main():
    parser = argparse.ArgumentParser(description="Prepare HF dataset")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub (requires huggingface-hub + token)")
    args = parser.parse_args()

    # Prepare splits
    creative = prepare_creative_outputs()
    hypotheses = prepare_hypothesis_results()

    # Write JSONL
    write_jsonl(creative, HF_DIR / "creative_outputs.jsonl")
    write_jsonl(hypotheses, HF_DIR / "hypothesis_results.jsonl")

    log.info("Dataset prepared in %s", HF_DIR)
    log.info("  creative_outputs: %d examples", len(creative))
    log.info("  hypothesis_results: %d examples", len(hypotheses))

    if args.upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = "SharathSPhD/pratyabhijna-world-model-outputs"
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=str(HF_DIR),
                repo_id=repo_id,
                repo_type="dataset",
            )
            log.info("Uploaded to https://huggingface.co/datasets/%s", repo_id)
        except ImportError:
            log.error("huggingface-hub not installed. Run: pip install huggingface-hub")
            sys.exit(1)
        except Exception as exc:
            log.error("Upload failed: %s", exc)
            sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
