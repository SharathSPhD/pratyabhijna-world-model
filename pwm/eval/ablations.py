"""
Ablation runner for Phase 1 hypotheses H1–H9 and ablations A1–A6.

Philosophical grounding:
  Viveka (BS 1.1.4, Shankara): Discrimination — knowing what contributes
  to the creative capacity and what does not. Ablations test each component
  in isolation to establish causal credit.

Ablation design:
  Each ablation is a named config override (config-driven, CLAUDE.md §8).
  The ablation runner spawns independent training jobs with the override,
  then compares final VFE and held-out MSE.

  H1: EFE actor > REINFORCE on sparse creative reward
  H2: Hopfield improves pattern completion (Phase 3)
  H3: Sleep reduces catastrophic forgetting (Phase 4)
  H4: Vimarśa bridge improves narration quality (Phase 5)
  H5: PWM > PCE v0.4 on creative quality
  H6: Camatkāra correlates with human aesthetic judgment
  H7: 3-level Trika > 1-level on long-horizon creativity (Phase 2+)
  H8: Mala regularisers prevent latent collapse
  H9: S_svātantrya correlates with human novelty ratings

  A1: No EFE (REINFORCE only)
  A2: No Hopfield (random retrieval)
  A3: No sleep
  A4: No vimarśa bridge (LLM disabled)
  A5: No Mala regularisers
  A6: Single-level Trika only
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    hypothesis_id: str
    description: str
    config_overrides: list[str]   # Hydra override strings, e.g. ["actor.type=reinforce"]
    phase: int = 1                # Minimum phase required
    seeds: list[int] = field(default_factory=lambda: [42, 1337, 0])
    n_steps: int = 50_000         # Steps per seed (reduced from 500K for speed)


# ── Phase 1 ablations (runnable now) ─────────────────────────────────────────

PHASE1_ABLATIONS: list[AblationConfig] = [
    AblationConfig(
        name="a5_no_mala",
        hypothesis_id="H8",
        description="No Mala regularisers — test latent collapse prevention",
        config_overrides=["reward.alpha_2=0.0", "reward.alpha_3=0.0"],
        phase=1,
    ),
    AblationConfig(
        name="a6_single_level",
        hypothesis_id="H7",
        description="Single-level Trika (Aparā only) — baseline for Phase 2 comparison",
        config_overrides=["world_model.levels=1"],
        phase=1,
    ),
]


# ── All ablations (some require later phases) ─────────────────────────────────

ALL_ABLATIONS: list[AblationConfig] = PHASE1_ABLATIONS + [
    AblationConfig(
        name="a1_no_efe",
        hypothesis_id="H1",
        description="REINFORCE only (no EFE actor) — baseline for Phase 2",
        config_overrides=["actor.type=reinforce"],
        phase=2,
    ),
    AblationConfig(
        name="a2_no_hopfield",
        hypothesis_id="H2",
        description="No Hopfield store — random retrieval baseline",
        config_overrides=["memory.enabled=false"],
        phase=3,
    ),
    AblationConfig(
        name="a3_no_sleep",
        hypothesis_id="H3",
        description="No sleep consolidation",
        config_overrides=["sleep.enabled=false"],
        phase=4,
    ),
    AblationConfig(
        name="a4_no_vimarsa",
        hypothesis_id="H4",
        description="No vimarśa bridge (LLM disabled entirely)",
        config_overrides=["llm.enabled=false"],
        phase=5,
    ),
]


def run_ablation(
    ablation: AblationConfig,
    train_script: Path,
    config_name: str = "phase1_apara",
    corpus_root: str = "",
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run a single ablation across all seeds.

    Returns per-seed results and aggregate statistics.
    """
    output_dir = output_dir or Path("benchmarks/results/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "name": ablation.name,
        "hypothesis_id": ablation.hypothesis_id,
        "description": ablation.description,
        "overrides": ablation.config_overrides,
        "seeds": [],
    }

    for seed in ablation.seeds:
        log.info("Running ablation '%s' seed=%d", ablation.name, seed)
        cmd = [
            "python", str(train_script),
            "--config-name", config_name,
            f"training.seed={seed}",
            f"training.max_steps={ablation.n_steps}",
            *ablation.config_overrides,
        ]

        env: dict[str, str] = {}
        if corpus_root:
            env["CORPUS_ROOT"] = corpus_root

        seed_result: dict[str, Any] = {"seed": seed, "overrides": ablation.config_overrides}

        if dry_run:
            log.info("[dry_run] cmd: %s", " ".join(cmd))
            seed_result["status"] = "dry_run"
        else:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,   # 2 hours max
                    env={**__import__("os").environ, **env},
                    cwd=str(train_script.parent.parent.parent),
                )
                # Parse final VFE from stdout
                final_vfe = None
                for line in proc.stdout.split("\n"):
                    if "Training complete" in line or "validation" in line:
                        parts = line.split("VFE=")
                        if len(parts) > 1:
                            try:
                                final_vfe = float(parts[1].strip().split()[0])
                            except ValueError:
                                pass
                seed_result["final_vfe"] = final_vfe
                seed_result["exit_code"] = proc.returncode
                seed_result["status"] = "ok" if proc.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                seed_result["status"] = "timeout"
            except Exception as e:
                seed_result["status"] = f"error: {e}"

        results["seeds"].append(seed_result)

    # Aggregate across seeds
    vfe_vals = [s.get("final_vfe") for s in results["seeds"] if s.get("final_vfe") is not None]
    if vfe_vals:
        results["final_vfe_mean"] = float(sum(vfe_vals) / len(vfe_vals))
        results["final_vfe_std"] = float(
            (sum((v - results["final_vfe_mean"]) ** 2 for v in vfe_vals) / len(vfe_vals)) ** 0.5
        )

    # Save result JSON
    out_path = output_dir / f"{ablation.name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Ablation '%s' saved: %s", ablation.name, out_path)

    return results


def run_phase1_ablations(
    train_script: Path,
    corpus_root: str = "",
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Run all Phase 1 ablations.

    Returns list of result dicts.
    """
    return [
        run_ablation(abl, train_script, corpus_root=corpus_root, output_dir=output_dir, dry_run=dry_run)
        for abl in PHASE1_ABLATIONS
    ]
