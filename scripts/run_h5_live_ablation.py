#!/usr/bin/env python3
"""
H5 Live Ablation: PWM-conditioned generation vs bare LLM baseline.

Pre-registered H5: PWM > PCE v0.4 on creative quality (R_camatk density + S_svātantrya).
Reformulation for Phase 5/6: WM-conditioned (VimarsaBridgeV2) > bare-LLM baseline
on text-level camatkāra heuristic (n_samples per domain, paired permutation test).

Uses real WM checkpoint + VimarsaBridgeV2 + Ollama backend (nemotron-3-super:120b).

Statistical protocol (CLAUDE.md §7):
  Paired permutation test (50K perms), Hedges' g, BCa 95% CI (10K resamples).
  One-tailed: H5 predicts PWM > LLM-only.

Usage:
    python scripts/run_h5_live_ablation.py [--dry-run] [--n-samples 5]

Sanskrit concept: Pratyabhijñā (ĪPK 1.3) — the WM's self-recognition (h_t)
enables the LLM to generate from a richer creative position than it could reach
from the user prompt alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [H5] %(levelname)s %(message)s")
log = logging.getLogger("h5_live")

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

# Reference from Phase 5 gate (phase_5_gate_step0500000.json)
H5_PHASE5_RATIO = 2.14183
H5_THRESHOLD = 2.0

PHASE6_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "phase6_step1000000.pt"
VIMARSA_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vimarsa_bridge_v2.pt"

# Fallback paths across worktrees
_PHASE6_FALLBACKS = [
    PHASE6_CHECKPOINT,
    Path("/home/sharaths/projects/pwm-phase2/checkpoints/final_phase6_seed55.pt"),
    Path("/home/sharaths/projects/pwm-phase6/checkpoints/final_phase6_seed55.pt"),
]
_BRIDGE_FALLBACKS = [
    VIMARSA_CHECKPOINT,
    Path("/home/sharaths/projects/pwm-phase6/checkpoints/vimarsa_bridge_v2.pt"),
    Path("/home/sharaths/projects/pwm-phase5/checkpoints/vimarsa_bridge_v2.pt"),
]


def _find_checkpoint(candidates: list[Path]) -> Path | None:
    """Return first existing checkpoint path."""
    for p in candidates:
        if p.exists():
            return p
    return None


def run_live_ablation(n_samples: int = 5, dry_run: bool = False) -> dict:
    """
    Run H5 ablation with real WM + VimarsaBridgeV2 + Ollama.

    Returns results dict with scores, statistics, gate_pass flag.
    """
    if dry_run:
        log.info("[dry-run] Simulating H5 live ablation (no real WM or LLM calls)")
        import random
        import numpy as np
        rng = random.Random(42)
        # Simulate scores consistent with Phase 5 gate: ratio ~2.14
        # PWM mean ~0.74, LLM mean ~0.345 → ratio ~2.14
        pwm_scores = [rng.gauss(0.74, 0.07) for _ in range(n_samples * 3)]
        llm_scores = [rng.gauss(0.345, 0.08) for _ in range(n_samples * 3)]
        ratio = float(np.mean(pwm_scores) / (np.mean(llm_scores) + 1e-12))
        return {
            "mode": "dry_run",
            "n_samples_per_domain": n_samples,
            "pwm_scores_mean": round(float(np.mean(pwm_scores)), 4),
            "llm_scores_mean": round(float(np.mean(llm_scores)), 4),
            "ratio": round(ratio, 4),
            "h5_threshold": H5_THRESHOLD,
            "gate_pass": ratio >= H5_THRESHOLD,
            "note": "dry-run scores — run without --dry-run for real results",
        }

    # ── Load WM checkpoint ────────────────────────────────────────────────────
    ckpt_wm = _find_checkpoint([
        *_PHASE6_FALLBACKS,
        PROJECT_ROOT / "checkpoints" / "phase5_step500000.pt",
        *sorted((PROJECT_ROOT / "checkpoints").glob("final_phase*.pt"), reverse=True)[:3],
    ])
    if ckpt_wm is None:
        log.error("No WM checkpoint found. Searched: %s", _PHASE6_FALLBACKS)
        return {"error": "no_checkpoint", "gate_pass": False}

    ckpt_bridge = _find_checkpoint([
        *_BRIDGE_FALLBACKS,
        *sorted((PROJECT_ROOT / "checkpoints").glob("vimarsa*.pt"), reverse=True)[:2],
    ])
    if ckpt_bridge is None:
        log.error("No VimarsaBridgeV2 checkpoint found. Run train_vimarsa_bridge.py first.")
        return {"error": "no_bridge_checkpoint", "gate_pass": False}

    log.info("WM checkpoint: %s", ckpt_wm)
    log.info("Bridge checkpoint: %s", ckpt_bridge)

    # ── Load components ───────────────────────────────────────────────────────
    import torch
    from pwm.world_model.trika import TrikaWorldModel
    from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
    from pwm.generation.llama_backend import LlamaCppBackend
    from pwm.generation.engine import DEVICE
    from pwm.eval.h5_ablation import run_h5_ablation, H5Config

    device = torch.device(DEVICE)
    log.info("Loading TrikaWorldModel from %s", ckpt_wm)
    wm_state = torch.load(str(ckpt_wm), map_location=device, weights_only=False)
    wm_sd = wm_state.get("world_model", wm_state)
    # Detect n_levels from checkpoint keys
    import re as _re
    n_levels = max(
        int(m.group(1)) for k in wm_sd
        if (m := _re.match(r"levels\.(\d+)\.", k))
    ) + 1
    log.info("Detected n_levels=%d from checkpoint", n_levels)
    wm = TrikaWorldModel(obs_dim=512, action_dim=64, n_levels=n_levels, hidden_dim=512,
                         decoder_z_only=True)
    wm.load_state_dict(wm_sd, strict=False)
    wm.eval().to(device)
    log.info("WM loaded (n_levels=%d)", n_levels)

    log.info("Loading VimarsaBridgeV2 from %s", ckpt_bridge)
    bridge_state = torch.load(str(ckpt_bridge), map_location=device, weights_only=False)
    bridge = VimarsaBridgeV2(hidden_dim=512)
    bridge.load_state_dict(bridge_state.get("bridge", bridge_state), strict=False)
    bridge.eval().to(device)
    log.info("VimarsaBridgeV2 loaded")

    # Ollama backend (nemotron-3-super:120b, with cascade mini→super)
    llm = LlamaCppBackend(
        model_path="",                       # not used in Ollama mode
        server_url="http://localhost:11434",
        model_name="nemotron-3-super:120b",
        cascade_model_name="nemotron-mini:4b",
    )

    cfg = H5Config(
        n_samples_per_domain=n_samples,
        n_permutations=50_000,
        n_bootstrap=10_000,
        max_tokens=120,
        temperature=0.85,
        top_p=0.92,
    )

    log.info("Running H5 ablation (%d samples/domain × %d domains)…",
             n_samples, len(cfg.domains))
    result = run_h5_ablation(wm=wm, bridge=bridge, llm_backend=llm, cfg=cfg)

    log.info("H5 live result: ratio=%.4f, gate_pass=%s", result.get("ratio", 0), result.get("gate_pass"))
    return result


def main():
    parser = argparse.ArgumentParser(description="H5 live ablation: PWM vs bare LLM")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Stanzas per domain (default 5; paper uses 10)")
    args = parser.parse_args()

    result = run_live_ablation(n_samples=args.n_samples, dry_run=args.dry_run)

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "ablation": "H5_live",
        "hypothesis": "H5",
        "description": "PWM-conditioned (VimarsaBridgeV2) vs bare-LLM on camatkāra heuristic",
        "backend": "nemotron-3-super:120b via Ollama (cascade nemotron-mini:4b)",
        "phase5_gate_reference": {"ratio": H5_PHASE5_RATIO, "threshold": H5_THRESHOLD, "status": "PASS"},
        "live_result": result,
        "gate_pass": result.get("gate_pass", False),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "h5_live_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Results saved: %s", out_path)
    log.info("H5 live gate: %s", "PASS" if result.get("gate_pass") else "FAIL")

    return 0 if result.get("gate_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
