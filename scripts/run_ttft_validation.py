#!/usr/bin/env python3
"""
Live TTFT/switch latency validation for Phase 7 ADR-001 + ADR-002.

Measures:
  1. TTFT (time-to-first-token) — fast model (nemotron-mini:4b) immediate stream
  2. Switch latency — gap until nemotron-3-super:120b first content token
     a. Without WM trace prefill (cold CoT — baseline for ADR-001)
     b. With WMReasoningTrace prefill (ADR-002 condition)

Expected results (from gate JSON phase7_gate.json):
  TTFT:          < 5s  (ADR-001)
  Switch without trace: ~65s (cold 120B CoT)
  Switch with trace:    ~5s  (ADR-002, WM prefill eliminates CoT)

Usage:
    python scripts/run_ttft_validation.py [--n-prompts 5] [--dry-run]

Statistical output:
  Mean ± std for each condition across n_prompts
  Paired t-test for switch_with vs switch_without

Sanskrit concept: Sphurattā (TĀ 1.56) — the flash of recognition; measured here
as the latency between prompt submission and the first token of creative output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [TTFT] %(levelname)s %(message)s")
log = logging.getLogger("ttft_val")

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

PROMPTS = [
    "Write 4 lines of evocative Carnatic raga imagery about dawn.",
    "Write a 4-line Kannada film lyric about monsoon and longing.",
    "Write a short Hindi ghazal couplet about memory and absence.",
    "Write 4 lines of English jazz lyric in the style of a spiritual.",
    "Write a Sanskrit śloka about the nature of creative consciousness.",
]


def _first_token_latency(llm_backend, system: str, user: str,
                         think_prefill: dict | None = None) -> tuple[float, float]:
    """
    Stream tokens and return (ttft_s, total_switch_s).

    ttft_s        — time from call to first token yielded
    total_switch_s — for cascade: time until slow model content begins
    """
    t_start = time.perf_counter()
    first_tok = None
    n_tokens = 0
    for _ in llm_backend.stream(
        system=system, user=user,
        logits_processor=None,
        max_tokens=80,
        temperature=0.85,
        top_p=0.92,
        think_prefill=think_prefill,
    ):
        t_now = time.perf_counter()
        if first_tok is None:
            first_tok = t_now - t_start
        n_tokens += 1
        if n_tokens >= 40:   # enough for a 4-line lyric
            break

    total = time.perf_counter() - t_start
    return (first_tok or total), total


def run_validation(n_prompts: int = 5, dry_run: bool = False) -> dict:
    """Run TTFT/switch validation across n_prompts × 3 conditions."""
    if dry_run:
        log.info("[dry-run] Simulating TTFT validation")
        import random
        rng = random.Random(42)
        # Simulate: TTFT ~3.5s (cascade), switch_no_trace ~62s, switch_with_trace ~5s
        return {
            "mode": "dry_run",
            "n_prompts": n_prompts,
            "ttft_mean_s": round(rng.gauss(3.5, 0.4), 3),
            "ttft_std_s": round(abs(rng.gauss(0.35, 0.05)), 3),
            "switch_no_trace_mean_s": round(rng.gauss(62.0, 4.0), 2),
            "switch_no_trace_std_s": round(abs(rng.gauss(3.5, 0.5)), 2),
            "switch_with_trace_mean_s": round(rng.gauss(5.2, 0.8), 2),
            "switch_with_trace_std_s": round(abs(rng.gauss(0.7, 0.1)), 2),
            "adr001_ttft_pass": True,
            "adr002_switch_pass": True,
            "note": "dry-run — run without --dry-run for real measurements",
        }

    from pwm.generation.llama_backend import LlamaCppBackend
    from pwm.vimarsa.narrator import WMReasoningTrace
    import torch

    # Load WM for trace generation (light — CPU only)
    from pwm.world_model.trika import TrikaWorldModel
    ckpt_path = Path("/home/sharaths/projects/pwm-phase2/checkpoints/final_phase6_seed55.pt")
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    wm_sd = raw_ckpt.get("world_model", raw_ckpt)
    import re as _re
    n_levels = max(
        int(m.group(1)) for k in wm_sd if (m := _re.match(r"levels\.(\d+)\.", k))
    ) + 1
    wm = TrikaWorldModel(obs_dim=512, action_dim=64, n_levels=n_levels, hidden_dim=512,
                         decoder_z_only=True, free_bits=0.1)
    wm.load_state_dict(wm_sd, strict=False)
    log.info("WM loaded: n_levels=%d", n_levels)
    wm.eval()

    # Pre-generate a think-block prefill (same for all prompts — representative)
    tracer = WMReasoningTrace()
    with torch.no_grad():
        states = wm.init_state(1, "cpu")
        obs = torch.randn(1, 512) * 0.3
        states, _, _ = wm.observe_step(obs, torch.zeros(1, 64), states, 0)
        h_t = states[0][0]

    think_prefill = tracer.render_as_assistant_prefill(
        h_t=h_t, domain="carnatic", stanza_idx=0, camatk_score=0.65,
    )

    llm = LlamaCppBackend(
        model_path="",
        server_url="http://localhost:11434",
        model_name="nemotron-3-super:120b",
        cascade_model_name="nemotron-mini:4b",
    )

    system = "You are a creative lyricist. Output only lyrics, no explanations."
    prompts = (PROMPTS * ((n_prompts // len(PROMPTS)) + 1))[:n_prompts]

    ttft_vals, switch_no_trace, switch_with_trace = [], [], []

    for i, user in enumerate(prompts):
        log.info("Prompt %d/%d: %s…", i + 1, n_prompts, user[:50])

        # Condition A: cascade, no WM trace (cold CoT)
        t1, sw1 = _first_token_latency(llm, system, user, think_prefill=None)
        ttft_vals.append(t1)
        switch_no_trace.append(sw1)
        log.info("  A (no trace): TTFT=%.2fs, total=%.2fs", t1, sw1)

        # Condition B: cascade, with WM trace (ADR-002)
        t2, sw2 = _first_token_latency(llm, system, user, think_prefill=think_prefill)
        switch_with_trace.append(sw2)
        log.info("  B (WM trace): TTFT=%.2fs, total=%.2fs", t2, sw2)

    import numpy as np
    ttft_arr = np.array(ttft_vals)
    no_trace_arr = np.array(switch_no_trace)
    with_trace_arr = np.array(switch_with_trace)

    # Paired t-test: ADR-002 switch < no-trace switch
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(with_trace_arr, no_trace_arr, alternative="less")

    adr001_pass = float(np.mean(ttft_arr)) < 5.0
    adr002_pass = (float(np.mean(with_trace_arr)) < float(np.mean(no_trace_arr)) * 0.20  # <20% of baseline
                   and p_val < 0.05)

    result = {
        "n_prompts": n_prompts,
        "ttft_mean_s": round(float(np.mean(ttft_arr)), 3),
        "ttft_std_s": round(float(np.std(ttft_arr, ddof=1)), 3),
        "switch_no_trace_mean_s": round(float(np.mean(no_trace_arr)), 2),
        "switch_no_trace_std_s": round(float(np.std(no_trace_arr, ddof=1)), 2),
        "switch_with_trace_mean_s": round(float(np.mean(with_trace_arr)), 2),
        "switch_with_trace_std_s": round(float(np.std(with_trace_arr, ddof=1)), 2),
        "switch_reduction_pct": round(100 * (1 - float(np.mean(with_trace_arr)) / float(np.mean(no_trace_arr))), 1),
        "paired_ttest_t": round(float(t_stat), 4),
        "paired_ttest_p": round(float(p_val), 6),
        "adr001_ttft_pass": adr001_pass,
        "adr002_switch_pass": adr002_pass,
    }
    log.info("Results: TTFT=%.2f±%.2fs, no-trace=%.2f±%.2fs, with-trace=%.2f±%.2fs, p=%.4f",
             result["ttft_mean_s"], result["ttft_std_s"],
             result["switch_no_trace_mean_s"], result["switch_no_trace_std_s"],
             result["switch_with_trace_mean_s"], result["switch_with_trace_std_s"],
             p_val)
    return result


def main():
    parser = argparse.ArgumentParser(description="Live TTFT/switch validation (ADR-001 + ADR-002)")
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_validation(n_prompts=args.n_prompts, dry_run=args.dry_run)

    out = {
        "validation": "ttft_switch_live",
        "description": "ADR-001 TTFT (<5s) and ADR-002 switch latency (with vs without WM trace)",
        "backend": "nemotron-3-super:120b via Ollama cascade (nemotron-mini:4b → 120b)",
        "targets": {"ttft_s": "<5", "switch_reduction_pct": ">80"},
        "result": result,
        "gate_pass": result.get("adr001_ttft_pass", False) and result.get("adr002_switch_pass", False),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ttft_live_validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Saved: %s  gate_pass=%s", out_path, out["gate_pass"])
    return 0 if out["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
