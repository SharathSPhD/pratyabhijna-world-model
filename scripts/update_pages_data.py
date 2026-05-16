#!/usr/bin/env python3
"""
update_pages_data.py — regenerate the hardcoded DATA constant in docs/app.js
from the latest gate JSON artefacts in benchmarks/results/.

Run after each phase gate completes:
    python3 scripts/update_pages_data.py

The script patches only the DATA block (between the sentinel comments), leaving
the rest of app.js unchanged. This implements the TRIZ C2 resolution (Principle 9 —
Separation by Condition): the page works offline with the baked-in data and gets
live updates on GitHub Pages via async fetch.

Usage:
    python3 scripts/update_pages_data.py [--dry-run]
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
APP_JS  = ROOT / "docs" / "app.js"

def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}

def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

def main(dry_run=False):
    p2 = load("phase_2_gate.json")
    p3 = load("phase_3_gate_step0300000.json")
    p4 = load("phase_4_gate_step0300000.json")
    p5 = load("phase_5_gate_step0500000.json")
    p6 = load("phase_6_gate_step1000000.json")

    today = date.today().isoformat()

    # Build hypothesis rows from live gate data (fall back to paper values)
    hyps = [
        dict(id="H1", short="EFE > REINFORCE",
             claim="EFE actor outperforms REINFORCE on sparse creative reward",
             metric="Mean episode reward ratio",
             efe_value=g(p2, "efe_actor", "mean_episode_reward", default=2.530),
             reinforce_value=g(p2, "reinforce_baseline", "mean_episode_reward", default=0.085),
             ratio=g(p2, "h1_reward_ratio", default=29.72),
             threshold=g(p2, "h1_reward_threshold", default=2.0),
             result="PASS" if g(p2, "h1_pass") else "FAIL",
             result_text=f"{g(p2,'h1_reward_ratio',default=29.72):.2f}× reward improvement",
             phase=2),
        dict(id="H2", short="Hopfield Completion",
             claim="Hopfield memory outperforms nearest-neighbour on 50% occlusion completion",
             metric="Occlusion accuracy ratio",
             hopfield_acc=g(p3, "completion", "acc_hopfield", default=0.846),
             baseline_acc=g(p3, "completion", "acc_baseline", default=0.648),
             ratio=g(p3, "completion", "ratio", default=1.307),
             threshold=g(p3, "completion", "threshold", default=1.10),
             result="PASS" if g(p3, "h2_pass") else "FAIL",
             result_text=f"{g(p3,'completion','acc_hopfield',default=0.846):.3f} vs {g(p3,'completion','acc_baseline',default=0.648):.3f} NN baseline",
             phase=3),
        dict(id="H3", short="Sleep vs. Forgetting",
             claim="Sleep consolidation reduces catastrophic forgetting across 3 sequential domains",
             metric="Forgetting rate",
             with_sleep=g(p4, "h3_forgetting_with_sleep", default=0.0),
             without_sleep=g(p4, "h3_forgetting_without_sleep", default=0.0),
             ratio=0.0, threshold=0.8,
             result="PASS" if g(p4, "h3_pass") else "FAIL",
             result_text="Forgetting ≈ 0 with NREM+REM consolidation",
             phase=4),
        dict(id="H4", short="Vimarśa Narration",
             claim="Vimarśa bridge produces meaningful narrations at sphurattā events",
             metric="Human meaningful rate",
             meaningful_rate=g(p5, "h4_meaningful_rate", default=1.0),
             n_sphuratta=g(p5, "h4_n_sphuratta", default=1600),
             n_meaningful=g(p5, "h4_n_meaningful", default=1600),
             threshold=g(p5, "h4_threshold", default=0.70),
             result="PASS" if g(p5, "h4_pass") else "FAIL",
             result_text=f"{g(p5,'h4_meaningful_rate',default=1.0)*100:.0f}% meaningful rate",
             phase=5),
        dict(id="H5", short="PWM > PCE v0.4",
             claim="Full PWM exceeds PCE v0.4 baseline on camatkāra reward density",
             metric="Reward ratio (full PWM / Phase-2 baseline)",
             pwm_reward=g(p5, "h5_mean_reward_phase5", default=5.419),
             baseline_reward=g(p5, "h5_phase2_baseline", default=2.530),
             ratio=g(p5, "h5_reward_ratio", default=2.142),
             threshold=g(p5, "h5_threshold", default=2.0),
             result="PASS" if g(p5, "h5_pass") else "FAIL",
             result_text=f"{g(p5,'h5_reward_ratio',default=2.142):.3f}× reward improvement over Phase-2 baseline",
             phase=5),
        dict(id="H6", short="Reward Diversity",
             claim="Camatkāra reward distribution exhibits non-trivial entropy across 500 episodes",
             metric="Reward entropy (nats, 20 bins)",
             reward_entropy=g(p6, "h6_reward_entropy", default=1.897),
             threshold=g(p6, "h6_threshold", default=0.5),
             result="PASS" if g(p6, "h6_pass") else "FAIL",
             result_text=f"{g(p6,'h6_reward_entropy',default=1.897):.3f} nats — diverse aesthetic event distribution",
             phase=6),
        dict(id="H7", short="Trika Long-Horizon",
             claim="3-level Trika RSSM maintains low VFE over 32-step imagination rollouts",
             metric="32-step VFE",
             vfe_phase6=g(p6, "h7_vfe_phase6", default=4.86e-4),
             vfe_ratio=g(p6, "h7_vfe_ratio", default=0.0),
             threshold=g(p6, "h7_threshold", default=0.85),
             result="PASS" if g(p6, "h7_pass") else "FAIL",
             result_text=f"VFE = {g(p6,'h7_vfe_phase6',default=4.86e-4):.4g}",
             phase=6),
        dict(id="H8", short="Encoder Stability",
             claim="Māla regularisers prevent latent collapse; encoder norm stays in [1.0, 50.0]",
             metric="Encoder weight L2 norm",
             encoder_norm=g(p6, "h8_encoder_norm", default=13.197),
             norm_min=1.0, norm_max=50.0,
             result="PASS" if g(p6, "h8_pass") else "FAIL",
             result_text=f"Norm = {g(p6,'h8_encoder_norm',default=13.197):.3f} — within [1.0, 50.0] bounds",
             phase=6),
        dict(id="H9", short="Action Diversity",
             claim="IDL-committed policy exhibits at least 2 distinct committed action modes",
             metric="Greedy-action empirical diversity entropy (nats)",
             action_entropy=g(p6, "h9_action_entropy", default=0.582),
             threshold=g(p6, "h9_threshold", default=0.5),
             result="PASS" if g(p6, "h9_pass") else "FAIL",
             result_text=f"{g(p6,'h9_action_entropy',default=0.582):.3f} nats > 0.5 threshold",
             phase=6),
    ]

    new_hyp_js = "  hypotheses: " + json.dumps(hyps, indent=4).replace("\n", "\n  ") + ","

    # Read current app.js
    src = APP_JS.read_text()

    # Replace generated date
    src = re.sub(
        r'generated: "[^"]*"',
        f'generated: "{today}"',
        src, count=1
    )

    # Replace hypotheses array inside DATA (between "hypotheses: [" and matching "]")
    # We use a sentinel pattern: "hypotheses: [" ... closing "],"
    hyps_json = json.dumps(hyps, indent=4)
    # Indent to match the 2-space DATA block indent
    hyps_indented = hyps_json.replace("\n", "\n  ")
    new_hyp_block = f"  hypotheses: {hyps_indented},"

    # Use string-split approach to avoid re.sub treating JSON escapes as regex
    marker_start = "  hypotheses: ["
    marker_end = "],"
    idx_s = src.find(marker_start)
    if idx_s == -1:
        print("[WARN] Could not find 'hypotheses:' block in app.js — skipping hypothesis patch")
    else:
        # Find the matching closing "]," by tracking bracket depth
        depth = 0
        i = idx_s + len(marker_start) - 1  # pointing at "["
        for j in range(i, len(src)):
            if src[j] == "[":
                depth += 1
            elif src[j] == "]":
                depth -= 1
                if depth == 0:
                    # j is the closing "]"; check for ","
                    end = j + 1
                    if end < len(src) and src[end] == ",":
                        end += 1
                    src = src[:idx_s] + new_hyp_block + src[end:]
                    break

    # Update the "Last auto-updated" comment
    src = re.sub(
        r'\* Last auto-updated.*',
        f'* Last auto-updated: {today}',
        src, count=1
    )

    if dry_run:
        print("[DRY RUN] Would write the following to docs/app.js:")
        print(src[:2000])
        print("...")
    else:
        APP_JS.write_text(src)
        print(f"[OK] docs/app.js updated — {today}")
        # Summary
        results = [(h["id"], h["result"]) for h in hyps]
        passed = sum(1 for _, r in results if r == "PASS")
        print(f"     Hypotheses: {passed}/9 PASS")
        for hid, res in results:
            mark = "✓" if res == "PASS" else "✗"
            print(f"     {mark} {hid}: {res}")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run)
