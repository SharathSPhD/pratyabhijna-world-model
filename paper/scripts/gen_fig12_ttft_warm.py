"""
Figure 12 — TTFT warm-model validation vs cold-start outliers.

Two panels: (left) per-prompt warm TTFT for Cond A (ADR-001 baseline) and
Cond B (with WM trace, ADR-002), with Ollama cold-start outliers annotated
but not aggregated; (right) the summary warm-mean comparison and the
percentage reduction.

Data: benchmarks/results/ttft_live_validation.json (raw_measurements +
warm_model_measurements).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (
    set_style,
    save,
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_GREY,
    COLOR_LIGHTGREY,
)

set_style()

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "benchmarks/results/ttft_live_validation.json"
data = json.loads(DATA_PATH.read_text())

raw = data["raw_measurements"]
warm = data["warm_model_measurements"]

prompts = sorted(raw.keys())
n = len(prompts)
cond_a_ttft = np.array([raw[p]["cond_A_ttft"] for p in prompts])
cond_b_ttft = np.array([raw[p]["cond_B_ttft"] for p in prompts])
cold_mask_a = cond_a_ttft > 60
cold_mask_b = cond_b_ttft > 60

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                         gridspec_kw={"width_ratios": [1.4, 1.0]})
fig.subplots_adjust(left=0.085, right=0.97, top=0.83, bottom=0.20, wspace=0.32)

# ── Panel A: per-prompt TTFT with cold outliers annotated ─────────────────
ax = axes[0]
x = np.arange(n)
width = 0.36
ax.bar(x - width / 2, np.where(cold_mask_a, 5.0, cond_a_ttft),
       width=width, color=COLOR_BLUE, edgecolor="white",
       label="Cond A: cascade only (ADR-001)")
ax.bar(x + width / 2, np.where(cold_mask_b, 5.0, cond_b_ttft),
       width=width, color=COLOR_GREEN, edgecolor="white",
       label="Cond B: cascade + WM trace (ADR-002)")

# Mark cold-start outliers with a star above the (capped) bar
for i, (a_val, b_val) in enumerate(zip(cond_a_ttft, cond_b_ttft)):
    if cold_mask_a[i]:
        ax.scatter(i - width / 2, 5.3, marker="*", s=70, color=COLOR_RED, zorder=4)
        ax.text(i - width / 2, 5.7, f"{a_val:.0f}s\ncold", ha="center", va="bottom",
                fontsize=6.3, color=COLOR_RED)
    if cold_mask_b[i]:
        ax.scatter(i + width / 2, 5.3, marker="*", s=70, color=COLOR_RED, zorder=4)
        ax.text(i + width / 2, 5.7, f"{b_val:.0f}s\ncold", ha="center", va="bottom",
                fontsize=6.3, color=COLOR_RED)

ax.axhline(5.0, color=COLOR_GREY, linestyle="--", linewidth=0.8, zorder=2)
ax.text(0.02, 5.15, "ADR-001 target < 5 s", fontsize=7, color=COLOR_GREY,
        transform=ax.get_yaxis_transform(), va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([f"P{i+1}" for i in range(n)], fontsize=8)
ax.set_ylim(0, 8.0)
ax.set_ylabel("Time-to-first-token (s)")
ax.set_title("A  Per-prompt warm TTFT (cold-start outliers asterisked)",
             fontsize=9, fontweight="bold", pad=6)
ax.legend(fontsize=7, loc="upper right", frameon=False)

# ── Panel B: warm-mean comparison ─────────────────────────────────────────
ax2 = axes[1]
means = np.array([warm["cond_A_warm_ttft_mean_s"], warm["cond_B_warm_ttft_mean_s"]])
labels = ["Cond A\n(cascade only)", "Cond B\n(cascade + WM trace)"]
bars = ax2.bar([0, 1], means, color=[COLOR_BLUE, COLOR_GREEN],
               edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
ax2.axhline(5.0, color=COLOR_GREY, linestyle="--", linewidth=0.8, zorder=2)
ax2.text(1.45, 5.0, "<5 s target", fontsize=7, color=COLOR_GREY,
         va="center", ha="left")
for x_, m in zip([0, 1], means):
    ax2.text(x_, m + 0.10, f"{m:.2f}s", ha="center", va="bottom",
             fontsize=8, fontweight="bold")

ax2.set_xticks([0, 1])
ax2.set_xticklabels(labels, fontsize=7.5)
ax2.set_ylabel("Warm-mean TTFT (s)")
ax2.set_ylim(0, 6.0)
reduction = warm["adr002_warm_reduction_pct"]
ax2.set_title(
    f"B  Warm means: ADR-002 reduces TTFT by {reduction:.1f}\\%",
    fontsize=9, fontweight="bold", pad=6,
)

handles = [
    mpatches.Patch(color=COLOR_BLUE, label="Cond A — ADR-001 cascade"),
    mpatches.Patch(color=COLOR_GREEN, label="Cond B — ADR-001 + ADR-002 WM trace"),
    mpatches.Patch(color=COLOR_RED, label="Cold-start outlier (Ollama 300/420 s timeout)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, 0.005))

save(fig, "fig12_ttft_warm_cold")
print("fig12_ttft_warm_cold done.")
