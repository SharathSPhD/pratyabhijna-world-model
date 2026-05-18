"""
Figure 11 — A6 ablation: 1-level (Aparā-only) vs 3-level Trika world model.

Bar chart over three 1-level seeds (51, 52, 53) with the 3-level Phase 6
reference. Data source: benchmarks/results/ablation_a6_1level_wm.json.
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
    COLOR_GREY,
    COLOR_LIGHTGREY,
)

set_style()

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "benchmarks/results/ablation_a6_1level_wm.json"
data = json.loads(DATA_PATH.read_text())

seeds = [s["seed"] for s in data["seed_results"]]
ratios_1l = [s["vfe_ratio_vs_phase3"] for s in data["seed_results"]]
steps = [s["steps_completed"] for s in data["seed_results"]]
ratio_3l = data["stats"]["vfe_ratio_3level_ref"]
advantage = data["stats"]["advantage_factor_3level_over_1level"]

fig, ax = plt.subplots(figsize=(5.0, 3.0))
fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.18)

x_positions = np.arange(len(seeds) + 1)
bar_heights = ratios_1l + [ratio_3l]
bar_labels = [f"1-level\nseed {s}" for s in seeds] + ["3-level Trika\n(Phase 6 ref)"]
bar_colors = [COLOR_ORANGE] * len(seeds) + [COLOR_GREEN]
bars = ax.bar(x_positions, bar_heights, color=bar_colors, width=0.6,
              edgecolor="white", linewidth=0.8, zorder=3)

for x, h, st in zip(x_positions[: len(seeds)], ratios_1l, steps):
    ax.text(x, h + 1.5e-4, f"{h*1000:.2f}×10⁻³", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=COLOR_GREY)
    ax.text(x, -0.00025, f"{st//1000}K steps", ha="center", va="top",
            fontsize=6.5, color=COLOR_GREY)

ax.text(x_positions[-1], ratio_3l + 1.5e-4, f"{ratio_3l*1000:.3f}×10⁻³",
        ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        color=COLOR_GREEN)

ax.set_xticks(x_positions)
ax.set_xticklabels(bar_labels, fontsize=7.5)
ax.set_ylabel("VFE ratio vs. Phase 3 baseline")
ax.set_ylim(0, max(ratios_1l) * 1.32)
ax.set_title(
    f"H7 ablation A6: 3-level Trika beats 1-level Aparā-only by ×{advantage:.1f}",
    fontsize=9, fontweight="bold", pad=8,
)

# Advantage annotation arrow
ax.annotate(
    "", xy=(len(seeds), ratio_3l), xytext=(len(seeds) - 0.4, np.mean(ratios_1l)),
    arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.4),
)
ax.text(len(seeds) - 0.85, np.mean(ratios_1l) * 0.82,
        f"{advantage:.1f}×\nadvantage",
        ha="center", va="top", fontsize=8, color=COLOR_GREEN,
        fontweight="bold")

# Threshold line: H7 passes if 1-level ratio > 1.18× 3-level
threshold = ratio_3l * 1.18
ax.axhline(threshold, linestyle="--", color=COLOR_GREY, linewidth=0.8, zorder=2)
ax.text(0.02, threshold * 1.05, f"H7 pass threshold (1.18× = {threshold*1000:.4f}×10⁻³)",
        fontsize=6.5, color=COLOR_GREY, transform=ax.get_yaxis_transform(),
        va="bottom")

handles = [
    mpatches.Patch(color=COLOR_ORANGE, label="1-level (Aparā only)"),
    mpatches.Patch(color=COLOR_GREEN, label="3-level Trika (Aparā + Parāparā + Parā)"),
]
ax.legend(handles=handles, fontsize=7.5, loc="upper right",
          edgecolor=COLOR_LIGHTGREY, framealpha=0.95)

save(fig, "fig11_a6_ablation")
print("fig11_a6_ablation done.")
