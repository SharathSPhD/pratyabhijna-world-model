"""
Figure 8 — Phase 7: C4 TRIZ Resolution — TTFT and Latency Comparison.

Two-panel figure:
  Left  — Time-to-First-Token (TTFT) across three system states
  Right — Full pipeline latency breakdown (first token + switch + quality ramp)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import set_style, save, COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_GREY, COLOR_LIGHTGREY

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

set_style()

# ── Data from phase7_gate.json + ADR-001/ADR-002 ─────────────────────────────
# All values in seconds.  "~" estimates are mid-range of documented ranges.
states = [
    "Baseline\n(120B cold CoT)",
    "S18 Cascade\n(ADR-001)",
    "S19+S20 Cascade\n+ WM Trace (ADR-002)",
]
ttft = [60.0, 3.5, 3.5]          # time to first visible token (s)
switch_lat = [0.0, 65.0, 5.0]    # additional latency until 120B content begins (s)
colors_ttft = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN]
colors_switch = [COLOR_LIGHTGREY, COLOR_RED, COLOR_ORANGE]

# ── IFR annotations ──────────────────────────────────────────────────────────
ifr_labels = ["IFR 0/4\n(baseline)", "IFR 3/4\n(Sketch B)", "IFR 4/4\n(Sketch A)"]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
fig.subplots_adjust(wspace=0.38)

# ── Panel A: TTFT ─────────────────────────────────────────────────────────────
ax = axes[0]
x = np.arange(len(states))
bars = ax.bar(x, ttft, color=colors_ttft, width=0.55, zorder=3,
              edgecolor="white", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(states, fontsize=7.5)
ax.set_ylabel("Time-to-First-Token (s)")
ax.set_title("A  Time-to-First-Token", fontsize=9, fontweight="bold", pad=6)
ax.set_ylim(0, 72)
for bar, val, lbl in zip(bars, ttft, ifr_labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
            lbl, ha="center", va="bottom", fontsize=6.5, color=COLOR_GREY)

ax.axhline(5.0, color=COLOR_GREEN, linestyle="--", linewidth=1.0, zorder=2)
ax.text(2.35, 5.8, "target <5 s", fontsize=6.5, color=COLOR_GREEN, va="bottom", ha="right")

# ── Panel B: Full latency stack (stacked bar: TTFT + switch gap) ──────────────
ax2 = axes[1]
# Three system states × two stacked components
ttft_arr = np.array(ttft)
switch_arr = np.array(switch_lat)

b1 = ax2.bar(x, ttft_arr, color=colors_ttft, width=0.55, zorder=3,
             edgecolor="white", linewidth=0.8, label="TTFT (fast model)")
b2 = ax2.bar(x, switch_arr, bottom=ttft_arr, color=colors_switch, width=0.55,
             zorder=3, edgecolor="white", linewidth=0.8, label="Switch gap (→ 120B)")

ax2.set_xticks(x)
ax2.set_xticklabels(states, fontsize=7.5)
ax2.set_ylabel("Latency (s)")
ax2.set_title("B  Full Pipeline Latency Stack", fontsize=9, fontweight="bold", pad=6)
ax2.set_ylim(0, 80)

totals = ttft_arr + switch_arr
total_labels = ["60 s", "68.5 s", "8.5 s"]
for i, (top, lbl) in enumerate(zip(totals, total_labels)):
    ax2.text(i, top + 1.5, lbl, ha="center", va="bottom", fontsize=8,
             fontweight="bold",
             color=COLOR_GREEN if i == 2 else COLOR_GREY)

# Improvement annotation arrow on panel B
ax2.annotate("", xy=(2.27, 8.5), xytext=(2.27, 68.5),
             arrowprops=dict(arrowstyle="<->", color=COLOR_GREEN, lw=1.5))
ax2.text(2.38, 38, "8×\nfaster", ha="left", va="center", fontsize=7.5,
         color=COLOR_GREEN, fontweight="bold")

legend_patches = [
    mpatches.Patch(color=COLOR_GREEN, label="TTFT — fast model (nemotron-mini:4b)"),
    mpatches.Patch(color=COLOR_ORANGE, label="Switch gap — 120B cold CoT"),
    mpatches.Patch(color=COLOR_RED, label="Switch gap — 120B WM-guided (ADR-002)"),
]
ax2.legend(handles=legend_patches, fontsize=6.5, loc="upper right",
           framealpha=0.9, edgecolor=COLOR_LIGHTGREY)

fig.suptitle(
    "Phase 7 — C4 TRIZ Resolution: TTFT/Latency Before vs. After Cascade Streaming + WM Reasoning Trace",
    fontsize=8, y=1.01, color=COLOR_GREY,
)

save(fig, "fig8_phase7_ttft")
print("fig8_phase7_ttft done.")
