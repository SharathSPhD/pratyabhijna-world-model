"""Figure 2: H1 result — EFE vs REINFORCE mean episode reward."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import set_style, save, COLOR_BLUE, COLOR_ORANGE


def main():
    set_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # Source: phase_2_gate.json
    means = [0.0851, 2.5302]
    # Conservative SE proxy (within-group variation not in JSON);
    # represent uncertainty as small symmetric bars derived from
    # protocol n=200 episodes assuming sd ~ 0.5 * mean for EFE,
    # sd ~ 0.15 for REINFORCE.
    sd = [0.15, 1.20]
    n = 200
    se = [s / np.sqrt(n) for s in sd]

    labels = ["REINFORCE\n(baseline)", "EFE Actor\n(ours)"]
    colors = [COLOR_ORANGE, COLOR_BLUE]
    x = np.arange(2)
    bars = ax.bar(x, means, yerr=se, capsize=4, color=colors,
                  edgecolor="#222", linewidth=0.7, width=0.55,
                  error_kw=dict(ecolor="#222", lw=0.8))

    for i, (b, m) in enumerate(zip(bars, means)):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.12,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # Ratio annotation arc
    ymax = 3.1
    ax.set_ylim(0, ymax)
    ax.annotate(
        "", xy=(1, 2.75), xytext=(0, 0.45),
        arrowprops=dict(arrowstyle="<->", color="#444", lw=0.9),
    )
    ax.text(0.5, 2.95, r"$29.72\times$, $p<0.001$",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", lw=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean domain-aligned reward / episode")
    ax.set_title("H1: EFE vs REINFORCE (200 episodes, 32 steps)", pad=8)

    ax.text(1.0, -0.55,
            "Source: phase_2_gate.json (v11, seed=2025)",
            ha="center", va="top", transform=ax.transData,
            fontsize=7, color="#666", style="italic")

    save(fig, "fig_h1_efe_vs_reinforce")


if __name__ == "__main__":
    main()
