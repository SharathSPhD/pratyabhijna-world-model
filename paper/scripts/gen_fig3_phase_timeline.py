"""Figure 3: Training phase timeline with gate metrics."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import set_style, save, PALETTE_PHASES


def main():
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    # Phases on a step axis 0 -> 1.0M
    phases = [
        ("Phase 0\nFoundation",  0,      50_000,   "Corpus 4.6M tok",            PALETTE_PHASES[0]),
        ("Phase 1\nAparā RSSM",  50_000, 200_000,  "WM/LSTM 1.47<2.0",           PALETTE_PHASES[1]),
        ("Phase 2\nEFE Actor",   200_000,400_000,  "H1: 29.72×",                 PALETTE_PHASES[2]),
        ("Phase 3\nHopfield",    400_000,550_000,  "H2: 1.307×",                 PALETTE_PHASES[3]),
        ("Phase 4\nSleep",       550_000,700_000,  "H3: ≈0 forgetting",          PALETTE_PHASES[4]),
        ("Phase 5\nVimarśa+LLM", 700_000,850_000,  "H4: 1.00  H5: 2.14×",        PALETTE_PHASES[5]),
        ("Phase 6\nPañcakṛtya",  850_000,1_000_000,"H6–H9 PASS",                PALETTE_PHASES[6]),
    ]

    bar_y = 0.55
    bar_h = 0.35
    for name, s, e, metric, color in phases:
        ax.add_patch(Rectangle((s, bar_y), e - s, bar_h,
                               facecolor=color, edgecolor="#222",
                               linewidth=0.6, alpha=0.9))
        cx = (s + e) / 2
        # Phase name above
        ax.text(cx, bar_y + bar_h + 0.08, name,
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        # Metric below
        ax.text(cx, bar_y - 0.08, metric,
                ha="center", va="top", fontsize=7.2, color="#1A1A1A")

    # Phase-gate tick marks
    for _, _, e, _, _ in phases[:-1]:
        ax.plot([e, e], [bar_y - 0.04, bar_y + bar_h + 0.04],
                color="#222", lw=0.6)
        ax.plot(e, bar_y + bar_h + 0.04, marker="v",
                color="#222", markersize=4)

    # X axis: steps
    ax.set_xlim(-25_000, 1_025_000)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    xticks = [0, 200_000, 400_000, 600_000, 800_000, 1_000_000]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "200k", "400k", "600k", "800k", "1.0M"])
    ax.set_xlabel("Training steps")
    ax.grid(axis="y", visible=False)

    ax.set_title("PWM Training Timeline — six phases, nine hypotheses all PASS", pad=10)
    ax.text(500_000, -0.32,
            "Sources: phase_{0..6}_gate JSONs (benchmarks/results/)",
            ha="center", va="top", fontsize=7, color="#666", style="italic")

    save(fig, "fig_phase_timeline")


if __name__ == "__main__":
    main()
