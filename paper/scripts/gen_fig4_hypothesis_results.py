"""Figure 4: All nine pre-registered hypotheses (3x3 grid)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import set_style, save, COLOR_GREEN, COLOR_LIGHTGREEN, COLOR_GREY


HYP = [
    # (id, title, measured, threshold, direction, unit/desc)
    ("H1", "EFE vs REINFORCE reward",     29.72,  2.00, ">=", "ratio"),
    ("H2", "Hopfield completion",          1.307, 1.10, ">=", "ratio"),
    ("H3", "Sleep forgetting",             0.00,  0.80, "<=", "rel. drop"),
    ("H4", "Vimarśa meaningful rate",      1.00,  0.70, ">=", "fraction"),
    ("H5", "Camatkāra reward (vs Φ2)",     2.14,  2.00, ">=", "ratio"),
    ("H6", "Reward entropy (diversity)",   1.897, 0.50, ">=", "nats"),
    ("H7", "Long-horizon VFE ratio",       0.00,  0.85, "<=", "ratio"),
    ("H8", "Encoder norm in [1,50]",       13.20, 1.00, "in",  "L2 norm"),
    ("H9", "Action entropy",               0.582, 0.50, ">=", "nats"),
]


def draw_hyp(ax, hid, title, measured, threshold, direction, unit):
    # Mini bar showing measured vs threshold
    if direction == "in":
        # Range [1, 50] for H8
        lo, hi = 1.0, 50.0
        ax.barh([0], [measured], color=COLOR_LIGHTGREEN,
                edgecolor=COLOR_GREEN, linewidth=0.7)
        ax.axvline(lo, color=COLOR_GREY, linestyle="--", lw=0.7)
        ax.axvline(hi, color=COLOR_GREY, linestyle="--", lw=0.7)
        ax.set_xlim(0, hi * 1.1)
    else:
        if direction == "<=":
            # For low-is-good, plot inverted: show measured & threshold
            ax.barh([0], [max(measured, 1e-3)], color=COLOR_LIGHTGREEN,
                    edgecolor=COLOR_GREEN, linewidth=0.7)
            ax.axvline(threshold, color=COLOR_GREY, linestyle="--", lw=0.7)
            xmax = max(threshold * 1.4, max(measured, 1e-3) * 1.4, 1.0)
            ax.set_xlim(0, xmax)
        else:  # >=
            ax.barh([0], [measured], color=COLOR_LIGHTGREEN,
                    edgecolor=COLOR_GREEN, linewidth=0.7)
            ax.axvline(threshold, color=COLOR_GREY, linestyle="--", lw=0.7)
            xmax = max(measured * 1.15, threshold * 1.4)
            ax.set_xlim(0, xmax)

    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", visible=False)

    # Title
    ax.set_title(f"{hid} — PASS", fontsize=9, fontweight="bold",
                 color=COLOR_GREEN, loc="left", pad=3)
    ax.text(0.0, 1.30, title, transform=ax.transAxes,
            fontsize=7.8, color="#222")

    # Measured value
    sym = "≤" if direction == "<=" else ("∈" if direction == "in" else "≥")
    thr_str = "[1, 50]" if direction == "in" else f"{threshold:g}"
    ax.text(1.0, -0.42, f"meas={measured:.3g}  {sym} {thr_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#333")


def main():
    set_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4))
    fig.subplots_adjust(hspace=0.95, wspace=0.35,
                        left=0.06, right=0.98, top=0.91, bottom=0.07)

    for ax, hyp in zip(axes.ravel(), HYP):
        draw_hyp(ax, *hyp)

    fig.suptitle("All nine pre-registered hypotheses PASS",
                 fontsize=11.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015,
             "Source: phase_{2..6}_gate JSONs. Dashed line = threshold; green bar = measured.",
             ha="center", fontsize=7, color="#666", style="italic")

    save(fig, "fig_hypothesis_results")


if __name__ == "__main__":
    main()
