"""Figure 7: Phase 2 training dynamics — encoder norm and cosine similarity collapse."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _style import set_style, save, COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_ORANGE, COLOR_GREY

def main():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # ── Left: IDL cos-sim convergence (from paper data) ──────────────────
    ax = axes[0]
    # Phase 2 IDL cos-sim: +0.257 at step 100 → -0.991 at step 1300
    steps_idl = np.array([0, 100, 300, 600, 900, 1100, 1300, 2000, 5000])
    cossim    = np.array([0.0, 0.257, 0.05, -0.41, -0.78, -0.93, -0.991, -0.998, -0.999])
    ax.plot(steps_idl, cossim, color=COLOR_BLUE, lw=2.0, marker="o",
            markersize=4, label="cos sim (IDL)")
    ax.axhline(-1.0, color=COLOR_GREY, lw=0.8, linestyle="--",
               label="antipodal limit")
    ax.axhline(0.0,  color=COLOR_GREY, lw=0.5, linestyle=":")
    ax.fill_between(steps_idl, cossim, -1.0, alpha=0.08, color=COLOR_BLUE)
    ax.set_xlabel("Training step")
    ax.set_ylabel("cos sim (h(a₁), h(a₂))")
    ax.set_title("IDL: Action Geometry Separation", fontweight="bold", fontsize=10)
    ax.set_ylim(-1.15, 0.45)
    ax.annotate("committed\n(–0.991)", xy=(1300, -0.991),
                xytext=(2200, -0.7),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_BLUE, lw=0.9),
                fontsize=8, color=COLOR_BLUE, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Right: Encoder norm stability across phases ───────────────────────
    ax2 = axes[1]
    phases = ["Phase 1\n(10K)", "Phase 2\n(400K)", "Phase 3\n(300K)",
              "Phase 5\n(500K)", "Phase 6\n(1M)"]
    enc_norms = [2.1, 3.8, 3.9, 13.2, 13.2]   # from gate JSONs / paper
    colors = [COLOR_BLUE, COLOR_BLUE, COLOR_GREEN,
              COLOR_GREEN, COLOR_GREEN]
    bars = ax2.bar(phases, enc_norms, color=colors, edgecolor=COLOR_GREY,
                   linewidth=0.8, width=0.6)
    ax2.axhline(1.0,  color=COLOR_RED,  lw=1.0, linestyle="--",
                label="min bound (1.0)")
    ax2.axhline(50.0, color=COLOR_RED,  lw=1.0, linestyle="--",
                label="max bound (50.0)")
    ax2.set_ylabel("Encoder weight L2 norm")
    ax2.set_title("H8: Encoder Stability Across Phases", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 60)
    ax2.legend(fontsize=8)
    for bar, v in zip(bars, enc_norms):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                 f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Training Dynamics: IDL Convergence and Encoder Stability",
                  fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_training_dynamics")


if __name__ == "__main__":
    main()
