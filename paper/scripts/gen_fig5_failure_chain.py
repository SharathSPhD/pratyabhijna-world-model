"""Figure 5: 11-layer H1 failure chain cascade diagram."""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import set_style, save, COLOR_RED, COLOR_GREEN, COLOR_GREY

LAYERS = [
    ("L1", "curr_vfe hardcoded 0.0", "ΔF = 0 always"),
    ("L2", "Zero-action replay",      "W_a blind to actions"),
    ("L3", "Passive corpus",           "p(o|a) = p(o): no coupling"),
    ("L4", "free_bits = 1.0 ceiling", "Prior entropy degenerate"),
    ("L5", "Encoder+prior+W_z collapse","Warm-start instability"),
    ("L6", "GRU posterior bypass",    "z_t ignored by decoder"),
    ("L7", "Gradient starvation",     "Encoder → 0 by step 100K"),
    ("L8", "‖h_t‖₂ action-invariant", "Actor stays uniform"),
    ("L9", "Cold-start deadlock",     "P(commit|H=13) ≈ 0%"),
    ("L10","Fresh-sample PG",         "log p(ã) ⊥ A_t: zero grad"),
    ("L11","Gate metric paradox",     "p95 rewards variance ≠ commit"),
]
FIX = "v11 Fix: IDL + domain-selective\ncorpus + decoder-z-only + WM-freeze\n→ H1 PASS (29.72×)"


def main():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.5),
                              gridspec_kw={"width_ratios": [1, 1]})

    for col, (ax, layers) in enumerate(zip(axes, [LAYERS[:6], LAYERS[6:]])):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, len(layers) * 1.4 + 0.4)
        ax.axis("off")

        for i, (lid, title, detail) in enumerate(layers):
            y = (len(layers) - i - 1) * 1.4 + 0.2
            # Box
            box = mpatches.FancyBboxPatch(
                (0.15, y), 3.7, 1.0,
                boxstyle="round,pad=0.05",
                linewidth=1.2,
                edgecolor="#993333",
                facecolor="#FFEEEE",
            )
            ax.add_patch(box)
            global_num = i + 1 + col * 6
            ax.text(0.45, y + 0.65, f"Layer {global_num}: {title}",
                    fontsize=8.5, fontweight="bold", color="#8B0000", va="center")
            ax.text(0.45, y + 0.28, detail,
                    fontsize=7.5, color="#444", va="center", style="italic")
            # Arrow down (not on last of each column)
            if i < len(layers) - 1:
                ax.annotate("", xy=(2.0, y - 0.02), xytext=(2.0, y),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR_GREY,
                                            lw=1.0))

    # Fix box spanning bottom
    fig.text(0.5, 0.02, FIX, ha="center", va="bottom", fontsize=9,
             fontweight="bold", color="#1A5C1A",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEFFEE",
                       edgecolor="#2E7D32", linewidth=1.5))

    axes[0].set_title("Layers 1–6: Environment & Model", fontsize=9,
                       fontweight="bold", pad=4)
    axes[1].set_title("Layers 7–11: Actor & Gate", fontsize=9,
                       fontweight="bold", pad=4)
    fig.suptitle("Eleven-Layer H1 Failure Chain", fontsize=11,
                  fontweight="bold", y=0.98)

    save(fig, "fig_failure_chain")


if __name__ == "__main__":
    main()
