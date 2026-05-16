"""Figure 1: PWM system overview schematic.

Six components in a flow: Trika RSSM -> EFE Actor -> Camatkara Reward
-> Hopfield CittaStore -> Sleep Consolidation -> Vimarsa Bridge.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _style import (
    set_style, save,
    COLOR_BLUE, COLOR_LIGHTBLUE, COLOR_GREEN, COLOR_LIGHTGREEN, COLOR_GREY,
)


def draw_box(ax, x, y, w, h, sanskrit, english, fc, ec, sub=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    ax.text(cx, cy + 0.18, sanskrit, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1A1A1A")
    ax.text(cx, cy - 0.05, f"({english})", ha="center", va="center",
            fontsize=8, color="#2A2A2A", style="italic")
    if sub:
        ax.text(cx, cy - 0.28, sub, ha="center", va="center",
                fontsize=7, color="#444")


def arrow(ax, p0, p1, color=COLOR_GREY, curve=0.0):
    a = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.1, color=color,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(a)


def main():
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    bw, bh = 1.65, 1.0

    # Row 1: top flow (3 boxes)
    boxes_top = [
        (0.3, 4.4, "Trika RSSM",       "3-level world model",      COLOR_LIGHTBLUE, COLOR_BLUE,  "Para / Pasyanti / Apara"),
        (3.4, 4.4, "EFE Actor",        "active inference",         COLOR_LIGHTBLUE, COLOR_BLUE,  "epistemic + pragmatic"),
        (6.5, 4.4, "Camatkara Reward", "aesthetic surprise",       COLOR_LIGHTGREEN, COLOR_GREEN, "DF + DI_Hop + Emp."),
    ]
    # Row 2: bottom flow (3 boxes)
    boxes_bot = [
        (6.5, 1.6, "Hopfield CittaStore", "episodic + semantic",  COLOR_LIGHTGREEN, COLOR_GREEN, "ālayavijñāna"),
        (3.4, 1.6, "Sleep Consolidation", "NREM / REM replay",    COLOR_LIGHTBLUE,  COLOR_BLUE,  "anti-forgetting"),
        (0.3, 1.6, "Vimarśa Bridge",      "WM x LLM attention",   COLOR_LIGHTGREEN, COLOR_GREEN, "narrative reflection"),
    ]

    for x, y, sk, en, fc, ec, sub in boxes_top + boxes_bot:
        draw_box(ax, x, y, 2.6, bh + 0.2, sk, en, fc, ec, sub)

    # Forward arrows along top row
    arrow(ax, (2.9, 4.95), (3.4, 4.95))
    arrow(ax, (6.0, 4.95), (6.5, 4.95))
    # Down arrow (right side)
    arrow(ax, (7.8, 4.4), (7.8, 2.8))
    # Bottom row backward (right to left)
    arrow(ax, (6.5, 2.2), (6.0, 2.2))
    arrow(ax, (3.4, 2.2), (2.9, 2.2))
    # Up arrow (left side, closes the loop -> Trika)
    arrow(ax, (1.6, 2.8), (1.6, 4.4))

    # Title
    ax.text(5.0, 5.75,
            "Pratyabhijñā World Model — Pañcakṛtya Cascade",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 5.45,
            "(sṛṣṭi → sthiti → saṃhāra → tirodhāna → anugraha)",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # Caption strip
    ax.text(5.0, 0.55,
            "Sanskrit primitives realised as differentiable modules; data flow forms a closed śakti loop.",
            ha="center", va="center", fontsize=8, color="#333")

    save(fig, "fig_system_overview")


if __name__ == "__main__":
    main()
