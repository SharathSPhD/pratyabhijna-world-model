"""
Figure 9 — WMReasoningTrace pipeline: PancakrtyaLoopV2 Acts 1-6 with ADR-002 integration.

A horizontal flow diagram showing the 6 acts of the pañcakṛtya loop with the
think-block prefill injection between Acts 5 (Jñāna) and 6 (Kriyā), and a
second row showing the 120B model receiving the condensed WM state vs. cold CoT.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import set_style, save, COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_GREY

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

set_style()

fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4))
fig.subplots_adjust(hspace=0.55)

# ── Shared layout ─────────────────────────────────────────────────────────────
ACT_W, ACT_H = 0.12, 0.38
X_STARTS = [0.02, 0.17, 0.32, 0.47, 0.62, 0.77]
ARROW_Y_MID = 0.55
ACT_Y = ARROW_Y_MID - ACT_H / 2

acts = [
    ("Cit\n(Act 1)", "WM\nobserve_step", COLOR_BLUE),
    ("Ānanda\n(Act 2)", "EFE\nactor", COLOR_BLUE),
    ("Icchā\n(Act 3)", "Hopfield\nrecall", COLOR_BLUE),
    ("Apohana\n(Act 4)", "Entropy\ngate", COLOR_BLUE),
    ("Jñāna\n(Act 5)", "VimarsaBridge\nlogits", COLOR_BLUE),
    ("Kriyā\n(Act 6)", "LLM\nstream()", COLOR_GREEN),
]

THINK_X = 0.72   # between Act 5 and Act 6
THINK_W = 0.04
THINK_H = 0.52

def draw_loop_row(ax, row_label, show_think_block, think_color, baseline_label):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(row_label, fontsize=8.5, fontweight="bold", loc="left", pad=4)

    for i, (act_name, sub_name, col) in enumerate(acts):
        x0 = X_STARTS[i]
        box = FancyBboxPatch((x0, ACT_Y), ACT_W, ACT_H,
                             boxstyle="round,pad=0.01",
                             facecolor=col if not (show_think_block and i == 5) else COLOR_GREEN,
                             edgecolor="white", linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(x0 + ACT_W / 2, ACT_Y + ACT_H * 0.62, act_name,
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", zorder=4)
        ax.text(x0 + ACT_W / 2, ACT_Y + ACT_H * 0.22, sub_name,
                ha="center", va="center", fontsize=5.8, color="white", zorder=4)

        if i < 5:
            next_x = X_STARTS[i + 1]
            ax.annotate("",
                        xy=(next_x, ARROW_Y_MID),
                        xytext=(x0 + ACT_W, ARROW_Y_MID),
                        arrowprops=dict(arrowstyle="->", color=COLOR_GREY,
                                        lw=1.0, connectionstyle="arc3,rad=0.0"),
                        zorder=2)

    # WM state label on Act 1
    ax.text(X_STARTS[0] + ACT_W / 2, ACT_Y - 0.08,
            "h_t, z_t\n(WM state)", ha="center", va="top", fontsize=6,
            color=COLOR_BLUE, style="italic")

    if show_think_block:
        # think-block box between Acts 5 and 6
        tb_x = X_STARTS[4] + ACT_W + 0.005
        tb_box = FancyBboxPatch((tb_x, ACT_Y - 0.05), THINK_W, ACT_H + 0.10,
                                boxstyle="round,pad=0.01",
                                facecolor=think_color, edgecolor=COLOR_ORANGE,
                                linewidth=1.5, zorder=5, alpha=0.92)
        ax.add_patch(tb_box)
        ax.text(tb_x + THINK_W / 2, ACT_Y + ACT_H / 2 + 0.04,
                "<think>", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=6)
        ax.text(tb_x + THINK_W / 2, ACT_Y + ACT_H / 2 - 0.04,
                "WM\nvimarśa", ha="center", va="center",
                fontsize=5.5, color="white", style="italic", zorder=6)
        ax.text(tb_x + THINK_W / 2, ACT_Y + ACT_H / 2 - 0.13,
                "</think>", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=6)
        ax.text(tb_x + THINK_W / 2, ACT_Y - 0.14,
                "~3 s\n(was ~60 s)", ha="center", va="top",
                fontsize=6, color=COLOR_ORANGE)

        # arrow from think-block to Act 6
        ax.annotate("",
                    xy=(X_STARTS[5], ARROW_Y_MID),
                    xytext=(tb_x + THINK_W, ARROW_Y_MID),
                    arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE,
                                    lw=1.3, connectionstyle="arc3,rad=0.0"),
                    zorder=4)
        ax.text(tb_x + THINK_W + 0.005, ARROW_Y_MID + 0.07,
                "think_prefill\n(assistant role)",
                ha="left", va="bottom", fontsize=5.8, color=COLOR_ORANGE)
    else:
        # plain arrow from Act 5 to Act 6
        ax.annotate("",
                    xy=(X_STARTS[5], ARROW_Y_MID),
                    xytext=(X_STARTS[4] + ACT_W, ARROW_Y_MID),
                    arrowprops=dict(arrowstyle="->", color=COLOR_GREY,
                                    lw=1.0, connectionstyle="arc3,rad=0.0"),
                    zorder=2)
        ax.text((X_STARTS[4] + ACT_W + X_STARTS[5]) / 2, ARROW_Y_MID + 0.08,
                "no prefill\n(~60 s CoT)", ha="center", va="bottom",
                fontsize=6, color=COLOR_RED)

    # Output label
    out_x = X_STARTS[5] + ACT_W + 0.005
    ax.annotate("",
                xy=(out_x + 0.06, ARROW_Y_MID),
                xytext=(out_x, ARROW_Y_MID),
                arrowprops=dict(arrowstyle="->", color=COLOR_GREEN,
                                lw=1.2, connectionstyle="arc3,rad=0.0"),
                zorder=2)
    ax.text(out_x + 0.07, ARROW_Y_MID,
            baseline_label, ha="left", va="center",
            fontsize=7, color=COLOR_GREEN, fontweight="bold")

    # Contract annotations
    ax.text(0.0, 0.08, "Contract 1: h_t/z_t shared across all 6 acts",
            ha="left", va="bottom", fontsize=6, color=COLOR_GREY, style="italic")
    ax.text(0.0, 0.0, "Contract 2: think_prefill internal only (not in SSE events)",
            ha="left", va="bottom", fontsize=6, color=COLOR_GREY, style="italic")


draw_loop_row(axes[0],
              "Baseline — 120B cold chain-of-thought (TTFT ~60s, switch ~60s)",
              show_think_block=False,
              think_color=COLOR_ORANGE,
              baseline_label="tokens\n(~60 s wait)")

draw_loop_row(axes[1],
              "ADR-002 (S19+S20) — WM Reasoning Trace prefill (TTFT <5s, switch ~5s)",
              show_think_block=True,
              think_color=COLOR_ORANGE,
              baseline_label="tokens\n(<5 s TTFT)")

fig.suptitle(
    "Figure 9 — WMReasoningTrace: Pañcakṛtya loop with think-block prefill injection (ADR-002)",
    fontsize=8, color=COLOR_GREY,
)

save(fig, "fig9_wm_reasoning_trace")
print("fig9_wm_reasoning_trace done.")
