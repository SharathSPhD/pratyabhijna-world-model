"""
Figure 10 — H5b live ablation per-domain breakdown.

Three-panel layout (one per domain) with per-sample paired dots and group
means. Data source: benchmarks/results/h5_live_ablation.json.
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
    COLOR_RED,
)

set_style()

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "benchmarks/results/h5_live_ablation.json"
data = json.loads(DATA_PATH.read_text())

pwm_scores = np.array(data["live_result"]["pwm_scores"], dtype=float)
llm_scores = np.array(data["live_result"]["llm_scores"], dtype=float)
domains = data["live_result"]["domains"]
n_per_domain = data["live_result"]["n_samples_per_domain"]

pretty = {
    "english_pop": "English pop",
    "carnatic": "Carnatic kṛti",
    "kannada_film": "Kannada film",
}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
fig.subplots_adjust(wspace=0.18, top=0.83, bottom=0.20, left=0.085, right=0.985)

for idx, domain in enumerate(domains):
    ax = axes[idx]
    start = idx * n_per_domain
    end = start + n_per_domain
    pwm_d = pwm_scores[start:end]
    llm_d = llm_scores[start:end]
    rng = np.random.default_rng(seed=42 + idx)
    jx0 = rng.uniform(-0.06, 0.06, size=n_per_domain)
    jx1 = rng.uniform(-0.06, 0.06, size=n_per_domain)

    # Paired connecting lines
    for j in range(n_per_domain):
        ax.plot([0 + jx0[j], 1 + jx1[j]], [pwm_d[j], llm_d[j]],
                color=COLOR_LIGHTGREY, linewidth=0.6, zorder=1)

    ax.scatter([0 + j for j in jx0], pwm_d, s=18, color=COLOR_BLUE,
               edgecolor="white", linewidth=0.6, zorder=3, label="PWM-conditioned")
    ax.scatter([1 + j for j in jx1], llm_d, s=18, color=COLOR_ORANGE,
               edgecolor="white", linewidth=0.6, zorder=3, label="Bare 120B LLM")

    # Group means as wide bars
    ax.hlines(pwm_d.mean(), -0.18, 0.18, colors=COLOR_BLUE, linewidth=2.0, zorder=4)
    ax.hlines(llm_d.mean(), 0.82, 1.18, colors=COLOR_ORANGE, linewidth=2.0, zorder=4)

    # Annotate means
    ax.text(-0.08, pwm_d.mean() + 0.035, f"μ={pwm_d.mean():.3f}",
            fontsize=7.5, color=COLOR_BLUE, ha="center")
    ax.text(1.08, llm_d.mean() + 0.035, f"μ={llm_d.mean():.3f}",
            fontsize=7.5, color=COLOR_ORANGE, ha="center")

    breakdown = data["domain_breakdown"][domain]
    pwm_wins = breakdown["pwm_wins"]
    ax.set_title(pretty[domain], fontsize=9, fontweight="bold", pad=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PWM", "LLM"], fontsize=8)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.3, 1.07)
    ax.text(0.5, 0.32, f"PWM wins: {pwm_wins}/{n_per_domain}",
            fontsize=7, ha="center", color=COLOR_GREY,
            transform=ax.transAxes,
            )

axes[0].set_ylabel("Camatkāra text score $R_{\\mathrm{camatk}}^{\\mathrm{text}}$")

# Aggregate stats banner
stats = data["live_result"]
banner = (
    "H5b live ablation: n=30 paired samples (3 domains × 10),  "
    f"Hedges' g = {stats['hedges_g']:.2f},  "
    f"BCa 95% CI [{stats['bca_ci_95'][0]:.2f}, {stats['bca_ci_95'][1]:.2f}],  "
    f"paired permutation p = {stats['p_value_permutation']:.3f}  →  H5b FAIL"
)
fig.suptitle(banner, fontsize=8, color=COLOR_RED, y=0.965, fontweight="bold")

handles = [
    mpatches.Patch(color=COLOR_BLUE, label="PWM-conditioned generation"),
    mpatches.Patch(color=COLOR_ORANGE, label="Unconditioned 120B LLM"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

save(fig, "fig10_h5_live_per_domain")
print("fig10_h5_live_per_domain done.")
