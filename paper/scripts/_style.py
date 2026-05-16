"""Shared matplotlib style for PWM paper figures."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Muted academic palette
COLOR_BLUE = "#3B6FA5"
COLOR_LIGHTBLUE = "#7FA9D1"
COLOR_GREEN = "#4A8C5F"
COLOR_LIGHTGREEN = "#8FBFA0"
COLOR_ORANGE = "#D08A4A"
COLOR_RED = "#B85450"
COLOR_GREY = "#4A4A4A"
COLOR_LIGHTGREY = "#CFCFCF"
COLOR_BG = "#FFFFFF"

PALETTE_PHASES = [
    "#5B8DBE", "#6FA8B9", "#7FB89A", "#A0C481",
    "#D6B26F", "#D08A6A", "#B86A6A",
]


def set_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str):
    """Save figure as both PDF (vector) and PNG (300dpi)."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")
