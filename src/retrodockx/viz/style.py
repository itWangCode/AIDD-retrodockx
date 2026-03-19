"""
src/retrodockx/viz/style.py
============================
Global visual style: Macaroon color palette + Times New Roman.
All figures in this project use this module for consistent styling.

Macaroon palette: soft pastels with clean contrast,
suitable for publication and presentation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ═══════════════════════════════════════════════════════
#  MACAROON COLOR PALETTE
# ═══════════════════════════════════════════════════════

MACAROON = {
    # Core pastels
    "rose":         "#F2A7BB",   # Soft rose pink
    "lavender":     "#C3B1E1",   # Lavender purple
    "mint":         "#A8D8B9",   # Mint green
    "sky":          "#A8D0E6",   # Sky blue
    "peach":        "#FFCBA4",   # Peach orange
    "lemon":        "#FFF0A0",   # Lemon yellow
    "lilac":        "#D4AAFF",   # Lilac
    "pistachio":    "#B5D99C",   # Pistachio green
    "blush":        "#FADADD",   # Blush pink
    "powder_blue":  "#B0C4DE",   # Powder blue

    # Darker accents (for text / borders on macaroon backgrounds)
    "rose_dark":      "#D4557A",
    "lavender_dark":  "#7B5EA7",
    "mint_dark":      "#4A9E6B",
    "sky_dark":       "#3A7DAF",
    "peach_dark":     "#E8874A",
    "lemon_dark":     "#C8A800",
    "lilac_dark":     "#8844CC",
    "pistachio_dark": "#5A8F3A",

    # Neutrals
    "cream":     "#FFF8F0",
    "white":     "#FFFFFF",
    "gray_soft": "#E8E8E8",
    "gray_mid":  "#AAAAAA",
    "gray_dark": "#555555",
    "charcoal":  "#2D2D2D",
}

# Ordered palette for cycling (most visually distinct)
PALETTE = [
    MACAROON["sky"],
    MACAROON["rose"],
    MACAROON["mint"],
    MACAROON["lavender"],
    MACAROON["peach"],
    MACAROON["pistachio"],
    MACAROON["lilac"],
    MACAROON["lemon"],
    MACAROON["blush"],
    MACAROON["powder_blue"],
]

PALETTE_DARK = [
    MACAROON["sky_dark"],
    MACAROON["rose_dark"],
    MACAROON["mint_dark"],
    MACAROON["lavender_dark"],
    MACAROON["peach_dark"],
    MACAROON["pistachio_dark"],
    MACAROON["lilac_dark"],
    MACAROON["lemon_dark"],
]

# Custom colormaps
def macaroon_cmap(name: str = "macaroon_seq") -> mcolors.LinearSegmentedColormap:
    """Sequential macaroon colormap: cream → sky → lavender."""
    colors_seq = [MACAROON["cream"], MACAROON["sky"], MACAROON["lavender"], MACAROON["rose_dark"]]
    return mcolors.LinearSegmentedColormap.from_list(name, colors_seq)

def macaroon_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Diverging: mint ↔ cream ↔ rose."""
    return mcolors.LinearSegmentedColormap.from_list(
        "macaroon_div",
        [MACAROON["mint_dark"], MACAROON["mint"], MACAROON["cream"],
         MACAROON["rose"], MACAROON["rose_dark"]]
    )

# ═══════════════════════════════════════════════════════
#  GLOBAL RCPARAMS
# ═══════════════════════════════════════════════════════

def set_style():
    """Apply global Times New Roman + macaroon style to matplotlib."""
    plt.rcParams.update({
        # ── Font ──
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"],
        "mathtext.fontset":   "stix",

        # ── Sizes ──
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "figure.titlesize":   16,

        # ── Bold ──
        "axes.titleweight":   "bold",
        "axes.labelweight":   "bold",
        "figure.titleweight": "bold",

        # ── Lines ──
        "axes.linewidth":     1.4,
        "grid.linewidth":     0.7,
        "lines.linewidth":    2.2,
        "patch.linewidth":    1.0,
        "lines.markersize":   7,

        # ── Color cycle ──
        "axes.prop_cycle":    plt.cycler(color=PALETTE),

        # ── Background ──
        "figure.facecolor":   MACAROON["white"],
        "axes.facecolor":     MACAROON["cream"],
        "axes.edgecolor":     MACAROON["gray_dark"],

        # ── Grid ──
        "axes.grid":          True,
        "grid.alpha":         0.5,
        "grid.color":         MACAROON["gray_soft"],
        "grid.linestyle":     "--",

        # ── Spines ──
        "axes.spines.top":    False,
        "axes.spines.right":  False,

        # ── Legend ──
        "legend.framealpha":  0.92,
        "legend.edgecolor":   MACAROON["gray_mid"],
        "legend.facecolor":   MACAROON["white"],

        # ── Resolution ──
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.18,
        "savefig.facecolor":  MACAROON["white"],
    })


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.04,
                fontsize: int = 16):
    """Add bold panel label (A), (B), etc. to subplot axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="top", ha="right",
            fontfamily="Times New Roman")


def styled_bar(ax, x, heights, labels=None, colors=None, edgecolor=None,
               width=0.6, alpha=0.88, annotate=True, fmt="{:.3f}"):
    """Draw styled bar chart with value annotations."""
    if colors is None:
        colors = PALETTE[:len(heights)]
    if edgecolor is None:
        edgecolor = MACAROON["gray_dark"]

    bars = ax.bar(x, heights, width=width, color=colors,
                  edgecolor=edgecolor, linewidth=0.9, alpha=alpha)

    if annotate:
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + max(heights) * 0.02,
                    fmt.format(h),
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    color=MACAROON["charcoal"])
    return bars


def add_significance_bracket(ax, x1, x2, y, h, text="*", lw=1.5):
    """Draw significance bracket between two bars."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, color=MACAROON["charcoal"])
    ax.text((x1 + x2) / 2, y + h * 1.1, text,
            ha="center", va="bottom", fontsize=13,
            fontweight="bold", color=MACAROON["charcoal"])
