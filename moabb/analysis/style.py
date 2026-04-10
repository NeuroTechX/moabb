"""MOABB visual style module.

Provides a consistent, publication-quality visual identity for all MOABB plots,
inspired by Economist-style design principles and the MOABB logo color scheme.
"""

from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sea

import moabb


# ---------------------------------------------------------------------------
# Brand colors — CVD-friendly categorical palette
# Designed for deuteranopia/protanopia/tritanopia separability while
# retaining the MOABB brand feel.  Based on Wong (2011) + ColorBrewer
# principles with hue-spread across navy, green-teal, sky-blue, purple,
# amber, and red axes.
# ---------------------------------------------------------------------------
MOABB_NAVY = "#2F3E5C"  # Deep navy — primary accent
MOABB_TEAL = "#1B9E77"  # Saturated green-teal
MOABB_SKY = "#56B4E9"  # Distinct sky blue
MOABB_PURPLE = "#7E63B8"  # Purple — new hue axis
MOABB_AMBER = "#E69F00"  # Warm gold — high contrast
MOABB_CORAL = "#D55E5E"  # Slightly cooler red

MOABB_PALETTE = [
    MOABB_NAVY,
    MOABB_TEAL,
    MOABB_SKY,
    MOABB_PURPLE,
    MOABB_AMBER,
    MOABB_CORAL,
]

# Semantic colors
MOABB_DARK_TEXT = "#1b3a57"  # Titles, labels
GRID_COLOR = "#758D99"  # Grid lines, subtitles, source text

# ---------------------------------------------------------------------------
# Font size specifications
# ---------------------------------------------------------------------------
FONT_SIZES = {
    "title": 18,
    "subtitle": 14,
    "axis_label": 13,
    "tick_label": 12,
    "legend": 12,
    "annotation": 11,
    "source": 10,
}


def get_moabb_palette(n: int) -> list[str]:
    """Return *n* colors from the MOABB palette, cycling if necessary.

    Parameters
    ----------
    n : int
        Number of colors requested.

    Returns
    -------
    list of str
        Hex color strings.
    """
    if n <= len(MOABB_PALETTE):
        return MOABB_PALETTE[:n]
    return [c for _, c in zip(range(n), cycle(MOABB_PALETTE))]


def set_moabb_defaults() -> None:
    """Configure seaborn/matplotlib defaults for the MOABB brand style.

    This replaces the previous ``sea.set(font="serif", style="whitegrid", ...)``
    call at module level in ``plotting.py``.
    """
    sea.set_theme(
        style="white",
        palette=MOABB_PALETTE,
        font="serif",
        rc={
            "figure.dpi": 100,
            "savefig.dpi": 150,
            # Grid
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": True,
            # Text colors
            "text.color": MOABB_DARK_TEXT,
            "axes.labelcolor": MOABB_DARK_TEXT,
            "xtick.color": MOABB_DARK_TEXT,
            "ytick.color": MOABB_DARK_TEXT,
            # Font sizes
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "legend.fontsize": FONT_SIZES["legend"],
        },
    )


_DEFAULT_SOURCE = (
    "Generated using MOABB v{version}"
    " \u2014 Chevallier et al. (2024)"
    " doi:10.48550/arXiv.2404.15319"
)


def apply_moabb_style(
    ax,
    title: str = "",
    subtitle: str = "",
    source: str | None = None,
    accent_line: bool = True,
    grid_axis: str = "y",
) -> None:
    """Apply the MOABB/Economist visual style to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    title : str
        Bold title placed at top-left of the figure.
    subtitle : str
        Lighter subtitle placed below the title.
    source : str
        Source attribution at bottom-left of the figure.
    accent_line : bool
        If True, draw a navy accent line and small rectangle at the top of
        the figure (Economist signature element in MOABB navy).
    grid_axis : str
        Which axis to show grid lines on: ``"y"``, ``"x"``, ``"both"``,
        or ``"none"``.
    """
    if source is None:
        source = _DEFAULT_SOURCE.format(version=moabb.__version__)

    fig = ax.get_figure()

    # --- Spines ---
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)

    # --- Grid ---
    if grid_axis == "none":
        ax.grid(False)
    elif grid_axis == "both":
        ax.grid(True, axis="both", color=GRID_COLOR, alpha=0.3, linewidth=0.6)
    elif grid_axis == "x":
        ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.3, linewidth=0.6)
        ax.grid(False, axis="y")
    else:  # default "y"
        ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3, linewidth=0.6)
        ax.grid(False, axis="x")

    # --- Tick styling ---
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelsize=FONT_SIZES["tick_label"],
        colors=MOABB_DARK_TEXT,
    )

    # --- Accent line at top ---
    if accent_line:
        # Full-width navy line
        fig.patches.append(
            plt.Rectangle(
                (0.08, 0.94),
                0.84,
                0.006,
                transform=fig.transFigure,
                clip_on=False,
                facecolor=MOABB_NAVY,
                edgecolor="none",
            )
        )
        # Small accent rectangle
        fig.patches.append(
            plt.Rectangle(
                (0.08, 0.935),
                0.04,
                0.005,
                transform=fig.transFigure,
                clip_on=False,
                facecolor=MOABB_TEAL,
                edgecolor="none",
            )
        )

    # --- Title ---
    if title:
        fig.text(
            0.08,
            0.92,
            title,
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            color=MOABB_DARK_TEXT,
            ha="left",
            va="top",
        )

    # --- Subtitle ---
    if subtitle:
        fig.text(
            0.08,
            0.89,
            subtitle,
            fontsize=FONT_SIZES["subtitle"],
            color=GRID_COLOR,
            ha="left",
            va="top",
        )

    # --- Source ---
    if source:
        fig.text(
            0.08,
            0.02,
            source,
            fontsize=FONT_SIZES["source"],
            fontstyle="italic",
            color=GRID_COLOR,
            ha="left",
            va="bottom",
        )

    # Remove the default matplotlib title (we use fig.text instead)
    ax.set_title("")


def style_legend(ax, **kwargs) -> None:
    """Restyle the legend on *ax* to match the MOABB visual identity.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose legend should be restyled.
    **kwargs
        Extra keyword arguments forwarded to ``ax.legend()``.
    """
    legend = ax.get_legend()
    if legend is None:
        return

    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]

    defaults = {
        "frameon": True,
        "edgecolor": "none",
        "facecolor": "white",
        "framealpha": 0.9,
        "fontsize": FONT_SIZES["legend"],
    }
    defaults.update(kwargs)
    ax.legend(handles, labels, **defaults)
