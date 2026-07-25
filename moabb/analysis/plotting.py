from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sea
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, RegularPolygon
from scipy.stats import t

from moabb.analysis._utils import _compute_n_trials, _match_float, _match_int
from moabb.analysis.meta_analysis import (
    collapse_session_scores,
    combine_effects,
    combine_pvalues,
)
from moabb.analysis.style import (
    FONT_SIZES,
    GRID_COLOR,
    MOABB_CORAL,
    MOABB_DARK_TEXT,
    MOABB_NAVY,
    MOABB_PALETTE,
    MOABB_SKY,
    MOABB_TEAL,
    apply_moabb_style,
    set_moabb_defaults,
    style_legend,
)


PIPELINE_PALETTE = MOABB_PALETTE  # backward-compat alias
set_moabb_defaults()

log = logging.getLogger(__name__)

# Line style definitions for adjusted chance level alpha thresholds
_CHANCE_COLOR = "#b0413e"  # muted red — distinct from data, not aggressive
_ALPHA_LINE_STYLES = {
    0.05: {"linestyle": "--", "color": _CHANCE_COLOR, "linewidth": 1.2, "alpha": 0.55},
    0.01: {"linestyle": "-.", "color": _CHANCE_COLOR, "linewidth": 1.2, "alpha": 0.45},
    0.001: {"linestyle": ":", "color": _CHANCE_COLOR, "linewidth": 1.2, "alpha": 0.35},
}


def _resolve_chance_levels(data, chance_level):
    """Resolve chance_level to per-dataset (theoretical, adjusted) dicts."""
    datasets = data["dataset"].unique()

    if chance_level is None:
        return dict.fromkeys(datasets, 0.5), None

    if chance_level == "auto":
        from moabb.analysis.chance_level import chance_by_chance

        return _resolve_chance_levels(data, chance_by_chance(data))

    if isinstance(chance_level, (int, float)):
        return {d: float(chance_level) for d in datasets}, None

    if isinstance(chance_level, dict):
        theoretical = {}
        adjusted = {}
        for d in datasets:
            val = chance_level.get(d)
            if isinstance(val, dict):
                theoretical[d] = val.get("theoretical", 0.5)
                if "adjusted" in val:
                    adjusted[d] = val["adjusted"]
            elif isinstance(val, (int, float)):
                theoretical[d] = float(val)
            else:
                theoretical[d] = 0.5
        return theoretical, adjusted if adjusted else None

    raise TypeError(
        f"chance_level must be None, float, 'auto', or dict, got {type(chance_level)}"
    )


def _chance_label_text(level):
    """Build the annotation string for a theoretical chance level line."""
    pct = f"{level:.0f}%" if level == int(level) else f"{level:.1f}%"
    return f"Chance level {pct} \u2014 Combrisson & Jerbi (2015)"


def _chance_by_chance_label_text(level):
    """Build the annotation string for an adjusted chance level band."""
    pct = f"{level:.1f}%"
    return f"Chance by chance ({pct}, p<0.05) \u2014 Combrisson & Jerbi (2015)"


_CHANCE_ANNOT_KW = {
    "fontsize": FONT_SIZES["source"],
    "color": _CHANCE_COLOR,
    "alpha": 0.9,
    "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
}

# Module-level colormap for significance heatmaps
_MOABB_SIGNIFICANCE_CMAP = LinearSegmentedColormap.from_list(
    "moabb_sig", ["white", MOABB_TEAL, MOABB_NAVY]
)
_MOABB_SIGNIFICANCE_CMAP.set_under(color=[1, 1, 1])
_MOABB_SIGNIFICANCE_CMAP.set_over(color=MOABB_CORAL)


def _max_adjusted_threshold(adjusted, alpha=0.05):
    """Max adjusted threshold across datasets for *alpha*, or None."""
    if not adjusted:
        return None
    vals = [lv[alpha] for lv in adjusted.values() if alpha in lv]
    return max(vals) if vals else None


def _to_percentage(data, theoretical, adjusted):
    """Convert scores and chance levels to percentages (returns copies)."""
    data = data.copy()
    data["score"] = data["score"] * 100
    theoretical = {k: v * 100 for k, v in theoretical.items()}
    if adjusted:
        adjusted = {
            k: {a: v * 100 for a, v in alphas.items()} for k, alphas in adjusted.items()
        }
    return data, theoretical, adjusted


def _draw_chance_lines(ax, chance_levels, datasets, orientation, adjusted=None):
    """Draw theoretical chance level lines and optional shaded band.

    If all datasets share the same level, draws a single spanning line.
    Otherwise, draws per-dataset line segments.  When *adjusted* levels
    are provided, a shaded band from the axis edge to the maximum
    adjusted threshold (alpha=0.05) is drawn and annotated with a
    "Chance by chance" label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    chance_levels : dict[str, float]
    datasets : array-like
        Ordered dataset names as they appear on the categorical axis.
    orientation : str
        'horizontal' or 'vertical'.
    adjusted : dict[str, dict[float, float]] or None
        Per-dataset adjusted levels ``{dataset: {alpha: threshold}}``.
    """
    unique_levels = set(chance_levels.values())

    # --- Draw shaded band for "chance by chance" region ---
    max_adj = _max_adjusted_threshold(adjusted)

    if max_adj is not None:
        if orientation in ("horizontal", "h"):
            ax.axvspan(
                ax.get_xlim()[0], max_adj, color=_CHANCE_COLOR, alpha=0.06, zorder=0
            )
        else:
            ax.axhspan(
                ax.get_ylim()[0], max_adj, color=_CHANCE_COLOR, alpha=0.06, zorder=0
            )

    # --- Draw chance level lines ---
    if len(unique_levels) == 1:
        level = unique_levels.pop()
        line_kw = {
            "linestyle": "--",
            "color": _CHANCE_COLOR,
            "linewidth": 1.5,
            "alpha": 0.75,
            "zorder": 2,
        }
        if orientation in ("horizontal", "h"):
            ax.axvline(level, **line_kw)
        else:
            ax.axhline(level, **line_kw)
    else:
        datasets_list = list(datasets)
        for i, d in enumerate(datasets_list):
            level = chance_levels.get(d, 0.5)
            line_kw = {
                "linestyle": "--",
                "color": _CHANCE_COLOR,
                "linewidth": 1.3,
                "alpha": 0.75,
                "zorder": 2,
            }
            if orientation in ("horizontal", "h"):
                ax.plot([level, level], [i - 0.4, i + 0.4], **line_kw)
            else:
                ax.plot([i - 0.4, i + 0.4], [level, level], **line_kw)

    # --- Annotate ---
    if max_adj is not None:
        label = _chance_by_chance_label_text(max_adj)
    else:
        ref_level = next(iter(chance_levels.values()), 50)
        label = _chance_label_text(ref_level)
        max_adj = ref_level  # use theoretical level for annotation position

    if orientation in ("horizontal", "h"):
        ax.annotate(
            label,
            xy=(max_adj, 0),
            xycoords=("data", "axes fraction"),
            xytext=(6, 8),
            textcoords="offset points",
            va="bottom",
            ha="left",
            **_CHANCE_ANNOT_KW,
        )
    else:
        ax.annotate(
            label,
            xy=(0, max_adj),
            xycoords=("axes fraction", "data"),
            xytext=(8, 6),
            textcoords="offset points",
            va="bottom",
            ha="left",
            **_CHANCE_ANNOT_KW,
        )


def _draw_adjusted_chance_lines(ax, adjusted_levels, datasets, orientation):
    """Draw adjusted significance threshold lines with value annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    adjusted_levels : dict[str, dict[float, float]]
        Mapping of dataset name to ``{alpha: threshold}``.
    datasets : array-like
        Ordered dataset names as they appear on the categorical axis.
    orientation : str
        'horizontal' or 'vertical'.
    """
    if not adjusted_levels:
        return

    # Collect all alpha values across datasets
    all_alphas = set()
    for levels in adjusted_levels.values():
        all_alphas.update(levels.keys())

    # Stagger annotation vertical offset so labels don't overlap
    alpha_list = sorted(all_alphas, reverse=True)

    for _rank, alpha_val in enumerate(alpha_list):
        style = _ALPHA_LINE_STYLES.get(
            alpha_val,
            {"linestyle": "--", "color": _CHANCE_COLOR, "linewidth": 1.2, "alpha": 0.5},
        )

        # Check if all datasets share the same adjusted level for this alpha
        per_dataset = {}
        for d in datasets:
            if d in adjusted_levels and alpha_val in adjusted_levels[d]:
                per_dataset[d] = adjusted_levels[d][alpha_val]

        if not per_dataset:
            continue

        unique_vals = set(per_dataset.values())
        line_alpha = style.get("alpha", 0.5)

        if len(unique_vals) == 1 and len(per_dataset) == len(datasets):
            level = unique_vals.pop()
            if orientation in ("horizontal", "h"):
                ax.axvline(level, **style)
            else:
                ax.axhline(level, **style)

            # Annotate with value and alpha — place at right edge
            pct = f"{level:.1f}%"
            label = f"p<{alpha_val} ({pct})"
            annot_kw = {
                "fontsize": FONT_SIZES["source"] - 1,
                "color": _CHANCE_COLOR,
                "alpha": line_alpha + 0.15,
                "bbox": {
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
            }
            if orientation in ("horizontal", "h"):
                ax.annotate(
                    label,
                    xy=(level, 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(6, -4),
                    textcoords="offset points",
                    va="top",
                    ha="left",
                    **annot_kw,
                )
            else:
                ax.annotate(
                    label,
                    xy=(1, level),
                    xycoords=("axes fraction", "data"),
                    xytext=(-6, 4),
                    textcoords="offset points",
                    va="bottom",
                    ha="right",
                    **annot_kw,
                )
        else:
            datasets_list = list(datasets)
            for i, d in enumerate(datasets_list):
                if d not in per_dataset:
                    continue
                level = per_dataset[d]
                if orientation in ("horizontal", "h"):
                    ax.plot([level, level], [i - 0.4, i + 0.4], **style)
                else:
                    ax.plot([i - 0.4, i + 0.4], [level, level], **style)


def _simplify_names(x):
    if len(x) > 10:
        return x.split(" ")[0]
    else:
        return x


def _prepare_plot_data(data, pipelines=None):
    """Collapse sessions, simplify dataset names, and filter pipelines.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Results dataframe.
    pipelines : list of str | None
        Pipelines to keep. If None, all pipelines are kept.

    Returns
    -------
    :class:`pandas.DataFrame`
        Preprocessed copy of the data.
    """
    data = collapse_session_scores(data)
    unique_ids = data["dataset"].apply(_simplify_names)
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Dataset names are too similar, turning off name shortening")
    else:
        data["dataset"] = unique_ids
    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]
    return data


def _extract_color_dict(handles, labels):
    """Build a color dictionary from legend handles.

    Parameters
    ----------
    handles : list
        Matplotlib legend handles.
    labels : list of str
        Corresponding labels.

    Returns
    -------
    dict
        Mapping of label to facecolor.
    """
    color_dict = {}
    for lb, h in zip(labels, handles):
        if hasattr(h, "get_facecolor"):
            fc = h.get_facecolor()
            color_dict[lb] = fc[0] if hasattr(fc, "__len__") and len(fc) > 0 else fc
        elif hasattr(h, "get_color"):
            color_dict[lb] = h.get_color()
        elif hasattr(h, "get_markerfacecolor"):
            color_dict[lb] = h.get_markerfacecolor()
        else:
            color_dict[lb] = "C0"
    return color_dict


def score_plot(data, pipelines=None, orientation="vertical", chance_level=None):
    """Plot scores for all pipelines and all datasets.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Output of ``Results.to_dataframe()``.
    pipelines : list of str | None
        Pipelines to include in this plot.
    orientation : str
        Plot orientation, one of ``["vertical", "v", "horizontal", "h"]``.
        Defaults to ``"vertical"``.
    chance_level : None, float, or dict
        Chance level to display on the plot. Defaults to ``None``.

        - ``None`` : defaults to 0.5 for all datasets (backward compatible).
        - ``float`` : uniform chance level for all datasets.
        - ``dict`` : per-dataset chance levels. Can be a simple
          ``{dataset_name: float}`` mapping or the output of
          ``chance_by_chance``.
          When the dict includes ``'adjusted'`` entries, adjusted
          significance threshold lines are also drawn.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Pyplot handle.
    color_dict : dict
        Dictionary with the facecolor for each pipeline.
    """
    data = _prepare_plot_data(data, pipelines)
    theoretical, adjusted = _resolve_chance_levels(data, chance_level)
    data, theoretical, adjusted = _to_percentage(data, theoretical, adjusted)

    if orientation in ["horizontal", "h"]:
        y, x = "dataset", "score"
        fig = plt.figure(figsize=(8.5, 11))
    elif orientation in ["vertical", "v"]:
        x, y = "dataset", "score"
        fig = plt.figure(figsize=(11, 9.5))
    else:
        raise ValueError("Invalid plot orientation selected!")

    ax = fig.add_subplot(111)
    sea.stripplot(
        data=data,
        y=y,
        x=x,
        jitter=0.15,
        palette=MOABB_PALETTE,
        hue="pipeline",
        dodge=True,
        ax=ax,
        alpha=0.7,
        size=7,
    )

    datasets_order = data["dataset"].unique()

    if orientation in ["horizontal", "h"]:
        ax.set_xlim([0, 100])
        ax.set_xlabel("Score (%)", fontsize=FONT_SIZES["axis_label"] + 2)
    else:
        ax.set_ylim([0, 100])
        ax.set_ylabel("Score (%)", fontsize=FONT_SIZES["axis_label"] + 2)

    if chance_level is not None:
        _draw_chance_lines(
            ax, theoretical, datasets_order, orientation, adjusted=adjusted
        )
        if adjusted:
            _draw_adjusted_chance_lines(ax, adjusted, datasets_order, orientation)

    handles, labels = ax.get_legend_handles_labels()
    color_dict = _extract_color_dict(handles, labels)

    apply_moabb_style(ax, title="Scores per dataset and algorithm", subtitle="")
    style_legend(ax)
    fig.subplots_adjust(top=0.85, bottom=0.14)

    return fig, color_dict


def distribution_plot(
    data, pipelines=None, orientation="vertical", chance_level=None, figsize=None
):
    """Plot score distributions using violin (KDE) and strip plots.

    Creates a combined violin plot and strip plot visualization that
    shows both the distribution shape (via KDE) and individual data
    points for each dataset/pipeline combination.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Output of ``Results.to_dataframe()``.
    pipelines : list of str | None
        Pipelines to include in this plot.
    orientation : str
        Plot orientation, one of ``["vertical", "v", "horizontal", "h"]``.
        Defaults to ``"vertical"``.
    chance_level : None, float, or dict
        Chance level to display on the plot. Defaults to ``None``.

        - ``None`` : defaults to 0.5 for all datasets.
        - ``float`` : uniform chance level for all datasets.
        - ``dict`` : per-dataset chance levels. Can be a simple
          ``{dataset_name: float}`` mapping or the output of
          ``chance_by_chance``.
          When the dict includes ``'adjusted'`` entries, adjusted
          significance threshold lines are also drawn.
    figsize : tuple of (float, float) | None
        Figure size. If None, defaults based on orientation.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Pyplot handle.
    color_dict : dict
        Dictionary with the facecolor for each pipeline.
    """
    data = _prepare_plot_data(data, pipelines)
    theoretical, adjusted = _resolve_chance_levels(data, chance_level)
    data, theoretical, adjusted = _to_percentage(data, theoretical, adjusted)

    if orientation in ["horizontal", "h"]:
        y, x = "dataset", "score"
        figsize = figsize or (8.5, 11)
    elif orientation in ["vertical", "v"]:
        x, y = "dataset", "score"
        figsize = figsize or (11, 9.5)
    else:
        raise ValueError("Invalid plot orientation selected!")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Violin plot for KDE density
    sea.violinplot(
        data=data,
        y=y,
        x=x,
        hue="pipeline",
        palette=MOABB_PALETTE,
        inner=None,
        alpha=0.3,
        dodge=True,
        ax=ax,
        cut=0,
        density_norm="width",
    )

    # Strip plot for individual data points
    sea.stripplot(
        data=data,
        y=y,
        x=x,
        jitter=0.15,
        palette=MOABB_PALETTE,
        hue="pipeline",
        dodge=True,
        ax=ax,
        alpha=0.7,
        size=5,
    )

    datasets_order = data["dataset"].unique()

    if orientation in ["horizontal", "h"]:
        ax.set_xlim([0, 100])
        ax.set_xlabel("Score (%)", fontsize=FONT_SIZES["axis_label"] + 2)
    else:
        ax.set_ylim([0, 100])
        ax.set_ylabel("Score (%)", fontsize=FONT_SIZES["axis_label"] + 2)

    if chance_level is not None:
        _draw_chance_lines(
            ax, theoretical, datasets_order, orientation, adjusted=adjusted
        )
        if adjusted:
            _draw_adjusted_chance_lines(ax, adjusted, datasets_order, orientation)

    # Deduplicate legend entries (violin + strip create duplicates)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles = []
    unique_labels = []
    for h, lb in zip(handles, labels):
        if lb not in seen:
            seen[lb] = h
            unique_handles.append(h)
            unique_labels.append(lb)
    ax.legend(unique_handles, unique_labels)

    color_dict = _extract_color_dict(unique_handles, unique_labels)

    apply_moabb_style(
        ax, title="Score distributions per dataset and algorithm", subtitle=""
    )
    style_legend(ax)
    fig.subplots_adjust(top=0.85, bottom=0.14)
    return fig, color_dict


def codecarbon_plot(
    data,
    order_list=None,
    pipelines=None,
    country="",
    include_efficiency=False,
    include_power_vs_score=False,
):
    """Plot code carbon consumption for results from the benchmark.

    Creates comprehensive emissions visualizations leveraging detailed CodeCarbon
    tracking data. By default, shows CO2 emissions per dataset and algorithm.
    Additional metrics can be enabled to show efficiency trade-offs and hardware
    utilization.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Output of Results.to_dataframe() containing benchmark results.
        Should include 'carbon_emission' and 'score' columns for enhanced analysis.
    order_list : list of str or None
        Order of pipelines to include in the plot. If None, uses default order.
        Defaults to ``None``.
    pipelines : list of str or None
        Specific pipelines to include in the plot. If None, includes all pipelines.
        Defaults to ``None``.
    country : str
        Country name to include in plot titles for geographic context.
        Defaults to ``""``.
    include_efficiency : bool
        If True, adds subplot showing energy efficiency (score per kg CO2).
        Highlights pipelines with best accuracy-to-emissions ratio.
        Defaults to ``False``.
    include_power_vs_score : bool
        If True, adds subplot showing accuracy vs emissions scatter plot.
        Useful for identifying Pareto-optimal pipelines balancing performance
        and sustainability.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Pyplot figure handle containing the requested visualizations.

    Notes
    -----
    The function expects CodeCarbon to have been enabled during benchmark with
    save_to_file=True to capture detailed emissions data. If detailed metrics
    are unavailable, falls back to basic CO2 emissions visualization.

    The plot uses logarithmic scale for CO2 emissions due to potential wide
    variance across different datasets and pipeline types.

    Examples
    --------
    Basic usage (emissions only):
    >>> results = benchmark(
    ...     pipelines="./pipelines/", codecarbon_config={"save_to_file": True}
    ... )
    >>> fig = codecarbon_plot(results)

    With efficiency metrics:
    >>> fig = codecarbon_plot(results, include_efficiency=True, country="France")

    With multiple views:
    >>> fig = codecarbon_plot(
    ...     results,
    ...     include_efficiency=True,
    ...     include_power_vs_score=True,
    ...     order_list=["CSP+SVM", "Tangent Space LR"],
    ...     pipelines=["CSP+SVM", "Tangent Space LR"],
    ... )
    """
    data = collapse_session_scores(data)
    unique_ids = data["dataset"].apply(_simplify_names)
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Dataset names are too similar, turning off name shortening")
    else:
        data["dataset"] = unique_ids

    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]

    # Rename for consistency
    if "carbon emission" in data.columns:
        data = data.rename(columns={"carbon emission": "carbon_emission"})

    # Prepare data for main plot
    plot_data = data.copy()

    # Determine number of subplots
    n_plots = 1
    if include_efficiency and "carbon_emission" in plot_data.columns:
        n_plots += 1
    if include_power_vs_score and "carbon_emission" in plot_data.columns:
        n_plots += 1

    # Create figure with subplots if needed
    if n_plots > 1:
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 8))
        # axes is already a numpy array of Axes objects
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8.5))
        axes = [ax]  # Wrap single axis in list for consistent indexing

    # Plot 1: Main CO2 emissions by dataset and algorithm
    ax = axes[0]
    pivot_data = (
        plot_data.groupby(["dataset", "pipeline"])["carbon_emission"].mean().reset_index()
    )

    # Get unique pipelines in the desired order
    unique_pipelines = (
        [p for p in order_list if p in pivot_data["pipeline"].unique()]
        if order_list
        else list(pivot_data["pipeline"].unique())
    )

    # Create bar plot
    for idx, pipeline in enumerate(unique_pipelines):
        pipeline_data = pivot_data[pivot_data["pipeline"] == pipeline]
        color = MOABB_PALETTE[idx % len(MOABB_PALETTE)]
        ax.bar(
            pipeline_data["dataset"],
            pipeline_data["carbon_emission"],
            label=pipeline,
            alpha=0.8,
            width=0.8 / len(unique_pipelines),
            color=color,
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"$CO_2$ Emission (kg, Log Scale)")
    ax.set_xlabel("Dataset")
    co2_title = r"$CO_2$ Emission per Dataset and Algorithm"
    if country:
        co2_title += f" {country}"
    ax.legend(title="Pipeline", bbox_to_anchor=(1.05, 1), loc="upper left")

    apply_moabb_style(
        ax, title=co2_title, subtitle="Average emissions by pipeline", accent_line=True
    )
    style_legend(ax)

    # Plot 2: Energy efficiency (score per kg CO2)
    if include_efficiency and n_plots > 1:
        ax = axes[1]
        efficiency_data = plot_data.groupby("pipeline").apply(
            lambda x: pd.Series(
                {
                    "avg_score": x["score"].mean(),
                    "avg_emissions": x["carbon_emission"].mean(),
                    "n_evals": len(x),
                }
            )
        )
        efficiency_data["efficiency"] = (
            efficiency_data["avg_score"] / efficiency_data["avg_emissions"]
        )
        efficiency_data = efficiency_data.sort_values("efficiency", ascending=False)

        colors = [
            (
                MOABB_PALETTE[unique_pipelines.index(p) % len(MOABB_PALETTE)]
                if p in unique_pipelines
                else MOABB_PALETTE[0]
            )
            for p in efficiency_data.index
        ]
        bars = ax.barh(efficiency_data.index, efficiency_data["efficiency"], color=colors)

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontsize=FONT_SIZES["annotation"],
            )

        ax.set_xlabel("Energy Efficiency (Accuracy / kg CO2)")
        apply_moabb_style(
            ax,
            title="Pipeline Energy Efficiency",
            subtitle="Higher is better",
            accent_line=False,
            source="",
            grid_axis="x",
        )

    # Plot 3: Accuracy vs Emissions scatter
    if include_power_vs_score and n_plots > 2:
        ax = axes[2]
        scatter_data = plot_data.groupby("pipeline").apply(
            lambda x: pd.Series(
                {
                    "avg_score": x["score"].mean(),
                    "avg_emissions": x["carbon_emission"].mean(),
                    "count": len(x),
                }
            )
        )

        for _idx, (pipeline, row) in enumerate(scatter_data.iterrows()):
            color = (
                MOABB_PALETTE[unique_pipelines.index(pipeline) % len(MOABB_PALETTE)]
                if pipeline in unique_pipelines
                else MOABB_PALETTE[0]
            )
            ax.scatter(
                row["avg_emissions"],
                row["avg_score"],
                s=300,
                alpha=0.7,
                color=color,
                edgecolors=MOABB_DARK_TEXT,
                linewidth=1.5,
            )
            ax.annotate(
                pipeline,
                (row["avg_emissions"], row["avg_score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=FONT_SIZES["annotation"],
                fontweight="bold",
            )

        ax.set_xlabel(r"Avg CO$_2$ Emissions (kg)")
        ax.set_ylabel("Avg Accuracy Score")
        ax.set_xscale("log")
        apply_moabb_style(
            ax,
            title="Accuracy vs Emissions Trade-off",
            subtitle="Upper-right is better",
            accent_line=False,
            source="",
            grid_axis="both",
        )

    fig.subplots_adjust(top=0.85, bottom=0.14)
    return fig


def emissions_summary(data, order_list=None, pipelines=None):
    """Generate a summary report of emissions metrics from benchmark results.

    Provides comprehensive analysis of energy consumption and sustainability
    metrics across pipelines, including efficiency rankings and trade-offs.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Output of Results.to_dataframe() containing benchmark results.
        Must include 'carbon_emission' and 'score' columns.
    order_list : list of str or None
        Order of pipelines to include in the summary. Defaults to ``None``.
    pipelines : list of str or None
        Specific pipelines to include in the summary. Defaults to ``None``.

    Returns
    -------
    summary : :class:`pandas.DataFrame`
        Summary statistics with columns:
        - pipeline : Pipeline name
        - avg_score : Average accuracy score
        - avg_emissions : Average CO2 emissions (kg)
        - total_emissions : Total CO2 emissions (kg)
        - efficiency : Score per kg CO2 (higher is better)
        - n_evaluations : Number of evaluations performed

    Examples
    --------
    >>> results = benchmark(pipelines="./pipelines/", ...)
    >>> summary = emissions_summary(results)
    >>> print(summary.sort_values("efficiency", ascending=False))
    """
    data = collapse_session_scores(data)

    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]

    if "carbon emission" in data.columns:
        data = data.rename(columns={"carbon emission": "carbon_emission"})

    if "carbon_emission" not in data.columns:
        log.warning("No carbon_emission data found in results")
        return None

    # Calculate summary statistics per pipeline
    summary = data.groupby("pipeline").apply(
        lambda x: pd.Series(
            {
                "avg_score": x["score"].mean(),
                "std_score": x["score"].std(),
                "avg_emissions": x["carbon_emission"].mean(),
                "total_emissions": x["carbon_emission"].sum(),
                "n_evaluations": len(x),
            }
        )
    )

    # Calculate efficiency metrics
    summary["efficiency"] = summary["avg_score"] / summary["avg_emissions"]
    summary["emissions_per_eval"] = summary["total_emissions"] / summary["n_evaluations"]

    # Reorder columns
    summary = summary[
        [
            "avg_score",
            "std_score",
            "avg_emissions",
            "total_emissions",
            "emissions_per_eval",
            "efficiency",
            "n_evaluations",
        ]
    ]

    # Apply ordering if provided
    if order_list is not None:
        existing_order = [p for p in order_list if p in summary.index]
        summary = summary.loc[existing_order]

    return summary


def _draw_paired_chance_region(ax, theoretical, adjusted, min_chance):
    """Draw chance level crosshair and significance band on a paired plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    theoretical : dict[str, float]
        Per-dataset theoretical chance levels (in percentage).
    adjusted : dict[str, dict[float, float]] or None
        Per-dataset adjusted levels (in percentage).
    min_chance : float
        Minimum theoretical chance level in percentage (used for crosshair).
    """
    # Draw crosshair at theoretical chance level
    ax.axhline(min_chance, linestyle="--", color=_CHANCE_COLOR, linewidth=1.2, alpha=0.5)
    ax.axvline(min_chance, linestyle="--", color=_CHANCE_COLOR, linewidth=1.2, alpha=0.5)

    # Draw shaded band if adjusted significance thresholds are available
    max_adj = _max_adjusted_threshold(adjusted)
    if max_adj is not None:
        ax.axhspan(ax.get_ylim()[0], max_adj, color=_CHANCE_COLOR, alpha=0.06, zorder=0)
        ax.axvspan(ax.get_xlim()[0], max_adj, color=_CHANCE_COLOR, alpha=0.06, zorder=0)

    # Annotate at the top-right edge of the shaded band
    if max_adj is not None:
        label = _chance_by_chance_label_text(max_adj)
        ax.annotate(
            label,
            xy=(1, max_adj),
            xycoords=("axes fraction", "data"),
            xytext=(-8, 6),
            textcoords="offset points",
            va="bottom",
            ha="right",
            **_CHANCE_ANNOT_KW,
        )
    else:
        pct = f"{min_chance:.0f}%"
        label = f"Chance level ({pct}) \u2014 Combrisson & Jerbi (2015)"
        ax.annotate(
            label,
            xy=(1, min_chance),
            xycoords=("axes fraction", "data"),
            xytext=(-8, 6),
            textcoords="offset points",
            va="bottom",
            ha="right",
            **_CHANCE_ANNOT_KW,
        )


def paired_plot(data, alg1, alg2, chance_level=None):
    """Generate a figure with a paired plot.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe obtained from evaluation.
    alg1 : str
        Name of a member of column ``data.pipeline``.
    alg2 : str
        Name of a member of column ``data.pipeline``.
    chance_level : None, float, or dict
        Chance level used to set axis limits and draw reference lines.
        Defaults to ``None``.

        - ``None`` : defaults to 0.5.
        - ``float`` : uniform chance level.
        - ``dict`` : per-dataset levels (the minimum value across datasets
          is used for axis limits).  When adjusted significance thresholds
          are included, a shaded band marks the "not significantly above
          chance" region.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Pyplot handle.
    """
    data = collapse_session_scores(data)
    data = data[data.pipeline.isin([alg1, alg2])]

    theoretical, adjusted = _resolve_chance_levels(data, chance_level)
    min_chance = min(theoretical.values()) if theoretical else 0.5
    min_chance *= 100
    _, theoretical, adjusted = _to_percentage(data, theoretical, adjusted)

    data = data.pivot_table(
        values="score", columns="pipeline", index=["subject", "dataset"]
    )
    data = data.reset_index()
    data[alg1] = data[alg1] * 100
    data[alg2] = data[alg2] * 100

    fig = plt.figure(figsize=(11, 9.5))
    ax = fig.add_subplot(111)
    ax.scatter(
        data[alg1],
        data[alg2],
        color=MOABB_PALETTE[0],
        edgecolors=MOABB_PALETTE[4],
        alpha=0.7,
        s=50,
        zorder=3,
    )
    ax.plot([min_chance, 100], [min_chance, 100], ls="--", c=GRID_COLOR, linewidth=1)
    ax.set_xlim([min_chance, 100])
    ax.set_ylim([min_chance, 100])
    if chance_level is not None:
        _draw_paired_chance_region(ax, theoretical, adjusted, min_chance)
    ax.set_xlabel(f"{alg1} (%)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(f"{alg2} (%)", fontsize=FONT_SIZES["axis_label"])

    apply_moabb_style(ax, title=f"{alg1} vs {alg2}", grid_axis="both")
    fig.subplots_adjust(top=0.85, bottom=0.14)
    return fig


def summary_plot(sig_df, effect_df, p_threshold=0.05, simplify=True):
    """Significance matrix to compare pipelines.

    Visualize significances as a heatmap with green/grey/red for significantly
    higher/significantly lower.

    Parameters
    ----------
    sig_df: :class:`pandas.DataFrame`
        DataFrame of pipeline x pipeline where each value is a p-value,
    effect_df: :class:`pandas.DataFrame`
        DataFrame where each value is an effect size

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
        Pyplot handle
    """
    if simplify:
        effect_df.columns = effect_df.columns.map(_simplify_names)
        sig_df.columns = sig_df.columns.map(_simplify_names)
    annot_df = effect_df.copy().astype(object)
    for row in annot_df.index:
        for col in annot_df.columns:
            if effect_df.loc[row, col] > 0:
                txt = "{:.2f}\np={:1.0e}".format(
                    effect_df.loc[row, col], sig_df.loc[row, col]
                )
            else:
                # we need the effect direction and p-value to coincide.
                # TODO: current is hack
                if sig_df.loc[row, col] < p_threshold:
                    sig_df.loc[row, col] = 1e-110
                txt = ""
            annot_df.loc[row, col] = txt
    fig = plt.figure(figsize=(10, 9.5))
    ax = fig.add_subplot(111)

    sea.heatmap(
        data=-np.log(sig_df),
        annot=annot_df,
        fmt="",
        cmap=_MOABB_SIGNIFICANCE_CMAP,
        linewidths=1,
        linecolor="0.8",
        annot_kws={"size": FONT_SIZES["annotation"]},
        cbar=False,
        vmin=-np.log(0.05),
        vmax=-np.log(1e-100),
    )
    for lb in ax.get_xticklabels():
        lb.set_rotation(45)
        lb.set_ha("right")
    ax.tick_params(axis="y", rotation=0.9)

    apply_moabb_style(
        ax,
        title="Algorithm comparison",
        subtitle="Significance matrix (effect size and p-values)",
        grid_axis="none",
    )
    fig.subplots_adjust(top=0.85, bottom=0.18)
    return fig


def meta_analysis_plot(stats_df, alg1, alg2):  # noqa: C901
    """Meta-analysis to compare two algorithms across several datasets.

    A meta-analysis style plot that shows the standardized effect with
    confidence intervals over all datasets for two algorithms.
    Hypothesis is that alg1 is larger than alg2

    Parameters
    ----------
    stats_df: :class:`pandas.DataFrame`
        DataFrame generated by compute_dataset_statistics
    alg1: str
        Name of first pipeline
    alg2: str
        Name of second pipeline

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
        Pyplot handle
    """

    def _marker(pval):
        if pval < 0.001:
            return "$***$", 100
        elif pval < 0.01:
            return "$**$", 70
        elif pval < 0.05:
            return "$*$", 30
        else:
            raise ValueError("insignificant pval {}".format(pval))

    if alg1 not in stats_df.pipe1.unique():  # was assert, now raises properly
        raise ValueError(f"algorithm '{alg1}' not found in stats_df")
    if alg2 not in stats_df.pipe1.unique():  # was assert
        raise ValueError(f"algorithm '{alg2}' not found in stats_df")
    df_fw = stats_df.loc[(stats_df.pipe1 == alg1) & (stats_df.pipe2 == alg2)]
    df_fw = df_fw.sort_values(by="pipe1")
    df_bk = stats_df.loc[(stats_df.pipe1 == alg2) & (stats_df.pipe2 == alg1)]
    df_bk = df_bk.sort_values(by="pipe1")
    dsets = df_fw.dataset.unique()
    simplify = True
    unique_ids = [_simplify_names(x) for x in dsets]
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Dataset names are too similar, turning off name shortening")
        simplify = False
    ci = []
    fig_height = max(5.5, 1.2 * (len(dsets) + 2.5))
    fig = plt.figure(figsize=(11, fig_height))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.65], wspace=0.06)
    sig_ind = []
    pvals = []
    ax = fig.add_subplot(gs[0, :-1])
    ax.set_yticks(np.arange(len(dsets) + 1))
    if simplify:
        ax.set_yticklabels(["Meta-effect"] + list(unique_ids))
    else:
        ax.set_yticklabels(["Meta-effect"] + list(dsets))
    pval_ax = fig.add_subplot(gs[0, -1], sharey=ax)
    plt.setp(pval_ax.get_yticklabels(), visible=False)
    _min = 0
    _max = 0
    for ind, d in enumerate(dsets):
        nsub = df_fw.loc[df_fw.dataset == d, "nsub"].item()
        t_dof = nsub - 1
        ci.append(t.ppf(0.95, t_dof) / np.sqrt(nsub))
        v = df_fw.loc[df_fw.dataset == d, "smd"].item()
        if v > 0:
            p = df_fw.loc[df_fw.dataset == d, "p"].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        else:
            p = df_bk.loc[df_bk.dataset == d, "p"].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        _min = _min if (_min < (v - ci[-1])) else (v - ci[-1])
        _max = _max if (_max > (v + ci[-1])) else (v + ci[-1])
        ax.plot(
            np.array([v - ci[-1], v + ci[-1]]), np.ones((2,)) * (ind + 1), c=MOABB_SKY
        )
    _range = max(abs(_min), abs(_max)) * 1.25  # extra breathing room
    ax.set_xlim((0 - _range, 0 + _range))
    final_effect = combine_effects(df_fw["smd"], df_fw["nsub"])
    ax.scatter(
        pd.concat([pd.Series([final_effect]), df_fw["smd"]]),
        np.arange(len(dsets) + 1),
        s=np.array([50] + [30] * len(dsets)),
        marker="D",
        c=[MOABB_NAVY] + [MOABB_SKY] * len(dsets),
    )
    for i, p in zip(sig_ind, pvals):
        m, s = _marker(p)
        ax.scatter(df_fw["smd"].iloc[i], i + 1.4, s=s, marker=m, color=MOABB_CORAL)
    # pvalues axis stuf
    pval_ax.set_xlim([-0.1, 0.1])
    pval_ax.grid(False)
    pval_fontsize = FONT_SIZES["tick_label"] + 2
    pval_ax.set_title("p-value", fontdict={"fontsize": FONT_SIZES["annotation"] + 2})
    pval_ax.set_xticks([])
    for spine in pval_ax.spines.values():
        spine.set_visible(False)
    for ind, p in zip(sig_ind, pvals):
        pval_ax.text(
            0,
            ind + 1,
            horizontalalignment="center",
            verticalalignment="center",
            s="{:.2e}".format(p),
            fontsize=pval_fontsize,
            color=MOABB_DARK_TEXT,
        )
    if final_effect > 0:
        p = combine_pvalues(df_fw["p"], df_fw["nsub"])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s, marker=m, c=MOABB_CORAL)
            pval_ax.text(
                0,
                0,
                horizontalalignment="center",
                verticalalignment="center",
                s="{:.2e}".format(p),
                fontsize=pval_fontsize,
                color=MOABB_DARK_TEXT,
            )
    else:
        p = combine_pvalues(df_bk["p"], df_bk["nsub"])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s, marker=m, c=MOABB_CORAL)
            pval_ax.text(
                0,
                0,
                horizontalalignment="center",
                verticalalignment="center",
                s="{:.2e}".format(p),
                fontsize=pval_fontsize,
                color=MOABB_DARK_TEXT,
            )

    ax.axvline(0, linestyle="--", c=GRID_COLOR)
    ax.axhline(0.5, linestyle="-", linewidth=3, c=MOABB_NAVY)
    xaxis_fontsize = FONT_SIZES["axis_label"] + 3
    ax.set_xlabel("Standardized Mean Difference", fontsize=xaxis_fontsize)
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] + 2)

    apply_moabb_style(ax, title=f"{alg1} vs {alg2}", subtitle="", grid_axis="none")
    fig.subplots_adjust(top=0.85, bottom=0.16)

    # Draw comparison caption anchored to x=0 so "|" is exactly on the zero line.
    # Split into three text artists to keep center stable regardless of name length.
    y_caption = 0.99
    gap_pts = 8.0
    caption_fontsize = FONT_SIZES["subtitle"] - 1
    base_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    left_transform = base_transform + mtransforms.ScaledTranslation(
        -gap_pts / 72.0, 0.0, fig.dpi_scale_trans
    )
    right_transform = base_transform + mtransforms.ScaledTranslation(
        gap_pts / 72.0, 0.0, fig.dpi_scale_trans
    )

    ax.text(
        0,
        y_caption,
        f"< {alg2} better",
        fontsize=caption_fontsize,
        color=GRID_COLOR,
        ha="right",
        va="bottom",
        transform=left_transform,
        clip_on=False,
    )
    ax.text(
        0,
        y_caption,
        "|",
        fontsize=caption_fontsize + 1,
        color=GRID_COLOR,
        ha="center",
        va="bottom",
        transform=base_transform,
        clip_on=False,
    )
    ax.text(
        0,
        y_caption,
        f"{alg1} better >",
        fontsize=caption_fontsize,
        color=GRID_COLOR,
        ha="left",
        va="bottom",
        transform=right_transform,
        clip_on=False,
    )

    return fig


def _get_hexa_grid(n, diameter, center):
    x = np.arange(n) - n // 2 + np.random.rand()  # TODO
    y = np.arange(n) - n // 2 + np.random.rand()
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    return (
        np.concatenate([x, x + 0.5]) * diameter + center[0],
        np.concatenate([y, y + 0.5]) * diameter * np.sqrt(3) + center[1],
    )


def _get_bubble_coordinates(n, diameter, center):
    x, y = _get_hexa_grid(n, diameter, center)
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    dort_idx = dist.argsort()
    x = x[dort_idx]
    y = y[dort_idx]
    return x[:n], y[:n]


def _plot_shape(shape, *args, **kwargs):
    if shape == "circle":
        return Circle(*args, **kwargs)
    elif shape == "hexagon":
        return RegularPolygon(*args, numVertices=6, **kwargs)
    else:
        raise ValueError(f"Unknown shape {shape}")


def _plot_hexa_bubbles(
    *,
    n: int,
    diameter: float,
    center: tuple[float, float] = (0.0, 0.0),
    ax,
    shape: Literal["circle", "hexagon"] = "circle",
    gap: float = 0.0,
    gid: str | None = None,
    **kwargs,
):
    x, y = _get_bubble_coordinates(n, diameter + gap, center)
    bubbles = [
        _plot_shape(shape, (xi, yi), radius=diameter / 2, **kwargs)
        for xi, yi in zip(x, y)
    ]
    collection = PatchCollection(bubbles, match_original=True)
    if gid is not None:
        collection.set_gid(gid)
    ax.add_collection(collection)
    return x, y


def _add_bubble_legend(scale, size_mode, color_map, alphas, fontsize, shape, x0, y0, ax):
    circles = []  # (text, diameter, alpha, color)
    alpha = alphas[0]
    # sizes
    if size_mode == "count":
        sizes = [("100 trials", 100), ("1000 trials", 1000), ("10000 trials", 10000)]
    elif size_mode == "duration":
        sizes = [("6 minutes", 60 * 6), ("1 hour", 60 * 60), ("10 hours", 60 * 60 * 10)]
    else:
        raise ValueError(f"Unknown size_mode {size_mode}")
    for desc, size in sizes:
        circles.append((desc, np.log(size) * scale, alpha, "black"))
    circles.append(None)
    # colour
    for paradigm, c in color_map.items():
        circles.append((paradigm, np.log(1000) * scale, alpha, c))
    circles.append(None)
    # intensity
    circles.append(("1 session", np.log(1000) * scale, alphas[0], "black"))
    circles.append(("3 sessions", np.log(1000) * scale, alphas[2], "black"))
    circles.append(("5 sessions", np.log(1000) * scale, alphas[4], "black"))

    for i, item in enumerate(reversed(circles)):
        if item is None:
            continue
        text, diameter, alpha, color = item
        y = i * fontsize / 2 + y0
        bubble = _plot_shape(
            shape,
            (x0, y),
            radius=diameter / 2,
            alpha=alpha,
            color=color,
            lw=0,
            gid=f"legend/bubble/{text}",
        )
        ax.add_patch(bubble)
        ax.text(
            x0 + 5,
            y,
            text,
            ha="left",
            va="center",
            fontsize=fontsize,
            gid=f"legend/text/{text}",
        )


def _get_dataset_parameters(dataset):
    row = dataset._summary_table
    dataset_name = dataset.__class__.__name__
    paradigm = dataset.paradigm
    n_subjects = len(dataset.subject_list)
    n_sessions = _match_int(row["#Sessions"])
    n_trials = _compute_n_trials(row, paradigm)
    if n_trials is None:
        # Fallback for unparsable trial counts
        n_trials = _match_int(row.get("#Trials / class", "1"), default=1)
    trial_len = _match_float(row["Trials length (s)"])
    return (dataset_name, paradigm, n_subjects, n_sessions, n_trials, trial_len)


def get_bubble_size(size_mode, n_sessions, n_trials, trial_len):
    if size_mode == "duration":
        return n_trials * n_sessions * trial_len
    elif size_mode == "count":
        return n_trials * n_sessions
    else:
        raise ValueError(f"Unknown size_mode {size_mode}")


def get_dataset_area(
    n_subjects: int,
    n_sessions: int,
    n_trials: int,
    trial_len: float,
    scale: float = 0.5,
    size_mode: Literal["count", "duration"] = "count",
    gap: float = 0.0,
):
    size = get_bubble_size(
        size_mode=size_mode, n_sessions=n_sessions, n_trials=n_trials, trial_len=trial_len
    )
    diameter = np.log(size) * scale + gap
    return n_subjects * 3 * 3**0.5 / 8 * diameter**2  # area of hexagons


def dataset_bubble_plot(
    dataset=None,
    center: tuple[float, float] = (0.0, 0.0),
    scale: float = 0.5,
    size_mode: Literal["count", "duration"] = "count",
    shape: Literal["circle", "hexagon"] = "circle",
    gap: float = 0.0,
    color_map: dict[str, Any] | None = None,
    alphas: Sequence[float] | None = None,
    title: bool = True,
    legend: bool = True,
    legend_position: tuple[float, float] | None = None,
    fontsize: int = 8,
    ax=None,
    scale_ax: bool = True,
    dataset_name: str | None = None,
    paradigm: str | None = None,
    n_subjects: int | None = None,
    n_sessions: int | None = None,
    n_trials: int | None = None,
    trial_len: float | None = None,
):
    """Plot a bubble plot for a dataset.

    Each bubble represents one subject. The size of the bubble is
    proportional to the number of trials per subject on a log scale,
    the color represents the paradigm, and the alpha is proportional to
    the number of sessions.

    You may pass a :class:`moabb.datasets.base.BaseDataset` object
    via the ``dataset`` parameret, and all the characteristics of this dataset
    will be extracted automatically.
    Alternatively, if you want to plot a dataset not present in MOABB,
    you can directly pass the characteristics of the dataset via the
    ``dataset_name``, ``paradigm``, ``n_subjects``, ``n_sessions``,
    ``n_trials``, and ``trial_len`` parameters.
    If you pass both the dataset object and some parameters, the parameters
    passed will override the ones extracted from the dataset object.

    Parameters
    ----------
    dataset: :class:`~moabb.datasets.base.BaseDataset`
        Dataset to plot
    center: tuple[float, float]
        Coordinates of the center of the plot
    scale: float
        Scaling factor applied to the bubble sizes.
    size_mode: Literal["count", "duration"]
        Specifies how the size of the bubbles is calculated.
        Either "count" (number of trials) or "duration"
        (number of trials times trial duration).
    shape: Literal["circle", "hexagon"]
        Shape of the bubbles. Either "circle" or "hexagon".
    gap: float
        Gap between the bubbles.
    color_map: dict[str, Any] | None
        Dictionary that maps paradigms to colors. If None,
        the tab10 color map is used.
    alphas: Sequence[float] | None
        List of alpha values for the bubbles. If None, a default
        list is used.
    title: bool
        Whether to display the dataset title in the center of the plot.
    legend: bool
        Whether to display the legend.
    legend_position : tuple[float, float] or None
        Coordinates of the bottom left corner of the legend.
        Defaults to ``None``.
        If None, the legend is placed at the bottom right of the plot.
    fontsize: int
        Font size of the legend text.
    ax: matplotlib.axes.Axes | None
        Axes to plot on. If None, the default axes are used.
    scale_ax: bool
        Whether to scale the axes to be equal and in the correct range.
    dataset_name: str | None
        Name of the dataset. Required if ``dataset`` is None.
    paradigm: str | None
        Paradigm name. Required if ``dataset`` is None.
    n_subjects: int | None
        Number of subjects. Required if ``dataset`` is None.
    n_sessions: int | None
        Number of sessions. Required if ``dataset`` is None.
    n_trials: int | None
        Number of trials per session. Required if ``dataset`` is None.
    trial_len: float | None
        Duration of one trial, in seconds. Required if ``dataset`` is None.
    """
    p = sea.color_palette("tab10", 5)
    color_map = color_map or dict(zip(["imagery", "p300", "ssvep", "cvep", "rstate"], p))

    alphas = alphas or [0.8, 0.65, 0.5, 0.35, 0.2]

    if dataset is not None:
        _dataset_name, _paradigm, _n_subjects, _n_sessions, _n_trials, _trial_len = (
            _get_dataset_parameters(dataset)
        )
        dataset_name = dataset_name or _dataset_name
        paradigm = paradigm or _paradigm
        n_subjects = n_subjects or _n_subjects
        n_sessions = n_sessions or _n_sessions
        n_trials = n_trials or _n_trials
        trial_len = trial_len or _trial_len
    else:
        if any(
            x is None for x in [dataset_name, n_subjects, n_sessions, n_trials, trial_len]
        ):
            raise ValueError(
                "If dataset is None, then dataset_name, n_subjects, n_sessions, "
                "n_trials and trial_len must be provided"
            )
    size = get_bubble_size(
        size_mode=size_mode, n_sessions=n_sessions, n_trials=n_trials, trial_len=trial_len
    )

    ax = ax or plt.gca()
    x, y = _plot_hexa_bubbles(
        n=n_subjects,
        color=color_map[paradigm],
        ax=ax,
        diameter=np.log(size) * scale,
        alpha=alphas[min(n_sessions, len(alphas)) - 1],
        lw=0,
        center=center,
        shape=shape,
        gap=gap,
        gid=f"bubbles/{dataset_name}",
    )
    if title:
        ax.text(
            center[0],
            center[1],
            dataset_name,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="black",
            bbox={
                "facecolor": "white",
                "alpha": 0.6,
                "linewidth": 0,
                "boxstyle": "round,pad=0.5",
            },
            gid=f"title/{dataset_name}",
        )
        # bbox is better than path_effects as the text is not converted to a path.
        # we can still select it in a pdf. Also the file is lighter.
    if legend:
        legend_position = legend_position or (x.max() + fontsize, y.min())
        _add_bubble_legend(
            scale=scale,
            size_mode=size_mode,
            color_map=color_map,
            alphas=alphas,
            fontsize=fontsize,
            x0=legend_position[0],
            y0=legend_position[1],
            ax=ax,
            shape=shape,
        )
    ax.axis("off")
    if scale_ax:
        ax.axis("equal")
        ax.autoscale()
    return ax
