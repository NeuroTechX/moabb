from __future__ import annotations

import logging
import re
from typing import Any, Literal, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, RegularPolygon
from scipy.stats import t

from moabb.analysis.meta_analysis import (
    collapse_session_scores,
    combine_effects,
    combine_pvalues,
)


PIPELINE_PALETTE = sea.color_palette("husl", 6)
sea.set(font="serif", style="whitegrid", palette=PIPELINE_PALETTE, color_codes=False)

log = logging.getLogger(__name__)


def _simplify_names(x):
    if len(x) > 10:
        return x.split(" ")[0]
    else:
        return x


def score_plot(data, pipelines=None, orientation="vertical"):
    """Plot scores for all pipelines and all datasets

    Parameters
    ----------
    data: output of Results.to_dataframe()
        results on datasets
    pipelines: list of str | None
        pipelines to include in this plot
    orientation: str, default="vertical"
        plot orientation, could be ["vertical", "v", "horizontal", "h"]

    Returns
    -------
    fig: Figure
        Pyplot handle
    color_dict: dict
        Dictionary with the facecolor
    """
    data = collapse_session_scores(data)
    unique_ids = data["dataset"].apply(_simplify_names)
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Dataset names are too similar, turning off name shortening")
    else:
        data["dataset"] = unique_ids

    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]

    if orientation in ["horizontal", "h"]:
        y, x = "dataset", "score"
        fig = plt.figure(figsize=(8.5, 11))
    elif orientation in ["vertical", "v"]:
        x, y = "dataset", "score"
        fig = plt.figure(figsize=(11, 8.5))
    else:
        raise ValueError("Invalid plot orientation selected!")

    # markers = ['o', '8', 's', 'p', '+', 'x', 'D', 'd', '>', '<', '^']
    ax = fig.add_subplot(111)
    sea.stripplot(
        data=data,
        y=y,
        x=x,
        jitter=0.15,
        palette=PIPELINE_PALETTE,
        hue="pipeline",
        dodge=True,
        ax=ax,
        alpha=0.7,
    )
    if orientation in ["horizontal", "h"]:
        ax.set_xlim([0, 1])
        ax.axvline(0.5, linestyle="--", color="k", linewidth=2)
    else:
        ax.set_ylim([0, 1])
        ax.axhline(0.5, linestyle="--", color="k", linewidth=2)
    ax.set_title("Scores per dataset and algorithm")
    handles, labels = ax.get_legend_handles_labels()
    color_dict = {lb: h.get_facecolor()[0] for lb, h in zip(labels, handles)}
    plt.tight_layout()

    return fig, color_dict


def codecarbon_plot(data, order_list=None, pipelines=None, country=""):
    """Plot code carbon consume for the results from the benchmark.

    Parameters
    ----------
    data: output of Results.to_dataframe()
        results on datasets
    order_list: list of str | None
        order of pipelines to include in this plot
    pipelines: list of str | None
        pipelines to include in this plot
    country: str
        country to include in the title
    pipelines: list of str | None
        pipelines to include in this plot

    Returns
    -------
    fig: Figure
        Pyplot handle
    """
    data = collapse_session_scores(data)
    unique_ids = data["dataset"].apply(_simplify_names)
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Dataset names are too similar, turning off name shortening")
    else:
        data["dataset"] = unique_ids

    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]

    data = data.rename(columns={"carbon emission": "carbon_emission"})

    fig = sea.catplot(
        kind="bar",
        data=data,
        x="dataset",
        y="carbon_emission",
        hue="pipeline",
        palette=PIPELINE_PALETTE,
        height=8.5,
        hue_order=order_list,
    ).set(title=r"$CO_2$ emission per dataset and algorithm" + country)
    fig.set(yscale="log")
    fig.tight_layout()
    fig.set_ylabels(r"$CO_2$ emission (Log Scale)")
    fig.set_xlabels("Dataset")

    return fig


def paired_plot(data, alg1, alg2):
    """Generate a figure with a paired plot.

    Parameters
    ----------
    data: DataFrame
        dataframe obtained from evaluation
    alg1: str
        Name of a member of column data.pipeline
    alg2: str
        Name of a member of column data.pipeline

    Returns
    -------
    fig: Figure
        Pyplot handle
    """
    data = collapse_session_scores(data)
    data = data[data.pipeline.isin([alg1, alg2])]
    data = data.pivot_table(
        values="score", columns="pipeline", index=["subject", "dataset"]
    )
    data = data.reset_index()
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    data.plot.scatter(alg1, alg2, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", c="k")
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    return fig


def summary_plot(sig_df, effect_df, p_threshold=0.05, simplify=True):
    """Significance matrix to compare pipelines.

    Visualize significances as a heatmap with green/grey/red for significantly
    higher/significantly lower.

    Parameters
    ----------
    sig_df: DataFrame
        DataFrame of pipeline x pipeline where each value is a p-value,
    effect_df: DataFrame
        DataFrame where each value is an effect size

    Returns
    -------
    fig: Figure
        Pyplot handle
    """
    if simplify:
        effect_df.columns = effect_df.columns.map(_simplify_names)
        sig_df.columns = sig_df.columns.map(_simplify_names)
    annot_df = effect_df.copy()
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    palette = sea.light_palette("green", as_cmap=True)
    palette.set_under(color=[1, 1, 1])
    palette.set_over(color=[0.5, 0, 0])
    sea.heatmap(
        data=-np.log(sig_df),
        annot=annot_df,
        fmt="",
        cmap=palette,
        linewidths=1,
        linecolor="0.8",
        annot_kws={"size": 10},
        cbar=False,
        vmin=-np.log(0.05),
        vmax=-np.log(1e-100),
    )
    for lb in ax.get_xticklabels():
        lb.set_rotation(45)
    ax.tick_params(axis="y", rotation=0.9)
    ax.set_title("Algorithm comparison")
    plt.tight_layout()
    return fig


def meta_analysis_plot(stats_df, alg1, alg2):  # noqa: C901
    """Meta-analysis to compare two algorithms across several datasets.

    A meta-analysis style plot that shows the standardized effect with
    confidence intervals over all datasets for two algorithms.
    Hypothesis is that alg1 is larger than alg2

    Parameters
    ----------
    stats_df: DataFrame
        DataFrame generated by compute_dataset_statistics
    alg1: str
        Name of first pipeline
    alg2: str
        Name of second pipeline

    Returns
    -------
    fig: Figure
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

    assert alg1 in stats_df.pipe1.unique()
    assert alg2 in stats_df.pipe1.unique()
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
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 5)
    sig_ind = []
    pvals = []
    ax = fig.add_subplot(gs[0, :-1])
    ax.set_yticks(np.arange(len(dsets) + 1))
    if simplify:
        ax.set_yticklabels(["Meta-effect"] + [d for d in unique_ids])
    else:
        ax.set_yticklabels(["Meta-effect"] + [d for d in dsets])
    pval_ax = fig.add_subplot(gs[0, -1], sharey=ax)
    plt.setp(pval_ax.get_yticklabels(), visible=False)
    _min = 0
    _max = 0
    for ind, d in enumerate(dsets):
        nsub = float(df_fw.loc[df_fw.dataset == d, "nsub"])
        t_dof = nsub - 1
        ci.append(t.ppf(0.95, t_dof) / np.sqrt(nsub))
        v = float(df_fw.loc[df_fw.dataset == d, "smd"])
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
            np.array([v - ci[-1], v + ci[-1]]), np.ones((2,)) * (ind + 1), c="tab:grey"
        )
    _range = max(abs(_min), abs(_max))
    ax.set_xlim((0 - _range, 0 + _range))
    final_effect = combine_effects(df_fw["smd"], df_fw["nsub"])
    ax.scatter(
        pd.concat([pd.Series([final_effect]), df_fw["smd"]]),
        np.arange(len(dsets) + 1),
        s=np.array([50] + [30] * len(dsets)),
        marker="D",
        c=["k"] + ["tab:grey"] * len(dsets),
    )
    for i, p in zip(sig_ind, pvals):
        m, s = _marker(p)
        ax.scatter(df_fw["smd"].iloc[i], i + 1.4, s=s, marker=m, color="r")
    # pvalues axis stuf
    pval_ax.set_xlim([-0.1, 0.1])
    pval_ax.grid(False)
    pval_ax.set_title("p-value", fontdict={"fontsize": 10})
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
            fontsize=8,
        )
    if final_effect > 0:
        p = combine_pvalues(df_fw["p"], df_fw["nsub"])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s, marker=m, c="r")
            pval_ax.text(
                0,
                0,
                horizontalalignment="center",
                verticalalignment="center",
                s="{:.2e}".format(p),
                fontsize=8,
            )
    else:
        p = combine_pvalues(df_bk["p"], df_bk["nsub"])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s, marker=m, c="r")
            pval_ax.text(
                0,
                0,
                horizontalalignment="center",
                verticalalignment="center",
                s="{:.2e}".format(p),
                fontsize=8,
            )

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, linestyle="--", c="k")
    ax.axhline(0.5, linestyle="-", linewidth=3, c="k")
    title = "< {} better{}\n{}{} better >".format(
        alg2, " " * (45 - len(alg2)), " " * (45 - len(alg1)), alg1
    )
    ax.set_title(title, ha="left", ma="right", loc="left")
    ax.set_xlabel("Standardized Mean Difference")
    fig.tight_layout()

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


def _match_int(s):
    """Match the first integer in a string."""
    match = re.search(r"(\d+)", str(s))
    assert match, f"Cannot parse number from '{s}'"
    return int(match.group(1))


def _get_dataset_parameters(dataset):
    row = dataset._summary_table
    dataset_name = dataset.__class__.__name__
    paradigm = dataset.paradigm
    n_subjects = len(dataset.subject_list)
    n_sessions = _match_int(row["#Sessions"])
    if paradigm in ["imagery", "ssvep"]:
        n_trials = _match_int(row["#Trials / class"]) * _match_int(row["#Classes"])
    elif paradigm == "rstate":
        n_trials = _match_int(row["#Classes"]) * _match_int(row["#Blocks / class"])
    elif paradigm == "cvep":
        n_trials = _match_int(row["#Trials / class"]) * _match_int(row["#Trial classes"])
    else:  # p300
        match = re.search(r"(\d+) NT / (\d+) T", row["#Trials / class"])
        if match is not None:
            n_trials = int(match.group(1)) + int(match.group(2))
        else:
            n_trials = _match_int(row["#Trials / class"])
    trial_len = float(row["Trials length (s)"])
    return (
        dataset_name,
        paradigm,
        n_subjects,
        n_sessions,
        n_trials,
        trial_len,
    )


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
        size_mode=size_mode,
        n_sessions=n_sessions,
        n_trials=n_trials,
        trial_len=trial_len,
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
    dataset: Dataset
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
    legend_position: tuple[float, float] | None, default=None
        Coordinates of the bottom left corner of the legend.
        If None, the legend is placed at the bottom right of the plot.
    fontsize: int
        Font size of the legend text.
    ax: Axes | None
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
        size_mode=size_mode,
        n_sessions=n_sessions,
        n_trials=n_trials,
        trial_len=trial_len,
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
            bbox=dict(
                facecolor="white", alpha=0.6, linewidth=0, boxstyle="round,pad=0.5"
            ),
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
