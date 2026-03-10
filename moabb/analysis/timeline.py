"""Stimulus protocol timeline diagram generator.

Generates vectorized timeline diagrams from dataset metadata,
showing the temporal structure of experimental paradigms (P300 oddball,
motor imagery cue sequences, SSVEP flicker, c-VEP code stimulation, etc.).

Public API
----------
plot_stimulus_timeline(dataset, ...)
    Generate a matplotlib Figure with the stimulus protocol timeline.
stimulus_timeline_svg(dataset, ...)
    Convenience wrapper returning the SVG string.
extract_stimulus_timeline(dataset)
    Extract a ``StimulusTimeline`` data model from dataset metadata.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TimelinePhase:
    """A single phase in the stimulus timeline."""

    label: str
    onset_s: float
    duration_s: float
    style: str
    icon: str | None = None


@dataclass
class TimelineAnnotation:
    """A timing annotation (brace) spanning a range."""

    start_s: float
    end_s: float
    label: str


@dataclass
class StimulusTimeline:
    """Complete timeline description for a dataset's stimulus protocol."""

    paradigm: str
    dataset_name: str
    phases: list[TimelinePhase]
    annotations: list[TimelineAnnotation]
    total_duration_s: float
    is_approximate: bool = False
    notes: list[str] | None = None


# ---------------------------------------------------------------------------
# Style registry
# ---------------------------------------------------------------------------

# Base styles (paradigm-neutral for shared phases)
_PHASE_STYLES: dict[str, dict] = {
    "standard": {"facecolor": "#AAAAAA", "edgecolor": "#2C3E50"},
    "target": {"facecolor": "#555555", "edgecolor": "#2C3E50"},
    "fixation": {"facecolor": "#D9D9D9", "edgecolor": "#7F7F7F"},
    "cue": {"facecolor": "#F39C12", "edgecolor": "#2C3E50"},
    "imagery": {"facecolor": "#6495ED", "edgecolor": "#2C3E50"},
    "feedback": {"facecolor": "#9B59B6", "edgecolor": "#2C3E50"},
    "rest": {"facecolor": "#F0F0F0", "edgecolor": "#BDBDBD"},
    "flicker": {"facecolor": "#4CAF50", "edgecolor": "#2C3E50"},
    "code_stim": {"facecolor": "#26A69A", "edgecolor": "#2C3E50", "hatch": "||"},
    "eyes_open": {"facecolor": "#AED6F1", "edgecolor": "#2C3E50"},
    "eyes_closed": {"facecolor": "#2C3E50", "edgecolor": "#2C3E50"},
}

# Per-paradigm colour overrides — consistent with MOABB's existing palette:
#   imagery = blue, p300 = red, ssvep = green, cvep = teal, rstate = grey
_PARADIGM_STYLES: dict[str, dict[str, dict]] = {
    "p300": {
        "standard": {"facecolor": "#F8A0A0", "edgecolor": "#C0392B"},
        "target": {"facecolor": "#D32F2F", "edgecolor": "#8B0000"},
        "cue": {"facecolor": "#EF9A9A", "edgecolor": "#C0392B"},
    },
    "erp": {
        "standard": {"facecolor": "#F8A0A0", "edgecolor": "#C0392B"},
        "target": {"facecolor": "#D32F2F", "edgecolor": "#8B0000"},
        "cue": {"facecolor": "#EF9A9A", "edgecolor": "#C0392B"},
    },
    "imagery": {
        "cue": {"facecolor": "#90CAF9", "edgecolor": "#1565C0"},
        "imagery": {"facecolor": "#4285F4", "edgecolor": "#1565C0"},
        "feedback": {"facecolor": "#1E88E5", "edgecolor": "#0D47A1"},
        "fixation": {"facecolor": "#BBDEFB", "edgecolor": "#64B5F6"},
    },
    "ssvep": {
        "cue": {"facecolor": "#A5D6A7", "edgecolor": "#2E7D32"},
        "flicker": {"facecolor": "#4CAF50", "edgecolor": "#1B5E20"},
    },
    "cvep": {
        "cue": {"facecolor": "#80CBC4", "edgecolor": "#00695C"},
        "code_stim": {
            "facecolor": "#00897B",
            "edgecolor": "#004D40",
            "hatch": "||",
        },
        "feedback": {"facecolor": "#4DB6AC", "edgecolor": "#00695C"},
    },
    "rstate": {
        "eyes_open": {"facecolor": "#B0BEC5", "edgecolor": "#455A64"},
        "eyes_closed": {"facecolor": "#455A64", "edgecolor": "#263238"},
    },
}


def _get_phase_style(style_key: str, paradigm: str | None = None) -> dict:
    """Return the style dict for a phase, with paradigm-specific overrides."""
    if paradigm and paradigm in _PARADIGM_STYLES:
        override = _PARADIGM_STYLES[paradigm].get(style_key)
        if override:
            return override
    return _PHASE_STYLES.get(style_key, _PHASE_STYLES["standard"])


# ---------------------------------------------------------------------------
# Timing string parser
# ---------------------------------------------------------------------------


def _parse_timing_ms(value: str) -> float | None:
    """Parse a timing string into milliseconds.

    Handles formats like ``"125"``, ``"2s"``, ``"0s (green cross)"``,
    ``"1.5-3.5s (random)"`` (returns midpoint).

    Parameters
    ----------
    value : str
        Timing value to parse.

    Returns
    -------
    float or None
        Timing in **milliseconds**, or None if unparsable.
    """
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None

    # Try range pattern first: "1.5-3.5s (...)"
    m = re.match(r"([\d.]+)\s*-\s*([\d.]+)\s*s", value)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2.0 * 1000.0

    # "Xs (...)" or "Xs"
    m = re.match(r"([\d.]+)\s*s", value)
    if m:
        return float(m.group(1)) * 1000.0

    # Plain number (assumed milliseconds)
    m = re.match(r"([\d.]+)", value)
    if m:
        return float(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

_BLOCK_HEIGHT = 0.75
_TIMELINE_Y = 0.0


def _draw_phase_block(
    ax: Axes,
    x: float,
    width: float,
    label: str,
    style_key: str,
    y: float = _TIMELINE_Y,
    height: float = _BLOCK_HEIGHT,
    paradigm: str | None = None,
) -> None:
    """Draw a colored rectangle with a centred label."""
    style = _get_phase_style(style_key, paradigm)
    hatch = style.get("hatch", None)
    rect = mpatches.FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04",
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        linewidth=1.2,
        hatch=hatch,
        mutation_aspect=1,
    )
    # Prevent degenerate Q curves in SVG that cause miter-join "ear" artifacts
    rect.set_joinstyle("round")
    ax.add_patch(rect)
    # Label text colour: white on dark backgrounds
    dark_styles = {"eyes_closed", "code_stim", "target", "imagery", "feedback", "flicker"}
    color = "white" if style_key in dark_styles else "#2C3E50"
    fontsize = 9.5 if len(label) > 12 else 10.5
    ax.text(
        x + width / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        clip_on=True,
    )


def _draw_timeline_arrow(
    ax: Axes, x_start: float, x_end: float, y: float = _TIMELINE_Y
) -> None:
    """Draw a horizontal arrow representing the time axis."""
    y_arrow = y - _BLOCK_HEIGHT / 2 - 0.15
    ax.annotate(
        "",
        xy=(x_end, y_arrow),
        xytext=(x_start, y_arrow),
        arrowprops=dict(arrowstyle="->", color="#7F7F7F", lw=1.2),
    )
    ax.text(
        x_end,
        y_arrow - 0.08,
        "time",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#7F7F7F",
        fontstyle="italic",
    )


def _draw_timing_brace(
    ax: Axes,
    x_start: float,
    x_end: float,
    label: str,
    y: float = _TIMELINE_Y,
    level: int = 0,
) -> None:
    """Draw a brace annotation below the timeline."""
    y_base = y - _BLOCK_HEIGHT / 2 - 0.35 - level * 0.25
    mid = (x_start + x_end) / 2
    ax.annotate(
        "",
        xy=(x_start, y_base),
        xytext=(x_end, y_base),
        arrowprops=dict(arrowstyle="|-|", color="#555555", lw=0.8),
    )
    ax.text(
        mid,
        y_base - 0.08,
        label,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )


def _draw_ellipsis(ax: Axes, x: float, y: float = _TIMELINE_Y) -> None:
    """Draw '...' continuation marker."""
    ax.text(
        x,
        y,
        "...",
        ha="center",
        va="center",
        fontsize=16,
        color="#7F7F7F",
        fontweight="bold",
    )


def _draw_time_tick(ax: Axes, x: float, label: str, y: float = _TIMELINE_Y) -> None:
    """Draw a time tick mark below a phase."""
    y_tick = y - _BLOCK_HEIGHT / 2 - 0.05
    ax.plot([x, x], [y_tick, y_tick - 0.06], color="#7F7F7F", lw=0.8)
    ax.text(
        x,
        y_tick - 0.12,
        label,
        ha="center",
        va="top",
        fontsize=8,
        color="#7F7F7F",
    )


def _format_time(seconds: float) -> str:
    """Format seconds to a human-friendly label."""
    if seconds == 0:
        return "0s"
    if seconds < 0.1:
        return f"{seconds * 1000:.0f}ms"
    if seconds == int(seconds):
        return f"{int(seconds)}s"
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Per-paradigm extractors
# ---------------------------------------------------------------------------


def _get_metadata(dataset: BaseDataset):
    """Return dataset METADATA or None.

    Checks the class-level ``METADATA`` attribute first, then falls back
    to the catalog-backed ``dataset.metadata`` property.
    """
    meta = getattr(dataset, "METADATA", None) or getattr(type(dataset), "METADATA", None)
    if meta is not None:
        return meta
    # Fallback: catalog-backed .metadata property (e.g. ErpCore2021_*, etc.)
    return getattr(dataset, "metadata", None)


def _get_stim_pres(metadata) -> dict[str, str]:
    """Return stimulus_presentation dict, or empty dict."""
    if metadata is None:
        return {}
    exp = getattr(metadata, "experiment", None)
    if exp is None:
        return {}
    return exp.stimulus_presentation or {}


def _get_paradigm_specific(metadata):
    """Return ParadigmSpecificMetadata or None."""
    if metadata is None:
        return None
    return getattr(metadata, "paradigm_specific", None)


# -- P300 / ERP --


def _extract_p300_timeline(metadata, dataset: BaseDataset) -> StimulusTimeline | None:
    ps = _get_paradigm_specific(metadata)
    sp = _get_stim_pres(metadata)

    isi_ms = None
    soa_ms = None
    stim_dur_ms = None

    if ps is not None:
        isi_ms = getattr(ps, "isi_ms", None)
        soa_ms = getattr(ps, "soa_ms", None)

    # Handle dict values (e.g. Kojima2024B has per-condition SOA)
    if isinstance(isi_ms, dict):
        isi_ms = next(iter(isi_ms.values()), None)
    if isinstance(soa_ms, dict):
        soa_ms = next(iter(soa_ms.values()), None)

    if sp:
        if isi_ms is None:
            isi_ms = _parse_timing_ms(sp.get("isi_ms", "") or sp.get("isi", ""))
        if soa_ms is None:
            soa_ms = _parse_timing_ms(sp.get("soa_ms", ""))
        if stim_dur_ms is None:
            stim_dur_ms = _parse_timing_ms(
                sp.get("stimulus_duration_ms", "") or sp.get("flash_duration", "")
            )

    # Fix inconsistent metadata: if SOA < ISI the SOA field is actually
    # the stimulus duration (e.g. EPFLP300: soa_ms=100, isi_ms=400).
    if soa_ms and isi_ms and soa_ms < isi_ms and stim_dur_ms is None:
        stim_dur_ms = soa_ms
        soa_ms = None

    # Derive missing values
    if soa_ms and stim_dur_ms and isi_ms is None:
        isi_ms = soa_ms - stim_dur_ms
    if soa_ms and isi_ms and stim_dur_ms is None:
        stim_dur_ms = soa_ms - isi_ms
    if stim_dur_ms and isi_ms and soa_ms is None:
        soa_ms = stim_dur_ms + isi_ms

    # Need at least one timing value to build a proper timeline
    if soa_ms is None and stim_dur_ms is None and isi_ms is None:
        return _extract_p300_fallback(metadata, dataset)

    if stim_dur_ms is None:
        stim_dur_ms = 100.0  # default
    if isi_ms is None:
        isi_ms = 0.0
    if soa_ms is None:
        soa_ms = stim_dur_ms + isi_ms

    # Clamp to non-negative
    stim_dur_ms = max(stim_dur_ms, 10.0)
    isi_ms = max(isi_ms, 0.0)
    soa_ms = max(soa_ms, stim_dur_ms)

    stim_dur_s = stim_dur_ms / 1000.0
    isi_s = isi_ms / 1000.0
    soa_s = soa_ms / 1000.0

    # Build a representative sequence: Std Std Std ... TARGET Std Std
    phases: list[TimelinePhase] = []
    annotations: list[TimelineAnnotation] = []

    labels = ["Std", "Std", "Std", "Target", "Std", "Std"]
    for lbl in labels:
        style = "target" if lbl == "Target" else "standard"
        phases.append(TimelinePhase(lbl, 0, stim_dur_s, style))

    # ISI / SOA annotations
    if isi_s > 0:
        annotations.append(TimelineAnnotation(0, 0, f"ISI = {_format_time(isi_s)}"))
    annotations.append(TimelineAnnotation(0, 0, f"SOA = {_format_time(soa_s)}"))

    notes = []
    n_targets = getattr(ps, "n_targets", None) if ps else None
    if n_targets:
        notes.append(f"{n_targets} targets")

    return StimulusTimeline(
        paradigm="p300",
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=annotations,
        total_duration_s=soa_s * 6,
        notes=notes or None,
    )


def _extract_p300_fallback(metadata, dataset: BaseDataset) -> StimulusTimeline:
    """P300 fallback when timing metadata is missing — show Target/NonTarget."""
    event_id = getattr(dataset, "event_id", None) or {}
    interval = getattr(dataset, "interval", None) or [0, 1]
    dur = float(interval[1] - interval[0])

    events = list(event_id.keys())
    has_target = any("target" in e.lower() for e in events)

    phases: list[TimelinePhase] = []
    if has_target:
        phases.append(TimelinePhase("Std", 0, dur, "standard"))
        phases.append(TimelinePhase("Std", 0, dur, "standard"))
        phases.append(TimelinePhase("Std", 0, dur, "standard"))
        phases.append(TimelinePhase("Target", 0, dur, "target"))
        phases.append(TimelinePhase("Std", 0, dur, "standard"))
    else:
        for ev in events[:3]:
            phases.append(TimelinePhase(ev, 0, dur, "standard"))

    notes = []
    ps = _get_paradigm_specific(metadata)
    n_targets = getattr(ps, "n_targets", None) if ps else None
    if n_targets:
        notes.append(f"{n_targets} targets")

    return StimulusTimeline(
        paradigm="p300",
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=[],
        total_duration_s=dur * len(phases),
        is_approximate=True,
        notes=notes or None,
    )


# -- Motor Imagery --


def _extract_mi_timeline(metadata, dataset: BaseDataset) -> StimulusTimeline | None:
    ps = _get_paradigm_specific(metadata)
    sp = _get_stim_pres(metadata)

    cross_onset_s = 0.0
    beep_s: float | None = None
    cue_onset_s: float | None = None
    cue_dur_s: float | None = None
    imagery_dur_s: float | None = None
    feedback_onset_s: float | None = None
    feedback_dur_s: float | None = None
    trial_dur_s: float | None = None

    if ps is not None:
        cue_dur_s = getattr(ps, "cue_duration_s", None)
        imagery_dur_s = getattr(ps, "imagery_duration_s", None)

    if sp:
        v = _parse_timing_ms(sp.get("cross_onset", ""))
        if v is not None:
            cross_onset_s = v / 1000.0
        v = _parse_timing_ms(sp.get("acoustic_signal", ""))
        if v is not None:
            beep_s = v / 1000.0
        v = _parse_timing_ms(sp.get("arrow_cue", ""))
        if v is not None:
            cue_onset_s = v / 1000.0
        v = _parse_timing_ms(sp.get("feedback_onset", ""))
        if v is not None:
            feedback_onset_s = v / 1000.0
        v = _parse_timing_ms(sp.get("feedback_duration", ""))
        if v is not None:
            feedback_dur_s = v / 1000.0
        v = _parse_timing_ms(sp.get("trial_duration", ""))
        if v is not None:
            trial_dur_s = v / 1000.0

    # Build from what we have; fall back to dataset.interval
    exp = getattr(metadata, "experiment", None) if metadata else None
    if trial_dur_s is None and exp is not None:
        trial_dur_s = getattr(exp, "trial_duration", None)
    interval = getattr(dataset, "interval", None)
    if trial_dur_s is None and interval is not None:
        trial_dur_s = float(interval[1] - interval[0])

    if trial_dur_s is None:
        return None

    phases: list[TimelinePhase] = []
    annotations: list[TimelineAnnotation] = []

    # Fixation cross — use explicit None check so 0.0 is honoured as a valid onset
    fix_end = (
        beep_s
        if beep_s is not None
        else (cue_onset_s if cue_onset_s is not None else 2.0)
    )
    phases.append(
        TimelinePhase(
            "Fixation +", cross_onset_s, fix_end - cross_onset_s, "fixation", "cross"
        )
    )

    # Cue arrow
    if cue_onset_s is not None:
        if cue_dur_s is None:
            cue_dur_s = (feedback_onset_s - cue_onset_s) if feedback_onset_s else 1.25
        phases.append(TimelinePhase("Cue", cue_onset_s, cue_dur_s, "cue", "arrow_left"))

    # Motor imagery / feedback
    if feedback_onset_s is not None:
        if feedback_dur_s is None:
            feedback_dur_s = trial_dur_s - feedback_onset_s
        phases.append(
            TimelinePhase("Motor Imagery", feedback_onset_s, feedback_dur_s, "imagery")
        )
    elif cue_onset_s is not None and cue_dur_s is not None:
        img_onset = cue_onset_s + cue_dur_s
        img_dur = imagery_dur_s or (trial_dur_s - img_onset)
        phases.append(TimelinePhase("Motor Imagery", img_onset, img_dur, "imagery"))
    else:
        # Simple: just event interval
        img_onset = fix_end
        phases.append(
            TimelinePhase("Motor Imagery", img_onset, trial_dur_s - img_onset, "imagery")
        )

    # Rest at end if there's remaining time
    last = phases[-1]
    end_of_last = last.onset_s + last.duration_s
    if trial_dur_s - end_of_last > 0.1:
        phases.append(
            TimelinePhase("Rest", end_of_last, trial_dur_s - end_of_last, "rest")
        )

    # Time ticks as annotations
    for p in phases:
        annotations.append(
            TimelineAnnotation(p.onset_s, p.onset_s, _format_time(p.onset_s))
        )
    annotations.append(
        TimelineAnnotation(trial_dur_s, trial_dur_s, _format_time(trial_dur_s))
    )

    return StimulusTimeline(
        paradigm="imagery",
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=annotations,
        total_duration_s=trial_dur_s,
    )


# -- SSVEP --


def _extract_ssvep_timeline(metadata, dataset: BaseDataset) -> StimulusTimeline | None:
    ps = _get_paradigm_specific(metadata)
    sp = _get_stim_pres(metadata)
    exp = getattr(metadata, "experiment", None) if metadata else None

    cue_dur_s = getattr(ps, "cue_duration_s", None) if ps else None
    freqs = getattr(ps, "stimulus_frequencies_hz", None) if ps else None
    trial_dur_s = getattr(exp, "trial_duration", None) if exp else None
    interval = getattr(dataset, "interval", None)

    if trial_dur_s is None and interval is not None:
        trial_dur_s = float(interval[1] - interval[0])
    if trial_dur_s is None:
        return None

    if cue_dur_s is None:
        cue_dur_s = 0.5  # common default

    phases: list[TimelinePhase] = []
    annotations: list[TimelineAnnotation] = []

    # Cue phase
    phases.append(TimelinePhase("Cue", 0, cue_dur_s, "cue"))

    # Flicker stimulation phase
    flicker_dur = trial_dur_s - cue_dur_s
    rest_dur = 0.5 if flicker_dur > 1.0 else 0.0
    flicker_dur -= rest_dur

    freq_label = "Flickering Stimulus"
    phases.append(TimelinePhase(freq_label, cue_dur_s, flicker_dur, "flicker"))

    # Rest
    if rest_dur > 0:
        phases.append(TimelinePhase("Rest", cue_dur_s + flicker_dur, rest_dur, "rest"))

    # Frequency info (notes only, no brace — avoids duplication)
    notes = []
    if freqs:
        if len(freqs) <= 6:
            freq_str = ", ".join(f"{f:.1f}" for f in freqs)
        else:
            freq_str = f"{freqs[0]:.1f}\u2013{freqs[-1]:.1f} Hz ({len(freqs)} targets)"
        notes.append(f"Frequencies: {freq_str}")

    # Display info
    display = sp.get("display", "")
    if display:
        notes.append(f"Display: {display}")

    return StimulusTimeline(
        paradigm="ssvep",
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=annotations,
        total_duration_s=trial_dur_s,
        notes=notes or None,
    )


# -- c-VEP --


def _extract_cvep_timeline(metadata, dataset: BaseDataset) -> StimulusTimeline | None:
    ps = _get_paradigm_specific(metadata)
    exp = getattr(metadata, "experiment", None) if metadata else None

    code_type = getattr(ps, "code_type", None) if ps else None
    code_length = getattr(ps, "code_length", None) if ps else None
    cue_dur_s = getattr(ps, "cue_duration_s", None) if ps else None
    stim_freqs = getattr(ps, "stimulus_frequencies_hz", None) if ps else None
    trial_dur_s = getattr(exp, "trial_duration", None) if exp else None
    interval = getattr(dataset, "interval", None)

    if trial_dur_s is None and interval is not None:
        trial_dur_s = float(interval[1] - interval[0])
    if trial_dur_s is None:
        return None

    if cue_dur_s is None:
        cue_dur_s = 0.5

    phases: list[TimelinePhase] = []
    annotations: list[TimelineAnnotation] = []
    notes: list[str] = []

    # Cue
    phases.append(TimelinePhase("Cue", 0, cue_dur_s, "cue"))

    # Code stimulation block
    feedback_dur = 0.5  # assume short feedback at end
    stim_dur = trial_dur_s - cue_dur_s - feedback_dur
    if stim_dur < 0.5:
        stim_dur = trial_dur_s - cue_dur_s
        feedback_dur = 0

    stim_label = "Code Stimulation"
    phases.append(TimelinePhase(stim_label, cue_dur_s, stim_dur, "code_stim"))

    if feedback_dur > 0:
        phases.append(
            TimelinePhase("Feedback", cue_dur_s + stim_dur, feedback_dur, "feedback")
        )

    # Code details annotation
    details = []
    if code_type:
        details.append(code_type)
        notes.append(f"Code type: {code_type}")
    if code_length:
        details.append(f"{code_length} bits")
        notes.append(f"Code length: {code_length}")
    rate_hz = stim_freqs[0] if stim_freqs else None
    if rate_hz:
        details.append(f"{rate_hz:.0f} Hz")
        notes.append(f"Presentation rate: {rate_hz:.0f} Hz")
        if code_length:
            cycle_dur = code_length / rate_hz
            notes.append(f"Cycle duration: {_format_time(cycle_dur)}")

    if details:
        annotations.append(
            TimelineAnnotation(
                cue_dur_s,
                cue_dur_s + stim_dur,
                " | ".join(details),
            )
        )

    return StimulusTimeline(
        paradigm="cvep",
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=annotations,
        total_duration_s=trial_dur_s,
        notes=notes or None,
    )


# -- Resting State --


def _extract_rstate_timeline(metadata, dataset: BaseDataset) -> StimulusTimeline | None:
    event_id = getattr(dataset, "event_id", None)
    interval = getattr(dataset, "interval", None)
    if event_id is None or interval is None:
        return None

    block_dur = float(interval[1] - interval[0])
    phases: list[TimelinePhase] = []
    annotations: list[TimelineAnnotation] = []

    events = list(event_id.keys())
    # Determine eye-related events
    has_open = any("open" in e.lower() for e in events)
    has_closed = any("closed" in e.lower() or "close" in e.lower() for e in events)

    if has_open and has_closed:
        # Alternating eyes open/closed
        for i in range(3):
            onset = i * block_dur
            if i % 2 == 0:
                phases.append(TimelinePhase("Eyes Open", onset, block_dur, "eyes_open"))
            else:
                phases.append(
                    TimelinePhase("Eyes Closed", onset, block_dur, "eyes_closed")
                )
        total = 3 * block_dur
    else:
        # Generic resting state events
        for i, ev in enumerate(events[:3]):
            onset = i * block_dur
            style = "rest" if i % 2 == 0 else "fixation"
            phases.append(TimelinePhase(ev, onset, block_dur, style))
        total = len(phases) * block_dur

    return StimulusTimeline(
        paradigm="rstate",
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=annotations,
        total_duration_s=total,
        notes=[f"Block duration: {_format_time(block_dur)}"],
    )


# -- Generic fallback --


def _extract_generic_timeline(dataset: BaseDataset) -> StimulusTimeline:
    interval = getattr(dataset, "interval", None) or [0, 1]
    event_id = getattr(dataset, "event_id", None) or {}
    tmin, tmax = float(interval[0]), float(interval[1])
    dur = tmax - tmin

    events = list(event_id.keys())
    label = " / ".join(events[:3]) if events else "Event"
    if len(events) > 3:
        label += " / \u2026"

    phases = [TimelinePhase(label, 0, dur, "standard")]
    annotations = [
        TimelineAnnotation(0, dur, f"interval = [{tmin}, {tmax}]s"),
    ]

    return StimulusTimeline(
        paradigm=getattr(dataset, "paradigm", "unknown"),
        dataset_name=_dataset_name(dataset),
        phases=phases,
        annotations=annotations,
        total_duration_s=dur,
        is_approximate=True,
    )


# ---------------------------------------------------------------------------
# Dispatcher tables
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, callable] = {
    "p300": _extract_p300_timeline,
    "erp": _extract_p300_timeline,
    "imagery": _extract_mi_timeline,
    "ssvep": _extract_ssvep_timeline,
    "cvep": _extract_cvep_timeline,
    "rstate": _extract_rstate_timeline,
}


def _dataset_name(dataset: BaseDataset) -> str:
    code = getattr(dataset, "code", None)
    if code:
        return code
    return type(dataset).__name__


# ---------------------------------------------------------------------------
# Per-paradigm renderers
# ---------------------------------------------------------------------------


def _render_p300(ax: Axes, timeline: StimulusTimeline) -> None:
    """Render P300 oddball sequence with visual spacing."""
    phases = timeline.phases
    if not phases:
        return

    # Use fixed visual dimensions, not raw time coordinates
    block_w = 0.65
    block_gap = 0.2  # gap between blocks within a group
    ellipsis_gap = 0.5  # wider gap for the "..." break

    # Determine which phases come before/after the ellipsis
    has_ellipsis = any(p.style == "standard" for p in phases) and any(
        p.style == "target" for p in phases
    )

    # Build visual layout: [Std] [Std] [Std] ... [Target] [Std] [Std]
    # We split the phases into pre-ellipsis and post-ellipsis groups
    # by finding the first target
    target_idx = next(
        (i for i, p in enumerate(phases) if p.style == "target"), len(phases) // 2
    )
    pre_group = phases[:target_idx]
    post_group = phases[target_idx:]

    drawn: list[tuple[float, TimelinePhase]] = []
    x = 0.0

    # Draw pre-ellipsis group
    for p in pre_group:
        _draw_phase_block(ax, x, block_w, p.label, p.style, paradigm=timeline.paradigm)
        drawn.append((x, p))
        x += block_w + block_gap

    # Draw ellipsis
    if has_ellipsis and pre_group and post_group:
        ellipsis_x = x + ellipsis_gap / 2 - 0.1
        _draw_ellipsis(ax, ellipsis_x)
        x += ellipsis_gap

    # Draw post-ellipsis group
    for p in post_group:
        _draw_phase_block(ax, x, block_w, p.label, p.style, paradigm=timeline.paradigm)
        drawn.append((x, p))
        x += block_w + block_gap

    total_x = x - block_gap  # remove trailing gap

    # Timeline arrow
    _draw_timeline_arrow(ax, -0.1, total_x + 0.3)

    # ISI and SOA annotations between the first two blocks
    if len(drawn) >= 2 and timeline.annotations:
        p0_x = drawn[0][0]
        p1_x = drawn[1][0]
        for ann in timeline.annotations:
            if "ISI" in ann.label:
                _draw_timing_brace(ax, p0_x + block_w, p1_x, ann.label, level=0)
            elif "SOA" in ann.label:
                _draw_timing_brace(ax, p0_x, p1_x, ann.label, level=1)

    ax.set_xlim(-0.2, total_x + 0.5)
    ax.set_ylim(-1.0, 0.6)


def _render_scaled_phases(
    ax: Axes,
    timeline: StimulusTimeline,
    *,
    show_braces: bool = False,
    show_ellipsis_end: bool = False,
    min_block_w: float = 0.45,
) -> None:
    """Shared renderer for phase-based timelines (MI, SSVEP, c-VEP, rstate).

    Handles minimum block widths and prevents tick label overlap.
    """
    total = timeline.total_duration_s
    phases = timeline.phases
    if not phases or total <= 0:
        return

    # Compute scale mapping total duration to a fixed visual width.
    scale = 6.0 / total

    # Place phases with enforced minimum width
    x = 0.0
    tick_positions: list[tuple[float, str]] = []
    for p in phases:
        w = max(p.duration_s * scale, min_block_w)
        _draw_phase_block(ax, x, w, p.label, p.style, paradigm=timeline.paradigm)
        tick_positions.append((x, _format_time(p.onset_s)))
        x += w
    tick_positions.append((x, _format_time(total)))
    total_x = x

    # Draw ticks, skipping overlaps (need at least 0.5 visual units apart)
    last_tick_x = -999.0
    for tx, tlabel in tick_positions:
        if tx - last_tick_x >= 0.45:
            _draw_time_tick(ax, tx, tlabel)
            last_tick_x = tx

    # Braces
    if show_braces:
        # Map annotation times to visual x positions
        cum_x = [0.0]
        for p in phases:
            w = max(p.duration_s * scale, min_block_w)
            cum_x.append(cum_x[-1] + w)
        for ann in timeline.annotations:
            # Find the visual x for start_s and end_s
            x_start = _time_to_visual_x(ann.start_s, phases, cum_x, scale, min_block_w)
            x_end = _time_to_visual_x(ann.end_s, phases, cum_x, scale, min_block_w)
            _draw_timing_brace(ax, x_start, x_end, ann.label)

    if show_ellipsis_end:
        _draw_ellipsis(ax, total_x + 0.25)
        _draw_timeline_arrow(ax, -0.1, total_x + 0.5)
        ax.set_xlim(-0.2, total_x + 0.7)
    else:
        _draw_timeline_arrow(ax, -0.1, total_x + 0.3)
        ax.set_xlim(-0.2, total_x + 0.5)

    y_low = -0.9 if (show_braces and timeline.annotations) else -0.7
    ax.set_ylim(y_low, 0.6)


def _time_to_visual_x(
    t: float,
    phases: list[TimelinePhase],
    cum_x: list[float],
    scale: float,
    min_block_w: float,
) -> float:
    """Map a time value to the visual x coordinate."""
    running_t = 0.0
    for i, p in enumerate(phases):
        w = max(p.duration_s * scale, min_block_w)
        if t <= running_t + p.duration_s:
            frac = (t - running_t) / p.duration_s if p.duration_s > 0 else 0
            return cum_x[i] + frac * w
        running_t += p.duration_s
    return cum_x[-1]


def _render_mi(ax: Axes, timeline: StimulusTimeline) -> None:
    """Render motor imagery trial phases."""
    _render_scaled_phases(ax, timeline)


def _render_ssvep(ax: Axes, timeline: StimulusTimeline) -> None:
    """Render SSVEP trial phases with frequency annotation."""
    _render_scaled_phases(ax, timeline, show_braces=True)


def _render_cvep(ax: Axes, timeline: StimulusTimeline) -> None:
    """Render c-VEP trial phases with code detail annotation."""
    _render_scaled_phases(ax, timeline, show_braces=True)


def _render_rstate(ax: Axes, timeline: StimulusTimeline) -> None:
    """Render resting state alternating blocks."""
    _render_scaled_phases(ax, timeline, show_ellipsis_end=True)


def _render_generic(ax: Axes, timeline: StimulusTimeline) -> None:
    """Render generic interval visualization."""
    total = timeline.total_duration_s
    scale = 6.0 / total if total > 0 else 1.0

    for p in timeline.phases:
        x = p.onset_s * scale
        w = p.duration_s * scale
        _draw_phase_block(ax, x, w, p.label, p.style, paradigm=timeline.paradigm)

    for ann in timeline.annotations:
        _draw_timing_brace(ax, ann.start_s * scale, ann.end_s * scale, ann.label)

    _draw_timeline_arrow(ax, -0.1, total * scale + 0.2)

    ax.set_xlim(-0.2, total * scale + 0.4)
    ax.set_ylim(-0.9, 0.6)


_RENDERERS: dict[str, callable] = {
    "p300": _render_p300,
    "erp": _render_p300,
    "imagery": _render_mi,
    "ssvep": _render_ssvep,
    "cvep": _render_cvep,
    "rstate": _render_rstate,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_stimulus_timeline(dataset: BaseDataset) -> StimulusTimeline:
    """Extract a normalised timeline from dataset metadata.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.

    Returns
    -------
    StimulusTimeline
        Structured timeline description.
    """
    paradigm = getattr(dataset, "paradigm", "unknown")
    metadata = _get_metadata(dataset)

    extractor = _EXTRACTORS.get(paradigm)
    timeline = None
    if extractor is not None:
        timeline = extractor(metadata, dataset)

    if timeline is None:
        timeline = _extract_generic_timeline(dataset)

    return timeline


def plot_stimulus_timeline(
    dataset: BaseDataset,
    *,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show_annotations: bool = True,
    title: str | None = None,
) -> Figure:
    """Generate a stimulus protocol timeline diagram for a dataset.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.  Default is ``(10, 2.5)``.
    ax : matplotlib Axes, optional
        If provided, draw into this axes instead of creating a new figure.
    show_annotations : bool
        Whether to draw timing annotations (ISI, SOA braces).
    title : str, optional
        Custom title.  Defaults to ``"<dataset_name> — Stimulus Protocol"``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the timeline diagram.
    """
    timeline = extract_stimulus_timeline(dataset)

    if figsize is None:
        figsize = (10, 2.5)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    renderer = _RENDERERS.get(timeline.paradigm, _render_generic)

    # Temporarily remove annotations if not wanted
    saved_annotations = timeline.annotations
    if not show_annotations:
        timeline.annotations = []

    renderer(ax, timeline)

    timeline.annotations = saved_annotations

    # Title
    if title is None:
        approx = " (approximate)" if timeline.is_approximate else ""
        title = f"{timeline.dataset_name} \u2014 Stimulus Protocol{approx}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Notes below the plot
    if timeline.notes:
        note_text = "  |  ".join(timeline.notes)
        ax.text(
            0.5,
            -0.15,
            note_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#7F7F7F",
            fontstyle="italic",
        )

    # Legend for P300 abbreviations (only when annotations are visible)
    has_soa = any("SOA" in a.label for a in timeline.annotations)
    has_isi = any("ISI" in a.label for a in timeline.annotations)
    if show_annotations and (has_soa or has_isi):
        legend_lines = []
        if has_soa:
            legend_lines.append("SOA: Stimulus Onset Asynchrony")
        if has_isi:
            legend_lines.append("ISI: Inter-Stimulus Interval")
        ax.text(
            1.0,
            -0.15,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#999999",
            fontstyle="italic",
        )

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def stimulus_timeline_svg(dataset: BaseDataset, **kwargs) -> str:
    """Return an SVG string of the stimulus protocol timeline.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    **kwargs
        Passed through to :func:`plot_stimulus_timeline`.

    Returns
    -------
    str
        SVG markup string.
    """
    fig = plot_stimulus_timeline(dataset, **kwargs)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Class balance chart
# ---------------------------------------------------------------------------

# Sequential blue palette for bars
_BAR_COLORS = [
    "#1565C0",
    "#1976D2",
    "#1E88E5",
    "#2196F3",
    "#42A5F5",
    "#64B5F6",
    "#90CAF9",
    "#BBDEFB",
]


def _normalize_class_label(label: str) -> str:
    """Return a normalized class label for robust metadata matching."""
    return re.sub(r"[^a-z0-9]+", "", str(label).strip().lower())


def plot_class_balance(
    dataset: BaseDataset,
    *,
    figsize: tuple[float, float] | None = None,
) -> Figure | None:
    """Generate a horizontal bar chart showing trial counts per class.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    figsize : tuple of float, optional
        Figure size. Default ``(6, 2.5)``.

    Returns
    -------
    Figure or None
        The figure, or None if insufficient data.
    """
    event_id = getattr(dataset, "event_id", None) or {}
    if not event_id:
        return None

    metadata = _get_metadata(dataset)

    class_names = list(event_id.keys())
    n_classes = len(class_names)

    # Try to get trial counts from metadata
    trials_per_class = None
    if metadata is not None:
        ds = getattr(metadata, "data_structure", None)
        if ds is not None:
            trials_per_class = getattr(ds, "n_trials_per_class", None)

    has_counts = isinstance(trials_per_class, dict) and bool(trials_per_class)

    if figsize is None:
        h = max(1.8, 0.45 * n_classes + 0.8)
        figsize = (6, h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if has_counts:
        # Normalize metadata keys to handle variants like
        # "NonTarget" vs "non-target".
        normalized_counts = {
            _normalize_class_label(key): value for key, value in trials_per_class.items()
        }

        # Map class names to counts
        counts = []
        for cn in class_names:
            count = trials_per_class.get(cn)
            if count is None:
                count = normalized_counts.get(_normalize_class_label(cn), 0)
            counts.append(count)

        # If most lookups resulted in zeros despite trials_per_class being
        # non-empty, the class labels didn't match well enough — fall back
        # to the "no counts" display instead of showing a misleading chart.
        n_matched = sum(1 for c in counts if c > 0)
        if trials_per_class and n_matched < len(counts) / 2:
            has_counts = False

    if has_counts:
        colors = [_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(n_classes)]
        y_pos = range(n_classes)
        bars = ax.barh(
            y_pos, counts, color=colors, edgecolor="#1565C0", linewidth=0.6, height=0.6
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(class_names, fontsize=10, fontweight="500")
        ax.invert_yaxis()

        # Count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + max(counts) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                ha="left",
                fontsize=9,
                color="#555555",
            )

        # Balance annotation
        if len(set(counts)) == 1:
            ax.set_title(
                f"Balanced: {counts[0]} trials/class",
                fontsize=10,
                fontweight="bold",
                color="#2E7D32",
                pad=8,
            )
        else:
            ax.set_title(
                "Trial counts per class",
                fontsize=10,
                fontweight="bold",
                color="#555555",
                pad=8,
            )

        ax.set_xlabel("Trials", fontsize=9, color="#7F7F7F")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.spines["left"].set_color("#CCCCCC")
        ax.tick_params(axis="x", colors="#999999", labelsize=8)

    else:
        # No counts — show class names only
        y_pos = range(n_classes)
        colors = [_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(n_classes)]
        ax.barh(
            y_pos,
            [1] * n_classes,
            color=colors,
            edgecolor="#1565C0",
            linewidth=0.6,
            height=0.6,
            alpha=0.4,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(class_names, fontsize=10, fontweight="500")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_title(
            "Classes (counts vary by subject)",
            fontsize=10,
            fontweight="bold",
            color="#999999",
            pad=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")

    fig.tight_layout()
    return fig


def class_balance_svg(dataset: BaseDataset, **kwargs) -> str | None:
    """Return an SVG string of the class balance chart.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.

    Returns
    -------
    str or None
        SVG markup, or None if no data.
    """
    fig = plot_class_balance(dataset, **kwargs)
    if fig is None:
        return None
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session structure diagram
# ---------------------------------------------------------------------------


def plot_session_structure(
    dataset: BaseDataset,
    *,
    figsize: tuple[float, float] | None = None,
) -> Figure | None:
    """Generate a small-multiples diagram showing session/run layout.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    figsize : tuple of float, optional
        Figure size. Default auto-calculated.

    Returns
    -------
    Figure or None
        The figure, or None if insufficient data.
    """
    n_sessions = getattr(dataset, "n_sessions", None)
    if n_sessions is None:
        return None

    metadata = _get_metadata(dataset)
    runs_per_session = None
    if metadata is not None:
        runs_per_session = getattr(metadata, "runs_per_session", None)

    if runs_per_session is None:
        runs_per_session = 1

    if figsize is None:
        h = max(1.5, 0.5 * n_sessions + 0.8)
        w = max(4, 0.6 * runs_per_session + 2)
        figsize = (w, h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    block_w = 0.8
    block_h = 0.35
    gap = 0.15
    session_gap = 0.6

    for s in range(n_sessions):
        y = -s * session_gap
        # Session label
        ax.text(
            -0.3,
            y,
            f"S{s + 1}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1565C0",
        )

        for r in range(runs_per_session):
            x = r * (block_w + gap)
            rect = mpatches.FancyBboxPatch(
                (x, y - block_h / 2),
                block_w,
                block_h,
                boxstyle="round,pad=0.04",
                facecolor="#90CAF9",
                edgecolor="#1565C0",
                linewidth=1.0,
            )
            ax.add_patch(rect)

        # Run count annotation
        run_label = "run" if runs_per_session == 1 else "runs"
        total_w = runs_per_session * (block_w + gap) - gap
        ax.text(
            total_w + 0.2,
            y,
            f"({runs_per_session} {run_label})",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#7F7F7F",
            fontstyle="italic",
        )

    # Title
    ax.set_title(
        f"{n_sessions} sessions \u00d7 {runs_per_session} "
        f"{'run' if runs_per_session == 1 else 'runs'}",
        fontsize=10,
        fontweight="bold",
        color="#555555",
        pad=10,
    )

    # Limits
    total_w = runs_per_session * (block_w + gap) - gap
    total_h = (n_sessions - 1) * session_gap
    ax.set_xlim(-0.8, total_w + 1.5)
    ax.set_ylim(-total_h - 0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def session_structure_svg(dataset: BaseDataset, **kwargs) -> str | None:
    """Return an SVG string of the session structure diagram.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.

    Returns
    -------
    str or None
        SVG markup, or None if no data.
    """
    fig = plot_session_structure(dataset, **kwargs)
    if fig is None:
        return None
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
