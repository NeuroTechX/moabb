"""Neural signature visualisation for MOABB datasets.

Generates interactive Plotly figures showing the key neurophysiological
signature for each BCI paradigm:

- **Motor Imagery** -- ERD/ERS time-frequency maps
- **P300 / ERP** -- Evoked response waveforms (Target vs NonTarget)
- **SSVEP** -- Power spectrum / SNR at stimulus frequencies
- **c-VEP** -- Code-modulated evoked response + PSD
- **Resting State** -- Band power PSD comparison across conditions

Public API
----------
generate_neural_signature(dataset, subjects=None, output_dir=None)
    High-level entry point: generates all HTML files for a dataset.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import plotly.graph_objects as go

import mne
import numpy as np

import moabb
from moabb.analysis.style import (
    _DEFAULT_SOURCE,
    GRID_COLOR,
    MOABB_AMBER,
    MOABB_CORAL,
    MOABB_DARK_TEXT,
    MOABB_NAVY,
    MOABB_PALETTE,
    MOABB_PURPLE,
    MOABB_SKY,
    MOABB_TEAL,
)


log = logging.getLogger(__name__)


def _check_plotly():
    """Raise a helpful error if plotly is not installed."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "plotly is required for neural signature figures. "
            "Install with: pip install moabb[interactive]"
        )


# Plotly style adapter
_FONT_FAMILY = "Georgia, Cambria, 'Times New Roman', serif"

_PLOT_PALETTE = [
    MOABB_NAVY,
    MOABB_CORAL,
    MOABB_TEAL,
    MOABB_PURPLE,
    MOABB_AMBER,
    MOABB_SKY,
]

_BG_TINT = "#FAFBFC"

_GRID_LIGHT = "rgba(117, 141, 153, 0.18)"


def get_plotly_template():
    """Return a Plotly layout template matching the MOABB visual identity."""
    import plotly.graph_objects as go

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(
            family=_FONT_FAMILY,
            color=MOABB_DARK_TEXT,
            size=12,
        ),
        title=dict(
            font=dict(size=20, color=MOABB_DARK_TEXT, family=_FONT_FAMILY),
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(b=12),
        ),
        colorway=MOABB_PALETTE,
        paper_bgcolor=_BG_TINT,
        plot_bgcolor=_BG_TINT,
        margin=dict(l=72, r=24, t=80, b=64),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=GRID_COLOR,
            linewidth=0.7,
            ticks="outside",
            ticklen=4,
            tickwidth=0.7,
            tickcolor=GRID_COLOR,
            title_font=dict(size=13, color=MOABB_DARK_TEXT),
            title_standoff=10,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID_LIGHT,
            gridwidth=0.5,
            griddash="solid",
            zeroline=False,
            showline=False,
            ticks="",
            title_font=dict(size=13, color=MOABB_DARK_TEXT),
            title_standoff=8,
        ),
        legend=dict(
            bgcolor="rgba(250, 251, 252, 0.92)",
            borderwidth=0,
            font=dict(size=11, family=_FONT_FAMILY),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=MOABB_NAVY,
            font=dict(size=12, family=_FONT_FAMILY, color=MOABB_DARK_TEXT),
        ),
    )
    return template


def get_plotly_colorscale():
    """Return a diverging blue-gray-red colorscale for ERD/ERS maps."""
    return [
        [0.00, "#2166AC"],  # ERD (blue)
        [0.25, "#67A9CF"],
        [0.50, "#F0F0F0"],  # neutral
        [0.75, "#EF8A62"],
        [1.00, "#B2182B"],  # ERS (red)
    ]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r, g, b, a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _get_montage_xy(
    ch_names: list[str],
    epochs: "mne.Epochs | None" = None,
) -> dict[str, tuple[float, float]]:
    """Return normalised 2-D (x, y) positions for *ch_names*."""
    pos_3d: dict[str, np.ndarray] = {}
    if epochs is not None:
        try:
            montage = epochs.get_montage()
            if montage is not None:
                pos_3d = montage.get_positions()["ch_pos"]
        except Exception:
            pass

    if not any(ch in pos_3d for ch in ch_names):
        montage = mne.channels.make_standard_montage("standard_1020")
        pos_3d = montage.get_positions()["ch_pos"]

    raw = {
        ch: (float(pos_3d[ch][0]), float(pos_3d[ch][1]))
        for ch in ch_names
        if ch in pos_3d
    }
    if not raw:
        return {}

    all_xy = np.array(list(raw.values()))
    cx, cy = float(all_xy[:, 0].mean()), float(all_xy[:, 1].mean())
    radius = (
        float(np.max(np.sqrt((all_xy[:, 0] - cx) ** 2 + (all_xy[:, 1] - cy) ** 2))) or 1.0
    )
    return {ch: ((x - cx) / radius, (y - cy) / radius) for ch, (x, y) in raw.items()}


def _build_event_label_map(dataset) -> dict[str, str]:
    """Invert ``dataset.event_id`` to map integer-string codes to names."""
    if not hasattr(dataset, "event_id") or not dataset.event_id:
        return {}
    label_map: dict[str, str] = {}
    for name, code in dataset.event_id.items():
        label_map.setdefault(str(code), name)
    return label_map


def _display_name(name: str) -> str:
    """Make an event name human-readable: 'left_hand' -> 'Left Hand'."""
    return name.replace("_", " ").title()


def _relabel_signature(sig: "NeuralSignatureData", label_map: dict[str, str]):
    """Replace integer-string event keys with human-readable names in-place."""
    if not label_map:
        return
    display_map = {k: _display_name(v) for k, v in label_map.items()}

    def _remap_dict(d: dict) -> dict:
        return {display_map.get(k, k): v for k, v in d.items()}

    for key in ("evokeds", "sems", "tfr", "psd", "snr", "band_powers"):
        if key in sig.data and isinstance(sig.data[key], dict):
            sig.data[key] = _remap_dict(sig.data[key])
    if "event_names" in sig.data:
        sig.data["event_names"] = [display_map.get(e, e) for e in sig.data["event_names"]]
    if "n_trials" in sig.metadata:
        sig.metadata["n_trials"] = _remap_dict(sig.metadata["n_trials"])


@dataclass
class NeuralSignatureData:
    """Container for computed neural signature data."""

    paradigm: str  # "imagery", "p300", "ssvep", "cvep", "rstate"
    dataset_name: str
    dataset_code: str
    signature_type: str  # "erd_ers", "erp", "psd_snr", "cvep_response", "rstate_psd"
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _compute_evokeds_and_sems(
    epochs: mne.Epochs,
    event_names: list[str],
) -> tuple[dict, dict, dict]:
    """Return ``(evokeds, sems, n_trials)`` dicts keyed by event name."""
    evokeds: dict[str, np.ndarray] = {}
    sems: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n = len(ep)
        n_trials[name] = n
        if n == 0:
            continue
        evk = ep.average()
        evokeds[name] = evk.data
        if n > 1:
            std = np.std(ep.get_data(), axis=0, ddof=1)
            sems[name] = std / np.sqrt(n) * 1.96
        else:
            sems[name] = np.zeros_like(evk.data)
    return evokeds, sems, n_trials


def _compute_psd_per_event(
    epochs: mne.Epochs,
    event_names: list[str],
    fmin: float = 1,
    fmax: float = 50,
    n_fft: int | None = None,
) -> tuple[dict, np.ndarray | None, dict]:
    """Return ``(psd_data, freqs, n_trials)`` via Welch PSD per event."""
    if n_fft is None:
        n_fft = len(epochs.times)
    psd_data: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    freqs = None
    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n_trials[name] = len(ep)
        if len(ep) == 0:
            continue
        spectrum = ep.compute_psd(
            method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False
        )
        psd_data[name] = spectrum.get_data().mean(axis=0).mean(axis=0)
        freqs = spectrum.freqs
    return psd_data, freqs, n_trials


def compute_erp_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
) -> NeuralSignatureData:
    """Compute P300/ERP evoked averages and 95% CI SEMs per event class."""
    if event_names is None:
        event_names = [e for e in ["Target", "NonTarget"] if e in epochs.event_id]
    if not event_names:
        event_names = list(epochs.event_id.keys())[:2]

    evokeds, sems, n_trials = _compute_evokeds_and_sems(epochs, event_names)

    return NeuralSignatureData(
        paradigm="p300",
        dataset_name="",
        dataset_code="",
        signature_type="erp",
        data=dict(
            evokeds=evokeds,
            sems=sems,
            times=epochs.times,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def _select_motor_channels(
    epochs: mne.Epochs,
    max_channels: int | None = None,
) -> list[str]:
    """Select sensorimotor channels: montage coords → label patterns → all."""
    ch_names = epochs.ch_names
    n_total = len(ch_names)

    if max_channels is None:
        if n_total <= 5:
            max_channels = n_total
        elif n_total <= 16:
            max_channels = min(9, n_total)
        elif n_total <= 32:
            max_channels = 9
        else:
            max_channels = 12

    # Strategy 1: coordinate-based
    try:
        montage = epochs.get_montage()
        if montage is not None:
            pos = montage.get_positions()["ch_pos"]
            dists = {}
            for ch in ch_names:
                if ch in pos:
                    x, y, _z = pos[ch]
                    dists[ch] = x**2 + (y * 1.5) ** 2  # favour central strip
            if dists:
                ranked = sorted(dists, key=dists.get)
                return ranked[:max_channels]
    except Exception:
        pass

    # Strategy 2: label-based
    _MOTOR_LABELS = [
        "C3",
        "C4",
        "Cz",
        "C1",
        "C2",
        "C5",
        "C6",
        "FC3",
        "FC4",
        "FCz",
        "FC1",
        "FC2",
        "FC5",
        "FC6",
        "CP3",
        "CP4",
        "CPz",
        "CP1",
        "CP2",
        "CP5",
        "CP6",
    ]
    selected = [ch for ch in _MOTOR_LABELS if ch in ch_names]
    if selected:
        return selected[:max_channels]

    # Strategy 3: all channels
    return ch_names[:max_channels]


def compute_erd_ers_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
    freqs: np.ndarray | None = None,
    baseline: tuple[float, float] | None = None,
    task_onset: float | None = None,
) -> NeuralSignatureData:
    """Compute ERD/ERS time-frequency maps via multitaper + percent baseline."""
    if event_names is None:
        event_names = list(epochs.event_id.keys())
    if freqs is None:
        freqs = np.arange(2, 36, 1.0)
    if baseline is None:
        t0 = epochs.times[0]
        if t0 < 0:
            baseline = (t0, 0.0)
        elif task_onset is not None and t0 < task_onset:
            baseline = (t0, task_onset)
        else:
            baseline = (t0, t0 + min(0.5, (epochs.times[-1] - t0) * 0.1))

    central_chs = _select_motor_channels(epochs)
    picks = mne.pick_channels(epochs.ch_names, central_chs, ordered=True)

    tfr_data = {}
    n_trials = {}
    times = epochs.times  # fallback if no events match

    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n_trials[name] = len(ep)
        if len(ep) == 0:
            continue

        n_cycles = np.clip(freqs.copy(), 2, None)

        tfr = ep.compute_tfr(
            method="multitaper",
            freqs=freqs,
            n_cycles=n_cycles,
            picks=picks,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=2,
            verbose=False,
        )
        tfr.apply_baseline(baseline=baseline, mode="percent", verbose=False)
        avg_data = tfr.average().data * 100  # fraction → percentage
        tfr_data[name] = avg_data
        times = tfr.times

    return NeuralSignatureData(
        paradigm="imagery",
        dataset_name="",
        dataset_code="",
        signature_type="erd_ers",
        data=dict(
            tfr=tfr_data,
            freqs=freqs,
            times=times,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=central_chs,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def _snr_spectrum(
    psd: np.ndarray,
    noise_n_neighbor_freqs: int = 3,
    noise_skip_neighbor_freqs: int = 1,
) -> np.ndarray:
    """Compute SNR spectrum (MNE SSVEP tutorial convolution-kernel approach)."""
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"),
        axis=-1,
        arr=psd,
    )
    edge = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge, edge)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


def compute_ssvep_signature(
    epochs: mne.Epochs,
    stimulus_frequencies: list[float] | None = None,
) -> NeuralSignatureData:
    """Compute SSVEP PSD and SNR at stimulus frequencies (MNE tutorial approach)."""
    event_names = list(epochs.event_id.keys())

    if stimulus_frequencies is None:
        stimulus_frequencies = []
        for name in event_names:
            try:
                stimulus_frequencies.append(float(name))
            except ValueError:
                pass

    psd_data = {}
    snr_data = {}
    n_trials = {}
    freqs = None

    n_fft = len(epochs.times)  # full epoch for max frequency resolution

    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n_trials[name] = len(ep)
        if len(ep) == 0:
            continue

        spectrum = ep.compute_psd(
            method="welch",
            fmin=1,
            fmax=90,
            n_fft=n_fft,
            n_overlap=0,
            n_per_seg=None,
            window="boxcar",
            verbose=False,
        )
        psds, freqs = spectrum.get_data(return_freqs=True)
        psd_mean = psds.mean(axis=(0, 1))
        psd_data[name] = psd_mean

        snr_all = _snr_spectrum(psds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            snr_mean = np.nanmean(snr_all, axis=(0, 1))

        try:
            target_freq = float(name)
        except ValueError:
            target_freq = None

        if target_freq is not None and freqs is not None:
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            snr_val = snr_mean[freq_idx]
            snr_data[name] = float(snr_val) if np.isfinite(snr_val) else 0.0

    return NeuralSignatureData(
        paradigm="ssvep",
        dataset_name="",
        dataset_code="",
        signature_type="psd_snr",
        data=dict(
            psd=psd_data,
            snr=snr_data,
            freqs=freqs,
            stimulus_frequencies=stimulus_frequencies,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def compute_cvep_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
) -> NeuralSignatureData:
    """Compute c-VEP evoked response and PSD."""
    if event_names is None:
        event_names = list(epochs.event_id.keys())

    evokeds, sems, n_trials_evk = _compute_evokeds_and_sems(epochs, event_names)
    psd_data, freqs, n_trials_psd = _compute_psd_per_event(
        epochs,
        event_names,
        fmin=1,
        fmax=50,
    )
    n_trials = {**n_trials_psd, **n_trials_evk}

    return NeuralSignatureData(
        paradigm="cvep",
        dataset_name="",
        dataset_code="",
        signature_type="cvep_response",
        data=dict(
            evokeds=evokeds,
            sems=sems,
            psd=psd_data,
            freqs=freqs,
            times=epochs.times,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def compute_rstate_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
) -> NeuralSignatureData:
    """Compute resting state PSD and relative band powers per condition."""
    if event_names is None:
        event_names = list(epochs.event_id.keys())

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    psd_data, freqs, n_trials = _compute_psd_per_event(
        epochs,
        event_names,
        fmin=1,
        fmax=50,
    )

    # Compute relative band powers from the PSD
    band_powers = {}
    for name, psd in psd_data.items():
        total_power = np.trapezoid(psd, freqs)
        bp = {}
        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            bp[band_name] = np.trapezoid(psd[mask], freqs[mask]) / total_power * 100
        band_powers[name] = bp

    return NeuralSignatureData(
        paradigm="rstate",
        dataset_name="",
        dataset_code="",
        signature_type="rstate_psd",
        data=dict(
            psd=psd_data,
            band_powers=band_powers,
            freqs=freqs,
            bands=bands,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def _make_figure(**kwargs):
    """Create a Figure with the MOABB template applied."""
    import plotly.graph_objects as go

    template = get_plotly_template()
    fig = go.Figure(**kwargs)
    fig.update_layout(template=template)
    return fig


def _add_branding(fig, title: str, subtitle: str = ""):
    """Add navy accent bar, teal pip, title/subtitle, and source line."""
    source = _DEFAULT_SOURCE.format(version=moabb.__version__)

    # Navy accent bar at top
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.0,
        y0=1.005,
        x1=1.0,
        y1=1.015,
        fillcolor=MOABB_NAVY,
        line_width=0,
        layer="above",
    )
    # Teal pip
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.0,
        y0=0.997,
        x1=0.04,
        y1=1.005,
        fillcolor=MOABB_TEAL,
        line_width=0,
        layer="above",
    )
    # Title
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b>"
                + (
                    f"<br><span style='font-size:13px;color:{GRID_COLOR}'>"
                    f"{subtitle}</span>"
                    if subtitle
                    else ""
                )
            ),
            font=dict(size=18, color=MOABB_DARK_TEXT, family=_FONT_FAMILY),
            x=0.0,
            xanchor="left",
            y=0.96,
            yanchor="top",
        ),
    )
    # Source footnote
    fig.add_annotation(
        text=f"<i>{source}</i>",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.08,
        xanchor="left",
        yanchor="top",
        font=dict(size=9, color=GRID_COLOR, family=_FONT_FAMILY),
    )


def _evoked_metrics(
    metrics: dict,
    name: str,
    evk: np.ndarray,
    times: np.ndarray | None,
    n: int,
) -> None:
    """Add evoked-response metrics (shared by ERP and c-VEP)."""
    evk_uv = evk * 1e6
    mean_evk = evk_uv.mean(axis=0)
    peak_idx = np.argmax(np.abs(mean_evk))
    metrics[f"{name}_peak_amplitude_uV"] = float(mean_evk[peak_idx])
    if times is not None:
        metrics[f"{name}_peak_latency_ms"] = float(times[peak_idx] * 1000)
        metrics[f"{name}_mean_amplitude_uV"] = float(np.mean(mean_evk))
    metrics[f"{name}_rms_uV"] = float(np.sqrt(np.mean(mean_evk**2)))
    metrics[f"{name}_n_trials"] = n


def compute_metrics(sig: NeuralSignatureData) -> dict[str, Any]:
    """Compute objective quality metrics for a neural signature."""
    metrics: dict[str, Any] = {}
    n_trials = sig.metadata.get("n_trials", {})

    if sig.signature_type == "erp":
        for name, evk in sig.data["evokeds"].items():
            _evoked_metrics(metrics, name, evk, sig.data["times"], n_trials.get(name, 0))

    elif sig.signature_type == "erd_ers":
        freqs = sig.data["freqs"]
        for name, tfr in sig.data["tfr"].items():
            mean_tfr = tfr.mean(axis=0)
            metrics[f"{name}_max_erd_pct"] = float(np.min(mean_tfr))
            metrics[f"{name}_max_ers_pct"] = float(np.max(mean_tfr))
            erd_freq_idx = np.unravel_index(np.argmin(mean_tfr), mean_tfr.shape)[0]
            metrics[f"{name}_erd_peak_freq_hz"] = float(freqs[erd_freq_idx])
            metrics[f"{name}_n_trials"] = n_trials.get(name, 0)
            for label, lo, hi in [("mu", 8, 13), ("beta", 13, 30)]:
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    metrics[f"{name}_{label}_band_erd_pct"] = float(
                        mean_tfr[mask, :].mean()
                    )

    elif sig.signature_type == "psd_snr":
        for name in sig.data["event_names"]:
            if name in sig.data["snr"]:
                metrics[f"{name}_snr"] = float(sig.data["snr"][name])
            if name in sig.data["psd"]:
                psd = sig.data["psd"][name]
                metrics[f"{name}_peak_power"] = float(np.max(psd))
                metrics[f"{name}_peak_freq_hz"] = float(sig.data["freqs"][np.argmax(psd)])
            metrics[f"{name}_n_trials"] = n_trials.get(name, 0)

    elif sig.signature_type == "cvep_response":
        for name, evk in sig.data["evokeds"].items():
            _evoked_metrics(metrics, name, evk, None, n_trials.get(name, 0))

    elif sig.signature_type == "rstate_psd":
        for name, bp in sig.data["band_powers"].items():
            for band, power in bp.items():
                metrics[f"{name}_{band}_pct"] = float(power)
            metrics[f"{name}_n_trials"] = n_trials.get(name, 0)
            if name in sig.data["psd"]:
                freqs, psd = sig.data["freqs"], sig.data["psd"][name]
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                if alpha_mask.any():
                    metrics[f"{name}_alpha_peak_hz"] = float(
                        freqs[alpha_mask][np.argmax(psd[alpha_mask])]
                    )

    return metrics


def _format_metric_value(key: str, val) -> str:
    """Format a metric value with units."""
    if isinstance(val, float):
        if "hz" in key.lower() or "freq" in key.lower():
            return f"{val:.1f} Hz"
        if "latency" in key.lower():
            return f"{val:.0f} ms"
        if "uv" in key.lower() or "amplitude" in key.lower():
            return f"{val:.2f} \u00b5V"
        if "snr" in key.lower():
            return f"{val:.2f}"
        if "pct" in key or "erd" in key or "ers" in key:
            return f"{val:.1f}%"
        return f"{val:.4g}"
    return str(val)


_GITHUB_BASE = "https://github.com/NeuroTechX/moabb/blob/develop"
_SRC_FILE = "moabb/analysis/neural_signatures.py"

# Map metric suffix → code snippet for HTML tooltip
_METRIC_CODE: dict[str, str] = {
    "peak_amplitude_uV": "mean_evk[np.argmax(np.abs(mean_evk))]  # V→µV",
    "peak_latency_ms": "times[np.argmax(np.abs(mean_evk))] * 1000",
    "mean_amplitude_uV": "np.mean(evk * 1e6, axis=(0,1))",
    "rms_uV": "np.sqrt(np.mean(mean_evk**2))",
    "n_trials": "len(epochs[event_name])",
    "max_erd_pct": "np.min(tfr.mean(axis=0)) * 100  # percent change",
    "max_ers_pct": "np.max(tfr.mean(axis=0)) * 100",
    "erd_peak_freq_hz": "freqs[np.argmin(mean_tfr)]",
    "mu_band_erd_pct": "mean_tfr[8-13 Hz, :].mean()",
    "beta_band_erd_pct": "mean_tfr[13-30 Hz, :].mean()",
    "snr": "psd / mean_noise  # convolution-kernel SNR",
    "peak_power": "np.max(psd)  # Welch",
    "peak_freq_hz": "freqs[np.argmax(psd)]",
    "alpha_peak_hz": "freqs[8-13Hz][np.argmax(psd[8-13Hz])]",
    **{
        f"{b}_pct": f"trapezoid(psd[{b}]) / trapezoid(psd[1-50Hz]) * 100"
        for b in ("delta", "theta", "alpha", "beta", "gamma")
    },
}


def _metric_tooltip(metric_suffix: str) -> str:
    """Return an HTML title attribute with the code snippet for a metric."""
    code = _METRIC_CODE.get(metric_suffix, "")
    if not code:
        return ""
    # Escape HTML entities for the title attribute
    escaped = code.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return f' title="{escaped}"'


def _build_metrics_table(metrics: dict[str, Any], sig_type: str) -> str:
    """Build an HTML metrics table."""
    if not metrics:
        return ""

    src_url = f"{_GITHUB_BASE}/{_SRC_FILE}"

    groups: dict[str, list[tuple[str, str, str]]] = {}
    for k, v in metrics.items():
        parts = k.split("_", 1)
        group = parts[0] if len(parts) > 1 else "General"
        suffix = parts[1] if len(parts) > 1 else k
        label = suffix.replace("_", " ").title()
        groups.setdefault(group, []).append((label, _format_metric_value(k, v), suffix))

    rows = []
    for group_name, items in groups.items():
        rows.append(
            f'<tr><td colspan="2" style="padding:8px 0 4px;font-weight:700;'
            f"color:{MOABB_NAVY};border-bottom:1px solid {MOABB_NAVY};"
            f'font-size:12px;letter-spacing:0.5px">{group_name}</td></tr>'
        )
        for label, val, suffix in items:
            tip = _metric_tooltip(suffix)
            rows.append(
                f'<tr class="metric-row"{tip}>'
                f'<td style="padding:3px 12px 3px 0;color:{GRID_COLOR};'
                f'font-size:11px">{label}</td>'
                f'<td style="padding:3px 0;font-weight:600;'
                f'color:{MOABB_DARK_TEXT};font-size:11px;text-align:right">'
                f"{val}</td></tr>"
            )

    src_link = (
        f'<div style="margin-top:12px;text-align:right">'
        f'<a href="{src_url}" target="_blank" '
        f'style="font-size:10px;color:{GRID_COLOR};text-decoration:none;'
        f'border-bottom:1px dotted {GRID_COLOR}">'
        f"View source: compute_metrics()</a></div>"
    )

    return (
        '<table style="border-collapse:collapse;font-family:Georgia,serif;'
        'width:100%">' + "".join(rows) + "</table>" + src_link
    )


def _build_head_svg(ch_names: list[str], active_ch: str | None = None) -> str:
    """Build an inline SVG scalp diagram with electrode positions."""
    positions = _get_montage_xy(ch_names)
    if not positions:
        return ""

    size = 100
    cx, cy = size / 2, size / 2
    r = size * 0.38

    parts = [
        f'<svg id="head-svg" viewBox="0 0 {size} {size}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;width:100%;height:100%">',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="#FAFBFC" stroke="{MOABB_NAVY}" stroke-width="1.5"/>',
        f'<polygon points="{cx-4},{cy-r} {cx},{cy-r-7} {cx+4},{cy-r}" '
        f'fill="none" stroke="{MOABB_NAVY}" stroke-width="1.5"/>',
        f'<path d="M{cx-r},{cy-5} Q{cx-r-4},{cy} {cx-r},{cy+5}" '
        f'fill="none" stroke="{MOABB_NAVY}" stroke-width="1.5"/>',
        f'<path d="M{cx+r},{cy-5} Q{cx+r+4},{cy} {cx+r},{cy+5}" '
        f'fill="none" stroke="{MOABB_NAVY}" stroke-width="1.5"/>',
    ]

    active_ch = active_ch or (ch_names[0] if ch_names else None)
    for ch, (nx, ny) in positions.items():
        px = cx + nx * r * 0.82
        py = cy - ny * r * 0.82
        is_active = ch == active_ch
        parts.append(
            f'<circle id="elec-{ch}" cx="{px:.1f}" cy="{py:.1f}" '
            f'r="{"5" if is_active else "2.5"}" '
            f'fill="{MOABB_CORAL if is_active else MOABB_NAVY}" '
            f'opacity="{"1" if is_active else "0.35"}" '
            f'data-ch="{ch}" data-cx="{px:.1f}" data-cy="{py:.1f}"/>'
        )

    if active_ch and active_ch in positions:
        apx = cx + positions[active_ch][0] * r * 0.82
        apy = cy - positions[active_ch][1] * r * 0.82
        parts.append(
            f'<text id="elec-label" x="{apx:.1f}" y="{apy + 10:.1f}" '
            f'text-anchor="middle" fill="{MOABB_CORAL}" '
            f'font-size="8" font-weight="700" '
            f'font-family="Georgia,serif">{active_ch}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_head_js(ch_names: list[str]) -> str:
    """Return JS to sync the SVG head with the Plotly channel selector."""
    if not ch_names:
        return ""
    return f"""<script>
(function() {{
  var chNames = {list(ch_names)};
  var chSet = new Set(chNames);
  var CORAL = "{MOABB_CORAL}";
  var NAVY  = "{MOABB_NAVY}";

  function highlightElectrode(name) {{
    if (!chSet.has(name)) return;
    chNames.forEach(function(ch) {{
      var el = document.getElementById("elec-" + ch);
      if (!el) return;
      var active = (ch === name);
      el.setAttribute("r", active ? "5" : "2.5");
      el.setAttribute("fill", active ? CORAL : NAVY);
      el.setAttribute("opacity", active ? "1" : "0.35");
    }});
    var lbl = document.getElementById("elec-label");
    var dot = document.getElementById("elec-" + name);
    if (lbl && dot) {{
      lbl.textContent = name;
      lbl.setAttribute("x", dot.getAttribute("data-cx"));
      lbl.setAttribute("y", String(parseFloat(dot.getAttribute("data-cy")) + 10));
    }}
  }}

  function bind() {{
    var plot = document.querySelector(".js-plotly-plot");
    if (!plot || !plot.on) {{ setTimeout(bind, 200); return; }}
    plot.on("plotly_sliderchange", function(e) {{
      if (e && e.step && e.step.label) {{
        highlightElectrode(e.step.label);
      }}
    }});
    plot.on("plotly_buttonclicked", function(e) {{
      if (e && e.button && e.button.label) {{
        highlightElectrode(e.button.label);
      }}
    }});
  }}
  if (document.readyState === "complete") bind();
  else window.addEventListener("load", bind);
}})();
</script>"""


def _wrap_branded_html(
    plotly_div: str,
    title: str,
    subtitle: str,
    metrics_html: str,
    ds_name: str,
    head_svg: str = "",
    head_js: str = "",
) -> str:
    """Wrap a Plotly div in a branded MOABB HTML page."""
    source = _DEFAULT_SOURCE.format(version=moabb.__version__)

    head_block = f'<div class="head-inset">{head_svg}</div>' if head_svg else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} \u2014 {ds_name} \u2014 MOABB</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');
  :root {{
    --navy: {MOABB_NAVY}; --teal: {MOABB_TEAL}; --coral: {MOABB_CORAL};
    --dark-text: {MOABB_DARK_TEXT}; --grid: {GRID_COLOR};
    --bg: #F4F6F8; --card-bg: #FFF;
    --shadow: 0 1px 4px rgba(0,0,0,0.06); --radius: 8px;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Source Serif 4',Georgia,serif; background:var(--bg); color:var(--dark-text); }}
  .page {{ max-width:1340px; margin:0 auto; padding:0 28px 56px; }}
  .top-bar {{ display:flex; height:5px; }}
  .top-bar .teal {{ width:60px; background:var(--teal); }}
  .top-bar .navy {{ flex:1; background:var(--navy); }}
  .header {{ padding:24px 0 20px; display:flex; justify-content:space-between; align-items:center; }}
  .header-left {{ display:flex; align-items:center; gap:18px; }}
  .head-inset {{ flex-shrink:0; width:76px; height:76px; background:var(--card-bg); border-radius:50%; padding:6px; box-shadow:var(--shadow); }}
  .header-text h1 {{ font-size:24px; font-weight:700; line-height:1.2; }}
  .header-text .subtitle {{ font-size:13px; color:var(--grid); margin-top:4px; }}
  .badge {{ padding:5px 14px; background:rgba(27,158,119,0.08); border:1px solid rgba(27,158,119,0.2); border-radius:20px; font-size:12px; font-weight:600; color:var(--teal); }}
  .content {{ display:grid; grid-template-columns:1fr 300px; gap:20px; align-items:start; }}
  @media (max-width:960px) {{ .content {{ grid-template-columns:1fr; }} }}
  .chart-card {{ background:var(--card-bg); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden; }}
  .chart-card .plotly-graph-div {{ width:100% !important; }}
  .metrics-card {{ background:var(--card-bg); border-radius:var(--radius); box-shadow:var(--shadow); padding:20px; position:sticky; top:16px; max-height:85vh; overflow-y:auto; }}
  .metrics-card h2 {{ font-size:12px; font-weight:700; color:var(--navy); text-transform:uppercase; letter-spacing:1.2px; margin-bottom:14px; padding-bottom:8px; border-bottom:2px solid var(--navy); }}
  .metric-row {{ cursor:help; }} .metric-row:hover {{ background:rgba(27,158,119,0.06); }}
  .source {{ margin-top:28px; padding-top:14px; border-top:1px solid #DFE3E8; font-size:11px; color:var(--grid); font-style:italic; text-align:center; }}
</style>
</head>
<body>
<div class="page">
  <div class="top-bar"><div class="teal"></div><div class="navy"></div></div>
  <div class="header">
    <div class="header-left">
      {head_block}
      <div class="header-text">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
      </div>
    </div>
    <div class="badge">{ds_name}</div>
  </div>
  <div class="content">
    <div class="chart-card">{plotly_div}</div>
    <div class="metrics-card">
      <h2>Objective Metrics</h2>
      {metrics_html}
    </div>
  </div>
  <div class="source">{source}</div>
  {head_js}
</div>
</body>
</html>"""


def plot_erp_interactive(
    sig: NeuralSignatureData,
    channel_idx: int = 0,
) -> "go.Figure":
    """Plot P300/ERP waveforms with confidence bands."""
    import plotly.graph_objects as go

    _check_plotly()

    times_ms = sig.data["times"] * 1000
    event_names = sig.data["event_names"]
    ch_names = sig.metadata["ch_names"]
    n_trials = sig.metadata["n_trials"]

    fig = _make_figure()

    buttons = []
    active_events = [e for e in event_names if e in sig.data["evokeds"]]
    n_ev = len(active_events)
    traces_per_ch = n_ev * 3  # line + upper/lower CI

    for ch_i, ch_name in enumerate(ch_names):
        for ev_i, name in enumerate(active_events):
            evk = sig.data["evokeds"][name][ch_i] * 1e6
            sem = sig.data["sems"][name][ch_i] * 1e6
            color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]
            n_t = n_trials.get(name, "?")
            visible = ch_i == channel_idx

            fig.add_trace(
                go.Scatter(
                    x=times_ms,
                    y=evk,
                    mode="lines",
                    name=f"{name}  n={n_t}",
                    line=dict(color=color, width=2.5),
                    visible=visible,
                    legendgroup=f"{name}_{ch_i}",
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=times_ms,
                    y=evk + sem,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    visible=visible,
                    legendgroup=f"{name}_{ch_i}",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=times_ms,
                    y=evk - sem,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, 0.10),
                    showlegend=False,
                    visible=visible,
                    legendgroup=f"{name}_{ch_i}",
                    hoverinfo="skip",
                )
            )

    for ch_i, ch_name in enumerate(ch_names):
        vis = []
        for c_i in range(len(ch_names)):
            vis.extend([c_i == ch_i] * traces_per_ch)
        buttons.append(dict(label=ch_name, method="update", args=[{"visible": vis}]))

    if len(ch_names) > 1:
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.12,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor=MOABB_NAVY,
                    borderwidth=1,
                    font=dict(size=11, family=_FONT_FAMILY, color=MOABB_DARK_TEXT),
                )
            ]
        )

    fig.add_vline(x=0, line_dash="dash", line_color=MOABB_NAVY, line_width=1, opacity=0.4)
    fig.add_annotation(
        text="onset",
        x=2,
        y=1.0,
        xref="x",
        yref="paper",
        showarrow=False,
        font=dict(size=9, color=GRID_COLOR),
    )

    _add_branding(
        fig,
        title=f"Evoked Response \u2014 {sig.dataset_name}",
        subtitle=f"Grand average across {sum(n_trials.values())} trials "
        f"\u00b7 Channel: {ch_names[channel_idx]}",
    )
    fig.update_layout(
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude (\u00b5V)",
        hovermode="x unified",
        height=480,
        margin=dict(t=100, b=60, l=72, r=24),
    )

    return fig


def plot_erd_ers_interactive(
    sig: NeuralSignatureData,
) -> "go.Figure":
    """Plot ERD/ERS time-frequency heatmaps for motor imagery."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_plotly()

    event_names = [e for e in sig.data["event_names"] if e in sig.data["tfr"]]
    ch_names = sig.metadata["ch_names"]
    n_trials = sig.metadata["n_trials"]
    times = sig.data["times"]
    freqs = sig.data["freqs"]
    colorscale = get_plotly_colorscale()

    n_events = len(event_names)
    if n_events == 0:
        fig = _make_figure()
        fig.update_layout(title="No ERD/ERS data available")
        return fig

    fig = make_subplots(
        rows=1,
        cols=n_events,
        subplot_titles=[
            f"<b>{name}</b>  <span style='color:{GRID_COLOR};font-size:11px'>"
            f"n={n_trials.get(name, '?')}</span>"
            for name in event_names
        ],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    fig.update_layout(template=get_plotly_template())

    all_values = [sig.data["tfr"][n] for n in event_names]
    vmax = max(np.percentile(np.abs(v), 95) for v in all_values) if all_values else 100

    steps = []
    for ch_i, ch_name in enumerate(ch_names):
        step_data = [sig.data["tfr"][name][ch_i] for name in event_names]
        steps.append(dict(method="update", label=ch_name, args=[{"z": step_data}]))

    for ev_i, name in enumerate(event_names):
        tfr = sig.data["tfr"][name][0]
        fig.add_trace(
            go.Heatmap(
                z=tfr,
                x=times,
                y=freqs,
                colorscale=colorscale,
                zmin=-vmax,
                zmax=vmax,
                colorbar=(
                    dict(
                        title=dict(text="ERD/ERS (%)", font=dict(size=11)),
                        thickness=12,
                        len=0.85,
                        tickfont=dict(size=10),
                        outlinewidth=0,
                    )
                    if ev_i == n_events - 1
                    else None
                ),
                showscale=(ev_i == n_events - 1),
                hovertemplate=(
                    "<b>%{y:.0f} Hz</b> at %{x:.2f}s"
                    "<br>ERD/ERS: %{z:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=ev_i + 1,
        )

    if len(ch_names) > 1:
        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(
                        prefix="Channel: ",
                        font=dict(size=12, color=MOABB_DARK_TEXT),
                    ),
                    pad=dict(t=40),
                    steps=steps,
                    bordercolor=MOABB_NAVY,
                    borderwidth=1,
                    activebgcolor=_hex_to_rgba(MOABB_TEAL, 0.25),
                    font=dict(size=11),
                )
            ]
        )

    for ev_i in range(n_events):
        for band_freq in (8, 13):
            fig.add_hline(
                y=band_freq,
                line_dash="dot",
                line_color="rgba(47, 62, 92, 0.25)",
                line_width=0.8,
                row=1,
                col=ev_i + 1,
            )

    for ev_i in range(n_events):
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="rgba(47, 62, 92, 0.35)",
            line_width=0.8,
            row=1,
            col=ev_i + 1,
        )

    fig.update_layout(
        height=700,
        margin=dict(t=40, b=80, l=72, r=80),
    )
    for i in range(1, n_events + 1):
        fig.update_xaxes(title_text="Time (s)", row=1, col=i)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)

    return fig


def plot_ssvep_interactive(
    sig: NeuralSignatureData,
) -> "go.Figure":
    """Plot SSVEP power spectra with stimulus frequency markers."""
    import plotly.graph_objects as go

    _check_plotly()

    freqs = sig.data["freqs"]
    stim_freqs = sig.data["stimulus_frequencies"]
    event_names = sig.data["event_names"]
    n_trials = sig.metadata["n_trials"]
    fig = _make_figure()

    for ev_i, name in enumerate(event_names):
        if name not in sig.data["psd"]:
            continue
        psd = sig.data["psd"][name]
        n_t = n_trials.get(name, "?")
        snr_val = sig.data["snr"].get(name, None)
        color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]

        label = f"{name} Hz  n={n_t}"
        if snr_val is not None and snr_val > 0:
            label += f"  SNR {snr_val:.1f}"

        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5),
            )
        )

    for freq in stim_freqs:
        for harmonic in [1, 2]:
            f = freq * harmonic
            if freqs is not None and f <= freqs[-1]:
                fig.add_vline(
                    x=f,
                    line_dash="dash" if harmonic == 1 else "dot",
                    line_color=_hex_to_rgba(MOABB_NAVY, 0.3),
                    line_width=1,
                )
                if harmonic == 1:
                    fig.add_annotation(
                        x=f,
                        y=1.06,
                        xref="x",
                        yref="paper",
                        text=f"<b>{f:.0f} Hz</b>",
                        showarrow=False,
                        font=dict(size=9, color=MOABB_NAVY),
                    )

    total_trials = sum(n_trials.values())
    _add_branding(
        fig,
        title=f"SSVEP Power Spectrum \u2014 {sig.dataset_name}",
        subtitle=f"Grand average \u00b7 {total_trials} trials \u00b7 Welch PSD",
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density (V\u00b2/Hz)",
        yaxis_type="log",
        hovermode="x unified",
        height=480,
        margin=dict(t=100, b=60, l=72, r=24),
    )

    return fig


def plot_cvep_interactive(
    sig: NeuralSignatureData,
    channel_idx: int = 0,
) -> "go.Figure":
    """Plot c-VEP evoked responses and PSD."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_plotly()

    event_names = [e for e in sig.data["event_names"] if e in sig.data["evokeds"]]
    ch_names = sig.metadata["ch_names"]
    n_trials = sig.metadata["n_trials"]
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "<b>Evoked Response</b>",
            "<b>Power Spectrum</b>",
        ],
        vertical_spacing=0.18,
        row_heights=[0.6, 0.4],
    )
    fig.update_layout(template=get_plotly_template())

    times_ms = sig.data["times"] * 1000
    for ev_i, name in enumerate(event_names):
        color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]
        n_t = n_trials.get(name, "?")
        ch_i = min(channel_idx, sig.data["evokeds"][name].shape[0] - 1)

        fig.add_trace(
            go.Scatter(
                x=times_ms,
                y=sig.data["evokeds"][name][ch_i] * 1e6,
                mode="lines",
                name=f"Code {name}  n={n_t}",
                line=dict(color=color, width=2.5),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )

        if name in sig.data["psd"] and sig.data["freqs"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=sig.data["freqs"],
                    y=sig.data["psd"][name],
                    mode="lines",
                    name=f"PSD {name}",
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (\u00b5V)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="PSD (V\u00b2/Hz)", type="log", row=2, col=1)

    ch_label = ch_names[channel_idx] if channel_idx < len(ch_names) else ""
    _add_branding(
        fig,
        title=f"c-VEP Signature \u2014 {sig.dataset_name}",
        subtitle=f"Channel: {ch_label} \u00b7 {sum(n_trials.values())} trials",
    )
    fig.update_layout(
        height=640,
        hovermode="x unified",
        margin=dict(t=110, b=60, l=72, r=24),
    )
    return fig


def plot_rstate_interactive(
    sig: NeuralSignatureData,
) -> "go.Figure":
    """Plot resting state PSD overlay and band power bar chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_plotly()

    event_names = [e for e in sig.data["event_names"] if e in sig.data["psd"]]
    n_trials = sig.metadata["n_trials"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Power Spectral Density</b>",
            "<b>Relative Band Power</b>",
        ],
        horizontal_spacing=0.14,
        column_widths=[0.6, 0.4],
    )
    fig.update_layout(template=get_plotly_template())

    freqs = sig.data["freqs"]
    band_names = list(sig.data["bands"].keys())
    band_labels = [b.capitalize() for b in band_names]

    for ev_i, name in enumerate(event_names):
        color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]
        n_t = n_trials.get(name, "?")

        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=sig.data["psd"][name],
                mode="lines",
                name=f"{name}  n={n_t}",
                line=dict(color=color, width=2.5),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )

        if name in sig.data["band_powers"]:
            bp = sig.data["band_powers"][name]
            fig.add_trace(
                go.Bar(
                    x=band_labels,
                    y=[bp[b] for b in band_names],
                    name=name,
                    marker_color=color,
                    marker_line_width=0,
                    legendgroup=name,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="PSD (V\u00b2/Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Relative Power (%)", row=1, col=2)

    total_trials = sum(n_trials.values())
    _add_branding(
        fig,
        title=f"Resting State Signature \u2014 {sig.dataset_name}",
        subtitle=f"Grand average \u00b7 {total_trials} trials \u00b7 Welch PSD",
    )
    fig.update_layout(
        height=480,
        hovermode="x unified",
        barmode="group",
        margin=dict(t=110, b=60, l=72, r=24),
    )
    return fig


_PARADIGM_HANDLERS = {
    "imagery": (compute_erd_ers_signature, plot_erd_ers_interactive),
    "p300": (compute_erp_signature, plot_erp_interactive),
    "ssvep": (compute_ssvep_signature, plot_ssvep_interactive),
    "cvep": (compute_cvep_signature, plot_cvep_interactive),
    "rstate": (compute_rstate_signature, plot_rstate_interactive),
}


def _get_paradigm_instance(paradigm_name: str, dataset=None):
    """Instantiate the appropriate paradigm for data loading."""
    from moabb.paradigms import (
        CVEP,
        P300,
        SSVEP,
        MotorImagery,
        RestingStateToP300Adapter,
    )

    _paradigm_factories = {
        "imagery": lambda _ds: MotorImagery(n_classes=2, fmin=1, fmax=45, resample=128),
        "p300": lambda _ds: P300(),
        "ssvep": lambda _ds: SSVEP(n_classes=2, resample=256),
        "cvep": lambda _ds: CVEP(n_classes=2, resample=256),
        "rstate": lambda ds: RestingStateToP300Adapter(
            events=(
                list(ds.event_id.keys())
                if ds is not None and hasattr(ds, "event_id")
                else None
            )
        ),
    }
    factory = _paradigm_factories.get(paradigm_name)
    if factory is None:
        raise ValueError(
            f"Unknown paradigm {paradigm_name!r}. "
            f"Supported: {list(_paradigm_factories)}"
        )
    return factory(dataset)


def _load_and_compute(
    dataset,
    subjects: list[int],
    compute_fn,
    plot_fn,
    *,
    collect_per_subject: bool = False,
) -> tuple["NeuralSignatureData | None", "go.Figure | None", dict]:
    """Load epochs, compute grand-average signature, and plot."""
    paradigm_name = dataset.paradigm
    paradigm = _get_paradigm_instance(paradigm_name, dataset)
    ds_name = dataset.__class__.__name__
    code = dataset.code

    # Widen interval to include a pre-stimulus baseline
    _task_onset = None
    if hasattr(dataset, "interval") and dataset.interval is not None:
        orig = dataset.interval
        _task_onset = orig[0]
        if orig[0] > 0:
            margin = min(orig[0], 1.0)
        else:
            margin = 0.5
        dataset.interval = [orig[0] - margin, orig[1]]

    per_subject_epochs: dict = {}
    grand_epochs = None

    for subj in subjects:
        log.info("Loading subject %s for %s", subj, ds_name)
        try:
            epochs, _, _ = paradigm.get_data(dataset, [subj], return_epochs=True)
        except Exception as exc:
            log.warning("Failed to load subject %s: %s", subj, exc)
            continue
        if collect_per_subject:
            per_subject_epochs[subj] = epochs
        if grand_epochs is None:
            grand_epochs = epochs
        else:
            try:
                grand_epochs = mne.concatenate_epochs(
                    [grand_epochs, epochs], verbose=False
                )
            except Exception:
                log.warning("Cannot concatenate epochs for subject %s", subj)

    if grand_epochs is None or len(grand_epochs) == 0:
        return None, None, per_subject_epochs

    log.info("Computing grand-average signature for %s", ds_name)
    kwargs = {}
    if _task_onset is not None and compute_fn is compute_erd_ers_signature:
        kwargs["task_onset"] = _task_onset
    sig = compute_fn(grand_epochs, **kwargs)
    sig.dataset_name = ds_name
    sig.dataset_code = code
    _relabel_signature(sig, _build_event_label_map(dataset))

    fig = plot_fn(sig)
    return sig, fig, per_subject_epochs


def generate_neural_signature(
    dataset,
    subjects: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate interactive neural signature HTML files for a dataset.

    Produces branded, standalone HTML files with the MOABB editorial style.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    subjects : list of int or None
        Subject IDs.  ``None`` uses all subjects.
    output_dir : str, Path, or None
        Output directory.  Defaults to ``./neural_signatures/``.

    Returns
    -------
    list of Path
        Paths to the generated HTML files.
    """
    _check_plotly()

    if output_dir is None:
        output_dir = Path("neural_signatures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = dataset.subject_list

    paradigm_name = dataset.paradigm
    if paradigm_name not in _PARADIGM_HANDLERS:
        raise ValueError(
            f"Unsupported paradigm {paradigm_name!r} for dataset "
            f"{dataset.__class__.__name__}. "
            f"Supported: {list(_PARADIGM_HANDLERS)}"
        )

    compute_fn, plot_fn = _PARADIGM_HANDLERS[paradigm_name]
    code = dataset.code
    ds_name = dataset.__class__.__name__
    generated = []

    sig, fig, all_evokeds_per_subject = _load_and_compute(
        dataset,
        subjects,
        compute_fn,
        plot_fn,
        collect_per_subject=True,
    )
    if sig is None:
        log.warning("No valid epochs for %s, skipping.", ds_name)
        return generated

    metrics = compute_metrics(sig)
    metrics_html = _build_metrics_table(metrics, sig.signature_type)

    _sig_titles = {
        "erd_ers": "ERD/ERS Time-Frequency",
        "erp": "Evoked Response",
        "psd_snr": "SSVEP Power Spectrum",
        "cvep_response": "c-VEP Signature",
        "rstate_psd": "Resting State",
    }
    title = f"{_sig_titles.get(sig.signature_type, 'Neural Signature')} \u2014 {ds_name}"
    n_total = sum(sig.metadata.get("n_trials", {}).values())
    subtitle = (
        f"Grand average across {len(subjects)} subject(s), " f"{n_total} total trials"
    )

    plotly_div = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": True, "displaylogo": False},
    )
    head_svg = ""
    head_js = ""
    ch_names = sig.metadata.get("ch_names", [])
    if ch_names:
        head_svg = _build_head_svg(ch_names, active_ch=ch_names[0])
        head_js = _build_head_js(ch_names)
    html = _wrap_branded_html(
        plotly_div,
        title,
        subtitle,
        metrics_html,
        ds_name,
        head_svg=head_svg,
        head_js=head_js,
    )
    path = output_dir / f"{code}_grand_average.html"
    path.write_text(html, encoding="utf-8")
    generated.append(path)
    log.info("Wrote %s", path)

    if paradigm_name in ("p300", "cvep"):
        fig_ch = _make_per_channel_figure(sig, paradigm_name)
        if fig_ch is not None:
            path_ch = output_dir / f"{code}_per_channel.html"
            fig_ch.write_html(str(path_ch), include_plotlyjs=True)
            generated.append(path_ch)
            log.info("Wrote %s", path_ch)

    if len(all_evokeds_per_subject) > 1:
        fig_subj = _make_per_subject_figure(
            all_evokeds_per_subject,
            compute_fn,
            plot_fn,
            paradigm_name,
            ds_name,
            code,
        )
        if fig_subj is not None:
            path_subj = output_dir / f"{code}_per_subject.html"
            fig_subj.write_html(str(path_subj), include_plotlyjs=True)
            generated.append(path_subj)
            log.info("Wrote %s", path_subj)

    return generated


def _make_per_channel_figure(sig, paradigm_name):
    """Create a per-channel overview figure for ERP-type signatures."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ch_names = sig.metadata["ch_names"]
    if len(ch_names) <= 1:
        return None

    n_ch = min(len(ch_names), 16)
    ncols = min(4, n_ch)
    nrows = (n_ch + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"<b>{ch}</b>" for ch in ch_names[:n_ch]],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )
    fig.update_layout(template=get_plotly_template())

    times_ms = sig.data["times"] * 1000
    event_names = sig.data.get("event_names", [])

    for ch_i in range(n_ch):
        row = ch_i // ncols + 1
        col = ch_i % ncols + 1
        for ev_i, name in enumerate(event_names):
            if name not in sig.data.get("evokeds", {}):
                continue
            evk = sig.data["evokeds"][name]
            if ch_i >= evk.shape[0]:
                continue
            fig.add_trace(
                go.Scatter(
                    x=times_ms,
                    y=evk[ch_i] * 1e6,
                    mode="lines",
                    name=name,
                    line=dict(color=_PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)], width=1.5),
                    showlegend=(ch_i == 0),
                    legendgroup=name,
                ),
                row=row,
                col=col,
            )

    _add_branding(fig, title=f"Per-Channel ERP \u2014 {sig.dataset_name}", subtitle="")
    fig.update_layout(height=220 * nrows + 80, hovermode="x unified")
    return fig


def _make_per_subject_figure(
    subjects_epochs,
    compute_fn,
    plot_fn,
    paradigm_name,
    ds_name,
    code,
):
    """Create a per-subject overlay figure."""
    import plotly.graph_objects as go

    _check_plotly()
    fig = _make_figure()

    for i, (subj, epochs) in enumerate(subjects_epochs.items()):
        try:
            sig = compute_fn(epochs)
            sig.dataset_name = ds_name
            sig.dataset_code = code
        except Exception:
            continue

        color = _PLOT_PALETTE[i % len(_PLOT_PALETTE)]

        data_key = {"imagery": "tfr", "p300": "evokeds", "cvep": "evokeds"}.get(
            paradigm_name, "psd"
        )
        event_names = [
            e for e in sig.data.get("event_names", []) if e in sig.data.get(data_key, {})
        ]
        if not event_names:
            continue

        if paradigm_name == "imagery":
            freqs = sig.data["freqs"]
            mu_mask = (freqs >= 8) & (freqs <= 13)
            tfr = sig.data["tfr"][event_names[0]]
            x, y = sig.data["times"], tfr[:, mu_mask, :].mean(axis=(0, 1))
        elif paradigm_name in ("p300", "cvep"):
            evk = sig.data["evokeds"][event_names[0]]
            x, y = sig.data["times"] * 1000, evk.mean(axis=0) * 1e6
        else:  # ssvep, rstate
            x, y = sig.data["freqs"], sig.data["psd"][event_names[0]]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Sub {subj}",
                line=dict(color=color, width=1.5),
                opacity=0.7,
            )
        )

    _add_branding(
        fig,
        title=f"Per-Subject Overview \u2014 {ds_name}",
        subtitle=f"{len(subjects_epochs)} subjects",
    )
    fig.update_layout(hovermode="x unified")

    if paradigm_name == "imagery":
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Mu-band (8\u201313 Hz) Power",
        )
    elif paradigm_name in ("p300", "cvep"):
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Amplitude (\u00b5V)")
        fig.add_vline(
            x=0, line_dash="dash", line_color=MOABB_NAVY, line_width=1, opacity=0.4
        )
    elif paradigm_name in ("ssvep", "rstate"):
        fig.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD (V\u00b2/Hz)",
            yaxis_type="log",
        )
    return fig


def neural_signature_html(
    dataset,
    subjects: list[int] | None = None,
) -> dict[str, str]:
    """Generate neural signature figures and return HTML strings.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    subjects : list of int or None
        Subject IDs.

    Returns
    -------
    dict of str -> str
        Mapping of figure name to HTML string.
    """
    _check_plotly()

    if subjects is None:
        subjects = dataset.subject_list

    paradigm_name = dataset.paradigm
    if paradigm_name not in _PARADIGM_HANDLERS:
        raise ValueError(f"Unsupported paradigm {paradigm_name!r}")

    compute_fn, plot_fn = _PARADIGM_HANDLERS[paradigm_name]

    sig, fig, _ = _load_and_compute(dataset, subjects, compute_fn, plot_fn)
    if sig is None:
        return {}

    return {"grand_average": fig.to_html(include_plotlyjs=True)}
