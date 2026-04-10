#!/usr/bin/env python3
"""Generate a unified interactive macro table HTML from DATASET_METADATA_CATALOG.

Outputs a self-contained HTML fragment (summary cards + DataTables table) to
``docs/source/_static/macro_table.html``.  Run this script before building the
Sphinx docs::

    python scripts/generate_macro_table.py
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd


# Ensure the repo root and sphinxext dir are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "docs" / "source" / "sphinxext"))

from dataset_constants import (  # noqa: E402
    PARADIGM_COLORS,
    PARADIGM_LABELS,
    country_flag,
    normalize_country,
    normalize_health,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARADIGM_LABELS = PARADIGM_LABELS
_PARADIGM_COLORS = PARADIGM_COLORS

_HEALTH_COLORS = {"healthy": "#2E7D32", "patients": "#E65100", "mixed": "#F9A825"}

_OUTPUT_PATH = _REPO_ROOT / "docs" / "source" / "_static" / "macro_table.html"


# ---------------------------------------------------------------------------
# Flatten metadata catalog → DataFrame
# ---------------------------------------------------------------------------


def _safe_get(obj, *attrs, default=None):
    """Safely traverse nested attributes."""
    for attr in attrs:
        if obj is None:
            return default
        obj = getattr(obj, attr, None)
    return obj if obj is not None else default


def catalog_to_dataframe(catalog=None) -> pd.DataFrame:
    """Flatten ``DATASET_METADATA_CATALOG`` into a pandas DataFrame.

    Parameters
    ----------
    catalog : dict-like, optional
        Override catalog.  Defaults to ``DATASET_METADATA_CATALOG``.

    Returns
    -------
    pd.DataFrame
        One row per dataset with all key metadata columns.
    """
    if catalog is None:
        from moabb.datasets.metadata import DATASET_METADATA_CATALOG

        catalog = DATASET_METADATA_CATALOG

    rows = []
    for name, meta in catalog.items():
        acq = meta.acquisition
        par = meta.participants
        exp = meta.experiment
        doc = meta.documentation
        aux = _safe_get(acq, "auxiliary_channels")

        ps = meta.paradigm_specific
        tags = meta.tags
        ds = meta.data_structure
        gender = _safe_get(par, "gender")
        handedness = _safe_get(par, "handedness")

        row = {
            # --- Core (visible by default) ---
            "dataset": name,
            "paradigm": _safe_get(exp, "paradigm", default=""),
            "n_subjects": _safe_get(par, "n_subjects", default=0),
            "n_channels": _safe_get(acq, "n_channels", default=0),
            "n_eeg_channels": (_safe_get(acq, "channel_types") or {}).get("eeg", 0),
            "n_classes": _safe_get(exp, "n_classes"),
            "sampling_rate": _safe_get(acq, "sampling_rate", default=0),
            "trial_duration": _safe_get(exp, "trial_duration"),
            "sessions": meta.sessions_per_subject or 1,
            "runs": meta.runs_per_session or 1,
            "health_status": _safe_get(par, "health_status", default=""),
            "doi": _safe_get(doc, "doi"),
            # --- Hidden by default (toggled via ColVis) ---
            # Experiment
            "class_labels": ", ".join(_safe_get(exp, "class_labels") or []),
            "stimulus_type": _safe_get(exp, "stimulus_type"),
            "primary_modality": _safe_get(exp, "primary_modality"),
            "feedback_type": _safe_get(exp, "feedback_type"),
            "synchronicity": _safe_get(exp, "synchronicity"),
            "mode": _safe_get(exp, "mode"),
            "study_design": _safe_get(exp, "study_design"),
            # Acquisition
            "hardware": _safe_get(acq, "hardware"),
            "reference": _safe_get(acq, "reference"),
            "sensor_type": _safe_get(acq, "sensor_type"),
            "montage": _safe_get(acq, "montage"),
            "line_freq": _safe_get(acq, "line_freq"),
            "filters": _safe_get(acq, "filters"),
            "cap_manufacturer": _safe_get(acq, "cap_manufacturer"),
            "software": _safe_get(acq, "software"),
            # Participants
            "clinical_population": _safe_get(par, "clinical_population"),
            "age_mean": _safe_get(par, "age_mean"),
            "age_min": _safe_get(par, "age_min"),
            "age_max": _safe_get(par, "age_max"),
            "gender": (
                ", ".join(f"{k}:{v}" for k, v in gender.items())
                if isinstance(gender, dict)
                else ""
            ),
            "handedness": (
                handedness
                if isinstance(handedness, str)
                else (
                    ", ".join(f"{k}:{v}" for k, v in handedness.items())
                    if isinstance(handedness, dict)
                    else ""
                )
            ),
            "bci_experience": _safe_get(par, "bci_experience"),
            # Documentation
            "license": _safe_get(doc, "license"),
            "country": _normalize_country(_safe_get(doc, "country")),
            "institution": _safe_get(doc, "institution"),
            "publication_year": _safe_get(doc, "publication_year"),
            "repository": _safe_get(doc, "repository"),
            "senior_author": _safe_get(doc, "senior_author"),
            "data_url": _safe_get(doc, "data_url"),
            # Tags
            "tags_pathology": ", ".join(_safe_get(tags, "pathology") or []),
            "tags_modality": ", ".join(_safe_get(tags, "modality") or []),
            "tags_type": ", ".join(_safe_get(tags, "type") or []),
            # Structure
            "file_format": meta.file_format or "",
            "has_eog": _safe_get(aux, "has_eog", default=False),
            "has_emg": _safe_get(aux, "has_emg", default=False),
            # Duration placeholder (to be filled manually)
            "duration_hours": None,
            # Data structure
            "n_trials": _fmt_trials(_safe_get(ds, "n_trials")),
            "n_blocks": _safe_get(ds, "n_blocks"),
            "trials_context": _safe_get(ds, "trials_context"),
            # Paradigm-specific
            "stimulus_frequencies": _fmt_freq_list(
                _safe_get(ps, "stimulus_frequencies_hz")
            ),
            "code_type": _safe_get(ps, "code_type"),
            "n_targets": _safe_get(ps, "n_targets"),
            "n_repetitions": _safe_get(ps, "n_repetitions"),
            "isi_ms": _safe_get(ps, "isi_ms"),
            "soa_ms": _safe_get(ps, "soa_ms"),
            "imagery_tasks": ", ".join(_safe_get(ps, "imagery_tasks") or []),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("dataset").reset_index(drop=True)
    return df


def _fmt_trials(val) -> str:
    """Format n_trials which can be int, dict, or str."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return str(int(val))
    if isinstance(val, dict):
        return ", ".join(f"{k}: {v}" for k, v in val.items())
    return str(val)


def _fmt_freq_list(val) -> str:
    """Format a list of stimulus frequencies compactly."""
    if not val:
        return ""
    if isinstance(val, list) and len(val) > 6:
        return f"{val[0]:g}–{val[-1]:g} Hz ({len(val)} freqs)"
    if isinstance(val, list):
        return ", ".join(f"{f:g}" for f in val)
    return str(val)


# Country/health helpers imported from dataset_constants
_normalize_country = normalize_country
_country_flag = country_flag


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


_CARD_ICONS = {
    "datasets": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>'
    ),
    "subjects": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>'
        '<circle cx="9" cy="7" r="4"/>'
        '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/>'
        '<path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
    ),
    "paradigms": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/></svg>'
    ),
    "countries": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>'
        '<path d="M2 12h20"/></svg>'
    ),
    "years": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
        '<line x1="16" y1="2" x2="16" y2="6"/>'
        '<line x1="8" y1="2" x2="8" y2="6"/>'
        '<line x1="3" y1="10" x2="21" y2="10"/></svg>'
    ),
}


def _build_paradigm_bar(df: pd.DataFrame) -> str:
    """Build an inline paradigm distribution bar."""
    counts = df["paradigm"].value_counts()
    total = len(df)
    segments = []
    for paradigm in ["imagery", "p300", "ssvep", "cvep", "rstate"]:
        n = counts.get(paradigm, 0)
        if n == 0:
            continue
        pct = n / total * 100
        color = _PARADIGM_COLORS.get(paradigm, "#757575")
        label = _PARADIGM_LABELS.get(paradigm, paradigm)
        segments.append(
            f'<div class="mt-bar-seg" style="width:{pct:.1f}%;background:{color}" '
            f'title="{label}: {n} datasets ({pct:.0f}%)">'
            f"</div>"
        )

    legend_items = []
    for paradigm in ["imagery", "p300", "ssvep", "cvep", "rstate"]:
        n = counts.get(paradigm, 0)
        if n == 0:
            continue
        color = _PARADIGM_COLORS.get(paradigm, "#757575")
        label = _PARADIGM_LABELS.get(paradigm, paradigm)
        legend_items.append(
            f'<span class="mt-bar-legend-item">'
            f'<span class="mt-bar-dot" style="background:{color}"></span>'
            f"{html.escape(label)} ({n})</span>"
        )

    return (
        f'<div class="mt-bar-container">'
        f'<div class="mt-bar">{"".join(segments)}</div>'
        f'<div class="mt-bar-legend">{"".join(legend_items)}</div>'
        f"</div>"
    )


def _build_summary_cards(df: pd.DataFrame) -> str:
    """Build HTML for the top-row summary metric cards."""
    total_datasets = len(df)
    total_subjects = int(df["n_subjects"].sum())
    n_paradigms = df["paradigm"].nunique()
    countries = df["country"].dropna().nunique()
    year_min = df["publication_year"].dropna()
    year_range = (
        f"{int(year_min.min())}–{int(year_min.max())}" if len(year_min) else "N/A"
    )

    cards_data = [
        ("datasets", "Datasets", str(total_datasets), "curated BCI datasets"),
        ("subjects", "Subjects", f"{total_subjects:,}", "total participants"),
        ("paradigms", "Paradigms", str(n_paradigms), "BCI paradigm types"),
        ("countries", "Countries", str(countries), "represented"),
        ("years", "Years", year_range, "publication span"),
    ]

    items = []
    for i, (icon_key, label, value, subtitle) in enumerate(cards_data):
        icon_svg = _CARD_ICONS.get(icon_key, "")
        items.append(
            f'<div class="mt-card" style="animation-delay:{i * 60}ms">'
            f'<div class="mt-card-icon">{icon_svg}</div>'
            f'<div class="mt-card-body">'
            f'<div class="mt-card-value">{html.escape(value)}</div>'
            f'<div class="mt-card-label">{html.escape(label)}</div>'
            f'<div class="mt-card-sub">{html.escape(subtitle)}</div>'
            f"</div></div>"
        )

    paradigm_bar = _build_paradigm_bar(df)

    return (
        f'<div class="mt-hero">'
        f'<div class="mt-cards" id="mt-cards">{"".join(items)}</div>'
        f"{paradigm_bar}"
        f"</div>"
    )


def _paradigm_tag(paradigm: str) -> str:
    """Render a color-coded paradigm pill."""
    label = _PARADIGM_LABELS.get(paradigm, paradigm)
    color = _PARADIGM_COLORS.get(paradigm, "#757575")
    return (
        f'<span class="mt-tag" style="--tag-color:{color}" '
        f'data-paradigm="{html.escape(paradigm)}">'
        f"{html.escape(label)}</span>"
    )


def _health_tag(status: str) -> str:
    """Render a health-status tag."""
    if not status:
        return ""
    normalized = normalize_health(status)
    color = _HEALTH_COLORS.get(normalized, "#757575")
    # Show original status as tooltip if it differs from normalized
    title = f' title="{html.escape(status)}"' if normalized != status else ""
    return (
        f'<span class="mt-tag mt-tag--health" style="--tag-color:{color}"{title}>'
        f"{html.escape(normalized)}</span>"
    )


_EXTERNAL_LINK_SVG = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2">'
    '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
    '<polyline points="15 3 21 3 21 9"/>'
    '<line x1="10" y1="14" x2="21" y2="3"/></svg>'
)


def _doi_link(doi: str | None) -> str:
    if not doi:
        return ""
    url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    return (
        f'<a class="mt-doi" href="{html.escape(url)}" '
        f'target="_blank" rel="noopener">'
        f"{_EXTERNAL_LINK_SVG} DOI</a>"
    )


def _dataset_link(name: str) -> str:
    """Link to the auto-generated dataset documentation page."""
    url = f"generated/moabb.datasets.{name}.html"
    return f'<a class="mt-dataset-link" href="{html.escape(url)}">{html.escape(name)}</a>'


def _fmt(val, fmt=None):
    """Format a value for table display."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if fmt:
        try:
            return fmt(val)
        except (ValueError, TypeError):
            return html.escape(str(val))
    return html.escape(str(val))


def _data_url_link(url: str | None) -> str:
    if not url:
        return ""
    return (
        f'<a class="mt-doi" href="{html.escape(url)}" '
        f'target="_blank" rel="noopener">'
        f"{_EXTERNAL_LINK_SVG} Link</a>"
    )


# Columns definition: (header, df_key, formatter, visible_by_default)
_TABLE_COLUMNS = [
    # --- Visible by default ---
    ("Dataset", "dataset", "link", True),
    ("Paradigm", "paradigm", "paradigm_tag", True),
    ("#Subj", "n_subjects", "int", True),
    ("#Chan", "n_channels", "int", True),
    ("#EEG", "n_eeg_channels", "int", True),
    ("#Classes", "n_classes", "num", True),
    ("Freq (Hz)", "sampling_rate", "num", True),
    ("Trial (s)", "trial_duration", "num", True),
    ("#Sess", "sessions", "int", True),
    ("#Runs", "runs", "int", True),
    ("Health", "health_status", "health_tag", True),
    ("#Trials", "n_trials", "str", True),
    ("Country", "country", "country", True),
    ("Year", "publication_year", "year", True),
    ("DOI", "doi", "doi_link", True),
    # --- Hidden by default ---
    ("Class Labels", "class_labels", "str", False),
    ("Stimulus", "stimulus_type", "str", False),
    ("Modality", "primary_modality", "str", False),
    ("Feedback", "feedback_type", "str", False),
    ("Sync", "synchronicity", "str", False),
    ("Mode", "mode", "str", False),
    ("Study Design", "study_design", "str", False),
    ("Hardware", "hardware", "str", False),
    ("Reference", "reference", "str", False),
    ("Sensor Type", "sensor_type", "str", False),
    ("Montage", "montage", "str", False),
    ("Line Freq", "line_freq", "num", False),
    ("Filters", "filters", "str", False),
    ("Cap Mfr", "cap_manufacturer", "str", False),
    ("Software", "software", "str", False),
    ("Clinical Pop.", "clinical_population", "str", False),
    ("Age Mean", "age_mean", "num", False),
    ("Age Min", "age_min", "num", False),
    ("Age Max", "age_max", "num", False),
    ("Gender", "gender", "str", False),
    ("Handedness", "handedness", "str", False),
    ("BCI Exp.", "bci_experience", "str", False),
    ("License", "license", "str", False),
    ("Institution", "institution", "str", False),
    ("Repository", "repository", "str", False),
    ("Duration (h)", "duration_hours", "num", False),
    ("Author", "senior_author", "str", False),
    ("Data URL", "data_url", "data_url", False),
    ("Pathology Tags", "tags_pathology", "str", False),
    ("Modality Tags", "tags_modality", "str", False),
    ("Type Tags", "tags_type", "str", False),
    ("File Format", "file_format", "str", False),
    ("EOG", "has_eog", "bool", False),
    ("EMG", "has_emg", "bool", False),
    # Data structure
    ("#Blocks", "n_blocks", "num", False),
    ("Trials Context", "trials_context", "str", False),
    # Paradigm-specific
    ("Stim. Freqs", "stimulus_frequencies", "str", False),
    ("Code Type", "code_type", "str", False),
    ("#Targets", "n_targets", "num", False),
    ("#Repetitions", "n_repetitions", "num", False),
    ("ISI (ms)", "isi_ms", "num", False),
    ("SOA (ms)", "soa_ms", "num", False),
    ("MI Tasks", "imagery_tasks", "str", False),
]


_TRUNCATE_LEN = 24


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    """Truncate text and wrap in a span with a CSS tooltip showing the full value."""
    if not text or len(text) <= max_len:
        return html.escape(text) if text else ""
    short = html.escape(text[:max_len].rstrip()) + "..."
    full_escaped = html.escape(text).replace('"', "&quot;")
    return f'<span class="mt-truncated" data-full="{full_escaped}">{short}</span>'


def _format_cell(value, fmt: str, row=None) -> str:
    """Format a single cell value according to its type."""
    if fmt == "link":
        return _dataset_link(value)
    if fmt == "paradigm_tag":
        return _paradigm_tag(value)
    if fmt == "health_tag":
        return _health_tag(value)
    if fmt == "doi_link":
        return _doi_link(value)
    if fmt == "data_url":
        return _data_url_link(value)
    if fmt == "country":
        if not value:
            return ""
        flag = _country_flag(value)
        return html.escape(f"{flag} {value}")
    if fmt == "year":
        return _fmt(value, lambda v: str(int(v)))
    if fmt == "int":
        return _fmt(value)
    if fmt == "num":
        return _fmt(value, lambda v: f"{v:g}")
    if fmt == "bool":
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        return ""
    # str — truncate long values
    text = (
        str(value)
        if value is not None and not (isinstance(value, float) and pd.isna(value))
        else ""
    )
    return _truncate(text)


def _build_table_html(df: pd.DataFrame) -> str:
    """Build the <table> element."""

    # Header
    header_cells = "".join(f"<th>{col[0]}</th>" for col in _TABLE_COLUMNS)
    thead = f"<thead><tr>{header_cells}</tr></thead>"

    # Body
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for _header, key, fmt, _vis in _TABLE_COLUMNS:
            cells.append(_format_cell(row.get(key), fmt, row))

        paradigm = row.get("paradigm", "")
        p_color = _PARADIGM_COLORS.get(paradigm, "#757575")
        tr_cells = "".join(f"<td>{c}</td>" for c in cells)
        body_rows.append(f'<tr style="--row-paradigm-color:{p_color}">{tr_cells}</tr>')

    tbody = f"<tbody>{''.join(body_rows)}</tbody>"
    return (
        f'<table id="moabb-macro-table" class="display compact nowrap" '
        f'style="width:100%">{thead}{tbody}</table>'
    )


_EXPERIMENTAL_BANNER = """\
<div class="mt-banner">
  <div class="mt-banner-icon">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86
              a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  </div>
  <div class="mt-banner-body">
    <strong>Experimental</strong> &mdash;
    This metadata catalog is under active consolidation. Some values may be
    incomplete or approximate. The information will be progressively validated
    to match the exact dataset properties.
    If you spot an error, please
    <a href="https://github.com/NeuroTechX/moabb/issues"
       target="_blank" rel="noopener">open an issue</a>.
  </div>
</div>
"""


def generate_html(df: pd.DataFrame) -> str:
    """Generate the full HTML fragment (cards + table)."""
    cards = _build_summary_cards(df)
    table = _build_table_html(df)

    return f"""\
<div class="mt-container">
{_EXPERIMENTAL_BANNER}
{cards}
<div class="mt-table-wrapper">
{table}
</div>
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Building metadata catalog...")
    df = catalog_to_dataframe()
    print(f"  {len(df)} datasets extracted, {len(df.columns)} columns")
    print(f"  Paradigms: {sorted(df['paradigm'].unique())}")
    print(f"  Total subjects: {int(df['n_subjects'].sum())}")

    html_content = generate_html(df)

    os.makedirs(_OUTPUT_PATH.parent, exist_ok=True)
    _OUTPUT_PATH.write_text(html_content, encoding="utf-8")
    print(f"  Written to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
