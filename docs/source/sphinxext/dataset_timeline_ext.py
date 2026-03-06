"""Sphinx extension: enhance dataset documentation pages.

1. Injects an enhanced dataset card (paradigm chips, stats, action buttons)
2. Injects an adaptive visual summary grid with panels for timeline, class
   balance, sessions, channels, and HED tags (when available)
3. Restructures the docstring into a tabbed layout (Overview, Code Examples,
   Metadata, Notes)
4. Shows inherited methods below tabs

Pre-generated SVG images live in ``_static/timelines/<ClassName>.svg`` and
``_static/viz/<ClassName>_classes.svg`` / ``<ClassName>_sessions.svg``.

To regenerate *all* SVGs (timelines + viz), run (from the repo root)::

    PYTHONPATH=. python scripts/generate_dataset_viz.py
"""

import csv
import functools
import inspect
import json
import math
import os
import re
import statistics
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote
from urllib.request import Request, urlopen


_PARADIGM_LABELS = {
    "p300": "P300 / ERP",
    "erp": "P300 / ERP",
    "imagery": "Motor Imagery",
    "ssvep": "SSVEP",
    "cvep": "c-VEP",
    "rstate": "Resting State",
}

_PARADIGM_COLORS = {
    "p300": "#D32F2F",
    "erp": "#D32F2F",
    "imagery": "#1565C0",
    "ssvep": "#2E7D32",
    "cvep": "#00695C",
    "rstate": "#546E7A",
}

_BENCHMARK_FILES = [
    ("within_session_mi_left_vs_right_hand.csv", "MI left vs right"),
    ("within_session_mi_all_classes.csv", "MI all classes"),
    ("within_session_mi_right_hand_vs_feet.csv", "MI right hand vs feet"),
    ("within_session_ssvep_all_classes.csv", "SSVEP all classes"),
    ("within_session_erp_p300_all_classes.csv", "ERP/P300 all classes"),
]
_BENCHMARK_CONTEXT_CACHE = {}
_DOI_METADATA_CACHE = {}
_DOI_CACHE_LOADED = False
_DOI_RE = re.compile(r"^10\.\d{4,}/", re.IGNORECASE)
_RST_INLINE_RE = re.compile(r"\*\*(.+?)\*\*|``(.+?)``|\*(.+?)\*")
_RST_FOOTNOTE_RE = re.compile(r"\s*\[\d+\]_\.?")
_RST_LIST_SPLIT_RE = re.compile(r"\s+- ")
_DATASET_PAGEVIEWS_CACHE = None
_DATASET_PAGEVIEWS_CACHE_SRC = None

# ---------------------------------------------------------------------------
# Official Creative Commons SVG icons (from creativecommons/cc-assets)
# Stored as individual .svg files in _static/icons/cc/.
# ---------------------------------------------------------------------------

_CC_ICONS_DIR = os.path.join(os.path.dirname(__file__), "..", "_static", "icons", "cc")


@functools.lru_cache(maxsize=None)
def _cc_icon_svg(icon_key, size=16):
    """Return an inline ``<svg>`` element for a Creative Commons icon.

    Reads the SVG file from ``_static/icons/cc/<icon_key>.svg`` (cached via
    ``lru_cache``) and injects ``width``/``height``/``aria-hidden`` attributes
    so it can be embedded inline.
    """
    svg_path = os.path.join(_CC_ICONS_DIR, f"{icon_key}.svg")
    try:
        with open(svg_path, "r") as fh:
            svg = fh.read().strip()
    except FileNotFoundError:
        return ""
    if not svg:
        return ""
    # Inject width/height/aria-hidden into the opening <svg> tag.
    return svg.replace(
        "<svg ",
        f'<svg width="{size}" height="{size}" aria-hidden="true" ',
        1,
    )


# ---------------------------------------------------------------------------
# License resolution
# ---------------------------------------------------------------------------

_LICENSE_INFO = {
    "cc-by-4.0": (
        "CC BY 4.0",
        "https://creativecommons.org/licenses/by/4.0/",
        ["cc", "by"],
    ),
    "cc-by-1.0": (
        "CC BY 1.0",
        "https://creativecommons.org/licenses/by/1.0/",
        ["cc", "by"],
    ),
    "cc-by-sa-4.0": (
        "CC BY-SA 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        ["cc", "by", "sa"],
    ),
    "cc-by-nc-4.0": (
        "CC BY-NC 4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/",
        ["cc", "by", "nc"],
    ),
    "cc-by-nc-sa-4.0": (
        "CC BY-NC-SA 4.0",
        "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        ["cc", "by", "nc", "sa"],
    ),
    "cc-by-nc-nd-4.0": (
        "CC BY-NC-ND 4.0",
        "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        ["cc", "by", "nc", "nd"],
    ),
    "cc-by-nd-4.0": (
        "CC BY-ND 4.0",
        "https://creativecommons.org/licenses/by-nd/4.0/",
        ["cc", "by", "nd"],
    ),
    "cc0-1.0": (
        "CC0 1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        ["cc", "zero"],
    ),
    "odc-by-1.0": ("ODC-By 1.0", "https://opendatacommons.org/licenses/by/1-0/", []),
    "gpl-3.0": ("GPL 3.0", "https://www.gnu.org/licenses/gpl-3.0.html", []),
    "unknown": ("Unknown", None, []),
}

# Aliases for non-standard license strings found in dataset metadata.
_LICENSE_ALIASES = {
    "creative commons attribution license": "cc-by-4.0",
    "cc by": "cc-by-4.0",
    "cc by 4.0": "cc-by-4.0",
}


def _normalize_license(raw):
    """Normalize a raw license string to a ``_LICENSE_INFO`` key (or *None*)."""
    if not raw:
        return None
    key = raw.strip().lower().replace(" ", "-")
    if key in _LICENSE_INFO:
        return key
    # Try alias lookup: space-preserved form first, then hyphenated form.
    alias_key = raw.strip().lower()
    return _LICENSE_ALIASES.get(alias_key) or _LICENSE_ALIASES.get(key)


def _is_concrete_dataset(obj):
    """Check if *obj* is a concrete (instantiable) MOABB dataset class."""
    try:
        from moabb.datasets.base import BaseDataset
    except Exception:
        return False
    return (
        isinstance(obj, type)
        and issubclass(obj, BaseDataset)
        and obj is not BaseDataset
        and not getattr(obj, "__abstractmethods__", set())
    )


def _repo_root():
    """Return repository root path (relative to this extension file)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _get_dataset_source_url(obj):
    """Build a GitHub source URL for a dataset class."""
    try:
        src_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        if not src_file:
            return None
        repo_root = _repo_root()
        rel_path = os.path.relpath(src_file, repo_root)
        if rel_path.startswith(".."):
            return None
        src_lines, start = inspect.getsourcelines(obj)
        end = start + len(src_lines) - 1
        rel_path = rel_path.replace(os.sep, "/")
        return (
            f"https://github.com/NeuroTechX/moabb/blob/develop/{rel_path}#L{start}-L{end}"
        )
    except Exception:
        return None


def _normalize_doi(value):
    """Normalize DOI values that may include URL prefixes."""
    if not value:
        return None
    text = str(value).strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if text.lower().startswith(prefix):
            return text[len(prefix) :]
    return text


def _dataset_dom_id(prefix, cls_name):
    """Return a stable DOM id fragment for a dataset class."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", cls_name).strip("-").lower()
    return f"{prefix}-{slug}"


def _is_likely_doi(value):
    """Return True if the value matches a DOI-like pattern."""
    norm = _normalize_doi(value)
    return bool(norm and _DOI_RE.match(norm))


def _load_doi_cache_once():
    """Load local DOI cache used by tests (if present)."""
    global _DOI_CACHE_LOADED
    if _DOI_CACHE_LOADED:
        return
    _DOI_CACHE_LOADED = True
    cache_path = os.path.join(_repo_root(), "moabb", "tests", "doi_cache.json")
    if not os.path.exists(cache_path):
        return
    try:
        with open(cache_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return

    for key, value in payload.items():
        if key == "_metadata":
            continue
        norm = _normalize_doi(key)
        if not norm:
            continue
        _DOI_METADATA_CACHE[norm.lower()] = value


def _resolve_doi_metadata(doi):
    """Resolve DOI metadata from local cache, then public DOI API as fallback."""
    norm = _normalize_doi(doi)
    if not norm or not _is_likely_doi(norm):
        return None
    key = norm.lower()

    _load_doi_cache_once()
    if key in _DOI_METADATA_CACHE:
        return _DOI_METADATA_CACHE[key]

    # Public API fallback via DOI content negotiation (citeproc JSON)
    try:
        req = Request(
            f"https://doi.org/{quote(norm)}",
            headers={"Accept": "application/citeproc+json"},
        )
        with urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in data.get("author", [])
            if isinstance(a, dict)
        ]
        issued = data.get("issued", {}).get("date-parts", [[None]])
        year = issued[0][0] if issued and issued[0] and issued[0][0] else None
        resolved = {
            "title": data.get("title"),
            "authors": authors,
            "year": year,
            "doi": norm,
        }
        _DOI_METADATA_CACHE[key] = resolved
        return resolved
    except Exception:
        _DOI_METADATA_CACHE[key] = None
        return None


def _format_resolved_citation(meta):
    """Build a compact one-line citation string from resolved metadata."""
    if not meta:
        return ""
    title = str(meta.get("title") or "").strip().rstrip(".")
    if len(title) > 170:
        title = title[:167].rstrip() + "..."
    authors = [a for a in (meta.get("authors") or []) if a]
    year = meta.get("year")

    author_text = ""
    if authors:
        surnames = [a.split()[-1] for a in authors if a.split()]
        if len(surnames) == 1:
            author_text = surnames[0]
        elif len(surnames) == 2:
            author_text = f"{surnames[0]} & {surnames[1]}"
        elif len(surnames) > 2:
            author_text = f"{surnames[0]} et al."

    lead = author_text or "Citation"
    if year:
        lead += f" ({year})"
    if title:
        return f"{lead}. {title}."
    return lead


def _select_preferred_paper_doi(dataset_doi, documentation_doi, associated_paper_doi):
    """Pick the DOI that should represent the associated paper.

    Priority:
    1) explicit documentation.associated_paper_doi
    2) documentation.doi
    3) dataset-level doi
    """
    candidates = [associated_paper_doi, documentation_doi, dataset_doi]
    for value in candidates:
        norm = _normalize_doi(value)
        if norm and _is_likely_doi(norm):
            return norm
    for value in candidates:
        norm = _normalize_doi(value)
        if norm:
            return norm
    return None


# ---------------------------------------------------------------------------
# Dataset info extraction
# ---------------------------------------------------------------------------


def _get_dataset_info(obj):
    """Try to instantiate the dataset and extract key info including metadata."""
    try:
        ds = obj()
        paradigm = getattr(ds, "paradigm", None)
        subject_list = getattr(ds, "subject_list", None)
        n_subjects = len(subject_list) if subject_list else None
        default_subject = subject_list[0] if subject_list else 1
        n_sessions = getattr(ds, "n_sessions", None)
        code = getattr(ds, "code", None)
        dataset_doi = getattr(ds, "doi", None)
        event_id = getattr(ds, "event_id", None) or {}
        interval = getattr(ds, "interval", None)

        # Extract richer stats from METADATA, falling back to catalog .metadata
        metadata = getattr(ds, "METADATA", None) or getattr(type(ds), "METADATA", None)
        if metadata is None:
            metadata = getattr(ds, "metadata", None)

        sampling_rate = None
        n_channels = None
        channel_types = None
        montage = None
        reference = None
        filter_range = None
        line_freq = None
        sensor_type = None
        filters = None
        n_classes = None
        class_labels = None
        trial_duration = None
        n_trials_per_class = None
        runs_per_session = None
        sessions_per_subject = None
        hed_tags = None
        exp = None
        investigators = None
        senior_author = None
        contact_info = None
        institution = None
        country = None
        publication_year = None
        paper_description = None
        documentation_doi = None
        associated_paper_doi = None
        license_str = None

        if metadata is not None:
            acq = getattr(metadata, "acquisition", None)
            if acq is not None:
                sampling_rate = getattr(acq, "sampling_rate", None)
                n_channels = getattr(acq, "n_channels", None)
                channel_types = getattr(acq, "channel_types", None)
                montage = getattr(acq, "montage", None)
                reference = getattr(acq, "reference", None)
                low_cut = getattr(acq, "low_cut_hz", None)
                high_cut = getattr(acq, "high_cut_hz", None)
                line_freq = getattr(acq, "line_freq", None)
                sensor_type = getattr(acq, "sensor_type", None)
                filters = getattr(acq, "filters", None)
                if low_cut is not None or high_cut is not None:
                    low_label = "?" if low_cut is None else f"{low_cut:g}"
                    high_label = "?" if high_cut is None else f"{high_cut:g}"
                    filter_range = f"{low_label}–{high_label} Hz"
                elif filters:
                    filter_range = str(filters)

            exp = getattr(metadata, "experiment", None)
            if exp is not None:
                n_classes = getattr(exp, "n_classes", None)
                class_labels = getattr(exp, "class_labels", None)
                trial_duration = getattr(exp, "trial_duration", None)
                hed_tags = getattr(exp, "hed_tags", None)

            data_struct = getattr(metadata, "data_structure", None)
            if data_struct is not None:
                n_trials_per_class = getattr(data_struct, "n_trials_per_class", None)

            runs_per_session = getattr(metadata, "runs_per_session", None)
            sessions_per_subject = getattr(metadata, "sessions_per_subject", None)

            doc = getattr(metadata, "documentation", None)
            if doc is not None:
                documentation_doi = getattr(doc, "doi", None)
                associated_paper_doi = getattr(doc, "associated_paper_doi", None)
                investigators = getattr(doc, "investigators", None)
                senior_author = getattr(doc, "senior_author", None)
                contact_info = getattr(doc, "contact_info", None)
                institution = getattr(doc, "institution", None)
                country = getattr(doc, "country", None)
                publication_year = getattr(doc, "publication_year", None)
                paper_description = getattr(doc, "description", None)
                license_str = getattr(doc, "license", None)

        paper_doi = _select_preferred_paper_doi(
            dataset_doi=dataset_doi,
            documentation_doi=documentation_doi,
            associated_paper_doi=associated_paper_doi,
        )

        # Fallbacks
        if n_classes is None and event_id:
            n_classes = len(event_id)
        if class_labels is None and event_id:
            class_labels = list(event_id.keys())
        if trial_duration is None and interval is not None:
            trial_duration = float(interval[1] - interval[0])
        if not hed_tags:
            try:
                from moabb.datasets.bids_interface import _build_hed_sidecar_annotations

                hed_tags = _build_hed_sidecar_annotations(ds)
            except Exception:
                hed_tags = hed_tags or None

        return {
            "paradigm": paradigm,
            "n_subjects": n_subjects,
            "default_subject": default_subject,
            "n_sessions": n_sessions,
            "code": code,
            "doi": dataset_doi,
            "dataset_doi": dataset_doi,
            "documentation_doi": documentation_doi,
            "associated_paper_doi": associated_paper_doi,
            "paper_doi": paper_doi,
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_types": channel_types,
            "montage": montage,
            "reference": reference,
            "filter_range": filter_range,
            "line_freq": line_freq,
            "sensor_type": sensor_type,
            "filters": filters,
            "n_classes": n_classes,
            "class_labels": class_labels,
            "trial_duration": trial_duration,
            "n_trials_per_class": n_trials_per_class,
            "event_id": event_id,
            "runs_per_session": runs_per_session,
            "sessions_per_subject": sessions_per_subject,
            "hed_tags": hed_tags,
            "investigators": investigators,
            "senior_author": senior_author,
            "contact_info": contact_info,
            "institution": institution,
            "country": country,
            "publication_year": publication_year,
            "paper_description": paper_description,
            "license": license_str,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _format_duration_seconds(seconds):
    """Return a short human-readable duration string."""
    if seconds is None:
        return None
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:g} s"


def _split_hed_top_level(hed_str):
    """Split a HED string by top-level commas (ignoring nested groups)."""
    if not hed_str:
        return []
    parts, buf = [], []
    depth = 0
    for ch in str(hed_str):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)
    piece = "".join(buf).strip()
    if piece:
        parts.append(piece)
    return parts


def _hed_token_label(element):
    """Extract a compact display token from a HED element."""
    token = str(element).strip().strip("()").strip()
    if not token:
        return ""
    token = token.split(",")[0].strip()
    if "/" in token:
        token = token.split("/", 1)[0].strip()
    return token


def _hed_element_to_tree(element):
    """Convert a HED element into a simple tree node dict."""
    text = str(element).strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        parts = _split_hed_top_level(inner)
        if not parts:
            return None
        head = {"label": _hed_token_label(parts[0]), "children": []}
        for child in parts[1:]:
            node = _hed_element_to_tree(child)
            if node:
                head["children"].append(node)
        return head
    return {"label": _hed_token_label(text), "children": []}


def _render_hed_tree_lines(nodes, prefix=""):
    """Render tree nodes as ASCII hierarchy lines."""
    lines = []
    valid_nodes = [n for n in nodes if n]
    for i, node in enumerate(valid_nodes):
        last = i == len(valid_nodes) - 1
        branch = "└─ " if last else "├─ "
        lines.append(f"{prefix}{branch}{node['label']}")
        if node.get("children"):
            child_prefix = prefix + ("   " if last else "│  ")
            lines.extend(_render_hed_tree_lines(node["children"], child_prefix))
    return lines


def _extract_score_mean(cell):
    """Parse a benchmark score string like '77.82±12.23' and return 77.82."""
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _get_benchmark_context(cls_name):
    """Return summary of benchmark tables containing this dataset."""
    if cls_name in _BENCHMARK_CONTEXT_CACHE:
        return _BENCHMARK_CONTEXT_CACHE[cls_name]

    root_dir = _repo_root()
    results_dir = os.path.join(root_dir, "results")
    col_name = f":class:`{cls_name}`"

    entries = []
    for fname, label in _BENCHMARK_FILES:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or col_name not in reader.fieldnames:
                    continue
                scores = []
                for row in reader:
                    score = _extract_score_mean(row.get(col_name))
                    if score is not None:
                        scores.append(score)
                if scores:
                    mean_val = statistics.mean(scores)
                    std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
                    entries.append(
                        {
                            "label": label,
                            "max": max(scores),
                            "median": statistics.median(scores),
                            "mean": mean_val,
                            "std": std_val,
                            "n_pipelines": len(scores),
                        }
                    )
        except Exception:
            continue

    context = {"n_tables": len(entries), "entries": entries}
    _BENCHMARK_CONTEXT_CACHE[cls_name] = context
    return context


def _estimate_relative_difficulty(info, benchmark_ctx):
    """Estimate a coarse relative difficulty score/label."""
    n_classes = info.get("n_classes")
    class_labels = info.get("class_labels") or []
    display_classes = len(class_labels) if class_labels else n_classes
    n_subjects = info.get("n_subjects")
    n_sessions = info.get("n_sessions")
    n_channels = info.get("n_channels")

    score = 0.0
    if display_classes is not None:
        if display_classes >= 4:
            score += 1.2
        elif display_classes >= 3:
            score += 0.7
    if n_subjects is not None and n_subjects < 12:
        score += 0.8
    if n_sessions == 1:
        score += 0.6
    if n_channels is not None and n_channels < 16:
        score += 0.5

    medians = [e["median"] for e in benchmark_ctx.get("entries", []) if "median" in e]
    if medians:
        global_median = statistics.median(medians)
        if global_median < 70:
            score += 0.8
        elif global_median < 80:
            score += 0.4
        elif global_median >= 88:
            score -= 0.4

    if score <= 0.6:
        return "Low", "●○○○○"
    if score <= 1.4:
        return "Medium", "●●○○○"
    if score <= 2.2:
        return "Moderate", "●●●○○"
    if score <= 3.0:
        return "High", "●●●●○"
    return "Very high", "●●●●●"


def _make_benchmark_context_html(cls_name, info):
    """Build benchmark-context card HTML for the dataset."""
    ctx = _get_benchmark_context(cls_name)
    if not ctx["entries"]:
        return ""
    n_subjects = info.get("n_subjects")
    n_sessions = info.get("n_sessions")
    sample_frame = ""
    if n_subjects is not None and n_sessions is not None:
        sample_frame = f"{n_subjects} subjects × {n_sessions} sessions"

    rows = []
    for entry in ctx["entries"][:4]:
        stats_str = (
            f'Max {entry["max"]:.2f} · '
            f'Median {entry["median"]:.2f} · '
            f'Mean {entry["mean"]:.2f} · '
            f'Std {entry["std"]:.2f}'
        )
        rows.append(
            "<li>"
            f'<span>{escape(entry["label"])} '
            f'<em>{entry["n_pipelines"]} pipelines</em></span>'
            f'<strong class="ds-bench-stats">{stats_str}</strong>'
            "</li>"
        )
    rows_html = "\n      ".join(rows)
    return (
        '<div class="ds-benchmark-context">'
        '<div class="ds-benchmark-head">'
        '<p class="ds-benchmark-title">Benchmark Context</p>'
        '<span class="ds-eval-pill">WithinSession</span>'
        "</div>"
        f'<p class="ds-benchmark-summary">Included in {ctx["n_tables"]} MOABB benchmark table(s). '
        "Scores are across available pipelines (WithinSession accuracy).</p>"
        f'<p class="ds-benchmark-meta"><span><strong>Sample frame:</strong> {escape(sample_frame or "N/A")}</span></p>'
        f"<ul>{rows_html}</ul>"
        "</div>"
    )


def _make_citation_impact_html(
    info,
    benchmark_ctx,
    *,
    live_citations=True,
    pageview_counts=None,
    pageview_rank=None,
    pageview_meta=None,
):
    """Build a compact citation, impact, and visibility block."""
    code = str(info.get("code") or "")
    paper_doi = _normalize_doi(info.get("paper_doi") or info.get("doi"))
    dataset_doi = _normalize_doi(info.get("dataset_doi") or info.get("doi"))
    if not code and not paper_doi and not dataset_doi:
        return ""

    items = []
    script_html = ""
    if paper_doi:
        doi_link_href = escape(f"https://doi.org/{quote(paper_doi, safe='')}", quote=True)
        items.append(
            f'<li><span>Paper DOI</span><a href="{doi_link_href}" '
            f'target="_blank" rel="noopener">{escape(paper_doi)}</a></li>'
        )
        if _is_likely_doi(paper_doi):
            if live_citations:
                items.append(
                    f'<li><span>Citations</span><strong class="ds-citation-count" data-doi="{escape(paper_doi)}">Loading…</strong></li>'
                )

                openalex_id = quote(f"https://doi.org/{paper_doi}", safe="")
                openalex_url = f"https://api.openalex.org/works/{openalex_id}"
                crossref_url = f"https://api.crossref.org/works/{quote(paper_doi)}"
                items.append(
                    "<li><span>Public API</span>"
                    f'<span class="ds-citation-links"><a href="{crossref_url}" target="_blank" rel="noopener">Crossref</a>'
                    "&nbsp;|&nbsp;"
                    f'<a href="{openalex_url}" target="_blank" rel="noopener">OpenAlex</a></span>'
                    "</li>"
                )
                script_html = """
<script>
(function () {
  if (window.__moabbCitationCountsInit) return;
  window.__moabbCitationCountsInit = true;

  function fmt(value) {
    return (typeof value === "number" && Number.isFinite(value))
      ? value.toLocaleString()
      : "N/A";
  }

  function setBoth(el, openalexCount, crossrefCount) {
    el.textContent = "OpenAlex: " + fmt(openalexCount) + " | Crossref: " + fmt(crossrefCount);
  }

  async function fetchOpenAlex(doi) {
    const id = encodeURIComponent("https://doi.org/" + doi);
    const resp = await fetch("https://api.openalex.org/works/" + id);
    if (!resp.ok) throw new Error("OpenAlex request failed");
    const data = await resp.json();
    return data && typeof data.cited_by_count === "number"
      ? data.cited_by_count
      : null;
  }

  async function fetchCrossref(doi) {
    const resp = await fetch("https://api.crossref.org/works/" + encodeURIComponent(doi));
    if (!resp.ok) throw new Error("Crossref request failed");
    const data = await resp.json();
    const count = data && data.message ? data.message["is-referenced-by-count"] : null;
    return typeof count === "number" ? count : null;
  }

  document.querySelectorAll(".ds-citation-count[data-doi]").forEach(async function (el) {
    const doi = (el.getAttribute("data-doi") || "").trim();
    if (!doi) {
      setBoth(el, null, null);
      return;
    }
    const [oaRes, crRes] = await Promise.allSettled([
      fetchOpenAlex(doi),
      fetchCrossref(doi),
    ]);
    const oa = oaRes.status === "fulfilled" ? oaRes.value : null;
    const cr = crRes.status === "fulfilled" ? crRes.value : null;
    setBoth(el, oa, cr);
  });
})();
</script>
"""
            else:
                doi_static_href = escape(
                    f"https://doi.org/{quote(paper_doi, safe='')}", quote=True
                )
                items.append(
                    f'<li><span>Citations</span><a href="{doi_static_href}" '
                    f'target="_blank" rel="noopener">See DOI</a></li>'
                )
    if dataset_doi and dataset_doi != paper_doi:
        data_doi_href = escape(
            f"https://doi.org/{quote(dataset_doi, safe='')}", quote=True
        )
        items.append(
            f'<li><span>Data DOI</span><a href="{data_doi_href}" '
            f'target="_blank" rel="noopener">{escape(dataset_doi)}</a></li>'
        )
    if benchmark_ctx and benchmark_ctx.get("n_tables"):
        items.append(
            f'<li><span>MOABB tables</span><strong>{benchmark_ctx["n_tables"]} (WithinSession)</strong></li>'
        )
    # --- Page Views row (single rich entry in the same list) ---
    if isinstance(pageview_counts, dict) and any(
        key in pageview_counts for key in ("last30", "all_time", "weekly_12")
    ):
        last30 = pageview_counts.get("last30")
        all_time = pageview_counts.get("all_time")
        updated_str = _format_updated_utc((pageview_meta or {}).get("generated_at_utc"))

        # Rank line
        if (
            isinstance(pageview_rank, dict)
            and pageview_rank.get("rank")
            and pageview_rank.get("total")
        ):
            rank_line = (
                f'<div class="ds-pv-rank">#{int(pageview_rank["rank"])} of '
                f'{int(pageview_rank["total"])} · Top {int(pageview_rank.get("top_percent", 0))}% most viewed</div>'
            )
        else:
            rank_line = '<div class="ds-pv-rank">Ranking: n/a</div>'

        # Sparkline
        sparkline_cell = ""
        weekly = pageview_counts.get("weekly_12")
        if weekly:
            sparkline_cell = (
                f'<div class="ds-pv-spark" aria-label="Page views trend (last 12 weeks)">'
                f"{_sparkline_svg(weekly)}</div>"
            )

        # Compose the rich right-hand value
        pv_value = (
            f'<div class="ds-pv-detail">'
            f'<div class="ds-pv-body">'
            f'<div class="ds-pv-metrics">'
            f"30d: <strong>{_format_count(last30)}</strong>"
            f' <span class="ds-provenance-sep">·</span> '
            f"all-time: <strong>{_format_count(all_time)}</strong>"
            f"</div>"
            f"{rank_line}"
            f'<div class="ds-pv-updated">Updated: {updated_str}</div>'
            f"</div>"
            f"{sparkline_cell}"
            f"</div>"
        )
        items.append(f'<li class="ds-pv-row"><span>Page Views</span>{pv_value}</li>')

    if not items:
        return ""

    list_html = "\n      ".join(items)
    return (
        '<div class="ds-citation-impact">'
        '<p class="ds-citation-title">Citation &amp; Impact</p>'
        f"<ul>{list_html}</ul>"
        f"{script_html}"
        "</div>"
    )


def _extract_description_text(lines):
    """Extract plain description lines from docstring, skipping admonitions/directives."""

    def _skip_directive_block(start_idx):
        directive_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        i = start_idx + 1
        while i < len(lines):
            if lines[i].strip() == "":
                i += 1
                continue
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            if line_indent > directive_indent:
                i += 1
                continue
            break
        return i

    desc = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Skip admonition blocks (directive + indented body)
        if stripped.startswith(".. admonition::"):
            i = _skip_directive_block(i)
            continue
        # Stop at rubrics and version directives
        if stripped.startswith(".. rubric::"):
            break
        if stripped.startswith(
            (".. versionadded::", ".. versionchanged::", ".. deprecated::")
        ):
            break
        if stripped.startswith(".. "):
            i = _skip_directive_block(i)
            continue
        # Stop at underline-style "references" or "References" header
        if (
            stripped.lower() in ("references", "references:")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= {"-", "=", "~"}
            and lines[i + 1].strip()
        ):
            break
        desc.append(stripped)
        i += 1
    # Strip leading/trailing blanks
    while desc and not desc[0]:
        desc.pop(0)
    while desc and not desc[-1]:
        desc.pop()
    return desc


def _rst_paragraph_to_html(text):
    """Convert a paragraph of reST-like text to simple HTML.

    Handles: **bold**, *italic*, ``code``, list items (- prefix),
    and strips footnote references like [1]_.
    """
    # Strip reST footnote references
    text = _RST_FOOTNOTE_RE.sub("", text)

    # Check if this is a list block (lines starting with "- ")
    if " - " in text and text.lstrip().startswith("- "):
        # Split on " - " pattern that indicates list items
        items = _RST_LIST_SPLIT_RE.split(text)
        items = [it.strip() for it in items if it.strip()]
        formatted = []
        for item in items:
            item = _rst_inline_to_html(item)
            formatted.append(f"<li>{item}</li>")
        return f'<ul class="ds-overview-list">{"".join(formatted)}</ul>'

    return f"<p>{_rst_inline_to_html(text)}</p>"


def _rst_inline_to_html(text):
    """Convert reST inline markup to HTML, escaping the rest."""
    parts = []
    pos = 0
    # Match **bold**, *italic*, ``code`` — process in order of appearance
    for m in _RST_INLINE_RE.finditer(text):
        # Escape text before this match
        parts.append(escape(text[pos : m.start()]))
        if m.group(1) is not None:
            parts.append(f"<strong>{escape(m.group(1))}</strong>")
        elif m.group(2) is not None:
            parts.append(f"<code>{escape(m.group(2))}</code>")
        elif m.group(3) is not None:
            parts.append(f"<em>{escape(m.group(3))}</em>")
        pos = m.end()
    parts.append(escape(text[pos:]))
    return "".join(parts)


def _make_overview_teaser_html(description_lines, cls_name):
    """Build a collapsible overview teaser panel with key facts."""
    if not description_lines:
        return ""

    # --- Parse all paragraphs and references ---
    all_paragraphs = []
    current = []
    ref_lines = []
    in_refs = False
    for line in description_lines:
        if re.match(r"^\.\.\s+rubric::\s*References", line):
            in_refs = True
            continue
        if in_refs:
            ref_lines.append(line)
            continue
        if not line:
            if current:
                all_paragraphs.append(" ".join(current))
                current = []
        else:
            # If this line starts a list item and we have non-list content
            # buffered, flush the buffer first
            if line.startswith("- ") and current and not current[0].startswith("- "):
                all_paragraphs.append(" ".join(current))
                current = []
            current.append(line)
    if current:
        all_paragraphs.append(" ".join(current))

    all_paragraphs = [p.strip() for p in all_paragraphs if p.strip()]

    if not all_paragraphs:
        return ""

    # --- Split into teaser (visible) vs overflow (hidden) ---
    # Show enough paragraphs to fill ~10-15 lines (~800 chars)
    TEASER_CHAR_LIMIT = 800
    teaser_paragraphs = []
    overflow_paragraphs = []
    char_count = 0
    for i, p in enumerate(all_paragraphs):
        if char_count < TEASER_CHAR_LIMIT or i == 0:
            teaser_paragraphs.append(p)
            char_count += len(p)
        else:
            overflow_paragraphs.append(p)

    # --- Build overflow HTML (hidden by default) ---
    full_html_parts = []
    for p in overflow_paragraphs:
        full_html_parts.append(_rst_paragraph_to_html(p))

    # References (collapsed inside expanded)
    ref_html = ""
    ref_text = [r for r in ref_lines if r.strip()]
    if ref_text:
        ref_content = "".join(f"<p>{_rst_inline_to_html(r)}</p>" for r in ref_text)
        ref_html = (
            '<details class="ds-overview-refs">'
            "<summary>References</summary>"
            f"{ref_content}"
            "</details>"
        )

    full_section = ""
    if full_html_parts or ref_html:
        full_content = "\n".join(full_html_parts)
        full_section = (
            f'<div class="ds-overview-full">' f"{full_content}" f"{ref_html}" f"</div>"
        )

    # --- Compose component ---
    overview_id = _dataset_dom_id("ds-overview", cls_name)

    teaser_parts = []
    for p in teaser_paragraphs:
        html = _rst_paragraph_to_html(p)
        # Add class to <p> and <ul> elements for consistent styling
        html = html.replace("<p>", '<p class="ds-overview-text">', 1)
        html = html.replace(
            '<ul class="ds-overview-list">',
            '<ul class="ds-overview-list ds-overview-text">',
            1,
        )
        teaser_parts.append(html)
    teaser_html = "".join(teaser_parts)

    # Only show expand controls if there's overflow content
    has_overflow = bool(full_section)

    toggle_btn = ""
    if has_overflow:
        toggle_btn = (
            f'<button class="ds-overview-toggle" type="button" '
            f'aria-expanded="false" aria-controls="{overview_id}" '
            f"onclick=\"var el=this.closest('.ds-overview-teaser');"
            f"el.classList.toggle('ds-expanded');"
            f"var exp=el.classList.contains('ds-expanded');"
            f"this.setAttribute('aria-expanded',exp?'true':'false');"
            f"this.textContent=exp?'Show less ▴':'Show more ▾';\">"
            f"Show more ▾</button>"
        )

    return (
        f'<div class="ds-overview-teaser">'
        f'<p class="ds-overview-title">Overview</p>'
        f"{teaser_html}"
        f"{full_section}"
        f'<div class="ds-overview-actions">'
        f"{toggle_btn}"
        f'<a class="ds-overview-tab-link" href="#{overview_id}" '
        f'onclick="event.preventDefault();'
        f"var tab=document.querySelector('.ds-doc-tabs .sd-tab-label');"
        f"if(tab){{tab.click();}}"
        f"var target=document.getElementById('{overview_id}');"
        f"if(target){{target.scrollIntoView({{behavior:'smooth',block:'start'}});}}"
        f'">Open in Overview tab →</a>'
        f"</div>"
        f"</div>"
    )


def _make_hed_summary_html(info):
    """Build HTML summary for embedded HED tags."""
    hed_map = info.get("hed_tags") if info else None
    event_id = info.get("event_id") if info else None
    event_total = len(event_id) if isinstance(event_id, dict) and event_id else None

    if not hed_map:
        return ""

    items = list(hed_map.items())
    tagged = len(items)
    denom = event_total if event_total else tagged
    coverage = f"{tagged}/{denom}"

    family_counts = {}
    event_rows = []
    tree_items = []
    for event_name, hed_str in items:
        elements = _split_hed_top_level(hed_str)
        tokens = []
        for elem in elements:
            t = _hed_token_label(elem)
            if t and t not in tokens:
                tokens.append(t)
        for t in tokens:
            family_counts[t] = family_counts.get(t, 0) + 1
        chip_html = "".join(
            f'<span class="ds-hed-chip">{escape(tok)}</span>' for tok in tokens[:5]
        )
        event_rows.append(
            '<div class="ds-hed-event-row">'
            f'<span class="ds-hed-event-name">{escape(str(event_name))}</span>'
            f'<div class="ds-hed-chip-wrap" title="{escape(str(hed_str))}">{chip_html}</div>'
            "</div>"
        )
        tree_nodes = [_hed_element_to_tree(e) for e in elements]
        tree_lines = _render_hed_tree_lines(tree_nodes)
        tree_text = "\n".join(tree_lines) if tree_lines else "(no tree)"
        tree_items.append(
            '<details class="ds-hed-tree-item">'
            f'<summary class="ds-hed-tree-summary">Tree · {escape(str(event_name))}</summary>'
            f'<pre class="ds-hed-tree-pre">{escape(tree_text)}</pre>'
            "</details>"
        )

    top_families = sorted(family_counts.items(), key=lambda x: (-x[1], x[0]))[:6]
    max_count = max([c for _, c in top_families], default=1)
    bar_rows = []
    for fam, count in top_families:
        width = int((count / max_count) * 100)
        bar_rows.append(
            '<div class="ds-hed-bar-row">'
            f'<span class="ds-hed-bar-label">{escape(fam)}</span>'
            f'<div class="ds-hed-bar"><i style="width:{width}%"></i></div>'
            f"<strong>{count}</strong>"
            "</div>"
        )

    return (
        '<div class="ds-hed-card">'
        '<div class="ds-hed-head">'
        '<span class="ds-hed-pill">HED tags</span>'
        f'<span class="ds-hed-meta">{coverage} events annotated</span>'
        "</div>"
        '<p class="ds-hed-source">Source: MOABB BIDS HED annotation mapping.</p>'
        f'<div class="ds-hed-bars">{"".join(bar_rows)}</div>'
        f'<div class="ds-hed-events">{"".join(event_rows)}</div>'
        '<div class="ds-hed-tree-block">'
        '<p class="ds-hed-tree-title">HED tree view</p>'
        f'{"".join(tree_items)}'
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Enhanced header card (Layer 1)
# ---------------------------------------------------------------------------


def _make_github_issue_url(cls_name):
    """Build a pre-filled GitHub issue URL for this dataset."""
    issue_title = quote(f"[Dataset] Issue with {cls_name}")
    issue_body = quote(
        f"## Dataset\n\n"
        f"- **Dataset ID:** {cls_name}\n\n"
        f"## Issue Description\n\n"
        f"Please describe the issue you encountered with this dataset:\n\n"
        f"## Steps to Reproduce\n\n"
        f"1. \n2. \n3. \n\n"
        f"## Expected Behavior\n\n\n"
        f"## Additional Context\n\n"
    )
    url = (
        f"https://github.com/NeuroTechX/moabb/issues/new"
        f"?title={issue_title}&body={issue_body}&labels=dataset"
    )
    return escape(url, quote=True)


_COUNTRY_TO_ISO2 = {
    "argentina": "AR",
    "australia": "AU",
    "austria": "AT",
    "belgium": "BE",
    "brazil": "BR",
    "canada": "CA",
    "chile": "CL",
    "china": "CN",
    "colombia": "CO",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "egypt": "EG",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "greece": "GR",
    "hungary": "HU",
    "india": "IN",
    "iran": "IR",
    "ireland": "IE",
    "israel": "IL",
    "italy": "IT",
    "japan": "JP",
    "malaysia": "MY",
    "mexico": "MX",
    "netherlands": "NL",
    "new zealand": "NZ",
    "norway": "NO",
    "pakistan": "PK",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "russia": "RU",
    "saudi arabia": "SA",
    "singapore": "SG",
    "south korea": "KR",
    "korea": "KR",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "taiwan": "TW",
    "thailand": "TH",
    "turkey": "TR",
    "uk": "GB",
    "united kingdom": "GB",
    "usa": "US",
    "united states": "US",
    "vietnam": "VN",
}


def _country_flag(country_str):
    """Return a flag emoji for a country name or ISO 3166-1 alpha-2 code."""
    if not country_str:
        return ""
    key = country_str.strip().lower()
    # Try direct lookup in name map, then treat input as ISO code
    iso2 = _COUNTRY_TO_ISO2.get(key, country_str.strip().upper())
    if len(iso2) != 2 or not iso2.isalpha():
        return ""
    # Convert to regional indicator symbols (Unicode flag)
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2.upper())


def _highlight_python(code):
    """Highlight a Python code string using Pygments, returning HTML."""
    from pygments import highlight as _pygments_highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    formatter = HtmlFormatter(nowrap=False, cssclass="highlight")
    return _pygments_highlight(code, PythonLexer(), formatter)


def _load_dataset_pageviews(srcdir):
    """Load GA4 dataset page views snapshot from docs static assets."""
    global _DATASET_PAGEVIEWS_CACHE, _DATASET_PAGEVIEWS_CACHE_SRC
    if _DATASET_PAGEVIEWS_CACHE is not None and _DATASET_PAGEVIEWS_CACHE_SRC == srcdir:
        return _DATASET_PAGEVIEWS_CACHE

    snapshot_path = os.path.join(srcdir, "_static", "analytics", "pageviews.json")
    payload = {
        "generated_at_utc": "",
        "status": "disabled",
        "reason": "",
        "counts": {},
        "ranks": {},
    }

    def _norm_name(name):
        return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

    canonical_name_map = {}
    try:
        from moabb.datasets.utils import dataset_list

        for ds_cls in dataset_list:
            canonical_name_map[_norm_name(ds_cls.__name__)] = ds_cls.__name__
    except Exception:
        canonical_name_map = {}

    try:
        with open(snapshot_path, encoding="utf-8") as f:
            raw_payload = json.load(f)
        payload["generated_at_utc"] = str(raw_payload.get("generated_at_utc", "") or "")
        payload["status"] = str(raw_payload.get("status", "") or "disabled")
        payload["reason"] = str(raw_payload.get("reason", "") or "")

        raw_counts = raw_payload.get("counts", {})
        merged_counts = {}
        if isinstance(raw_counts, dict):
            for cls_name, values in raw_counts.items():
                if not isinstance(values, dict):
                    continue
                canonical = canonical_name_map.get(_norm_name(cls_name), str(cls_name))
                entry = merged_counts.setdefault(
                    canonical, {"last30": 0, "all_time": 0, "weekly_12": [0] * 12}
                )
                if "last30" in values:
                    try:
                        entry["last30"] += int(values["last30"])
                    except (TypeError, ValueError):
                        pass
                if "all_time" in values:
                    try:
                        entry["all_time"] += int(values["all_time"])
                    except (TypeError, ValueError):
                        pass
                weekly = values.get("weekly_12")
                if isinstance(weekly, list):
                    for i, val in enumerate(weekly[:12]):
                        try:
                            entry["weekly_12"][i] += int(val)
                        except (TypeError, ValueError):
                            pass
        payload["counts"] = merged_counts

        ranked = sorted(
            payload["counts"].items(),
            key=lambda kv: (-int(kv[1].get("all_time", 0)), kv[0]),
        )
        total = len(ranked)
        ranks = {}
        if total > 0:
            for idx, (name, _) in enumerate(ranked, start=1):
                ranks[name] = {
                    "rank": idx,
                    "total": total,
                    "top_percent": max(1, math.ceil((idx / total) * 100)),
                }
        payload["ranks"] = ranks
    except Exception:
        pass

    _DATASET_PAGEVIEWS_CACHE = payload
    _DATASET_PAGEVIEWS_CACHE_SRC = srcdir
    return payload


def _get_dataset_pageview_counts(srcdir, cls_name):
    """Return page view counts for a dataset class name (if available)."""
    return _load_dataset_pageviews(srcdir).get("counts", {}).get(cls_name, {})


def _get_dataset_pageview_rank(srcdir, cls_name):
    """Return pageview rank metadata for a dataset class name (if available)."""
    return _load_dataset_pageviews(srcdir).get("ranks", {}).get(cls_name, {})


def _get_dataset_pageview_meta(srcdir):
    """Return GA pageview snapshot metadata."""
    payload = _load_dataset_pageviews(srcdir)
    return {
        "generated_at_utc": payload.get("generated_at_utc", ""),
        "status": payload.get("status", ""),
        "reason": payload.get("reason", ""),
    }


def _format_count(value):
    """Return a thousands-separated integer string, or 'n/a'."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _format_updated_utc(iso_text):
    """Format ISO timestamp into YYYY-MM-DD UTC."""
    if not iso_text:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(str(iso_text).replace("Z", "+00:00"))
        parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%Y-%m-%d UTC")
    except Exception:
        return "n/a"


def _sparkline_svg(values):
    """Return an inline SVG sparkline for a sequence of numeric values."""
    if not isinstance(values, list) or len(values) < 2:
        return ""
    nums = []
    for val in values[:12]:
        try:
            nums.append(max(0, int(val)))
        except (TypeError, ValueError):
            nums.append(0)
    if len(nums) < 2:
        return ""

    width, height = 110, 28
    pad = 2
    min_y = pad
    max_y = height - pad
    max_val = max(nums) if nums else 0
    denom = max_val if max_val > 0 else 1
    step = (width - 2 * pad) / (len(nums) - 1)

    points = []
    for i, val in enumerate(nums):
        x = pad + i * step
        y = max_y - ((val / denom) * (max_y - min_y))
        points.append((x, y))

    line_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    area_path = (
        f"M {points[0][0]:.2f} {max_y:.2f} "
        + " ".join(f"L {x:.2f} {y:.2f}" for x, y in points)
        + f" L {points[-1][0]:.2f} {max_y:.2f} Z"
    )
    return (
        '<svg class="ds-views-spark" viewBox="0 0 110 28" '
        'role="img" aria-label="Weekly page views over the last 12 weeks">'
        f'<path class="ds-views-spark-area" d="{area_path}"></path>'
        f'<polyline class="ds-views-spark-line" points="{line_points}"></polyline>'
        "</svg>"
    )


def _make_provenance_html(info):
    """Build the author provenance byline block for the card header."""
    investigators = info.get("investigators") or []
    senior_author = info.get("senior_author")
    contact_info = info.get("contact_info") or []
    institution = info.get("institution")
    country = info.get("country")
    publication_year = info.get("publication_year")

    if not investigators and not senior_author:
        return ""

    # Build author name spans — all authors listed, senior author highlighted
    senior_lower = (senior_author or "").strip().lower()
    author_spans = []
    for name in investigators:
        safe = escape(name)
        if name.strip().lower() == senior_lower:
            author_spans.append(
                f'<span class="ds-author-name ds-author-senior">{safe}</span>'
            )
        else:
            author_spans.append(f'<span class="ds-author-name">{safe}</span>')

    authors_line = ""
    if author_spans:
        authors_line = (
            '<p class="ds-authors">'
            '<span class="ds-authors-label">Authors</span>'
            f'{", ".join(author_spans)}'
            "</p>"
        )

    # Build provenance meta line: flag institution, country · year · email
    meta_parts = []
    if institution:
        flag = _country_flag(country) if country else ""
        flag_prefix = f"{flag}\u2002" if flag else ""
        inst_str = f"{flag_prefix}{escape(institution)}"
        if country:
            inst_str += f", {escape(country)}"
        meta_parts.append(f"<span>{inst_str}</span>")
    if publication_year:
        meta_parts.append(f"<span>{int(publication_year)}</span>")
    if contact_info:
        email = contact_info[0]
        safe_email = escape(email)
        meta_parts.append(f'<a href="mailto:{safe_email}">{safe_email}</a>')

    meta_line = ""
    if meta_parts:
        sep = '<span class="ds-provenance-sep">\u00b7</span>'
        meta_line = f'<div class="ds-provenance-meta">{sep.join(meta_parts)}</div>'

    return f'<div class="ds-provenance">{authors_line}{meta_line}</div>'


def _make_header_html(
    cls_name,
    info,
    source_url=None,
    *,
    live_citations=True,
    pageview_counts=None,
    pageview_rank=None,
    pageview_meta=None,
    description_lines=None,
):
    """Build the enhanced dataset card HTML (Layer 1)."""
    paradigm = info.get("paradigm") or "unknown"
    label = _PARADIGM_LABELS.get(paradigm, paradigm.title())
    color = _PARADIGM_COLORS.get(paradigm, "#546E7A")
    n_subj = info.get("n_subjects")
    n_sess = info.get("n_sessions")
    paper_doi = _normalize_doi(info.get("paper_doi") or info.get("doi"))
    sampling_rate = info.get("sampling_rate")
    n_channels = info.get("n_channels")
    channel_types = info.get("channel_types")
    n_classes = info.get("n_classes")
    class_labels = info.get("class_labels")
    trial_duration = info.get("trial_duration")
    default_subject = info.get("default_subject", 1)
    subject_literal = repr(default_subject)
    code = info.get("code")
    quickstart_id = _dataset_dom_id("ds-quickstart", cls_name)
    quickstart_btn_id = _dataset_dom_id("ds-quickstart-btn", cls_name)
    source_html = ""
    if source_url:
        source_html = (
            f'<a class="ds-card-source" href="{escape(source_url)}" '
            f'target="_blank" rel="noopener">[source]</a>'
        )

    # --- Subtitle: auto-generated from paradigm + classes ---
    # Use the actual count of class labels when available
    display_n_classes = n_classes
    if class_labels:
        display_n_classes = len(class_labels)
    subtitle_parts = [label]
    if display_n_classes is not None:
        subtitle_parts.append(f"{display_n_classes} classes")
    if class_labels and len(class_labels) <= 6:
        safe_labels = [escape(str(lbl)) for lbl in class_labels[:6]]
        subtitle_parts.append("(" + " vs ".join(safe_labels) + ")")
    subtitle = ", ".join(subtitle_parts[:2])
    if len(subtitle_parts) > 2:
        subtitle += " " + subtitle_parts[2]

    # --- Stat chips ---
    chips = []
    chips.append(f'<span class="ds-chip" style="--chip-color: {color}">{label}</span>')
    if code:
        chips.append(f'<span class="ds-chip ds-chip-muted">Code: {code}</span>')
    if n_subj is not None:
        chips.append(f'<span class="ds-chip ds-chip-muted">{n_subj} subjects</span>')
    if n_sess is not None:
        sess_label = "session" if n_sess == 1 else "sessions"
        chips.append(f'<span class="ds-chip ds-chip-muted">{n_sess} {sess_label}</span>')

    # Channel chip
    if n_channels is not None:
        ch_detail = ""
        if channel_types and isinstance(channel_types, dict):
            eeg_count = channel_types.get("eeg", channel_types.get("EEG", 0))
            if eeg_count and eeg_count != n_channels:
                ch_detail = f" ({eeg_count} EEG)"
        chips.append(
            f'<span class="ds-chip ds-chip-muted">{n_channels} ch{ch_detail}</span>'
        )

    # Sampling rate chip
    if sampling_rate is not None:
        sr_display = (
            f"{int(sampling_rate)}"
            if sampling_rate == int(sampling_rate)
            else f"{sampling_rate:g}"
        )
        chips.append(f'<span class="ds-chip ds-chip-muted">{sr_display} Hz</span>')

    # Classes chip
    if display_n_classes is not None:
        chips.append(
            f'<span class="ds-chip ds-chip-muted">{display_n_classes} classes</span>'
        )

    # Trial duration chip
    if trial_duration is not None:
        dur_display = (
            f"{trial_duration:g}"
            if trial_duration != int(trial_duration)
            else f"{int(trial_duration)}.0"
        )
        chips.append(f'<span class="ds-chip ds-chip-muted">{dur_display} s trials</span>')

    # License chip
    license_raw = info.get("license")
    license_key = _normalize_license(license_raw)
    if license_key:
        display_name, license_url, icon_keys = _LICENSE_INFO[license_key]
        icons_html = "".join(_cc_icon_svg(k) for k in icon_keys)
        if license_url:
            chips.append(
                f'<a class="ds-chip ds-chip-license" href="{escape(license_url)}" '
                f'target="_blank" rel="noopener" title="{escape(display_name)}">'
                f"{icons_html}{escape(display_name)}</a>"
            )
        else:
            chips.append(
                f'<span class="ds-chip ds-chip-license" title="{escape(display_name)}">'
                f"{icons_html}{escape(display_name)}</span>"
            )

    chips_html = "\n      ".join(chips)
    benchmark_html = _make_benchmark_context_html(cls_name, info)
    benchmark_ctx = _get_benchmark_context(cls_name)
    citation_html = _make_citation_impact_html(
        info,
        benchmark_ctx,
        live_citations=live_citations,
        pageview_counts=pageview_counts,
        pageview_rank=pageview_rank,
        pageview_meta=pageview_meta,
    )
    compare_anchor_map = {
        "imagery": "motor-imagery",
        "p300": "p300-erp",
        "erp": "p300-erp",
        "ssvep": "ssvep",
        "cvep": "c-vep",
        "rstate": "resting-states",
    }
    compare_anchor = compare_anchor_map.get(paradigm)
    compare_href = "../dataset_summary.html"
    if compare_anchor:
        compare_href += f"#{compare_anchor}"

    # --- Optional class-label line ---
    class_line = ""
    if class_labels:
        preview = ", ".join(escape(str(lbl)) for lbl in class_labels[:8])
        if len(class_labels) > 8:
            preview += ", ..."
        class_line = (
            f'<p class="ds-class-line"><span class="ds-class-line-label">'
            f"Class Labels:</span> {preview}</p>"
        )

    # --- Action buttons ---
    actions = []
    # Quickstart button toggles the code panel
    actions.append(
        (
            f'<button id="{quickstart_btn_id}" class="ds-btn ds-btn-primary ds-btn-toggle" type="button" '
            f'aria-controls="{quickstart_id}" aria-expanded="false" '
            f"onclick=\"var el=document.getElementById('{quickstart_id}');"
            "if(el){var expanded=this.getAttribute('aria-expanded')==='true';"
            "var next=!expanded;"
            "this.setAttribute('aria-expanded',next?'true':'false');"
            "el.hidden=!next;"
            "el.setAttribute('aria-hidden',next?'false':'true');}\">"
            "Quickstart"
            "</button>"
        )
    )
    if paper_doi:
        doi_href = escape(f"https://doi.org/{quote(paper_doi, safe='')}", quote=True)
        actions.append(
            f'<a class="ds-btn" href="{doi_href}" '
            f'target="_blank" rel="noopener">Read Paper</a>'
        )
    actions.append(f'<a class="ds-btn" href="{compare_href}">Compare Similar</a>')
    github_url = _make_github_issue_url(cls_name)
    actions.append(
        f'<a class="ds-btn" href="{github_url}" '
        f'target="_blank" rel="noopener">Report Issue</a>'
    )
    actions_html = "\n      ".join(actions)

    # --- Quickstart code block (Pygments-highlighted) ---
    quickstart_code = (
        f"from moabb.datasets import {cls_name}\n\n"
        f"dataset = {cls_name}()\n"
        f"data = dataset.get_data(subjects=[{subject_literal}])\n"
        f"print(data[{subject_literal}])"
    )
    hl_code = _highlight_python(quickstart_code)
    quickstart = (
        f'<div id="{quickstart_id}" class="ds-quickstart" role="region" '
        f'aria-labelledby="{quickstart_btn_id}" aria-hidden="true" hidden>\n'
        f'  <div class="ds-quickstart-code">{hl_code}</div>\n'
        f"</div>"
    )

    # --- Alt name (paper description) ---
    alt_name_html = ""
    paper_desc = info.get("paper_description")
    if paper_desc:
        alt_name_html = f'<p class="ds-card-alt-name">{escape(paper_desc)}</p>'

    # --- Author provenance ---
    provenance_html = _make_provenance_html(info)

    # --- Overview teaser ---
    overview_teaser = _make_overview_teaser_html(description_lines or [], cls_name)

    return f"""\
<div class="ds-card" role="region" aria-label="{cls_name} dataset overview">
  {source_html}
  <div class="ds-card-head">
    <p class="ds-card-kicker">Dataset Snapshot</p>
    <p class="ds-card-title">{cls_name}</p>
    {alt_name_html}
    <p class="ds-subtitle">{subtitle}</p>
    {provenance_html}
  </div>
  <div class="ds-stats">
      {chips_html}
  </div>
  {class_line}
  <div class="ds-actions">
      {actions_html}
  </div>
  {overview_teaser}
  {quickstart}
  {benchmark_html}
  {citation_html}
</div>"""


# ---------------------------------------------------------------------------
# Visual summary grid (Layer 2)
# ---------------------------------------------------------------------------


def _make_visual_grid_lines(cls_name, info, srcdir):
    """Build RST lines for the adaptive visual summary grid."""
    lines = []
    paradigm = info.get("paradigm") or "unknown"
    paradigm_label = _PARADIGM_LABELS.get(paradigm, paradigm.title())
    n_classes = info.get("n_classes")
    class_labels = info.get("class_labels") or []
    display_n_classes = len(class_labels) if class_labels else n_classes
    runs_per_session = info.get("runs_per_session")
    n_sessions = info.get("n_sessions")
    trial_duration = info.get("trial_duration")
    has_hed = bool(info.get("hed_tags")) if info else False
    hed_html = _make_hed_summary_html(info) if has_hed else ""

    # Check which SVGs exist
    timeline_svg = os.path.join(srcdir, "_static", "timelines", f"{cls_name}.svg")

    has_timeline = os.path.exists(timeline_svg)
    # Build channel summary HTML
    channel_html = _make_channel_summary_html(info)

    # Count how many grid items we have (timeline gets full width, others share row)
    n_items = sum([has_timeline, has_hed, bool(channel_html)])
    if n_items == 0:
        if not has_timeline:
            return []

    # Timeline gets its own full-width row; remaining items share a 2-col row
    n_cols = 2 if (n_items - int(has_timeline)) >= 2 else 1

    lines.extend(
        [
            "",
            f".. grid:: {n_cols}",
            "   :gutter: 3",
            "",
        ]
    )

    # Panel footnotes
    protocol_bits = []
    if trial_duration is not None:
        protocol_bits.append(f"{trial_duration:g}s task window per trial")
    if display_n_classes is not None:
        protocol_bits.append(
            f"{display_n_classes}-class {paradigm_label.lower()} paradigm"
        )
    if runs_per_session is not None and n_sessions is not None:
        protocol_bits.append(
            f"{runs_per_session} runs/session across {n_sessions} sessions"
        )
    protocol_note = " \u00b7 ".join(protocol_bits)

    if has_timeline:
        lines.extend(
            [
                "   .. grid-item-card:: Stimulus Protocol",
                "      :columns: 12",
                "      :class-card: ds-viz-card",
                "",
                f"      .. image:: /_static/timelines/{cls_name}.svg",
                "         :width: 100%",
                "         :class: timeline-diagram",
                "",
            ]
        )
        if protocol_note:
            lines.extend(
                [
                    "      .. raw:: html",
                    "",
                    f'         <p class="ds-viz-note">{escape(protocol_note)}</p>',
                    "",
                ]
            )

    if has_hed:
        lines.extend(
            [
                "   .. grid-item-card:: HED Event Tags",
                "      :class-card: ds-viz-card",
                "",
                "      .. raw:: html",
                "",
            ]
        )
        for hed_line in hed_html.split("\n"):
            lines.append(f"         {hed_line}")
        lines.append("")

    if channel_html:
        lines.extend(
            [
                "   .. grid-item-card:: Channel Summary",
                "      :class-card: ds-viz-card",
                "",
                "      .. raw:: html",
                "",
            ]
        )
        for ch_line in channel_html.split("\n"):
            lines.append(f"         {ch_line}")
        lines.append("")

    # Timeline disclaimer
    if has_timeline:
        lines.extend(
            [
                ".. raw:: html",
                "",
                '   <p class="timeline-disclaimer">'
                "This diagram is automatically generated from MOABB metadata. "
                "Please consult the original publication to confirm "
                "the experimental protocol details.</p>",
                "",
            ]
        )

    return lines


def _make_channel_summary_html(info):
    """Build a small HTML card summarising channel configuration."""
    n_channels = info.get("n_channels") if info else None
    channel_types = info.get("channel_types") if info else None
    montage = info.get("montage") if info else None
    sampling_rate = info.get("sampling_rate") if info else None
    reference = info.get("reference") if info else None
    filter_range = info.get("filter_range") if info else None
    line_freq = info.get("line_freq") if info else None
    sensor_type = info.get("sensor_type") if info else None

    if (
        n_channels is None
        and montage is None
        and sampling_rate is None
        and reference is None
        and filter_range is None
        and line_freq is None
        and sensor_type is None
    ):
        return ""

    rows = []
    if n_channels is not None:
        rows.append(("Total channels", f"{float(n_channels):g}"))

    if channel_types and isinstance(channel_types, dict):
        sorted_types = sorted(channel_types.items(), key=lambda x: (-x[1], x[0]))
        for ctype, count in sorted_types[:4]:
            if str(ctype).lower() == "eeg" and sensor_type:
                rows.append((ctype.upper(), f"{float(count):g} ({sensor_type})"))
            else:
                rows.append((ctype.upper(), f"{float(count):g}"))

    if montage is not None:
        rows.append(("Montage", "10-05" if montage == "standard_1005" else str(montage)))

    if sampling_rate is not None:
        sr_display = (
            f"{int(sampling_rate)} Hz"
            if sampling_rate == int(sampling_rate)
            else f"{sampling_rate:g} Hz"
        )
        rows.append(("Sampling", sr_display))

    if reference:
        rows.append(("Reference", str(reference)))

    if filter_range:
        rows.append(("Filter", str(filter_range)))
    if line_freq is not None:
        line_display = (
            f"{float(line_freq):g} Hz"
            if isinstance(line_freq, (int, float))
            else str(line_freq)
        )
        rows.append(("Notch / line", line_display))

    if not rows:
        return ""

    row_html = "\n".join(
        f'<div class="ds-channel-row"><span>{escape(str(key))}</span><strong>{escape(str(val))}</strong></div>'
        for key, val in rows
    )
    return f'<div class="ds-channel-card">{row_html}</div>'


# ---------------------------------------------------------------------------
# Tabbed docstring restructuring (Layer 3)
# ---------------------------------------------------------------------------


def _restructure_docstring_lines(lines, cls_name, default_subject=1):
    """Reorganize docstring lines into a tabbed layout.

    Scans lines for section markers and groups content into:
    - Overview (description + references)
    - Code Examples (code snippet)
    - Metadata (admonition cards)
    - Notes (notes, version directives)

    Returns modified lines wrapped in sphinx-design tab-set.
    """
    # Classify lines into buckets
    metadata_lines = []
    description_lines = []
    reference_lines = []
    notes_lines = []
    current_bucket = "description"
    in_admonition = False
    admonition_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect admonition starts (metadata cards + feedback)
        if stripped.startswith(".. admonition::"):
            title = stripped[len(".. admonition::") :].strip()
            metadata_titles = {
                "Dataset summary",
                "Participants",
                "Equipment",
                "Preprocessing",
                "Data Access",
                "Experimental Protocol",
            }
            if title in metadata_titles:
                current_bucket = "metadata"
                in_admonition = True
                admonition_indent = len(line) - len(line.lstrip())
                metadata_lines.append(line)
                i += 1
                continue
            elif "Found an issue" in title:
                # Discard feedback section — "Report Issue" is now in
                # the card header action bar.
                current_bucket = "discard_feedback"
                in_admonition = True
                admonition_indent = len(line) - len(line.lstrip())
                i += 1
                continue

        # Detect rubric sections
        if stripped.startswith(".. rubric::"):
            rubric_title = stripped[len(".. rubric::") :].strip()
            if rubric_title == "References":
                current_bucket = "references"

                in_admonition = False
                reference_lines.append(line)
                i += 1
                continue
            elif rubric_title in ("Notes", "Notes:"):
                current_bucket = "notes"

                in_admonition = False
                notes_lines.append(line)
                i += 1
                continue

        # Detect version directives → notes
        if (
            stripped.startswith(".. versionadded::")
            or stripped.startswith(".. versionchanged::")
            or stripped.startswith(".. deprecated::")
        ):
            current_bucket = "notes"
            notes_lines.append(line)
            i += 1
            continue

        # If in an admonition, check if we've left it (by indentation)
        if in_admonition:
            if stripped == "":
                # Blank lines can be part of admonition
                if current_bucket == "metadata":
                    metadata_lines.append(line)
                # discard_feedback: silently skip
                i += 1
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent > admonition_indent:
                # Still inside admonition
                if current_bucket == "metadata":
                    metadata_lines.append(line)
                # discard_feedback: silently skip
                i += 1
                continue
            else:
                # Exited admonition
                in_admonition = False
                current_bucket = "description"

        # Route to current bucket
        if current_bucket == "references":
            # Stay in references until a new section starts or double blank
            reference_lines.append(line)
        elif current_bucket == "notes":
            notes_lines.append(line)
        elif current_bucket == "metadata":
            metadata_lines.append(line)
        else:
            description_lines.append(line)

        i += 1

    # Clean up: strip trailing blanks from each bucket
    def _strip_trailing_blanks(lst):
        while lst and lst[-1].strip() == "":
            lst.pop()
        return lst

    description_lines = _strip_trailing_blanks(description_lines)
    metadata_lines = _strip_trailing_blanks(metadata_lines)
    reference_lines = _strip_trailing_blanks(reference_lines)
    notes_lines = _strip_trailing_blanks(notes_lines)

    # If we have very little content, don't restructure
    has_metadata = bool(metadata_lines)
    has_description = any(line.strip() for line in description_lines)
    if not has_metadata and not has_description:
        return None  # Don't restructure

    def _reindent(block, base_indent):
        """Re-indent a block of lines to a new base indentation.

        Finds the minimum indentation in the block and shifts all lines
        so that minimum becomes ``base_indent``.  Blank lines stay blank.
        """
        # Determine minimum indentation of non-blank lines
        min_indent = None
        for bline in block:
            if bline.strip():
                indent = len(bline) - len(bline.lstrip())
                if min_indent is None or indent < min_indent:
                    min_indent = indent
        if min_indent is None:
            min_indent = 0

        out = []
        for bline in block:
            if not bline.strip():
                out.append("")
            else:
                # Strip the common prefix, add the new base indent
                current_indent = len(bline) - len(bline.lstrip())
                extra = current_indent - min_indent
                out.append(" " * (base_indent + extra) + bline.lstrip())
        return out

    # The tab-item content needs 6 spaces of indentation (3 for tab-set + 3 for tab-item)
    TAB_INDENT = 6

    # Build the tabbed layout
    new_lines = []

    # Tab-set directive
    new_lines.append("")
    new_lines.append(".. tab-set::")
    new_lines.append("   :class: ds-doc-tabs")
    new_lines.append("")

    # --- Tab: Overview ---
    new_lines.append("   .. tab-item:: Overview")
    new_lines.append("")
    new_lines.append(
        " " * TAB_INDENT + f".. _{_dataset_dom_id('ds-overview', cls_name)}:"
    )
    new_lines.append("")
    if description_lines:
        new_lines.extend(_reindent(description_lines, TAB_INDENT))
        new_lines.append("")
    if reference_lines:
        new_lines.extend(_reindent(reference_lines, TAB_INDENT))
        new_lines.append("")
    # If overview is empty, add a placeholder
    if not description_lines and not reference_lines:
        new_lines.append(" " * TAB_INDENT + "*No description available.*")
        new_lines.append("")

    # --- Tab: Code Examples ---
    new_lines.append("   .. tab-item:: Code Examples")
    new_lines.append("")
    new_lines.append(" " * TAB_INDENT + ".. code-block:: python")
    new_lines.append("")
    new_lines.append(" " * (TAB_INDENT + 3) + f"from moabb.datasets import {cls_name}")
    new_lines.append(" " * (TAB_INDENT + 3) + f"dataset = {cls_name}()")
    subject_literal = repr(default_subject)
    new_lines.append(
        " " * (TAB_INDENT + 3) + f"data = dataset.get_data(subjects=[{subject_literal}])"
    )
    new_lines.append(" " * (TAB_INDENT + 3) + f"print(data[{subject_literal}])")
    new_lines.append("")

    # --- Tab: Metadata ---
    if has_metadata:
        new_lines.append("   .. tab-item:: Metadata")
        new_lines.append("")
        new_lines.extend(_reindent(metadata_lines, TAB_INDENT))
    new_lines.append("")

    # --- Tab: Notes ---
    if notes_lines:
        new_lines.append("   .. tab-item:: Notes")
        new_lines.append("")
        new_lines.extend(_reindent(notes_lines, TAB_INDENT))
        new_lines.append("")

    return new_lines


# ---------------------------------------------------------------------------
# Legacy timeline lines (kept for when grid is not used)
# ---------------------------------------------------------------------------


def _make_timeline_lines(cls_name, srcdir):
    """Build RST lines for the timeline image + disclaimer."""
    svg_path = os.path.join(srcdir, "_static", "timelines", f"{cls_name}.svg")
    if not os.path.exists(svg_path):
        return []

    return [
        "",
        ".. rubric:: Stimulus Protocol Timeline",
        "",
        f".. image:: /_static/timelines/{cls_name}.svg",
        "   :width: 100%",
        "   :class: timeline-diagram",
        "",
        ".. raw:: html",
        "",
        '   <p class="timeline-disclaimer">'
        "This diagram is automatically generated from MOABB metadata. "
        "Please consult the original publication to confirm "
        "the experimental protocol details.</p>",
        "",
    ]


# ---------------------------------------------------------------------------
# Main docstring processor
# ---------------------------------------------------------------------------


def _is_autosummary_context():
    """Return True if we are called from autosummary's summary extraction."""
    import traceback

    for frame_info in traceback.extract_stack():
        if "autosummary" in frame_info.filename and frame_info.name == "get_items":
            return True
    return False


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Enhance dataset class docstrings with card, grid, and tabs."""
    if what != "class":
        return
    if not _is_concrete_dataset(obj):
        return

    # Skip heavy restructuring when autosummary is extracting one-line
    # summaries for the API table — it only needs the first paragraph.
    if _is_autosummary_context():
        return

    cls_name = obj.__name__
    info = _get_dataset_info(obj)
    source_url = _get_dataset_source_url(obj)

    # --- Extract description lines for teaser (before restructuring) ---
    desc_lines = _extract_description_text(lines)

    # --- Layer 1: Enhanced card (inserted at top) ---
    top_block = []
    if info:
        live_citations = getattr(app.config, "dataset_card_live_citations", True)
        pageview_counts = _get_dataset_pageview_counts(app.srcdir, cls_name)
        pageview_rank = _get_dataset_pageview_rank(app.srcdir, cls_name)
        pageview_meta = _get_dataset_pageview_meta(app.srcdir)
        header_html = _make_header_html(
            cls_name,
            info,
            source_url=source_url,
            live_citations=live_citations,
            pageview_counts=pageview_counts,
            pageview_rank=pageview_rank,
            pageview_meta=pageview_meta,
            description_lines=desc_lines,
        )
        top_block.append(".. raw:: html")
        top_block.append("")
        for h_line in header_html.split("\n"):
            top_block.append(f"   {h_line}")
        top_block.append("")

    # --- Layer 2: Visual summary grid ---
    if info:
        grid_lines = _make_visual_grid_lines(cls_name, info, app.srcdir)
        top_block.extend(grid_lines)

    # --- Layer 3: Restructure remaining docstring into tabs ---
    default_subject = info.get("default_subject", 1) if info else 1
    restructured = _restructure_docstring_lines(
        lines, cls_name, default_subject=default_subject
    )
    if restructured is not None:
        # Replace all existing lines with restructured content
        lines.clear()
        lines.extend(restructured)

    # Insert the card + grid at position 0
    for i, line in enumerate(top_block):
        lines.insert(i, line)


def source_read_add_inherited(app, docname, source):
    """Inject :inherited-members: and __init__ into dataset page RST sources.

    Auto-generated RST files from autosummary only have :members:.
    For dataset classes we also need inherited methods (get_data, download, etc.)
    and __init__ shown explicitly.
    """
    if not docname.startswith("generated/moabb.datasets."):
        return
    # Skip non-class pages (e.g. function pages, module pages)
    if not re.search(r"\.\. autoclass::", source[0]):
        return

    # Remove sidebars on dataset pages for a focused, wide layout.
    # PyData theme reads this from document metadata field lists.
    source[0] = (
        ":html_theme.sidebar_primary.remove:\n"
        ":html_theme.sidebar_secondary.remove:\n\n" + source[0]
    )

    # Add :inherited-members: and :show-inheritance: after :members:
    source[0] = source[0].replace(
        "   :members:\n",
        "   :members:\n   :inherited-members:\n   :show-inheritance:\n",
    )

    # Add __init__ to :special-members: so the constructor is documented
    source[0] = re.sub(
        r"(:special-members:.*)",
        r"\1,__init__",
        source[0],
    )


def _generate_all_svgs(app):
    """Generate stimulus timeline SVGs.

    Runs once at the start of the Sphinx build (builder-inited event).
    SVGs are written to ``_static/timelines/``.

    Controlled by the ``dataset_card_generate_svgs`` config value
    (default ``True``).  When ``False``, SVG generation is skipped entirely.
    Existing SVG files are never overwritten.
    """
    if not getattr(app.config, "dataset_card_generate_svgs", True):
        return

    import traceback

    srcdir = app.srcdir
    timeline_dir = os.path.join(srcdir, "_static", "timelines")
    os.makedirs(timeline_dir, exist_ok=True)

    try:
        from moabb.analysis.timeline import stimulus_timeline_svg
        from moabb.datasets.utils import dataset_list
    except ImportError:
        traceback.print_exc()
        print(
            "[dataset_timeline_ext] Could not import timeline functions. "
            "Make sure moabb is installed from the current repo."
        )
        return

    for ds_cls in dataset_list:
        name = ds_cls.__name__
        try:
            ds = ds_cls()
        except Exception:
            continue

        # Timeline
        timeline_path = os.path.join(timeline_dir, f"{name}.svg")
        if not os.path.exists(timeline_path):
            try:
                svg = stimulus_timeline_svg(ds)
                with open(timeline_path, "w", encoding="utf-8") as f:
                    f.write(svg)
            except Exception:
                pass


def setup(app):
    app.add_config_value("dataset_card_live_citations", True, "html")
    app.add_config_value("dataset_card_generate_svgs", True, "html")
    app.connect("builder-inited", _generate_all_svgs)
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("source-read", source_read_add_inherited)
    return {"version": "1.0", "parallel_read_safe": True}
