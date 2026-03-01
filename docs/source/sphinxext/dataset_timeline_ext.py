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
import inspect
import json
import os
import re
import statistics
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
        doi = getattr(ds, "doi", None)
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
            "doi": doi,
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
                    entries.append(
                        {
                            "label": label,
                            "median": statistics.median(scores),
                            "best": max(scores),
                            "n_pipelines": len(scores),
                        }
                    )
        except Exception:
            continue

    context = {"n_tables": len(entries), "entries": entries}
    _BENCHMARK_CONTEXT_CACHE[cls_name] = context
    return context


def _make_known_caveats_html(info):
    """Build a compact known-caveats list from available metadata."""
    caveats = []
    n_subj = info.get("n_subjects")
    n_sessions = info.get("n_sessions")
    n_classes = info.get("n_classes")
    montage = info.get("montage")
    n_channels = info.get("n_channels")
    code = str(info.get("code") or "")

    if n_subj is not None and n_subj < 12:
        caveats.append(f"Small cohort: {n_subj} subjects")
    if n_sessions == 1:
        caveats.append("Single-session recordings")
    if montage and montage != "standard_1005":
        caveats.append("Custom montage (non-10-05)")
    if n_channels is not None and n_channels < 16:
        caveats.append("Low channel density for spatial decoding")
    if n_classes is not None and n_classes >= 4:
        caveats.append(f"{n_classes}-class paradigm increases task complexity")
    if "bnci" in code.lower():
        caveats.append("Competition dataset in controlled lab conditions")

    if not caveats:
        return ""

    items = "\n      ".join(f"<li>{escape(c)}</li>" for c in caveats[:4])
    return (
        '<div class="ds-caveats">'
        '<p class="ds-caveats-title">Known Caveats</p>'
        f"<ul>{items}</ul>"
        "</div>"
    )


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
        rows.append(
            "<li>"
            f'<span>{escape(entry["label"])} '
            f'<em>{entry["n_pipelines"]} pipelines</em></span>'
            f"<strong>{entry['median']:.1f} WS</strong>"
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
        "Scores are per-dataset medians across available pipelines.</p>"
        f'<p class="ds-benchmark-meta"><span><strong>Sample frame:</strong> {escape(sample_frame or "N/A")}</span></p>'
        f"<ul>{rows_html}</ul>"
        "</div>"
    )


def _make_citation_impact_html(info, benchmark_ctx, *, live_citations=True):
    """Build a compact citation and impact block."""
    code = str(info.get("code") or "")
    doi = _normalize_doi(info.get("doi"))
    if not code and not doi:
        return ""

    pwc_slug = code.lower().replace("_", "-") if code else ""
    pwc_url = (
        f"https://paperswithcode.com/dataset/{quote(pwc_slug)}-moabb-1"
        if pwc_slug
        else ""
    )

    items = []
    script_html = ""
    if doi:
        doi_link_href = escape(f"https://doi.org/{quote(doi, safe='')}", quote=True)
        items.append(
            f'<li><span>DOI</span><a href="{doi_link_href}" '
            f'target="_blank" rel="noopener">{escape(doi)}</a></li>'
        )
        if _is_likely_doi(doi):
            if live_citations:
                items.append(
                    f'<li><span>Citations</span><strong class="ds-citation-count" data-doi="{escape(doi)}">Loading…</strong></li>'
                )

                openalex_id = quote(f"https://doi.org/{doi}", safe="")
                openalex_url = f"https://api.openalex.org/works/{openalex_id}"
                crossref_url = f"https://api.crossref.org/works/{quote(doi)}"
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
                    f"https://doi.org/{quote(doi, safe='')}", quote=True
                )
                items.append(
                    f'<li><span>Citations</span><a href="{doi_static_href}" '
                    f'target="_blank" rel="noopener">See DOI</a></li>'
                )
    if pwc_url:
        items.append(
            f'<li><span>PapersWithCode</span><a href="{pwc_url}" target="_blank" '
            f'rel="noopener">Dataset page</a></li>'
        )
    if benchmark_ctx and benchmark_ctx.get("n_tables"):
        items.append(
            f'<li><span>MOABB tables</span><strong>{benchmark_ctx["n_tables"]} (WithinSession)</strong></li>'
        )
    if not items:
        return ""

    list_html = "\n      ".join(items)
    return (
        '<div class="ds-citation-impact">'
        '<p class="ds-citation-title">Citation & Impact</p>'
        f"<ul>{list_html}</ul>"
        f"{script_html}"
        "</div>"
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


def _make_header_html(cls_name, info, source_url=None, *, live_citations=True):
    """Build the enhanced dataset card HTML (Layer 1)."""
    paradigm = info.get("paradigm") or "unknown"
    label = _PARADIGM_LABELS.get(paradigm, paradigm.title())
    color = _PARADIGM_COLORS.get(paradigm, "#546E7A")
    n_subj = info.get("n_subjects")
    n_sess = info.get("n_sessions")
    doi = _normalize_doi(info.get("doi"))
    sampling_rate = info.get("sampling_rate")
    n_channels = info.get("n_channels")
    channel_types = info.get("channel_types")
    n_classes = info.get("n_classes")
    class_labels = info.get("class_labels")
    trial_duration = info.get("trial_duration")
    default_subject = info.get("default_subject", 1)
    subject_literal = repr(default_subject)
    code = info.get("code")
    quickstart_id = (
        "ds-quickstart-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", cls_name).strip("-").lower()
    )
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

    chips_html = "\n      ".join(chips)
    caveats_html = _make_known_caveats_html(info)
    benchmark_html = _make_benchmark_context_html(cls_name, info)
    benchmark_ctx = _get_benchmark_context(cls_name)
    citation_html = _make_citation_impact_html(
        info, benchmark_ctx, live_citations=live_citations
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
    # Quickstart button toggles details panel
    actions.append(
        (
            f'<button class="ds-btn ds-btn-primary ds-btn-toggle" type="button" '
            f'aria-controls="{quickstart_id}" aria-expanded="false" '
            f"onclick=\"var el=document.getElementById('{quickstart_id}');"
            "if(el){el.open=!el.open;"
            "this.setAttribute('aria-expanded',el.open?'true':'false');}\">"
            "Quickstart"
            "</button>"
        )
    )
    if doi:
        doi_href = escape(f"https://doi.org/{quote(doi, safe='')}", quote=True)
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

    # --- Quickstart code block ---
    quickstart = (
        f'<details id="{quickstart_id}" class="ds-quickstart">\n'
        f'  <summary class="ds-quickstart-summary">Toggle quickstart code</summary>\n'
        f'  <pre class="ds-quickstart-code"><code>'
        f"from moabb.datasets import {cls_name}\n\n"
        f"dataset = {cls_name}()\n"
        f"data = dataset.get_data(subjects=[{subject_literal}])\n"
        f"print(data[{subject_literal}])"
        f"</code></pre>\n"
        f"</details>"
    )

    return f"""\
<div class="ds-card" role="region" aria-label="{cls_name} dataset overview">
  {source_html}
  <div class="ds-card-head">
    <p class="ds-card-kicker">Dataset Snapshot</p>
    <p class="ds-card-title">{cls_name}</p>
    <p class="ds-subtitle">{subtitle}</p>
  </div>
  <div class="ds-stats">
      {chips_html}
  </div>
  {class_line}
  <div class="ds-actions">
      {actions_html}
  </div>
  {quickstart}
  {benchmark_html}
  {citation_html}
  {caveats_html}
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
    n_trials_per_class = info.get("n_trials_per_class")
    runs_per_session = info.get("runs_per_session")
    n_sessions = info.get("n_sessions")
    trial_duration = info.get("trial_duration")
    has_hed = bool(info.get("hed_tags")) if info else False
    hed_html = _make_hed_summary_html(info) if has_hed else ""

    # Check which SVGs exist
    timeline_svg = os.path.join(srcdir, "_static", "timelines", f"{cls_name}.svg")
    sessions_svg = os.path.join(srcdir, "_static", "viz", f"{cls_name}_sessions.svg")
    classes_svg = os.path.join(srcdir, "_static", "viz", f"{cls_name}_classes.svg")

    has_timeline = os.path.exists(timeline_svg)
    has_sessions = os.path.exists(sessions_svg)
    has_classes = os.path.exists(classes_svg)
    # Build channel summary HTML
    channel_html = _make_channel_summary_html(info)

    # Count how many grid items we have
    n_items = sum([has_timeline, has_hed, has_classes, has_sessions, bool(channel_html)])
    if n_items == 0:
        # At minimum show the timeline if it exists, else skip grid
        if not has_timeline:
            return []

    # Determine grid columns — use 2 if 2+ items, else 1
    n_cols = 2 if n_items >= 2 else 1

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

    sessions_bits = []
    if n_sessions is not None:
        sessions_bits.append(f"{n_sessions} sessions/subject")
    if runs_per_session is not None:
        sessions_bits.append(f"{runs_per_session} runs/session")
    if (
        isinstance(n_trials_per_class, (int, float))
        and display_n_classes is not None
        and n_sessions
        and runs_per_session
        and trial_duration
    ):
        try:
            trials_per_session = (n_trials_per_class * display_n_classes) / n_sessions
            trials_per_run = trials_per_session / runs_per_session
            run_active_seconds = trials_per_run * trial_duration
            sessions_bits.append(
                f"~{_format_duration_seconds(run_active_seconds)} active time/run (no inter-trial gaps)"
            )
        except Exception:
            pass
    sessions_note = " \u00b7 ".join(sessions_bits)

    if has_timeline:
        lines.extend(
            [
                "   .. grid-item-card:: Stimulus Protocol",
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

    if has_classes:
        lines.extend(
            [
                "   .. grid-item-card:: Class Balance",
                "      :class-card: ds-viz-card",
                "",
                f"      .. image:: /_static/viz/{cls_name}_classes.svg",
                "         :width: 100%",
                "         :class: viz-diagram",
                "",
            ]
        )

    if has_sessions:
        lines.extend(
            [
                "   .. grid-item-card:: Sessions & Blocks",
                "      :class-card: ds-viz-card",
                "",
                f"      .. image:: /_static/viz/{cls_name}_sessions.svg",
                "         :width: 100%",
                "         :class: viz-diagram",
                "",
            ]
        )
        if sessions_note:
            lines.extend(
                [
                    "      .. raw:: html",
                    "",
                    f'         <p class="ds-viz-note">{escape(sessions_note)}</p>',
                    "",
                ]
            )

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
    pwc_lines = []  # PapersWithCode link

    current_bucket = "description"
    in_admonition = False
    admonition_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect PapersWithCode link at top
        if stripped.startswith("**PapersWithCode leaderboard:**"):
            pwc_lines.append(line)
            i += 1
            continue

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
    new_lines.append("")

    # --- Tab: Overview ---
    new_lines.append("   .. tab-item:: Overview")
    new_lines.append("")
    if pwc_lines:
        new_lines.extend(_reindent(pwc_lines, TAB_INDENT))
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

    # --- Layer 1: Enhanced card (inserted at top) ---
    top_block = []
    if info:
        live_citations = getattr(app.config, "dataset_card_live_citations", True)
        header_html = _make_header_html(
            cls_name, info, source_url=source_url, live_citations=live_citations
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

    # Add :inherited-members: after :members:
    source[0] = source[0].replace(
        "   :members:\n",
        "   :members:\n   :inherited-members:\n",
    )

    # Add __init__ to :special-members: so the constructor is documented
    source[0] = re.sub(
        r"(:special-members:.*)",
        r"\1,__init__",
        source[0],
    )


def _generate_all_svgs(app):
    """Generate timeline, class-balance, and session-structure SVGs.

    Runs once at the start of the Sphinx build (builder-inited event).
    SVGs are written to ``_static/timelines/`` and ``_static/viz/``.

    Controlled by the ``dataset_card_generate_svgs`` config value
    (default ``True``).  When ``False``, SVG generation is skipped entirely.
    Existing SVG files are never overwritten.
    """
    if not getattr(app.config, "dataset_card_generate_svgs", True):
        return

    import traceback

    srcdir = app.srcdir
    timeline_dir = os.path.join(srcdir, "_static", "timelines")
    viz_dir = os.path.join(srcdir, "_static", "viz")
    os.makedirs(timeline_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    try:
        from moabb.analysis.timeline import (
            class_balance_svg,
            session_structure_svg,
            stimulus_timeline_svg,
        )
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

        # Class balance
        classes_path = os.path.join(viz_dir, f"{name}_classes.svg")
        if not os.path.exists(classes_path):
            try:
                svg = class_balance_svg(ds)
                if svg:
                    with open(classes_path, "w", encoding="utf-8") as f:
                        f.write(svg)
            except Exception:
                pass

        # Session structure
        sessions_path = os.path.join(viz_dir, f"{name}_sessions.svg")
        if not os.path.exists(sessions_path):
            try:
                svg = session_structure_svg(ds)
                if svg:
                    with open(sessions_path, "w", encoding="utf-8") as f:
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
