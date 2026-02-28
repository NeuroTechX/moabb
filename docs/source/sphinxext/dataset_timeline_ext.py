"""Sphinx extension: enhance dataset documentation pages.

1. Injects an enhanced dataset card (paradigm chips, stats, action buttons)
2. Injects a 2x2 visual summary grid (timeline, classes, sessions, channels)
3. Restructures the docstring into a tabbed layout (Overview, Metadata, Code, Notes)
4. Shows inherited methods below tabs

Pre-generated SVG images live in ``_static/timelines/<ClassName>.svg`` and
``_static/viz/<ClassName>_classes.svg`` / ``<ClassName>_sessions.svg``.

To regenerate *all* SVGs (timelines + viz), run (from the repo root)::

    PYTHONPATH=. python scripts/generate_dataset_viz.py
"""

import os
import re
from urllib.parse import quote


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
        n_sessions = getattr(ds, "n_sessions", None)
        code = getattr(ds, "code", None)
        doi = getattr(ds, "doi", None)
        event_id = getattr(ds, "event_id", None) or {}
        interval = getattr(ds, "interval", None)

        # Extract richer stats from METADATA
        metadata = getattr(ds, "METADATA", None) or getattr(type(ds), "METADATA", None)

        sampling_rate = None
        n_channels = None
        channel_types = None
        montage = None
        n_classes = None
        class_labels = None
        trial_duration = None
        n_trials_per_class = None
        runs_per_session = None
        sessions_per_subject = None

        if metadata is not None:
            acq = getattr(metadata, "acquisition", None)
            if acq is not None:
                sampling_rate = getattr(acq, "sampling_rate", None)
                n_channels = getattr(acq, "n_channels", None)
                channel_types = getattr(acq, "channel_types", None)
                montage = getattr(acq, "montage", None)

            exp = getattr(metadata, "experiment", None)
            if exp is not None:
                n_classes = getattr(exp, "n_classes", None)
                class_labels = getattr(exp, "class_labels", None)
                trial_duration = getattr(exp, "trial_duration", None)

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

        return {
            "paradigm": paradigm,
            "n_subjects": n_subjects,
            "n_sessions": n_sessions,
            "code": code,
            "doi": doi,
            "sampling_rate": sampling_rate,
            "n_channels": n_channels,
            "channel_types": channel_types,
            "montage": montage,
            "n_classes": n_classes,
            "class_labels": class_labels,
            "trial_duration": trial_duration,
            "n_trials_per_class": n_trials_per_class,
            "event_id": event_id,
            "runs_per_session": runs_per_session,
            "sessions_per_subject": sessions_per_subject,
        }
    except Exception:
        return None


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
    return (
        f"https://github.com/NeuroTechX/moabb/issues/new"
        f"?title={issue_title}&body={issue_body}&labels=dataset"
    )


def _make_header_html(cls_name, info):
    """Build the enhanced dataset card HTML (Layer 1)."""
    paradigm = info.get("paradigm") or "unknown"
    label = _PARADIGM_LABELS.get(paradigm, paradigm.title())
    color = _PARADIGM_COLORS.get(paradigm, "#546E7A")
    n_subj = info.get("n_subjects")
    n_sess = info.get("n_sessions")
    doi = info.get("doi")
    sampling_rate = info.get("sampling_rate")
    n_channels = info.get("n_channels")
    channel_types = info.get("channel_types")
    n_classes = info.get("n_classes")
    class_labels = info.get("class_labels")
    trial_duration = info.get("trial_duration")

    # --- Subtitle: auto-generated from paradigm + classes ---
    # Use the actual count of class labels when available
    display_n_classes = n_classes
    if class_labels:
        display_n_classes = len(class_labels)
    subtitle_parts = [label]
    if display_n_classes is not None:
        subtitle_parts.append(f"{display_n_classes} classes")
    if class_labels and len(class_labels) <= 6:
        subtitle_parts.append("(" + " vs ".join(class_labels[:6]) + ")")
    subtitle = ", ".join(subtitle_parts[:2])
    if len(subtitle_parts) > 2:
        subtitle += " " + subtitle_parts[2]

    # --- Stat chips ---
    chips = []
    chips.append(f'<span class="ds-chip" style="--chip-color: {color}">{label}</span>')
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

    # --- Action buttons ---
    actions = []
    # Quickstart Code button (toggles details element via CSS)
    actions.append(
        '<a class="ds-btn ds-btn-primary" href="#ds-quickstart" '
        "onclick=\"var el=document.getElementById('ds-quickstart');"
        "if(el){el.open=!el.open;}"
        'return false;">Quickstart Code</a>'
    )
    if doi:
        actions.append(
            f'<a class="ds-btn" href="https://doi.org/{doi}" '
            f'target="_blank" rel="noopener">Paper</a>'
        )
    github_url = _make_github_issue_url(cls_name)
    actions.append(
        f'<a class="ds-btn" href="{github_url}" '
        f'target="_blank" rel="noopener">Report Issue</a>'
    )
    actions_html = "\n      ".join(actions)

    # --- Quickstart code block ---
    quickstart = (
        f'<details id="ds-quickstart" class="ds-quickstart">\n'
        f'  <summary class="ds-quickstart-summary">Quickstart Code</summary>\n'
        f'  <pre class="ds-quickstart-code"><code>'
        f"from moabb.datasets import {cls_name}\n\n"
        f"dataset = {cls_name}()\n"
        f"data = dataset.get_data(subjects=[1])"
        f"</code></pre>\n"
        f"</details>"
    )

    return f"""\
<div class="ds-card">
  <p class="ds-subtitle">{subtitle}</p>
  <div class="ds-stats">
      {chips_html}
  </div>
  <div class="ds-actions">
      {actions_html}
  </div>
  {quickstart}
</div>"""


# ---------------------------------------------------------------------------
# Visual summary grid (Layer 2)
# ---------------------------------------------------------------------------


def _make_visual_grid_lines(cls_name, info, srcdir):
    """Build RST lines for the 2x2 visual summary grid."""
    lines = []

    # Check which SVGs exist
    timeline_svg = os.path.join(srcdir, "_static", "timelines", f"{cls_name}.svg")
    classes_svg = os.path.join(srcdir, "_static", "viz", f"{cls_name}_classes.svg")
    sessions_svg = os.path.join(srcdir, "_static", "viz", f"{cls_name}_sessions.svg")

    has_timeline = os.path.exists(timeline_svg)
    has_classes = os.path.exists(classes_svg)
    has_sessions = os.path.exists(sessions_svg)

    # Build channel summary HTML
    channel_html = _make_channel_summary_html(info)

    # Count how many grid items we have
    n_items = sum([has_timeline, has_classes, has_sessions, bool(channel_html)])
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

    if has_classes:
        lines.extend(
            [
                "   .. grid-item-card:: Classes & Trials",
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
    parts = []

    n_channels = info.get("n_channels") if info else None
    channel_types = info.get("channel_types") if info else None
    montage = info.get("montage") if info else None
    sampling_rate = info.get("sampling_rate") if info else None

    if n_channels is None and montage is None and sampling_rate is None:
        return ""

    # Line 1: N-channel + montage
    line1_parts = []
    if n_channels is not None:
        line1_parts.append(f"{n_channels}-channel")
    if montage is not None and montage != "standard_1005":
        line1_parts.append(f"{montage} montage")
    elif montage is not None:
        line1_parts.append("10-05 montage")
    if line1_parts:
        parts.append(" &middot; ".join(line1_parts))

    # Line 2: Channel type breakdown
    if channel_types and isinstance(channel_types, dict):
        type_strs = []
        for ctype, count in sorted(channel_types.items(), key=lambda x: -x[1]):
            type_strs.append(f"{count} {ctype.upper()}")
        if type_strs:
            parts.append(" &middot; ".join(type_strs))

    # Line 3: Sampling rate
    if sampling_rate is not None:
        sr_display = (
            f"{int(sampling_rate)} Hz"
            if sampling_rate == int(sampling_rate)
            else f"{sampling_rate:g} Hz"
        )
        parts.append(sr_display)

    if not parts:
        return ""

    inner = "<br>".join(parts)
    return f'<div class="ds-channel-card">{inner}</div>'


# ---------------------------------------------------------------------------
# Tabbed docstring restructuring (Layer 3)
# ---------------------------------------------------------------------------


def _restructure_docstring_lines(lines, cls_name):
    """Reorganize docstring lines into a tabbed layout.

    Scans lines for section markers and groups content into:
    - Overview (description + references)
    - Metadata (admonition cards)
    - Code Examples (quickstart snippet)
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

    # --- Tab: Metadata ---
    if has_metadata:
        new_lines.append("   .. tab-item:: Metadata")
        new_lines.append("")
        new_lines.extend(_reindent(metadata_lines, TAB_INDENT))
        new_lines.append("")

    # --- Tab: Code Examples ---
    new_lines.append("   .. tab-item:: Code Examples")
    new_lines.append("")
    new_lines.append(" " * TAB_INDENT + ".. code-block:: python")
    new_lines.append("")
    new_lines.append(" " * (TAB_INDENT + 3) + f"from moabb.datasets import {cls_name}")
    new_lines.append(" " * (TAB_INDENT + 3) + f"dataset = {cls_name}()")
    new_lines.append(" " * (TAB_INDENT + 3) + "data = dataset.get_data(subjects=[1])")
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


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Enhance dataset class docstrings with card, grid, and tabs."""
    if what != "class":
        return
    if not _is_concrete_dataset(obj):
        return

    cls_name = obj.__name__
    info = _get_dataset_info(obj)

    # --- Layer 1: Enhanced card (inserted at top) ---
    top_block = []
    if info:
        header_html = _make_header_html(cls_name, info)
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
    restructured = _restructure_docstring_lines(lines, cls_name)
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

    # Remove the right-sidebar "On this page" ToC on dataset pages
    source[0] = ".. meta::\n   :html_theme.sidebar_secondary.remove:\n\n" + source[0]

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
    """
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
        print("[dataset_timeline_ext] Could not import timeline functions. "
              "Make sure moabb is installed from the current repo.")
        return

    for ds_cls in dataset_list:
        name = ds_cls.__name__
        try:
            ds = ds_cls()
        except Exception:
            continue

        # Timeline
        try:
            svg = stimulus_timeline_svg(ds)
            with open(os.path.join(timeline_dir, f"{name}.svg"), "w") as f:
                f.write(svg)
        except Exception:
            pass

        # Class balance
        try:
            svg = class_balance_svg(ds)
            if svg:
                with open(os.path.join(viz_dir, f"{name}_classes.svg"), "w") as f:
                    f.write(svg)
        except Exception:
            pass

        # Session structure
        try:
            svg = session_structure_svg(ds)
            if svg:
                with open(os.path.join(viz_dir, f"{name}_sessions.svg"), "w") as f:
                    f.write(svg)
        except Exception:
            pass


def setup(app):
    app.connect("builder-inited", _generate_all_svgs)
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("source-read", source_read_add_inherited)
    return {"version": "1.0", "parallel_read_safe": True}
