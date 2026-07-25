"""Sphinx extension: inject pre-generated macro table into dataset_summary.

The heavy lifting (metadata extraction, HTML generation) is done by
``scripts/generate_macro_table.py``, which writes a self-contained HTML
fragment to ``_static/macro_table.html``.  This extension simply reads
that file and replaces the ``<!-- MACRO_TABLE -->`` placeholder in
``dataset_summary.rst`` at source-read time.
"""

import logging
import os


logger = logging.getLogger(__name__)


def _on_source_read(app, docname, source):
    if docname != "dataset_summary":
        return

    placeholder = "<!-- MACRO_TABLE -->"
    if placeholder not in source[0]:
        return

    html_path = os.path.join(app.srcdir, "_static", "macro_table.html")
    if not os.path.isfile(html_path):
        logger.warning(
            "macro_table_ext: %s not found. "
            "Run 'python scripts/generate_macro_table.py' first.",
            html_path,
        )
        source[0] = source[0].replace(placeholder, "")
        return

    with open(html_path, encoding="utf-8") as fh:
        macro_html = fh.read()

    # Wrap in raw HTML directive for RST processing
    raw_block = "\n.. raw:: html\n\n"
    raw_block += "\n".join("   " + line for line in macro_html.splitlines())
    raw_block += "\n\n"

    source[0] = source[0].replace(placeholder, raw_block)


def setup(app):
    app.connect("source-read", _on_source_read)
    return {"version": "0.1", "parallel_read_safe": True}
