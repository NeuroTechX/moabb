"""
Convert Python example scripts to Jupyter notebooks for Colab integration.

This script converts Sphinx-Gallery style Python examples to Jupyter notebooks.
It is adapted from Braindecode's docs workflow utilities.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat
from sphinx_gallery import gen_gallery
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery.py_source_parser import split_code_and_text_blocks


def convert_script_to_notebook(
    src_file: Path, output_file: Path, gallery_conf: dict
) -> None:
    """Convert a single Python script to a Jupyter notebook."""
    # Parse the Python file
    _file_conf, blocks = split_code_and_text_blocks(str(src_file))

    # Convert to notebook (returns a dict, not a notebook object)
    example_nb_dict = jupyter_notebook(blocks, gallery_conf, str(src_file.parent))

    # Convert dict to nbformat notebook object
    example_nb = nbformat.from_dict(example_nb_dict)

    # Prepend an installation cell for moabb (unless the example already does so)
    first_source = ""
    if getattr(example_nb, "cells", None):
        try:
            first_source = example_nb.cells[0].source
        except (IndexError, AttributeError):
            first_source = ""

    install_cmd = "%pip install moabb"
    if "pip install" not in first_source or "moabb" not in first_source:
        install_cell = nbformat.v4.new_code_cell(source=install_cmd)
        install_cell.metadata["language"] = "python"
        example_nb.cells.insert(0, install_cell)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_notebook(example_nb, output_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Python example script to a Jupyter notebook."
    )
    parser.add_argument("--input", required=True, help="Path to the Python script.")
    parser.add_argument("--output", required=True, help="Path to the output notebook.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    gallery_conf = copy.deepcopy(gen_gallery.DEFAULT_GALLERY_CONF)

    convert_script_to_notebook(input_path, output_path, gallery_conf)
    print(f"Notebook saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
