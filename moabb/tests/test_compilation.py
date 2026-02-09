"""Smoke tests for Python compilation."""

import py_compile
from pathlib import Path

import pytest


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_FILES = sorted(
    path for path in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts
)


@pytest.mark.parametrize(
    "py_file",
    _PYTHON_FILES,
    ids=lambda p: str(p.relative_to(_PACKAGE_ROOT)),
)
def test_python_file_compiles(py_file: Path):
    """Ensure source files are syntactically valid."""
    py_compile.compile(str(py_file), doraise=True)
