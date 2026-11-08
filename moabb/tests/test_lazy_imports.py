"""Tests for lazy imports across MOABB packages (Issue #1153)."""

import subprocess
import sys

import pytest

import moabb


def _run_python(code: str) -> str:
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def test_import_moabb_is_lazy():
    """Importing top-level moabb should not pull heavy subpackages or third-party libs."""
    out = _run_python(
        """
import sys
import moabb

eager = [
    m for m in (
        "moabb.datasets",
        "moabb.paradigms",
        "moabb.evaluations",
        "moabb.pipelines",
        "moabb.analysis",
        "moabb.utils",
        "sklearn",
        "pyriemann",
        "matplotlib",
    )
    if m in sys.modules
]
print(",".join(eager))
"""
    )
    assert out == ""


def test_import_single_dataset_is_lazy():
    """Importing one dataset must not import all other dataset modules."""
    out = _run_python(
        """
import sys
from moabb.datasets import BNCI2014_001

eager = [
    m for m in (
        "moabb.datasets.physionet_mi",
        "moabb.datasets.gigadb",
        "moabb.datasets.schirrmeister2017",
        "moabb.datasets.compound_dataset",
        "moabb.paradigms",
        "moabb.evaluations",
        "matplotlib",
    )
    if m in sys.modules
]
print(",".join(eager))
"""
    )
    assert out == ""


def test_import_single_paradigm_is_lazy():
    """Importing LeftRightImagery must not import unrelated paradigms or initialize dataset_list."""
    out = _run_python(
        """
import sys
from moabb.paradigms import LeftRightImagery

eager = [
    m for m in (
        "moabb.paradigms.p300",
        "moabb.paradigms.ssvep",
        "moabb.paradigms.cvep",
        "moabb.datasets.physionet_mi",
        "moabb.datasets.gigadb",
    )
    if m in sys.modules
]
print(",".join(eager))
"""
    )
    assert out == ""


def test_lazy_attributes_and_dir():
    """All top-level and subpackage lazy exports resolve and appear in dir()."""
    assert callable(moabb.benchmark)
    assert callable(moabb.set_log_level)
    assert "datasets" in dir(moabb)
    assert "paradigms" in dir(moabb)
    assert "BNCI2014_001" in dir(moabb.datasets)
    assert "LeftRightImagery" in dir(moabb.paradigms)
    assert "WithinSessionEvaluation" in dir(moabb.evaluations)
    assert "FilterBank" in dir(moabb.pipelines)
    assert "Results" in dir(moabb.analysis)

    with pytest.raises(AttributeError, match="DemonsP300 has been removed"):
        _ = moabb.datasets.DemonsP300

    with pytest.raises(AttributeError, match="braindecode"):
        _ = moabb.pipelines.KerasEEGNet_8_2
