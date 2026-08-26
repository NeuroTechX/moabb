"""Importing moabb must not restyle the caller's matplotlib.

``moabb.analysis.plotting`` themes the global rcParams at import time. That is
fine for someone who asked for plotting, but it used to reach anyone who merely
imported a dataset, via ``moabb.datasets.bids_interface`` -> ``analysis.results``.
"""

import json
import subprocess
import sys

import pytest


# Read at axes creation, so a caller cannot undo them afterwards.
_WATCHED = ("font.family", "axes.grid", "axes.spines.left")

_PROBE = """
import json, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
before = {{k: repr(plt.rcParams[k]) for k in {watched!r}}}
import {module}  # noqa: F401
after = {{k: repr(plt.rcParams[k]) for k in {watched!r}}}
json.dump([before, after], sys.stdout)
"""


def _around_import(module):
    """rcParams before and after importing ``module`` in a fresh interpreter."""
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, watched=_WATCHED)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_importing_moabb_does_not_restyle_matplotlib():
    for module in ("moabb", "moabb.datasets", "moabb.analysis"):
        before, after = _around_import(module)
        assert before == after, f"import {module} mutated rcParams: {before} -> {after}"

    # The theme must be kept, not lost -- only stopped from leaking.
    before, after = _around_import("moabb.analysis.plotting")
    assert before != after


def test_lazily_exported_names_still_resolve():
    import moabb.analysis

    for name in ("codecarbon_plot", "distribution_plot", "emissions_summary"):
        assert callable(getattr(moabb.analysis, name)) and name in dir(moabb.analysis)

    # Exercise the __getattr__ this PR added, not the interpreter's own.
    with pytest.raises(AttributeError, match="definitely_not_real"):
        _ = moabb.analysis.definitely_not_real
