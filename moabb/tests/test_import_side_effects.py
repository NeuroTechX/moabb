"""Importing moabb must not restyle the caller's matplotlib.

``moabb.analysis.plotting`` used to call ``seaborn.set_theme`` at import time,
which gave every figure of any program importing it (braindecode does, through
``moabb.datasets``) a serif font, grids and zero-length ticks. MOABB's plotting
functions now apply that style only while they draw.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest


_PROBE = """
import json, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
before = {{k: repr(v) for k, v in plt.rcParams.items()}}
import {module}  # noqa: F401
after = {{k: repr(v) for k, v in plt.rcParams.items()}}
json.dump({{k: [before[k], after[k]] for k in before if before[k] != after[k]}}, sys.stdout)
"""


def _changed_by_import(module):
    """rcParams changed by importing ``module`` in a fresh interpreter."""
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize(
    "module", ["moabb", "moabb.datasets", "moabb.analysis", "moabb.analysis.plotting"]
)
def test_importing_moabb_does_not_restyle_matplotlib(module):
    changed = _changed_by_import(module)
    assert changed == {}, f"import {module} changed rcParams: {sorted(changed)}"


def test_plotting_draws_in_moabb_style_without_leaking():
    import matplotlib.pyplot as plt

    from moabb.analysis.plotting import summary_plot

    names = ["A", "B"]
    sig = pd.DataFrame([[1.0, 0.01], [0.01, 1.0]], index=names, columns=names)
    effect = pd.DataFrame([[0.0, 0.5], [-0.5, 0.0]], index=names, columns=names)
    before = dict(plt.rcParams)
    fig = summary_plot(sig, effect)
    try:
        assert dict(plt.rcParams) == before
        # The style is kept for MOABB's own figure, not lost.
        label = fig.axes[0].get_xticklabels()[0]
        assert label.get_fontfamily() == ["serif"]
    finally:
        plt.close(fig)


def test_lazily_exported_names_still_resolve():
    import moabb.analysis

    for name in ("codecarbon_plot", "distribution_plot", "emissions_summary"):
        assert callable(getattr(moabb.analysis, name)) and name in dir(moabb.analysis)

    # The eager import used to bind the submodule as an attribute; deferring it
    # must not take that away.
    assert moabb.analysis.plotting.__name__ == "moabb.analysis.plotting"
    assert "plotting" in dir(moabb.analysis)

    # Exercise the __getattr__ this PR added, not the interpreter's own.
    with pytest.raises(AttributeError, match="definitely_not_real"):
        _ = moabb.analysis.definitely_not_real
