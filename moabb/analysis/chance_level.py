"""Chance level computation utilities.

Implements adjusted chance levels based on Combrisson & Jerbi (2015).
"""

from __future__ import annotations

from typing import Any

from scipy.stats import binom


def adjusted_chance_level(n_classes: int, n_trials: int, alpha: float = 0.05) -> float:
    """Adjusted chance level via binomial inverse survival function."""
    # theoretical chance level: 1 / n_classes
    return binom.isf(alpha, n_trials, 1.0 / n_classes) / n_trials


def chance_by_chance(
    data,
    alpha: float | list[float] = 0.05,
) -> dict[str, dict[str, Any]]:
    """Compute chance levels from ``samples_test`` and ``n_classes`` columns."""
    if isinstance(alpha, (int, float)):
        alpha = [alpha]
    result = {}
    for dname, grp in data.groupby("dataset"):
        n_classes = int(grp["n_classes"].iloc[0])
        n_trials = int(grp["samples_test"].iloc[0])
        result[dname] = {
            # theoretical chance level: 1 / n_classes
            "theoretical": 1.0 / n_classes,
            "adjusted": {a: adjusted_chance_level(n_classes, n_trials, a) for a in alpha},
        }
    return result
