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
    data, alpha: float | list[float] = 0.05
) -> dict[str, dict[str, Any]]:
    """Compute chance levels from ``samples_test`` and ``n_classes`` columns."""
    if isinstance(alpha, (int, float)):
        alpha = [alpha]
    result = {}
    for dname, grp in data.groupby("dataset"):
        if grp[["n_classes", "samples_test"]].isna().any().any():
            raise ValueError(
                "Adjusted chance level requires non-missing n_classes and "
                f"samples_test values for every row of dataset {dname!r}."
            )
        n_classes_values = grp["n_classes"].unique()
        n_trials_values = grp["samples_test"].unique()
        if len(n_classes_values) != 1 or len(n_trials_values) != 1:
            raise ValueError(
                "Adjusted chance level requires one n_classes and one samples_test "
                f"value per dataset, but {dname!r} has "
                f"n_classes={n_classes_values.tolist()} and "
                f"samples_test={n_trials_values.tolist()}."
            )
        n_classes = int(n_classes_values[0])
        n_trials = int(n_trials_values[0])
        result[dname] = {
            # theoretical chance level: 1 / n_classes
            "theoretical": 1.0 / n_classes,
            "adjusted": {a: adjusted_chance_level(n_classes, n_trials, a) for a in alpha},
        }
    return result
