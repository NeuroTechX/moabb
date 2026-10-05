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
        if len(n_classes_values) != 1:
            raise ValueError(
                "Dataset-level chance requires one n_classes value per dataset, "
                f"but {dname!r} has {n_classes_values.tolist()}."
            )
        n_classes = int(n_classes_values[0])
        # Exact-binomial thresholds are discrete and are not strictly monotone
        # in the number of trials. Build a dataset-level envelope by evaluating
        # every fold size that actually occurs and taking the strictest
        # threshold for each alpha, rather than assuming the smallest fold wins.
        n_trials_values = [int(value) for value in grp["samples_test"].unique()]
        result[dname] = {
            # theoretical chance level: 1 / n_classes
            "theoretical": 1.0 / n_classes,
            "adjusted": {
                a: max(
                    adjusted_chance_level(n_classes, n_trials, a)
                    for n_trials in n_trials_values
                )
                for a in alpha
            },
        }
    return result
