"""Tests for moabb.analysis.chance_level module."""

import pandas as pd

from moabb.analysis.chance_level import adjusted_chance_level, chance_by_chance


def test_adjusted_chance_level():
    # Adjusted threshold should exceed theoretical (1/n_classes)
    assert adjusted_chance_level(2, 20, 0.05) > 0.5
    assert adjusted_chance_level(2, 50, 0.01) > adjusted_chance_level(2, 50, 0.05)


def test_chance_by_chance():
    data = pd.DataFrame(
        {
            "dataset": ["A", "A", "B", "B"],
            "samples_test": [50, 50, 100, 100],
            "n_classes": [2, 2, 4, 4],
        }
    )
    levels = chance_by_chance(data, alpha=0.05)
    assert levels["A"]["theoretical"] == 0.5
    assert levels["A"]["adjusted"][0.05] > 0.5
    assert levels["B"]["theoretical"] == 0.25
