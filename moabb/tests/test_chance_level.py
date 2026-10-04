"""Tests for moabb.analysis.chance_level module."""

import pandas as pd
import pytest

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


def test_chance_by_chance_uses_conservative_smallest_test_fold():
    data = pd.DataFrame(
        {
            "dataset": ["A", "A"],
            "samples_test": [100, 50],
            "n_classes": [2, 2],
        }
    )

    levels = chance_by_chance(data, alpha=0.05)

    expected = adjusted_chance_level(2, 50, 0.05)
    assert levels["A"]["adjusted"][0.05] == expected

    reversed_levels = chance_by_chance(
        data.iloc[::-1].reset_index(drop=True), alpha=0.05
    )
    assert reversed_levels == levels


def test_chance_by_chance_rejects_ambiguous_class_count():
    data = pd.DataFrame(
        {
            "dataset": ["A", "A"],
            "samples_test": [50, 100],
            "n_classes": [2, 3],
        }
    )

    with pytest.raises(ValueError, match="requires one n_classes"):
        chance_by_chance(data)


@pytest.mark.parametrize("column", ["samples_test", "n_classes"])
def test_chance_by_chance_rejects_missing_metadata(column):
    data = pd.DataFrame(
        {"dataset": ["A", "A"], "samples_test": [50.0, 50.0], "n_classes": [2.0, 2.0]}
    )
    data.loc[1, column] = float("nan")

    with pytest.raises(ValueError, match="requires non-missing"):
        chance_by_chance(data)
