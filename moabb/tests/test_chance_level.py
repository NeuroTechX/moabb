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



@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("samples_test", [50, 100]),
        ("n_classes", [2, 3]),
    ],
)
def test_chance_by_chance_rejects_ambiguous_dataset_level_threshold(column, values):
    data = pd.DataFrame(
        {
            "dataset": ["A", "A"],
            "samples_test": [50, 50],
            "n_classes": [2, 2],
        }
    )
    data[column] = values

    with pytest.raises(ValueError, match="requires one n_classes and one samples_test"):
        chance_by_chance(data)


def test_chance_by_chance_is_row_order_invariant_for_valid_input():
    data = pd.DataFrame(
        {
            "dataset": ["A", "A", "B", "B"],
            "samples_test": [50, 50, 100, 100],
            "n_classes": [2, 2, 4, 4],
        }
    )

    forward = chance_by_chance(data, alpha=[0.05, 0.01])
    reversed_rows = chance_by_chance(
        data.iloc[::-1].reset_index(drop=True), alpha=[0.05, 0.01]
    )

    assert forward == reversed_rows
