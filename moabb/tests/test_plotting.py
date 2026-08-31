import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure


matplotlib.use("Agg")

from moabb.analysis.plotting import (
    _get_bubble_coordinates,
    _get_dataset_parameters,
    _get_hexa_grid,
    dataset_bubble_plot,
    distribution_plot,
    paired_plot,
    score_plot,
)
from moabb.datasets.utils import dataset_list


@pytest.mark.parametrize(
    "dataset_class", [pytest.param(d, id=d.__name__) for d in dataset_list]
)
def test_get_dataset_parameters(dataset_class):
    if "Fake" in dataset_class.__name__:
        pytest.skip(
            f"Skipping test for {dataset_class.__name__} as it is a fake dataset."
        )
    dataset = dataset_class()
    dataset_name, paradigm, n_subjects, n_sessions, n_trials, trial_len = (
        _get_dataset_parameters(dataset)
    )
    assert isinstance(dataset_name, str)
    assert isinstance(paradigm, str)
    assert isinstance(n_subjects, int)
    assert isinstance(n_sessions, int)
    assert isinstance(n_trials, int)
    assert isinstance(trial_len, float)


def _make_df():
    rng = np.random.RandomState(42)
    rows = [
        {
            "dataset": ds,
            "pipeline": pipe,
            "subject": subj,
            "session": "0",
            "score": rng.uniform(0.4, 1.0),
            "time": 0.1,
            "n_samples": 100,
            "n_channels": 10,
            "samples_test": 50,
            "n_classes": 2,
        }
        for ds in ["D0", "D1"]
        for pipe in ["P0", "P1"]
        for subj in range(1, 4)
    ]
    return pd.DataFrame(rows)


def test_score_plot_auto():
    fig, _ = score_plot(_make_df(), chance_level="auto")
    assert isinstance(fig, Figure)


def test_distribution_plot():
    fig, _ = distribution_plot(_make_df(), chance_level="auto")
    assert isinstance(fig, Figure)


def test_paired_plot():
    fig = paired_plot(_make_df(), "P0", "P1", chance_level="auto")
    assert isinstance(fig, Figure)


def test_hexa_grid_is_reproducible():
    x1, y1 = _get_hexa_grid(4, 1.0, (0.0, 0.0), random_state=42)
    x2, y2 = _get_hexa_grid(4, 1.0, (0.0, 0.0), random_state=42)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_hexa_grid_accepts_generator():
    x1, y1 = _get_hexa_grid(4, 1.0, (0.0, 0.0), random_state=np.random.default_rng(42))
    x2, y2 = _get_hexa_grid(4, 1.0, (0.0, 0.0), random_state=np.random.default_rng(42))

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_bubble_coordinates_change_with_seed():
    x1, y1 = _get_bubble_coordinates(6, 1.0, (0.0, 0.0), random_state=1)
    x2, y2 = _get_bubble_coordinates(6, 1.0, (0.0, 0.0), random_state=2)

    assert not (np.array_equal(x1, x2) and np.array_equal(y1, y2))


def test_dataset_bubble_plot_is_reproducible():
    kwargs = {
        "dataset_name": "TestDataset",
        "paradigm": "imagery",
        "n_subjects": 8,
        "n_sessions": 1,
        "n_trials": 100,
        "trial_len": 2.0,
        "legend": False,
        "random_state": 42,
    }
    fig1, ax1 = plt.subplots()
    dataset_bubble_plot(**kwargs, ax=ax1)
    fig1.canvas.draw()
    image1 = np.asarray(fig1.canvas.buffer_rgba()).copy()
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    dataset_bubble_plot(**kwargs, ax=ax2)
    fig2.canvas.draw()
    image2 = np.asarray(fig2.canvas.buffer_rgba()).copy()
    plt.close(fig2)

    np.testing.assert_array_equal(image1, image2)
