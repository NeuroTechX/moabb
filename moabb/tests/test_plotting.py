import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.pyplot import Figure


matplotlib.use("Agg")

from moabb.analysis.plotting import (
    _get_bubble_coordinates,
    _get_dataset_parameters,
    _get_hexa_grid,
    _nemenyi_critical_difference,
    dataset_bubble_plot,
    distribution_plot,
    paired_plot,
    plot_critical_difference,
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


def _make_df(pipelines=("P0", "P1")):
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
        for pipe in pipelines
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


def test_nemenyi_critical_difference_matches_demsar_2006():
    # Demšar (2006), Sec. 3.2: k=4 classifiers on N=14 datasets at
    # alpha=0.05 gives a Nemenyi critical difference of 1.25.
    assert _nemenyi_critical_difference(4, 14, 0.05) == pytest.approx(1.25, abs=0.01)


def test_plot_critical_difference_uses_subject_balanced_complete_blocks():
    rows = []
    scores = {
        "D0": {"P0": 0.65, "P1": 0.7, "P2": 0.5},
        "D1": {"P0": 0.7, "P1": 0.9, "P2": 0.5},
        "D2": {"P0": 0.9, "P1": 0.5, "P2": 0.7},
        "D3": {"P0": 0.7, "P1": 0.5, "P2": 0.9},
    }
    for dataset, pipelines in scores.items():
        for pipeline, score in pipelines.items():
            if (dataset, pipeline) == ("D0", "P0"):
                session_scores = {
                    "0": [score + 0.3, score + 0.3, score + 0.3],
                    "1": [score - 0.3],
                }
            else:
                session_scores = {"0": [score], "1": [score]}
            for subject, subject_scores in session_scores.items():
                for session, session_score in enumerate(subject_scores):
                    rows.append(
                        {
                            "dataset": dataset,
                            "pipeline": pipeline,
                            "subject": subject,
                            "session": str(session),
                            "score": session_score,
                        }
                    )

    fig = plot_critical_difference(pd.DataFrame(rows))
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert any("Friedman p" in text.get_text() for text in fig.texts)
    ranks = sorted(
        tuple(collection.get_offsets()[0])
        for collection in ax.collections
        if isinstance(collection, PathCollection)
        and len(collection.get_offsets()) == 1
        and collection.get_offsets()[0][1] == 0
    )
    np.testing.assert_allclose([rank for rank, _ in ranks], [1.75, 2.0, 2.25])
    plt.close(fig)


def test_plot_critical_difference_requires_three_pipelines():
    data = _make_df(pipelines=("P0", "P1"))
    with pytest.raises(ValueError, match="At least three pipelines"):
        plot_critical_difference(data)


def test_plot_critical_difference_rejects_incomplete_benchmarks():
    data = _make_df(pipelines=("P0", "P1", "P2")).query(
        "not (dataset == 'D0' and pipeline == 'P1')"
    )
    with pytest.raises(ValueError, match="requires every pipeline"):
        plot_critical_difference(data)


def test_plot_critical_difference_rejects_unbalanced_subject_sets():
    data = _make_df(pipelines=("P0", "P1", "P2"))
    data = data.query("not (dataset == 'D0' and pipeline == 'P1' and subject == 3)")

    with pytest.raises(ValueError, match="same subjects"):
        plot_critical_difference(data)


def test_plot_critical_difference_rejects_missing_identifiers():
    data = _make_df(pipelines=("P0", "P1", "P2"))
    data.loc[data.index[0], "subject"] = np.nan

    with pytest.raises(ValueError, match="identifier columns"):
        plot_critical_difference(data)


def test_plot_critical_difference_rejects_mixed_evaluation_protocols():
    data = _make_df(pipelines=("P0", "P1", "P2"))
    data["evaluation"] = "WithinSession"
    data.loc[data["dataset"] == "D1", "evaluation"] = "CrossSubject"

    with pytest.raises(ValueError, match="single evaluation protocol"):
        plot_critical_difference(data)


def test_plot_critical_difference_rejects_missing_evaluation_identity():
    data = _make_df(pipelines=("P0", "P1", "P2"))
    data["evaluation"] = "WithinSession"
    data.loc[data.index[0], "evaluation"] = np.nan

    with pytest.raises(ValueError, match="evaluation must not contain missing"):
        plot_critical_difference(data)


def test_plot_critical_difference_rejects_missing_requested_pipeline():
    data = _make_df(pipelines=("P0", "P1", "P2"))

    with pytest.raises(ValueError, match="Requested pipelines are missing"):
        plot_critical_difference(data, pipelines=["P0", "P1", "P2", "P3"])


def test_plot_critical_difference_handles_identical_pipelines():
    data = _make_df(pipelines=("P0", "P1", "P2"))
    data["score"] = 0.5
    fig = plot_critical_difference(data)
    assert any("Friedman p = 1" in text.get_text() for text in fig.texts)
    plt.close(fig)


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
