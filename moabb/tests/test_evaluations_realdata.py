"""Real-data regression tests for parallel vs legacy evaluation paths.

These tests download and use real EEG datasets (BNCI2014_001) to verify
that the parallel and legacy code paths produce identical scores.

Run with::

    pytest moabb/tests/test_evaluations_realdata.py --run-slow -v
"""

import tempfile

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb.datasets import BNCI2014_001
from moabb.evaluations.evaluations import (
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
    WithinSessionEvaluation,
)
from moabb.paradigms.motor_imagery import LeftRightImagery


@pytest.fixture(scope="module")
def real_dataset():
    """BNCI2014_001 with 2 subjects for fast but real testing."""
    dataset = BNCI2014_001()
    dataset.subject_list = dataset.subject_list[:2]
    return dataset


@pytest.fixture(scope="module")
def paradigm():
    return LeftRightImagery()


@pytest.fixture(scope="module")
def pipelines():
    try:
        from mne.decoding import CSP

        return {"CSP+LDA": make_pipeline(CSP(n_components=8), LDA())}
    except ImportError:
        return {"LDA": make_pipeline(LDA())}


def _run_and_get_scores(eval_class, paradigm, dataset, pipelines, seed=42, legacy=False):
    """Run an evaluation (parallel or legacy) and return sorted score DataFrame."""
    with tempfile.TemporaryDirectory() as tmp:
        ev = eval_class(
            paradigm=paradigm,
            datasets=[dataset],
            random_state=seed,
            n_jobs=1,
            overwrite=True,
            hdf5_path=tmp,
        )
        if legacy:
            results = ev._process_legacy(
                pipelines, param_grid=None, postprocess_pipeline=None
            )
        else:
            results = ev.process(pipelines)
    keys = ["subject", "session", "pipeline"]
    return results[keys + ["score"]].sort_values(keys).reset_index(drop=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "eval_class",
    [WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation],
    ids=["WithinSession", "CrossSession", "CrossSubject"],
)
def test_parallel_legacy_scores_match(eval_class, real_dataset, paradigm, pipelines):
    """Verify parallel and legacy paths produce identical scores on real data."""
    parallel = _run_and_get_scores(eval_class, paradigm, real_dataset, pipelines)
    legacy = _run_and_get_scores(
        eval_class, paradigm, real_dataset, pipelines, legacy=True
    )
    np.testing.assert_allclose(
        parallel["score"].to_numpy(),
        legacy["score"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"{eval_class.__name__} parallel vs legacy scores differ on real data",
    )
