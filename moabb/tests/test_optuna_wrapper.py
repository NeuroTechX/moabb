"""Optuna search wrapper behavior tests."""

import pickle

import pytest


optuna = pytest.importorskip("optuna", reason="Optuna is required for these tests.")
from optuna.distributions import FloatDistribution
from sklearn.base import is_classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from moabb.evaluations.base import optuna_available
from moabb.evaluations.utils import OptunaSearchCVClassifier, check_search_available


pytestmark = pytest.mark.skipif(
    not optuna_available, reason="Optuna is required for these tests."
)


def _make_search_cv(**kwargs):
    search_methods, available = check_search_available()
    assert available
    optuna_search_cls = search_methods["optuna"]
    return optuna_search_cls(
        estimator=LogisticRegression(max_iter=200),
        param_distributions={"C": FloatDistribution(0.1, 1.0)},
        n_trials=1,
        cv=2,
        random_state=0,
        **kwargs,
    )


def test_optuna_wrapper_reports_classifier_tags():
    search_cv = _make_search_cv()

    assert isinstance(search_cv, OptunaSearchCVClassifier)
    assert is_classifier(search_cv)

    tags = search_cv.__sklearn_tags__()
    estimator_type = (
        tags.get("estimator_type")
        if isinstance(tags, dict)
        else getattr(tags, "estimator_type", None)
    )
    assert estimator_type == "classifier"


def test_optuna_wrapper_pickles_after_fit():
    search_cv = _make_search_cv()
    data, target = load_iris(return_X_y=True)

    search_cv.fit(data, target)

    payload = pickle.dumps(search_cv)
    assert isinstance(payload, bytes)
