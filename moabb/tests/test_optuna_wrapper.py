import pickle

import pytest
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression

from moabb.evaluations import utils
from moabb.evaluations.utils import check_search_available


pytest.importorskip("optuna")


@pytest.mark.filterwarnings("ignore:Sparse CSR_:UserWarning")
def test_optuna_wrapper_reports_classifier_tags_and_pickles():
    search_methods, available = check_search_available()
    assert available
    optuna_search = search_methods.get("optuna")
    assert optuna_search is not None

    estimator = LogisticRegression()
    param_distributions = {"C": [0.1, 1.0]}

    optuna_cv = optuna_search(estimator, param_distributions, n_trials=1, cv=2)

    assert is_classifier(optuna_cv)

    tags = optuna_cv.__sklearn_tags__()
    if isinstance(tags, dict):
        assert tags.get("estimator_type") == "classifier"
    else:
        assert getattr(tags, "estimator_type", None) == "classifier"

    pickle.dumps(optuna_cv)


@pytest.mark.filterwarnings("ignore:Sparse CSR_:UserWarning")
def test_optuna_wrapper_sets_classifier_tag_on_dict_tags(monkeypatch):
    search_methods, available = check_search_available()
    assert available
    optuna_search = search_methods.get("optuna")
    assert optuna_search is not None

    base_cls = utils._BaseOptunaSearchCV
    original = base_cls.__sklearn_tags__

    def fake_sklearn_tags(self):
        return {}

    monkeypatch.setattr(base_cls, "__sklearn_tags__", fake_sklearn_tags)

    try:
        optuna_cv = optuna_search(LogisticRegression(), {"C": [0.5]}, n_trials=1, cv=2)
        tags = optuna_cv.__sklearn_tags__()
        assert isinstance(tags, dict)
        assert tags.get("estimator_type") == "classifier"
    finally:
        monkeypatch.setattr(base_cls, "__sklearn_tags__", original)
