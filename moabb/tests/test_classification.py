import pytest
from sklearn.utils.validation import NotFittedError, check_is_fitted

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import SSVEP
from moabb.pipelines import SSVEP_MsetCCA


class TestSSVEP_MsetCCA:
    def setup_method(self):
        # Use moabb generated dataset for test
        dataset = FakeDataset(n_sessions=1, n_runs=1, n_subjects=1, paradigm="ssvep")
        paradigm = SSVEP(n_classes=3)
        X, y, _ = paradigm.get_data(dataset)
        self.freqs = paradigm.used_events(dataset)
        self.n_filters = 2
        self.X = X
        self.y = y
        self.clf = SSVEP_MsetCCA(freqs=self.freqs, n_filters=self.n_filters)

    def test_fit(self):
        self.clf.fit(self.X, self.y)
        assert hasattr(self.clf, "classes_")
        assert hasattr(self.clf, "one_hot")
        assert hasattr(self.clf, "Ym")

    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        assert len(y_pred) == len(self.X)

    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        assert P.shape[0] == len(self.X)
        assert P.shape[1] == len(self.freqs)

    def test_fit_predict_is_fitted(self):
        with pytest.raises(NotFittedError):
            self.clf.predict(self.X)
        with pytest.raises(NotFittedError):
            self.clf.predict_proba(self.X)

        self.clf.fit(self.X, self.y)
        check_is_fitted(self.clf, attributes=["classes_", "one_hot", "Ym"])
