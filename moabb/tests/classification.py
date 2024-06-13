import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import SSVEP
from moabb.pipelines import SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA


class TestSSVEP_CCA(unittest.TestCase):
    def setUp(self):
        # Use moabb generated dataset for test
        dataset = FakeDataset(n_sessions=1, n_runs=1, n_subjects=1, paradigm="ssvep")
        paradigm = SSVEP(n_classes=3)
        X, y, _ = paradigm.get_data(dataset)
        self.freqs = paradigm.used_events(dataset)
        self.n_harmonics = 3
        self.X = X
        self.y = y
        self.clf = SSVEP_CCA(n_harmonics=self.n_harmonics)

    def test_fit(self):
        self.clf.fit(self.X, self.y)
        self.assertTrue(hasattr(self.clf, "freqs_"))
        self.assertTrue(hasattr(self.clf, "classes_"))
        self.assertTrue(hasattr(self.clf, "le_"))
        self.assertTrue(hasattr(self.clf, "one_hot_"))
        self.assertTrue(hasattr(self.clf, "slen_"))

    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], len(self.freqs_))

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)
        self.clf.fit(self.X, self.y)
        check_is_fitted(
            self.clf, attributes=["classes_", "one_hot_", "slen_", "freqs_", "le_"]
        )


class TestSSVEP_TRCA(unittest.TestCase):
    def setUp(self):
        # Use moabb generated dataset for test
        dataset = FakeDataset(n_sessions=1, n_runs=1, n_subjects=1, paradigm="ssvep")
        self.n_classes = 3
        paradigm = SSVEP(n_classes=self.n_classes)
        X, y, _ = paradigm.get_data(dataset)
        self.freqs = paradigm.used_events(dataset)
        self.n_fbands = 3
        self.X = X
        self.y = y

    def test_fit(self):
        for method in ["original", "riemann", "logeuclid"]:
            for estimator in ["scm", "lwf", "oas"]:
                self.clf = SSVEP_TRCA(
                    n_fbands=self.n_fbands, method=method, estimator=estimator
                )
                self.clf.fit(self.X, self.y)
                self.assertTrue(hasattr(self.clf, "freqs_"))
                self.assertTrue(hasattr(self.clf, "peaks_"))
                self.assertTrue(hasattr(self.clf, "classes_"))
                self.assertTrue(hasattr(self.clf, "n_classes"))
                self.assertTrue(hasattr(self.clf, "le_"))
                self.assertTrue(hasattr(self.clf, "one_hot_"))
                self.assertTrue(hasattr(self.clf, "one_inv_"))
                self.assertTrue(hasattr(self.clf, "sfreq_"))

    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], len(self.n_classes))

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)
        self.clf.fit(self.X, self.y)
        check_is_fitted(
            self.clf,
            attributes=[
                "classes_",
                "n_classes",
                "peaks_",
                "one_hot_",
                "one_inv_",
                "freqs_",
                "le_",
                "sfreq_",
            ],
        )


class TestSSVEP_MsetCCA(unittest.TestCase):
    def setUp(self):
        # Use moabb generated dataset for test
        dataset = FakeDataset(n_sessions=1, n_runs=1, n_subjects=1, paradigm="ssvep")
        paradigm = SSVEP(n_classes=3)
        X, y, _ = paradigm.get_data(dataset)
        self.freqs = paradigm.used_events(dataset)
        self.n_filters = 2
        self.X = X
        self.y = y
        self.clf = SSVEP_MsetCCA(n_filters=self.n_filters)

    def test_fit(self):
        self.clf.fit(self.X, self.y)
        self.assertTrue(hasattr(self.clf, "freqs_"))
        self.assertTrue(hasattr(self.clf, "classes_"))
        self.assertTrue(hasattr(self.clf, "le_"))
        self.assertTrue(hasattr(self.clf, "one_hot_"))
        self.assertTrue(hasattr(self.clf, "Ym"))

    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], len(self.classes_))

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)
        self.clf.fit(self.X, self.y)
        check_is_fitted(
            self.clf, attributes=["classes_", "one_hot_", "Ym", "freqs_", "le_"]
        )


if __name__ == "__main__":
    unittest.main()
