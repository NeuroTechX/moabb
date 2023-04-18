import unittest

import numpy as np
from sklearn.utils.validation import NotFittedError, check_is_fitted

from moabb.pipelines import SSVEP_MsetCCA


class TestSSVEP_MsetCCA(unittest.TestCase):
    def setUp(self):
        self.freqs = {1: 6.66, 2: 7.5, 3: 8.57}  # Example frequency dictionary
        self.n_filters = 2
        self.n_components = 3
        self.X = np.random.rand(10, 8, 100)  # Example input data
        self.y = np.random.choice(list(self.freqs.keys()), size=10)  # Example labels
        self.clf = SSVEP_MsetCCA(
            freqs=self.freqs, n_filters=self.n_filters, n_components=self.n_components
        )

    def test_fit(self):
        self.clf.fit(self.X, self.y)
        self.assertTrue(hasattr(self.clf, "classes_"))
        self.assertTrue(hasattr(self.clf, "one_hot"))
        self.assertTrue(hasattr(self.clf, "Ym"))

    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], len(self.freqs))

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)
        self.clf.fit(self.X, self.y)
        check_is_fitted(self.clf, attributes=["classes_", "one_hot", "Ym"])


if __name__ == "__main__":
    unittest.main()
