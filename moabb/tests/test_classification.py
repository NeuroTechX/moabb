import unittest

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import NotFittedError, check_is_fitted

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import SSVEP
from moabb.pipelines import (
    SSVEP_CCA,
    SSVEP_SSCOR,
    SSVEP_TDCA,
    SSVEP_TRCA,
    SSVEP_TRCA_R,
    SSVEP_eCCA,
    SSVEP_itCCA,
    SSVEP_MsetCCA,
)


@pytest.fixture(scope="module")
def ssvep_epochs_data():
    dataset = FakeDataset(
        n_sessions=1,
        n_runs=1,
        n_subjects=1,
        paradigm="ssvep",
        event_list=("13", "17", "21"),
        sfreq=256,
    )
    paradigm = SSVEP(n_classes=3)
    X, y, _ = paradigm.get_data(dataset, return_epochs=True)
    return X, np.asarray(y), paradigm.used_events(dataset)


def _build_ssvep_classifier(name):
    if name == "SSVEP_CCA":
        return SSVEP_CCA(n_harmonics=3)
    if name == "SSVEP_MsetCCA":
        return SSVEP_MsetCCA(n_filters=2)
    if name == "SSVEP_itCCA":
        return SSVEP_itCCA()
    if name == "SSVEP_eCCA":
        return SSVEP_eCCA(n_harmonics=3)
    raise ValueError(f"Unknown classifier {name}")


_CCA_STYLE_CLASSIFIERS = ("SSVEP_CCA", "SSVEP_MsetCCA", "SSVEP_itCCA", "SSVEP_eCCA")

_EXPECTED_FIT_ATTRIBUTES = {
    "SSVEP_CCA": ["classes_", "one_hot_", "slen_", "freqs_", "le_"],
    "SSVEP_MsetCCA": ["classes_", "one_hot_", "Ym", "freqs_", "le_"],
    "SSVEP_itCCA": ["classes_", "one_hot_", "templates_", "freqs_", "le_"],
    "SSVEP_eCCA": ["classes_", "one_hot_", "slen_", "freqs_", "le_", "templates_", "Yf"],
}


@pytest.mark.parametrize("clf_name", _CCA_STYLE_CLASSIFIERS)
def test_ssvep_cca_style_fit_attributes(clf_name, ssvep_epochs_data):
    X, y, _ = ssvep_epochs_data
    clf = _build_ssvep_classifier(clf_name).fit(X, y)
    check_is_fitted(clf, attributes=_EXPECTED_FIT_ATTRIBUTES[clf_name])


@pytest.mark.parametrize("clf_name", _CCA_STYLE_CLASSIFIERS)
def test_ssvep_cca_style_predict_contract(clf_name, ssvep_epochs_data):
    X, y, _ = ssvep_epochs_data
    clf = _build_ssvep_classifier(clf_name).fit(X, y)
    y_pred = np.asarray(clf.predict(X))
    assert len(y_pred) == len(X)
    assert set(y_pred).issubset(set(clf.classes_))


@pytest.mark.parametrize("clf_name", _CCA_STYLE_CLASSIFIERS)
def test_ssvep_cca_style_predict_proba_contract(clf_name, ssvep_epochs_data):
    X, y, freqs = ssvep_epochs_data
    clf = _build_ssvep_classifier(clf_name).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] == len(freqs)

    y_pred = np.asarray(clf.predict(X))
    y_from_proba = clf.classes_[np.argmax(proba, axis=1)]
    assert np.array_equal(y_pred, y_from_proba)


@pytest.mark.parametrize("clf_name", _CCA_STYLE_CLASSIFIERS)
def test_ssvep_cca_style_raises_when_not_fitted(clf_name, ssvep_epochs_data):
    X, _, _ = ssvep_epochs_data
    clf = _build_ssvep_classifier(clf_name)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


@pytest.mark.parametrize("clf_name", ("SSVEP_CCA", "SSVEP_eCCA"))
def test_ssvep_cca_frequency_mapping_required_for_encoded_labels(
    clf_name, ssvep_epochs_data
):
    X, y, _ = ssvep_epochs_data
    y_encoded = LabelEncoder().fit_transform(y)
    clf = _build_ssvep_classifier(clf_name)
    with pytest.raises(ValueError):
        clf.fit(X, y_encoded)


@pytest.mark.parametrize("clf_name", ("SSVEP_CCA", "SSVEP_eCCA"))
def test_ssvep_cca_frequency_mapping_override_for_encoded_labels(
    clf_name, ssvep_epochs_data
):
    X, y, _ = ssvep_epochs_data
    y_encoded = LabelEncoder().fit_transform(y)
    freq_map = {0: 13.0, 1: 17.0, 2: 21.0}
    if clf_name == "SSVEP_CCA":
        clf = SSVEP_CCA(n_harmonics=3, freq_map=freq_map)
    else:
        clf = SSVEP_eCCA(n_harmonics=3, freq_map=freq_map)
    clf.fit(X, y_encoded)
    inferred_freqs = set(np.round(list(clf.class_freqs_.values()), 8))
    assert inferred_freqs == {13.0, 17.0, 21.0}
    y_pred = np.asarray(clf.predict(X))
    assert set(y_pred).issubset(set(clf.classes_))


class TestSSVEP_TRCA(unittest.TestCase):
    def setUp(self):
        # Use moabb generated dataset for test with SSVEP frequencies as event names
        # and a higher sampling rate for proper filter design
        dataset = FakeDataset(
            n_sessions=1,
            n_runs=1,
            n_subjects=1,
            paradigm="ssvep",
            event_list=("13", "17", "21"),  # SSVEP frequencies in Hz
            sfreq=256,  # Higher sampling rate for filterbank
        )
        self.n_classes = 3
        paradigm = SSVEP(n_classes=self.n_classes)
        X, y, _ = paradigm.get_data(dataset, return_epochs=True)
        self.freqs = paradigm.used_events(dataset)
        self.n_fbands = 3
        self.X = X
        self.y = y
        self.clf = SSVEP_TRCA(n_fbands=self.n_fbands)

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
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

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], self.n_classes)

    def test_fit_predict_is_fitted(self):
        # Test that predict raises NotFittedError before fit
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)
        # Note: fit() testing is done in test_fit() which is xfail due to filterbank issues


class TestSSVEP_TRCA_R(unittest.TestCase):
    def setUp(self):
        dataset = FakeDataset(
            n_sessions=1,
            n_runs=1,
            n_subjects=1,
            paradigm="ssvep",
            event_list=("13", "17", "21"),
            sfreq=256,
        )
        self.n_classes = 3
        paradigm = SSVEP(n_classes=self.n_classes)
        X, y, _ = paradigm.get_data(dataset, return_epochs=True)
        self.freqs = paradigm.used_events(dataset)
        self.n_fbands = 3
        self.X = X
        self.y = y
        self.clf = SSVEP_TRCA_R(n_fbands=self.n_fbands, n_harmonics=3)

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_fit(self):
        self.clf.fit(self.X, self.y)
        self.assertTrue(hasattr(self.clf, "freqs_"))
        self.assertTrue(hasattr(self.clf, "peaks_"))
        self.assertTrue(hasattr(self.clf, "classes_"))
        self.assertTrue(hasattr(self.clf, "n_classes"))
        self.assertTrue(hasattr(self.clf, "le_"))
        self.assertTrue(hasattr(self.clf, "one_hot_"))
        self.assertTrue(hasattr(self.clf, "one_inv_"))
        self.assertTrue(hasattr(self.clf, "sfreq_"))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], self.n_classes)

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)


class TestSSVEP_SSCOR(unittest.TestCase):
    def setUp(self):
        dataset = FakeDataset(
            n_sessions=1,
            n_runs=1,
            n_subjects=1,
            paradigm="ssvep",
            event_list=("13", "17", "21"),
            sfreq=256,
        )
        self.n_classes = 3
        paradigm = SSVEP(n_classes=self.n_classes)
        X, y, _ = paradigm.get_data(dataset, return_epochs=True)
        self.freqs = paradigm.used_events(dataset)
        self.n_fbands = 3
        self.X = X
        self.y = y
        self.clf = SSVEP_SSCOR(n_fbands=self.n_fbands)

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_fit(self):
        self.clf.fit(self.X, self.y)
        self.assertTrue(hasattr(self.clf, "freqs_"))
        self.assertTrue(hasattr(self.clf, "peaks_"))
        self.assertTrue(hasattr(self.clf, "classes_"))
        self.assertTrue(hasattr(self.clf, "n_classes"))
        self.assertTrue(hasattr(self.clf, "le_"))
        self.assertTrue(hasattr(self.clf, "one_hot_"))
        self.assertTrue(hasattr(self.clf, "one_inv_"))
        self.assertTrue(hasattr(self.clf, "sfreq_"))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], self.n_classes)

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)


class TestSSVEP_TDCA(unittest.TestCase):
    def setUp(self):
        dataset = FakeDataset(
            n_sessions=1,
            n_runs=1,
            n_subjects=1,
            paradigm="ssvep",
            event_list=("13", "17", "21"),
            sfreq=256,
        )
        self.n_classes = 3
        paradigm = SSVEP(n_classes=self.n_classes)
        X, y, _ = paradigm.get_data(dataset, return_epochs=True)
        self.freqs = paradigm.used_events(dataset)
        self.n_fbands = 3
        self.X = X
        self.y = y
        self.clf = SSVEP_TDCA(n_fbands=self.n_fbands, n_components=1, n_delay=3)

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_fit(self):
        self.clf.fit(self.X, self.y)
        self.assertTrue(hasattr(self.clf, "freqs_"))
        self.assertTrue(hasattr(self.clf, "peaks_"))
        self.assertTrue(hasattr(self.clf, "classes_"))
        self.assertTrue(hasattr(self.clf, "n_classes"))
        self.assertTrue(hasattr(self.clf, "le_"))
        self.assertTrue(hasattr(self.clf, "one_hot_"))
        self.assertTrue(hasattr(self.clf, "one_inv_"))
        self.assertTrue(hasattr(self.clf, "sfreq_"))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict(self):
        self.clf.fit(self.X, self.y)
        y_pred = self.clf.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X))

    @pytest.mark.xfail(
        reason="Filterbank design parameters may fail with some frequency combinations"
    )
    def test_predict_proba(self):
        self.clf.fit(self.X, self.y)
        P = self.clf.predict_proba(self.X)
        self.assertEqual(P.shape[0], len(self.X))
        self.assertEqual(P.shape[1], self.n_classes)

    def test_fit_predict_is_fitted(self):
        self.assertRaises(NotFittedError, self.clf.predict, self.X)
        self.assertRaises(NotFittedError, self.clf.predict_proba, self.X)
