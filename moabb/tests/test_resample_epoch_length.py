"""Resampled epochs have the same number of samples whatever the native rate.

MNE gives ``round(n * new / old)`` samples when it resamples ``n`` samples, so
a 0-3 s epoch has 601 samples at 200 Hz when it was recorded at 160 Hz
(481 samples) but 602 when it was recorded at 128 Hz (385 samples), the last
one at 3.005 s, and subjects recorded at different native rates could not be
stacked. No downloads: the data are synthetic.
"""

import logging
from operator import methodcaller

import numpy as np
import pytest
from mne.decoding import Vectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb.datasets.fake import FakeDataset
from moabb.datasets.preprocessing import NamedFunctionTransformer
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import FilterBankLeftRightImagery, LeftRightImagery
from moabb.paradigms import base as paradigm_base


_EVENTS = ("left_hand", "right_hand")


class MixedRateFakeDataset(FakeDataset):
    """FakeDataset whose subjects were recorded at different sampling rates."""

    def __init__(self, rates, **kwargs):
        self.rates = list(rates)
        kwargs.setdefault("n_sessions", 1)
        kwargs.setdefault("n_runs", 1)
        kwargs.setdefault("n_events", 12)
        kwargs.setdefault("duration", 60)
        kwargs.setdefault("code", "MixedRateFakeDataset")
        super().__init__(
            event_list=_EVENTS,
            paradigm="imagery",
            n_subjects=len(self.rates),
            seed=7,
            **kwargs,
        )

    def _get_single_subject_data(self, subject):
        self.sfreq = self.rates[subject - 1]
        return super()._get_single_subject_data(subject)


def _resample_step_before_fix(sfreq, **kwargs):
    """The resample step before this fix: MNE resampling, nothing else."""
    return NamedFunctionTransformer(
        func=methodcaller("resample", sfreq=sfreq, verbose=False),
        display_name=f"Resample ({sfreq} Hz)",
    )


def _get_data_before_fix(paradigm, dataset, monkeypatch, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(paradigm_base, "get_resample_pipeline", _resample_step_before_fix)
        return paradigm.get_data(dataset, **kwargs)


# (native rates of the subjects, resample, samples of every 0-3 s epoch)
MIXED = [
    # 385 samples at 128 Hz -> 602 at 200 Hz, 481 at 160 Hz -> 601
    pytest.param((128, 160), 200, 601, id="128+160Hz-to-200Hz"),
    # 481 samples at 160 Hz -> 385 at 128 Hz, 751 at 250 Hz -> 385
    pytest.param((160, 250), 128, 385, id="160+250Hz-to-128Hz"),
    # 751 samples at 250 Hz -> 385 at 128 Hz, 769 at 256 Hz -> 384
    pytest.param((250, 256), 128, 384, id="250+256Hz-to-128Hz"),
    # 385 at 128 Hz -> 602 at 200 Hz, 3001 at 1000 Hz -> 600
    pytest.param((128, 1000), 200, 600, id="128+1000Hz-to-200Hz"),
]


@pytest.mark.parametrize("rates, resample, n_times", MIXED)
def test_mixed_native_rates_stack(rates, resample, n_times):
    dataset = MixedRateFakeDataset(rates)
    paradigm = LeftRightImagery(resample=resample)
    X, y, meta = paradigm.get_data(dataset, subjects=[1, 2])
    assert X.shape[-1] == n_times
    assert sorted(set(meta.subject)) == [1, 2]
    assert len(X) == len(y) == len(meta)

    epochs, _, _ = paradigm.get_data(dataset, subjects=[1, 2], return_epochs=True)
    assert epochs.info["sfreq"] == resample
    assert len(epochs.times) == n_times
    assert epochs.times[0] == 0.0
    assert epochs.times[-1] <= 3.0
    np.testing.assert_allclose(epochs.times, np.arange(n_times) / resample)


@pytest.mark.parametrize("rates, resample, n_times", MIXED)
def test_mixed_native_rates_keep_the_samples_of_each_subject(
    rates, resample, n_times, monkeypatch
):
    """Stacking only drops samples at the end of the window; the others are the
    samples MNE's resampling gives for the subject."""
    dataset = MixedRateFakeDataset(rates)
    paradigm = LeftRightImagery(resample=resample)
    X, _, meta = paradigm.get_data(dataset, subjects=[1, 2])
    for subject in (1, 2):
        X_before, _, _ = _get_data_before_fix(
            paradigm, dataset, monkeypatch, subjects=[subject]
        )
        assert X_before.shape[-1] >= n_times
        np.testing.assert_array_equal(
            X[(meta.subject == subject).to_numpy()], X_before[..., :n_times]
        )


def test_shortest_length_is_logged(caplog):
    dataset = MixedRateFakeDataset((250, 256))
    paradigm = LeftRightImagery(resample=128)
    X1, _, _ = paradigm.get_data(dataset, subjects=[1])
    X2, _, _ = paradigm.get_data(dataset, subjects=[2])
    # 0-3 s at 128 Hz: 385 samples, but 769 samples at 256 Hz resample to 384
    assert (X1.shape[-1], X2.shape[-1]) == (385, 384)
    with caplog.at_level(logging.WARNING, logger="moabb.paradigms.base"):
        X, _, _ = paradigm.get_data(dataset, subjects=[1, 2])
    assert X.shape[-1] == 384
    assert "keeping the first 384 samples" in caplog.text


@pytest.mark.parametrize(
    "rate, resample, n_times",
    [
        pytest.param(160, 200, 601, id="160Hz-to-200Hz"),
        pytest.param(250, 128, 385, id="250Hz-to-128Hz"),
        pytest.param(512, 128, 384, id="512Hz-to-128Hz"),
        pytest.param(128, 128, 385, id="128Hz-unchanged-rate"),
    ],
)
@pytest.mark.parametrize("return_epochs", [False, True])
def test_single_rate_output_is_unchanged(
    rate, resample, n_times, return_epochs, monkeypatch
):
    """Single-rate data whose resampled epochs stay inside the window are
    bit-identical to the output before the fix."""
    dataset = MixedRateFakeDataset((rate, rate))
    paradigm = LeftRightImagery(resample=resample)
    kwargs = {"subjects": [1, 2], "return_epochs": return_epochs}
    after, y_after, meta_after = paradigm.get_data(dataset, **kwargs)
    before, y_before, meta_before = _get_data_before_fix(
        paradigm, dataset, monkeypatch, **kwargs
    )
    if return_epochs:
        np.testing.assert_array_equal(after.times, before.times)
        after, before = after.get_data(), before.get_data()
    assert after.shape[-1] == n_times
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(y_after, y_before)
    assert meta_after.equals(meta_before)


def test_single_rate_upsampling_drops_only_samples_after_tmax(monkeypatch):
    """128 Hz to 200 Hz: MNE gives 602 samples for 0-3 s, the last at 3.005 s."""
    dataset = MixedRateFakeDataset((128, 128))
    paradigm = LeftRightImagery(resample=200)
    X, _, _ = paradigm.get_data(dataset, subjects=[1, 2])
    X_before, _, _ = _get_data_before_fix(paradigm, dataset, monkeypatch, subjects=[1, 2])
    assert X_before.shape[-1] == 602
    assert X.shape[-1] == 601
    np.testing.assert_array_equal(X, X_before[..., :601])


def test_window_off_the_target_sample_grid():
    """0.0625-3.0625 s: the first sample lies half a 200 Hz sample off the
    multiples of 5 ms, where cropping by time is ambiguous."""
    dataset = MixedRateFakeDataset((128, 160))
    paradigm = LeftRightImagery(tmin=0.0625, tmax=3.0625, resample=200)
    X, _, _ = paradigm.get_data(dataset, subjects=[1, 2])
    assert X.shape[-1] == 601
    epochs, _, _ = paradigm.get_data(dataset, subjects=[1, 2], return_epochs=True)
    assert len(epochs.times) == 601
    # mne.concatenate_epochs puts the first sample on the 200 Hz grid (0.06 s)
    assert abs(epochs.times[0] - 0.0625) <= 0.5 / 200 + 1e-9
    np.testing.assert_allclose(np.diff(epochs.times), 1 / 200)
    assert epochs.times[-1] <= 3.0625


def test_baseline_outside_the_window():
    """The window is cropped from the baseline window before resampling."""
    dataset = MixedRateFakeDataset((128, 160))
    paradigm = LeftRightImagery(tmin=0.5, tmax=3.0, baseline=(0.0, 0.5), resample=200)
    epochs, _, _ = paradigm.get_data(dataset, subjects=[1, 2], return_epochs=True)
    assert len(epochs.times) == 501
    assert epochs.times[0] == 0.5
    assert epochs.times[-1] == 3.0


def test_filter_bank_bands_have_the_same_length():
    dataset = MixedRateFakeDataset((128, 160))
    paradigm = FilterBankLeftRightImagery(filters=[[8, 12], [12, 16]], resample=200)
    X, _, meta = paradigm.get_data(dataset, subjects=[1, 2])
    assert X.shape[2:] == (601, 2)
    assert sorted(set(meta.subject)) == [1, 2]


def test_cross_subject_evaluation_with_mixed_native_rates(tmp_path):
    """Leave-one-subject-out needs every subject in one array."""
    dataset = MixedRateFakeDataset((128, 160, 160))
    evaluation = CrossSubjectEvaluation(
        paradigm=LeftRightImagery(resample=200),
        datasets=[dataset],
        hdf5_path=str(tmp_path),
        overwrite=True,
    )
    results = evaluation.process({"Vec+LDA": make_pipeline(Vectorizer(), LDA())})
    assert sorted(results.subject.astype(str).unique()) == ["1", "2", "3"]
    assert np.isfinite(results.score.to_numpy(dtype=float)).all()


def test_caches_of_the_previous_resampling_are_not_reused(tmp_path, monkeypatch):
    """The resample step is part of the cache key: 602-sample epochs cached
    before the fix are recomputed, not read back."""
    dataset = MixedRateFakeDataset((128,))
    paradigm = LeftRightImagery(resample=200)
    cache = {
        "use": True,
        "path": str(tmp_path),
        "save_raw": False,
        "save_epochs": True,
        "save_array": True,
        "overwrite_raw": False,
        "overwrite_epochs": False,
        "overwrite_array": False,
    }
    X_before, _, _ = _get_data_before_fix(
        paradigm, dataset, monkeypatch, subjects=[1], cache_config=cache
    )
    assert X_before.shape[-1] == 602
    X, _, _ = paradigm.get_data(dataset, subjects=[1], cache_config=cache)
    assert X.shape[-1] == 601
    np.testing.assert_array_equal(X, X_before[..., :601])
    epochs, _, _ = paradigm.get_data(
        dataset, subjects=[1], cache_config=cache, return_epochs=True
    )
    assert len(epochs.times) == 601
