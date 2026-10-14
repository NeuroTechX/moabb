"""Synthetic regression test for the Hinss2021 trial layout (#663).

The released EEGLAB files hold 2 s epochs at 250 Hz, but the loader built the
continuous recording at a hard-coded 500 Hz and put each event one sample after
its epoch's start. Every trial therefore spanned two released epochs (4 s of
data labelled as 2 s, all frequencies doubled), and the last trial of each
session ran past the end and was dropped by MOABB and braindecode alike.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import Hinss2021


_SFREQ = 250.0
_N_TIMES = 500  # 2 s
_N_CH = 3
_COUNTS = {"RS": 2, "MATBeasy": 3, "MATBmed": 3, "MATBdiff": 3}
_LABELS = {"RS": "rs", "MATBeasy": "easy", "MATBmed": "medium", "MATBdiff": "diff"}


def _source_epochs(event_file, session):
    """Epoch ``k`` of a file holds a constant value unique to (session, file, k)."""
    offset = 1000 * session + 100 * list(_COUNTS).index(event_file)
    n = _COUNTS[event_file]
    data = np.ones((n, _N_CH, _N_TIMES)) * (offset + np.arange(n))[:, None, None]
    info = mne.create_info([f"E{i}" for i in range(_N_CH)], _SFREQ, "eeg")
    return mne.EpochsArray(data * 1e-6, info)


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setattr(Hinss2021, "data_path", lambda self, subject, **kw: ["/x"])
    monkeypatch.setattr(
        Hinss2021,
        "_get_epochs",
        lambda self, path, subject, session, event_file: _source_epochs(
            event_file, session
        ),
    )
    return Hinss2021()


def test_trials_are_released_epochs(dataset):
    sessions = dataset._get_single_subject_data(1)
    n_epochs = sum(_COUNTS.values())
    for session, runs in sessions.items():
        raw = runs["0"]
        assert raw.info["sfreq"] == _SFREQ
        events = mne.find_events(raw, shortest_event=0)
        assert len(events) == n_epochs
        tmin, tmax = dataset.interval
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None)
        data = epochs.get_data(picks="eeg") * 1e6
        # nothing dropped, one released epoch per trial
        assert data.shape == (n_epochs, _N_CH, _N_TIMES)
        expected = []
        labels = []
        for event_file, n in _COUNTS.items():
            src = _source_epochs(event_file, int(session)).get_data() * 1e6
            expected.append(src)
            labels += [dataset.event_id[_LABELS[event_file]]] * n
        np.testing.assert_allclose(data, np.concatenate(expected))
        np.testing.assert_array_equal(events[:, 2], labels)
