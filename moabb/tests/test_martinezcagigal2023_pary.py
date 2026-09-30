import mne
import numpy as np
import pytest

from moabb.datasets.martinezcagigal2023_pary_cvep import (
    DECLARED_SFREQ,
    MartinezCagigal2023Pary,
)


def _make_recording(sfreq, n_trials=2, n_cycles=3, duration_s=30.0):
    """Build a minimal synthetic BSON ``rec`` in train mode.

    The onsets are absolute timestamps compatible with ``times`` starting at
    10 s, so the ``- times[0]`` offset in the loader is exercised. Cycle
    indices repeat per trial and the last element is the maximum, so
    ``_trim_unfinished_trial`` is a no-op.
    """
    n_cycles_total = n_trials * n_cycles
    n_samples = int(duration_s * sfreq)
    n_channels = 4
    rng = np.random.default_rng(0)
    trial_of_cycle = [i // n_cycles for i in range(n_cycles_total)]
    command_per_trial = [0, 1]

    return {
        "cvepspellerdata": {
            "mode": "train",
            "fps_resolution": 120,
            "onsets": [11.0 + 2.0 * i for i in range(n_cycles_total)],
            "cycle_idx": [i % n_cycles for i in range(n_cycles_total)],
            "level_idx": [0] * n_cycles_total,
            "matrix_idx": [0] * n_cycles_total,
            "trial_idx": trial_of_cycle,
            "unit_idx": [0] * n_cycles_total,
            "command_idx": [command_per_trial[t] for t in trial_of_cycle],
            "commands_info": {
                0: {
                    "0": {"sequence": [0, 1, 0, 1], "label": "cmd0"},
                    "1": {"sequence": [1, 0, 1, 0], "label": "cmd1"},
                }
            },
        },
        "eeg": {
            "signal": rng.standard_normal((n_samples, n_channels)),
            "times": 10.0 + np.arange(n_samples) / sfreq,
            "fs": sfreq,
            "channel_set": {"l_cha": ["FPZ", "FZ", "CZ", "PZ"]},
        },
        "date": "2023-01-01 12:00:00",
        "subject_id": 1,
        # Real recordings carry an empty recording id; a non-empty string here
        # makes mne's info merge (EEG raw + stim raw) fail on `description`.
        "recording_id": "",
    }


@pytest.fixture
def dataset():
    return MartinezCagigal2023Pary()


@pytest.mark.parametrize(
    "input_sfreq", [256.0, 600.0], ids=["at-declared-rate", "off-rate-resampled"]
)
def test_convert_resamples_to_declared_rate(dataset, monkeypatch, input_sfreq):
    """Off-rate recordings (subject ``zdvm``, bases 2/3/5/7) load at 256 Hz."""
    rec = _make_recording(input_sfreq)
    monkeypatch.setattr(
        MartinezCagigal2023Pary, "_load_bson_recording", staticmethod(lambda path: rec)
    )

    raw = dataset._convert_to_mne_format("synthetic.bson")

    assert raw.info["sfreq"] == pytest.approx(DECLARED_SFREQ)
    # Stimulus channels survive the conversion with markers on the trial grid.
    assert "stim_trial" in raw.ch_names
    assert "stim_epoch" in raw.ch_names
    events = mne.find_events(raw, stim_channel="stim_trial", verbose=False)
    assert len(events) == 2
    np.testing.assert_array_equal(events[:, 2], [200, 201])


def test_convert_preserves_trial_onset_seconds(dataset, monkeypatch):
    """Resampling must not shift trial onsets: they stay in seconds."""
    rec = _make_recording(600.0)
    monkeypatch.setattr(
        MartinezCagigal2023Pary, "_load_bson_recording", staticmethod(lambda path: rec)
    )

    raw = dataset._convert_to_mne_format("synthetic.bson")

    # First trial starts 1 s after the recording start, whatever the rate.
    events = mne.find_events(raw, stim_channel="stim_trial", verbose=False)
    assert events[0, 0] == pytest.approx(DECLARED_SFREQ * 1.0, abs=1)
    assert events[1, 0] == pytest.approx(DECLARED_SFREQ * 7.0, abs=1)
