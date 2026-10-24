"""Synthetic regression tests for Han2024Fatigue label handling (#1184).

The fatigue MAT blocks do not store trials in the documented target order, so
the loader must warn about unreliable session-'1' labels while keeping the
session loadable, and must leave the training-session event order untouched.
"""

import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets.ssvep_han2024 import Han2024Fatigue


mne.set_log_level("ERROR")

_N_TRAIN_BLOCKS = 2
_N_FATIGUE_BLOCKS = 2


@pytest.fixture(scope="module")
def mat_paths(tmp_path_factory):
    """Write tiny synthetic per-condition MAT files in the real layout.

    The arrays keep the documented (16, 64, 3000, n_blocks) shape so the
    loader's reshape path is exercised unmodified; contents are noise since
    only event codes and warning behaviour are asserted.
    """
    root = tmp_path_factory.mktemp("han2024_synth")
    rng = np.random.default_rng(0)
    paths = {}
    for dir_name, _, phase in [
        ("low_frequency_train_data", "low", "train"),
        ("low_frequency_fatigue_data", "low", "fatigue"),
        ("high_frequency_train_data", "high", "train"),
        ("high_frequency_fatigue_data", "high", "fatigue"),
    ]:
        n_blocks = _N_TRAIN_BLOCKS if phase == "train" else _N_FATIGUE_BLOCKS
        d = root / dir_name
        d.mkdir()
        data = rng.standard_normal((16, 64, 3000, n_blocks)).astype(np.float32)
        savemat(d / "S1.mat", {"data": data})
        paths[dir_name] = str(d / "S1.mat")
    return paths


@pytest.fixture
def patch_paths(monkeypatch, mat_paths):
    monkeypatch.setattr(
        Han2024Fatigue, "data_path", lambda self, subject, **kwargs: mat_paths
    )


def test_fatigue_session_load_warns(patch_paths):
    """Loading emits the unreliable-fatigue-labels UserWarning (#1184)."""
    dataset = Han2024Fatigue()
    with pytest.warns(UserWarning, match="issue #1184"):
        sessions = dataset._get_single_subject_data(1)
    assert set(sessions) == {"0", "1"}


def test_training_session_event_order_is_documented(patch_paths):
    """Session-'0' stim codes stay low-band 1..16 then high-band 17..32,
    block-major, exactly as documented (regression guard for #1184)."""
    dataset = Han2024Fatigue()
    with pytest.warns(UserWarning):
        sessions = dataset._get_single_subject_data(1)
    events = mne.find_events(sessions["0"]["0"], stim_channel="STI", shortest_event=1)
    codes = events[:, 2]
    low = np.tile(np.arange(1, 17), _N_TRAIN_BLOCKS)
    high = np.tile(np.arange(17, 33), _N_TRAIN_BLOCKS)
    np.testing.assert_array_equal(codes, np.concatenate([low, high]))


def test_both_sessions_carry_all_trials(patch_paths):
    """Both sessions stay fully loadable (train and fatigue block counts)."""
    dataset = Han2024Fatigue()
    with pytest.warns(UserWarning):
        sessions = dataset._get_single_subject_data(1)
    for sess, n_blocks in (("0", _N_TRAIN_BLOCKS), ("1", _N_FATIGUE_BLOCKS)):
        events = mne.find_events(
            sessions[sess]["0"], stim_channel="STI", shortest_event=1
        )
        assert len(events) == n_blocks * 32  # 16 low-band + 16 high-band per block
