"""Synthetic regression test for the Zhou2020 channel layout (#1120).

The S-subjects (1-12) used to load as ``EEG1``..``EEG41`` without montage,
taking the first 41 columns of ``signal`` as EEG. Per the authors' files on
IEEE DataPort, columns 26-27 hold hEOG/vEOG, 28-29 the M1/M2 mastoids and the
15 extra S-cohort electrodes columns 30-44, so two EOG and two mastoid
channels were served as EEG and POz, PO4, Oz, F6 were dropped.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import Zhou2020
from moabb.datasets.zhou2020 import _26CH_NAMES, _S_EXTRA_NAMES


_N_SAMPLES = 6000  # 12 s at 500 Hz


def _write_npz(path, n_cols):
    # Column c holds the constant (c + 1) * 10 microvolts, so every loaded
    # channel can be traced back to its source column.
    signal = np.tile((np.arange(n_cols, dtype=float) + 1) * 10.0, (_N_SAMPLES, 1))
    signal[:, -1] = 0.0  # trigger column
    mark = np.array([[1000, 769], [3500, 770]])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signal=signal, MarkOnSignal=mark, SampleRate=np.array([500]))


@pytest.fixture(scope="module")
def zhou_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("zhou2020_synth")
    _write_npz(root / "S01" / "session_1" / "run_01.npz", n_cols=46)
    _write_npz(root / "S13" / "session_1" / "run_01.npz", n_cols=29)
    return root


@pytest.fixture
def dataset(zhou_root, monkeypatch):
    monkeypatch.setattr(Zhou2020, "data_path", lambda self, subject, **kw: str(zhou_root))
    return Zhou2020(return_all_modalities=True)


def _column_of(raw, name):
    """Recover the source column from the synthetic constant value."""
    value = raw.get_data(picks=[name])[0, 0] * 1e6
    return int(round(value / 10.0)) - 1


def test_s_subject_layout(dataset):
    raw = dataset._get_single_subject_data(1)["0"]["0"]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=False, eog=True)]
    misc = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=False, misc=True)]
    assert eeg == _26CH_NAMES + _S_EXTRA_NAMES
    assert len(eeg) == 41
    assert eog == ["hEOG", "vEOG"]
    assert misc == ["M1", "M2"]
    assert not any(n.startswith("EEG") for n in raw.ch_names)
    expected = {"F3": 0, "C3": 13, "Cz": 15, "C4": 17, "CP6": 25, "hEOG": 26}
    expected.update({"vEOG": 27, "M1": 28, "M2": 29, "F5": 30, "AF3": 31})
    expected.update({"Pz": 36, "PO4": 42, "Oz": 43, "F6": 44})
    for name, col in expected.items():
        assert _column_of(raw, name) == col, name
    pos = raw.get_montage().get_positions()["ch_pos"]
    assert all(np.isfinite(pos[n]).all() for n in eeg)


def test_a_subject_layout(dataset):
    raw = dataset._get_single_subject_data(13)["0"]["0"]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    assert eeg == _26CH_NAMES
    assert _column_of(raw, "C4") == 17
    assert _column_of(raw, "vEOG") == 27
    assert "M1" not in raw.ch_names


def test_events_and_default_eeg_only(zhou_root, monkeypatch):
    monkeypatch.setattr(Zhou2020, "data_path", lambda self, subject, **kw: str(zhou_root))
    # get_data() would otherwise prefetch the NEMAR sourcedata tree (network)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(
        Zhou2020, "_prefetch_nemar_sourcedata", lambda self, *a, **kw: None
    )
    data = Zhou2020().get_data([1])
    raw = data[1]["0"]["0"]
    types = raw.get_channel_types()
    # EOG and mastoids are not loaded without return_all_modalities
    assert "eog" not in types and "misc" not in types
    assert _column_of(raw, "F6") == 44
    assert types.count("eeg") == 41
    events, _ = mne.events_from_annotations(raw)
    assert len(events) == 2
