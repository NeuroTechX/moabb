"""Synthetic regression test for the Zhou2020 channel layout (#1120).

The S-subjects (1-12) used to load as ``EEG1``..``EEG41`` without montage,
taking the first 41 columns of ``signal`` as EEG. Per the authors' files on
IEEE DataPort, columns 26-27 hold hEOG/vEOG, 28-29 the M1/M2 mastoids and the
15 extra S-cohort electrodes columns 30-44, so two EOG and two mastoid
channels were served as EEG and POz, PO4, Oz, F6 were dropped.

The two cohorts then returned different channel sets (41 vs 26 EEG), which
crashed every paradigm and evaluation stacking S- and A-subjects. The default
``cohort=None`` now returns the 26 shared channels for all subjects;
``cohort="S"``/``"A"`` select one group with its full montage.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import Zhou2020
from moabb.datasets.base import is_abbrev, is_camel_kebab_case
from moabb.datasets.zhou2020 import _26CH_NAMES, _S_EXTRA_NAMES
from moabb.paradigms import MotorImagery


_N_SAMPLES = 6000  # 12 s at 500 Hz


def _write_npz(path, n_cols, n_samples=_N_SAMPLES, mark=((1000, 769), (3500, 770))):
    # Column c holds the constant (c + 1) * 10 microvolts, so every loaded
    # channel can be traced back to its source column.
    signal = np.tile((np.arange(n_cols, dtype=float) + 1) * 10.0, (n_samples, 1))
    signal[:, -1] = 0.0  # trigger column
    mark = np.array(mark)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signal=signal, MarkOnSignal=mark, SampleRate=np.array([500]))


@pytest.fixture(scope="module")
def zhou_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("zhou2020_synth")
    _write_npz(root / "S01" / "session_1" / "run_01.npz", n_cols=46)
    _write_npz(root / "S13" / "session_1" / "run_01.npz", n_cols=29)
    return root


@pytest.fixture
def offline(zhou_root, monkeypatch):
    """Serve ``zhou_root`` and keep get_data() off the network."""
    monkeypatch.setattr(Zhou2020, "data_path", lambda self, subject, **kw: str(zhou_root))
    # get_data() would otherwise prefetch the NEMAR sourcedata tree (network)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(
        Zhou2020, "_prefetch_nemar_sourcedata", lambda self, *a, **kw: None
    )
    return zhou_root


@pytest.fixture
def dataset(offline):
    return Zhou2020(cohort="S", return_all_modalities=True)


def _names(raw, **picks):
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, **picks)]


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


@pytest.mark.parametrize("cohort", [None, "A"])
def test_a_subject_layout(offline, cohort):
    dataset = Zhou2020(cohort=cohort, return_all_modalities=True)
    raw = dataset._get_single_subject_data(13)["0"]["0"]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    assert eeg == _26CH_NAMES
    assert _column_of(raw, "C4") == 17
    assert _column_of(raw, "vEOG") == 27
    assert "M1" not in raw.ch_names


def test_events_and_default_eeg_only(offline):
    data = Zhou2020().get_data([1])
    raw = data[1]["0"]["0"]
    types = raw.get_channel_types()
    # EOG and mastoids are not loaded without return_all_modalities
    assert "eog" not in types and "misc" not in types
    # the default returns the 26 shared channels for S-subjects too
    assert _names(raw, eeg=True) == _26CH_NAMES
    assert _column_of(raw, "CP6") == 25
    events, _ = mne.events_from_annotations(raw)
    assert len(events) == 2

    raw = Zhou2020(cohort="S").get_data([1])[1]["0"]["0"]
    types = raw.get_channel_types()
    assert "eog" not in types and "misc" not in types
    assert _column_of(raw, "F6") == 44
    assert types.count("eeg") == 41


def test_default_same_channels_for_both_cohorts(offline):
    """S- and A-subjects get identical channels by default, also with EOG."""
    for kwargs in ({}, {"return_all_modalities": True}):
        data = Zhou2020(**kwargs).get_data([1, 13])
        s_raw, a_raw = data[1]["0"]["0"], data[13]["0"]["0"]
        assert s_raw.ch_names == a_raw.ch_names
        assert s_raw.get_channel_types() == a_raw.get_channel_types()
        assert "M1" not in s_raw.ch_names and "F6" not in s_raw.ch_names
        for name in ("F3", "Cz", "CP6"):
            assert _column_of(s_raw, name) == _column_of(a_raw, name)
    assert _names(s_raw, eeg=False, eog=True) == ["hEOG", "vEOG"]
    pos = s_raw.get_montage().get_positions()["ch_pos"]
    assert all(np.isfinite(pos[n]).all() for n in _26CH_NAMES)


@pytest.fixture(scope="module")
def epoch_root(tmp_path_factory):
    # 24 s runs with four 5 s trials, all inside the run for the [0, 5] s interval.
    root = tmp_path_factory.mktemp("zhou2020_epochs")
    mark = [(500, 769), (3200, 770), (5900, 769), (8600, 770)]
    for subject, n_cols in [(1, 46), (2, 46), (13, 29), (14, 29)]:
        path = root / f"S{subject:02d}" / "session_1" / "run_01.npz"
        _write_npz(path, n_cols, n_samples=12000, mark=mark)
    return root


@pytest.mark.parametrize(
    "cohort, subjects, n_chans",
    [
        (None, [1, 13], 26),
        (None, [2, 1, 14, 13], 26),
        ("S", [1, 2], 41),
        ("A", [13, 14], 26),
    ],
)
def test_paradigm_get_data_one_channel_set(
    epoch_root, monkeypatch, tmp_path, cohort, subjects, n_chans
):
    """MotorImagery stacks mixed S+A selections; develop crashed here (41 vs 26)."""
    monkeypatch.setattr(
        Zhou2020, "data_path", lambda self, subject, **kw: str(epoch_root)
    )
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(
        Zhou2020, "_prefetch_nemar_sourcedata", lambda self, *a, **kw: None
    )
    paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
    epochs, y, meta = paradigm.get_data(
        Zhou2020(cohort=cohort), subjects=subjects, return_epochs=True
    )
    expected = _26CH_NAMES + (_S_EXTRA_NAMES if cohort == "S" else [])
    assert epochs.ch_names == expected
    assert len(expected) == n_chans
    assert sorted(meta.subject.unique()) == sorted(subjects)
    assert len(y) == 4 * len(subjects)
    X, _, _ = paradigm.get_data(Zhou2020(cohort=cohort), subjects=subjects)
    assert X.shape[:2] == (4 * len(subjects), n_chans)


def test_cohort_codes_subjects_and_metadata():
    expected = {
        None: ("Zhou2020", list(range(1, 21)), {"eeg": 26, "eog": 2}, _26CH_NAMES),
        "S": (
            "Zhou-s2020",
            list(range(1, 13)),
            {"eeg": 41, "eog": 2, "misc": 2},
            _26CH_NAMES + _S_EXTRA_NAMES,
        ),
        "A": ("Zhou-a2020", list(range(13, 21)), {"eeg": 26, "eog": 2}, _26CH_NAMES),
    }
    codes = set()
    for cohort, (code, subjects, channel_types, sensors) in expected.items():
        ds = Zhou2020(cohort=cohort)
        assert ds.code == code
        assert is_camel_kebab_case(code) and is_abbrev("Zhou2020", code)
        assert ds.subject_list == subjects
        meta = ds.metadata
        assert meta is ds.METADATA or cohort is None
        assert meta.participants.n_subjects == len(subjects)
        assert meta.acquisition.channel_types == channel_types
        assert meta.acquisition.sensors == sensors
        assert meta.data_structure.n_trials == len(subjects) * 7 * 6 * 40
        assert meta.sessions_per_subject == ds.n_sessions
        assert meta.experiment.class_labels == list(ds.event_id)
        codes.add(code)
    assert len(codes) == 3
    # The default instance carries the class-level METADATA (catalog source)
    assert Zhou2020().METADATA is Zhou2020.METADATA
    assert Zhou2020.METADATA.participants.n_subjects == 20
    assert Zhou2020.METADATA.acquisition.channel_types == {"eeg": 26, "eog": 2}
    # Cohort variants do not touch the class attribute
    Zhou2020(cohort="S")
    assert Zhou2020.METADATA.participants.n_subjects == 20


def test_out_of_cohort_subject_raises(offline):
    with pytest.raises(ValueError, match="cohort"):
        Zhou2020(cohort="B")
    with pytest.raises(ValueError, match=r"cohort='S'.*\[13\]"):
        Zhou2020(cohort="S", subjects=[1, 13])
    with pytest.raises(ValueError, match=r"cohort='A'.*\[1\]"):
        Zhou2020(cohort="A").get_data([1])
    with pytest.raises(ValueError, match=r"cohort='A'.*\[12\]"):
        MotorImagery().get_data(Zhou2020(cohort="A"), subjects=[12])
    assert Zhou2020(subjects=[1, 13]).subject_list == [1, 13]


def test_cohorts_share_download_folder(tmp_path):
    subj = tmp_path / "MNE-zhou2020-data" / "S01" / "session_1"
    subj.mkdir(parents=True)
    np.savez(subj / "run_01.npz", signal=np.zeros((2, 2)))
    default = Zhou2020().data_path(1, path=str(tmp_path))
    assert Zhou2020(cohort="S").data_path(1, path=str(tmp_path)) == default
    with pytest.raises(ValueError, match="cohort"):
        Zhou2020(cohort="S").data_path(13, path=str(tmp_path))
