"""Offline regressions for the Kodera2023 recording setups (``cohort`` option).

The archive holds 2020-2021 recordings with 16 scalp + 2 auxiliary channels at
500 Hz and 2021-2023 recordings with 9 channels at 1000 Hz. MOABB paradigms
stack every subject into one array, so a dataset object must return one
channel set at one rate. No downloads: BrainVision reads are synthetic.
"""

from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import Kodera2023, kodera2023
from moabb.datasets.base import BaseDataset, is_abbrev, is_camel_kebab_case
from moabb.paradigms import LeftRightImagery


_FULL = list(kodera2023._FULL_EEG_CHANNELS) + ["17", "18"]
_SHARED = list(kodera2023._COMMON_EEG_CHANNELS)
# Subject 1 = ("01_12_2020", "1z"), 2020-2021 setup; subject 3 =
# ("02_09_2021", "S03"), 2021-2023 setup.
_RUNS = {
    1: ["01_12_2020/1z01122020lh1.vhdr", "01_12_2020/1z01122020rh1.vhdr"],
    3: [
        "02_09_2021/HR_02092021_03_x_leva.vhdr",
        "02_09_2021/HR_02092021_03_x_prava.vhdr",
    ],
}
_CUES = [2.0, 10.0, 18.0]


def _fake_run(path, *args, **kwargs):
    """A run of the setup its date folder belongs to, with S 1 cues."""
    folder = Path(path).parent.name
    if kodera2023._FOLDER_SETUP[folder] == "16ch":
        names, sfreq = _FULL, 500.0
    else:
        names, sfreq = _SHARED, 1000.0
    rng = np.random.default_rng(len(path))
    info = mne.create_info(names, sfreq, "eeg")
    raw = mne.io.RawArray(
        rng.standard_normal((len(names), int(30 * sfreq))) * 1e-5, info, verbose=False
    )
    raw.set_annotations(
        mne.Annotations(_CUES + [25.0], 0, ["Stimulus/S  1"] * 3 + ["Stimulus/S  2"])
    )
    return raw


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    monkeypatch.setattr(mne.io, "read_raw_brainvision", _fake_run)

    def make(**kw):
        ds = Kodera2023(**kw)
        ds.data_path = lambda subject, *a, **k: [f"/x/{p}" for p in _RUNS[subject]]
        return ds

    return make


def test_setups_match_archive_headers():
    # Subjects of each setup, as found in every header of the Zenodo archive.
    assert kodera2023._COHORTS["16ch"]["subjects"] == [
        1, 2, 11, 12, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26,
    ]  # fmt: skip
    assert kodera2023._COHORTS["9ch"]["subjects"] == [
        3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 22, 23, 27, 28, 29,
    ]  # fmt: skip
    assert len(kodera2023.SUBJECTS) == 29


def test_default_returns_shared_channels_at_500hz(offline):
    ds = offline()
    for subject in (1, 3):
        runs = ds._get_single_subject_data(subject)["0"]
        for raw in runs.values():
            assert raw.ch_names == _SHARED
            assert raw.get_channel_types() == ["eeg"] * 9
            assert raw.info["sfreq"] == 500.0
            assert raw.n_times == 30 * 500
            # Cue times survive the resampling; S 2 is dropped.
            np.testing.assert_allclose(raw.annotations.onset, _CUES)
        assert [r.annotations.description[0] for r in runs.values()] == [
            "left_hand",
            "right_hand",
        ]


def test_resampling_preserves_low_frequency_content(offline, monkeypatch):
    path = "/x/02_09_2021/HR_02092021_03_x_leva.vhdr"
    native = _fake_run(path)
    native._data[:] = 1e-5 * np.sin(2 * np.pi * 10 * native.times)
    monkeypatch.setattr(mne.io, "read_raw_brainvision", Mock(return_value=native))
    raw = offline()._read_run(path)
    assert raw.info["sfreq"] == 500.0
    expected = 1e-5 * np.sin(2 * np.pi * 10 * raw.times)
    np.testing.assert_allclose(
        raw.get_data()[:, 500:-500], np.tile(expected[500:-500], (9, 1)), atol=1e-7
    )


def test_paradigm_stacks_both_setups(offline):
    """The call that crashed: subjects of both setups in one paradigm call."""
    ds = offline()
    X, y, meta = LeftRightImagery().get_data(ds, subjects=[1, 3])
    assert X.shape == (12, 9, 2001)
    assert sorted(set(meta.subject)) == [1, 3]
    assert (y == "left_hand").sum() == (y == "right_hand").sum() == 6
    epochs, _, _ = LeftRightImagery().get_data(ds, subjects=[3, 1], return_epochs=True)
    assert epochs.ch_names == _SHARED
    assert epochs.info["sfreq"] == 500.0


def test_cohort_16ch_full_montage(offline):
    ds = offline(cohort="16ch")
    assert ds.code == "Kodera-16ch2023"
    assert ds.subject_list == kodera2023._COHORTS["16ch"]["subjects"]
    raw = ds._get_single_subject_data(1)["0"]["0"]
    assert raw.ch_names == list(kodera2023._FULL_EEG_CHANNELS)
    assert raw.info["sfreq"] == 500.0
    X, _, _ = LeftRightImagery().get_data(ds, subjects=[1])
    assert X.shape[1:] == (16, 2001)
    md = ds.metadata
    assert md.participants.n_subjects == 14
    assert md.acquisition.channel_types == {"eeg": 16}
    assert md.acquisition.sampling_rate == 500.0
    assert md.acquisition.sensors == list(kodera2023._FULL_EEG_CHANNELS)


def test_cohort_9ch_native_rate(offline):
    ds = offline(cohort="9ch")
    assert ds.code == "Kodera-9ch2023"
    raw = ds._get_single_subject_data(3)["0"]["0"]
    assert raw.ch_names == _SHARED
    assert raw.info["sfreq"] == 1000.0
    X, _, _ = LeftRightImagery().get_data(ds, subjects=[3])
    assert X.shape[1:] == (9, 4001)
    md = ds.metadata
    assert md.participants.n_subjects == 15
    assert md.acquisition.sampling_rate == 1000.0
    assert md.acquisition.channel_types == {"eeg": 9}


def test_default_metadata_untouched_by_cohorts():
    Kodera2023(cohort="16ch")
    md = Kodera2023.METADATA
    assert md.participants.n_subjects == 29
    assert md.acquisition.sampling_rate == 500.0
    assert md.acquisition.channel_types == {"eeg": 9}
    assert Kodera2023().metadata.participants.n_subjects == 29


def test_codes_follow_naming_rules():
    for cohort in (None, "16ch", "9ch"):
        code = kodera2023._COHORTS[cohort]["code"]
        assert is_camel_kebab_case(code)
        assert is_abbrev("Kodera2023", code)


def test_out_of_cohort_subjects_raise(offline, tmp_path):
    with pytest.raises(ValueError, match="other setup"):
        Kodera2023(cohort="16ch", subjects=[3])
    ds = offline(cohort="9ch")
    with pytest.raises(ValueError, match="other setup"):
        ds.get_data([1])
    with pytest.raises(ValueError, match="other setup"):
        LeftRightImagery().get_data(ds, subjects=[1])
    with pytest.raises(ValueError, match="other setup"):
        Kodera2023(cohort="9ch").data_path(1, path=tmp_path)
    with pytest.raises(ValueError, match="cohort must be"):
        Kodera2023(cohort="A")


def test_unexpected_rate_fails_loudly(monkeypatch):
    raw = _fake_run("/x/02_09_2021/HR_02092021_03_x_leva.vhdr").resample(250)
    monkeypatch.setattr(mne.io, "read_raw_brainvision", Mock(return_value=raw))
    with pytest.raises(ValueError, match="250 Hz"):
        Kodera2023()._read_run("/x/02_09_2021/HR_02092021_03_x_leva.vhdr")


def test_download_folder_shared_by_cohorts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        kodera2023,
        "download_and_extract_subject_zip",
        lambda url, sign, *a, **k: calls.append(sign),
    )
    for kw, subject in (({}, 1), ({"cohort": "16ch"}, 1), ({"cohort": "9ch"}, 3)):
        with pytest.raises(FileNotFoundError, match=r"MNE-kodera2023-data"):
            Kodera2023(**kw).data_path(subject, path=tmp_path)
    assert calls == ["Kodera2023"] * 3
