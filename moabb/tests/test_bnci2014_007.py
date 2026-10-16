"""Offline tests for the BNCI2014_007 (Affective Pacman) loader."""

import numpy as np
import pytest
from scipy.io import loadmat

from moabb.datasets import BNCI2014_007
from moabb.datasets.bnci.bnci_2014_007 import _CH_NAMES_RECORDED, SFREQ
from moabb.paradigms import MotorImagery


SFREQ_INT = int(SFREQ)


def _write_mat(path, *, markers_by_block, sfreq=SFREQ_INT, n_pad_start=128):
    """Write a synthetic recording with the SXX.mat layout.

    markers_by_block maps block id -> (condition, {sample_in_block: marker}).
    Blocks are separated by 64-sample unlabelled padding (block=-1).
    """
    block_len = 2 * sfreq
    gap = 64
    total = n_pad_start + len(markers_by_block) * (block_len + gap)
    X = (10.0 * np.ones((47, total))).astype(np.float32)  # uV
    Y = np.zeros(total)
    info = np.zeros((6, total))
    info[0] = np.arange(total) / sfreq  # time row
    info[1] = -1
    info[2] = -1
    info[3:] = -1

    start = n_pad_start
    for block_id, (condition, markers) in sorted(markers_by_block.items()):
        span = slice(start, start + block_len)
        info[1, span] = block_id
        info[2, span] = condition
        for sample, marker in markers.items():
            Y[start + sample] = marker
        start += block_len + gap

    ch_names = list(_CH_NAMES_RECORDED)
    mat = {
        "X": X,
        "Y": Y.reshape(1, -1),
        "I": info,
        "chann": np.array([ch_names], dtype=object),
        "id_lab": np.array(
            [
                [
                    "time",
                    "block",
                    "frustration-condition",
                    "valence",
                    "arousal",
                    "dominance",
                ]
            ],
            dtype=object,
        ),
        "markers": np.array([[1, 2], ["left", "right"]], dtype=object),
    }
    from scipy.io import savemat

    savemat(str(path), mat)
    return path


@pytest.fixture
def mat_file(tmp_path):
    """Three blocks: normal_0, frustration_0, frustration_1."""
    return _write_mat(
        tmp_path / "S00.mat",
        markers_by_block={
            0: (0, {128: 1, 180: 12}),  # accepted left press, avatar died
            1: (1, {128: 2, 150: 3}),  # accepted right press, ignored left press
            2: (1, {100: 5, 128: 2, 150: 4}),  # freeze, accepted right, ignored right
        },
    )


@pytest.fixture
def dataset(mat_file, monkeypatch):
    monkeypatch.setattr(
        "moabb.datasets.download.data_dl",
        lambda url, sign, path=None, force_update=False, verbose=None: str(mat_file),
    )
    return BNCI2014_007()


def test_metadata():
    ds = BNCI2014_007()
    assert ds.code == "BNCI2014-007"
    assert ds.paradigm == "imagery"
    assert ds.interval == [-0.5, 0.5]
    assert ds.nemar_id is None
    assert ds.METADATA.acquisition.sampling_rate == 128.0
    assert ds.METADATA.documentation.license == "CC-BY-NC-ND-4.0"


def test_subject_mapping_skips_withheld_files(mat_file, monkeypatch):
    """Public subjects 1-10 map to files S00-S11 minus the withheld S3/S8."""
    urls = {}

    def fake_dl(url, sign, path=None, force_update=False, verbose=None):
        urls.clear()
        urls["url"] = url
        return str(mat_file)

    monkeypatch.setattr("moabb.datasets.download.data_dl", fake_dl)
    ds = BNCI2014_007()
    ds.data_path(1)
    assert urls["url"].endswith("/007-2014/S00.mat")
    ds.data_path(4)
    assert urls["url"].endswith("/007-2014/S04.mat")  # S03 withheld
    ds.data_path(8)
    assert urls["url"].endswith("/007-2014/S09.mat")  # S08 withheld


def test_invalid_subject_raises(dataset):
    with pytest.raises(ValueError, match="between 1 and 10"):
        dataset.data_path(11)


def test_runs_and_annotations(dataset):
    sessions = dataset._get_single_subject_data(1)
    assert list(sessions.keys()) == ["0"]
    runs = sessions["0"]
    assert list(runs.keys()) == ["0normal0", "1frustration0", "2frustration1"]

    normal = runs["0normal0"]
    assert len(normal.ch_names) == 32
    assert normal.info["sfreq"] == SFREQ
    desc = list(normal.annotations.description)
    assert desc == ["left_hand", "avatar_died"]

    frustration = runs["2frustration1"]
    desc = list(frustration.annotations.description)
    assert desc == ["screen_freeze", "right_hand", "ignored_right_hand"]

    # onsets are expressed in seconds from the block start
    assert normal.annotations.onset[0] == pytest.approx(1.0)


def test_units_are_converted_to_volts(dataset):
    raw = dataset._get_single_subject_data(1)["0"]["0normal0"]
    assert raw.get_data().mean() == pytest.approx(10e-6, rel=1e-5)


def test_return_all_modalities(mat_file, monkeypatch):
    monkeypatch.setattr(
        "moabb.datasets.download.data_dl",
        lambda url, sign, path=None, force_update=False, verbose=None: str(mat_file),
    )
    ds = BNCI2014_007(return_all_modalities=True)
    raw = ds._get_single_subject_data(1)["0"]["0normal0"]
    assert raw.get_channel_types().count("eeg") == 32
    assert raw.get_channel_types().count("eog") == 4
    assert raw.get_channel_types().count("emg") == 4
    assert raw.get_channel_types().count("misc") == 4
    assert not {"GSR2", "Erg1", "Erg2"}.intersection(raw.ch_names)


def test_unused_noise_channels_dropped_by_default(dataset):
    raw = dataset._get_single_subject_data(1)["0"]["0normal0"]
    assert not {"GSR2", "Erg1", "Erg2"}.intersection(raw.ch_names)


def test_mixed_condition_block_is_skipped(tmp_path, monkeypatch):
    mat_file = _write_mat(
        tmp_path / "S00.mat",
        markers_by_block={
            0: (0, {128: 1}),
            1: (0, {128: 2}),  # becomes mixed below
            2: (1, {128: 2}),
        },
    )
    # manually corrupt block 1 to hold both conditions
    mat = loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
    info = mat["I"]
    span = np.flatnonzero(info[1] == 1)
    info[2, span[len(span) // 2 :]] = 1
    from scipy.io import savemat

    savemat(str(mat_file), mat)

    monkeypatch.setattr(
        "moabb.datasets.download.data_dl",
        lambda url, sign, path=None, force_update=False, verbose=None: str(mat_file),
    )
    ds = BNCI2014_007()
    runs = ds._get_single_subject_data(1)["0"]
    assert list(runs.keys()) == ["0normal0", "2frustration0"]


def test_paradigm_end_to_end(dataset):
    paradigm = MotorImagery()
    assert paradigm.is_valid(dataset)
    X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
    assert set(labels) == {"left_hand", "right_hand"}
    assert list(labels).count("left_hand") == 1
    assert list(labels).count("right_hand") == 2
    assert len(metadata) == 3
    assert set(metadata["run"]) == {"0normal0", "1frustration0", "2frustration1"}
