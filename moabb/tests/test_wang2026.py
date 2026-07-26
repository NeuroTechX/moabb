"""Tests for the Wang2026 dataset."""

import io
import zipfile

import mne
import numpy as np
import pytest
from scipy.io import savemat

import moabb.datasets as datasets
import moabb.datasets.wang2026 as wang2026
from moabb.datasets.wang2026 import (
    _CHANNELS,
    _EVENTS,
    _GROUPS,
    _SUBJECT_MAPPING,
    Wang2026,
)


_COHORTS = [
    ("joint_learning", "JointLearning", 64710999, range(1, 16)),
    ("bci2000_control", "BCI2000Control", 64710750, range(16, 24)),
    ("tactile_control", "TactileControl", 64710993, range(24, 32)),
    ("eegnet_control", "EEGNetControl", 64710990, range(32, 40)),
]


def test_wang2026_is_one_dataset_with_global_subject_ids():
    dataset = Wang2026()

    assert datasets.Wang2026 is Wang2026
    assert dataset.subject_list == list(range(1, 40))
    assert dataset.subject_mapping == _SUBJECT_MAPPING
    assert len(set(dataset.subject_mapping.values())) == 39
    assert dataset.metadata.participants.n_subjects == 39
    assert dataset.doi == "10.1184/R1/32293995.v1"

    for old_name in (
        "Wang2026JointLearning",
        "Wang2026BCI2000Control",
        "Wang2026TactileControl",
        "Wang2026EEGNetControl",
    ):
        assert not hasattr(datasets, old_name)


@pytest.mark.parametrize("group, archive, file_id, subjects", _COHORTS)
def test_group_filter_and_subject_mapping(group, archive, file_id, subjects):
    dataset = Wang2026(group=group)
    first = subjects.start

    assert dataset.subject_list == list(subjects)
    assert _GROUPS[group] == (archive, file_id, subjects)
    assert dataset.subject_mapping[first] == (archive, "S001")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"group": "unknown"}, "group must be one of"),
        ({"group": "joint_learning", "subjects": [16]}, "not in group"),
    ],
)
def test_invalid_selection(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Wang2026(**kwargs)


@pytest.mark.parametrize("group, archive, file_id, subjects", _COHORTS)
def test_data_path_extracts_and_reuses_one_subject(
    monkeypatch, tmp_path, group, archive, file_id, subjects
):
    subject = subjects.start
    member = f"{archive}/S001/S001_run0.mat"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zip_file:
        zip_file.writestr(member, b"mat-data")

    calls = []

    def remote_zip(url, **kwargs):
        calls.append((url, kwargs))
        return zipfile.ZipFile(io.BytesIO(payload.getvalue()))

    monkeypatch.setattr(wang2026, "RemoteZip", remote_zip)
    monkeypatch.setattr(wang2026.dl, "get_dataset_path", lambda code, path: str(tmp_path))

    dataset = Wang2026(group=group, subjects=[subject])
    expected = tmp_path / "MNE-wang2026-data" / archive / "S001" / "S001_run0.mat"
    assert dataset.data_path(subject) == [str(expected)]
    assert expected.read_bytes() == b"mat-data"
    assert calls[0][0] == wang2026._FIGSHARE_FILE.format(file_id)

    dataset.data_path(subject)
    assert len(calls) == 1
    dataset.data_path(subject, force_update=True)
    assert len(calls) == 2


def _write_run(path, layout, labels, lengths):
    rng = np.random.default_rng(42)
    if layout == "fixed":
        signal = rng.standard_normal((lengths[0], len(_CHANNELS), len(labels)))
        targets = np.repeat(np.asarray(labels)[None, :], lengths[0], axis=0)
        run_data = {"trialSignal": signal, "trialTargetClass": targets}
    else:
        signal = np.empty(len(labels), dtype=object)
        targets = np.empty(len(labels), dtype=object)
        for index, (label, length) in enumerate(zip(labels, lengths)):
            signal[index] = rng.standard_normal((length, len(_CHANNELS)))
            targets[index] = np.full(length, label)
        run_data = {"trialSignal": signal, "trialTargetCode": targets}
    savemat(path, {"runData": run_data})


@pytest.mark.parametrize(
    "layout, filename, labels, lengths, session, expected_events",
    [
        (
            "fixed",
            "S001_sess01_run01.mat",
            (0, 3),
            (20, 20),
            "1",
            [_EVENTS["left_hand"], _EVENTS["rest"]],
        ),
        (
            "variable",
            "S001_run0.mat",
            (1, 2),
            (20, 10),
            "0baseline",
            [_EVENTS["right_hand"]],
        ),
        (
            "variable",
            "S001_sess03_run01UD.mat",
            (1, 2),
            (20, 20),
            "3",
            [_EVENTS["hands"], _EVENTS["rest"]],
        ),
    ],
)
def test_read_supported_run_layouts(
    monkeypatch, tmp_path, layout, filename, labels, lengths, session, expected_events
):
    path = tmp_path / filename
    _write_run(path, layout, labels, lengths)
    monkeypatch.setattr(wang2026, "_NOMINAL_TRIAL_SAMPLES", 20)

    dataset = Wang2026(subjects=[1])
    monkeypatch.setattr(dataset, "data_path", lambda subject: [str(path)])
    sessions = dataset._get_single_subject_data(1)
    raw = sessions[session]["0"]
    events = mne.find_events(raw, shortest_event=1, verbose=False)

    assert events[:, -1].tolist() == expected_events
    assert raw.ch_names == [*_CHANNELS, "STI"]
    assert raw.info["subject_info"]["his_id"] == "JointLearning/S001"


def test_run_without_five_second_trials_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "S001_run0.mat"
    _write_run(path, "variable", (1, 2), (19, 19))
    monkeypatch.setattr(wang2026, "_NOMINAL_TRIAL_SAMPLES", 20)

    dataset = Wang2026(subjects=[1])
    monkeypatch.setattr(dataset, "data_path", lambda subject: [str(path)])
    with pytest.raises(FileNotFoundError, match="No five-second trials"):
        dataset._get_single_subject_data(1)


def test_data_path_rejects_invalid_subject_before_network(monkeypatch):
    dataset = Wang2026(group="joint_learning")
    monkeypatch.setattr(
        wang2026,
        "RemoteZip",
        lambda *args, **kwargs: pytest.fail("network access must not start"),
    )
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(16)
