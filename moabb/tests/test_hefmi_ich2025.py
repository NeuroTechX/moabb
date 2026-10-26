"""HefmiIch2025 subjects and sessions follow the depositor's folders (offline).

The figshare file names (``1_epo.mat``, ``2_epo.mat``, ``3_epo.mat``) carry no
subject; MOABB ships each file's folder, taken from the ``folder_structure`` of
the figshare record (article 28955456, version 4). These tests mock the
downloads with synthetic data.
"""

import hashlib
import os
import socket
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import savemat

import moabb.datasets.download as dl
from moabb.datasets import hefmi_ich2025
from moabb.datasets.hefmi_ich2025 import _EPOCH_FILES, HefmiIch2025, _subject


@pytest.fixture
def mne_data(tmp_path, monkeypatch):
    fake_home = tmp_path / "mne-home"
    fake_home.mkdir()
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(fake_home))
    for key in tuple(os.environ):
        if key.startswith("MNE_DATASETS_"):
            monkeypatch.delenv(key)
    root = tmp_path / "mne_data"
    root.mkdir()
    monkeypatch.setenv("MNE_DATA", str(root))
    return root


def _forbid_connect(self, address):
    raise AssertionError(f"network touched: {address}")


def _subjects_sessions(rows):
    """{subject: sorted session keys} of a file map."""
    sessions = {}
    for _id, group, folder_subject, session, _size, _md5 in rows:
        subject = _subject(group, folder_subject)
        sessions.setdefault(subject, []).append(str(session - 1))
    return {subject: sorted(keys) for subject, keys in sessions.items()}


@pytest.mark.parametrize(
    ("file_id", "folder", "subject", "session"),
    [
        (54452000, "Patients/EEG/subject9/2_epo.mat", 26, "1"),
        (54452621, "Normals/EEG/subject1/1_epo.mat", 1, "0"),
        (54452189, "Normals/EEG/subject7/3_epo.mat", 7, "2"),
        (54452075, "Patients/EEG/subject8/1_epo.mat", 25, "0"),
    ],
)
def test_known_files(file_id, folder, subject, session):
    rows = {row[0]: row for row in _EPOCH_FILES}
    _id, group, folder_subject, k, _size, _md5 = rows[file_id]
    assert f"{group}/EEG/subject{folder_subject}/{k}_epo.mat" == folder
    assert _subject(group, folder_subject) == subject
    assert str(k - 1) == session


def test_subject_and_session_counts():
    sessions = _subjects_sessions(_EPOCH_FILES)
    assert sorted(sessions) == list(range(1, 38)) == HefmiIch2025().subject_list
    assert sum(len(keys) for keys in sessions.values()) == 98
    # Sessions are numbered from "0" without gaps, at most 3 per subject.
    assert all(keys == [str(i) for i in range(len(keys))] for keys in sessions.values())
    assert Counter(len(keys) for keys in sessions.values()) == {3: 25, 2: 11, 1: 1}
    assert [s for s, keys in sessions.items() if len(keys) == 2] == [
        1,
        23,
        24,
        26,
        28,
        32,
        33,
        34,
        35,
        36,
        37,
    ]
    assert sessions[25] == ["0"]
    assert Counter(row[3] for row in _EPOCH_FILES) == {1: 37, 2: 36, 3: 25}
    # Subjects 1-17 are the healthy controls, 18-37 the patients.
    groups = {_subject(row[1], row[2]): row[1] for row in _EPOCH_FILES}
    assert {groups[s] for s in range(1, 18)} == {"Normals"}
    assert {groups[s] for s in range(18, 38)} == {"Patients"}
    assert len({row[0] for row in _EPOCH_FILES}) == 98
    meta = HefmiIch2025.METADATA
    assert meta.participants.n_subjects == 37
    assert meta.sessions_per_subject == 3
    assert meta.data_structure.n_trials == 98 * 30


@pytest.fixture
def synthetic(mne_data, monkeypatch):
    """Three subjects' files with synthetic bytes and their shipped map."""
    payload = {}
    rows = []
    for file_id, group, folder_subject, session in (
        (31, "Normals", 1, 2),
        (12, "Normals", 1, 1),
        (20, "Patients", 9, 1),
        (11, "Patients", 9, 2),
        (40, "Normals", 2, 1),
    ):
        data = f"{group}/{folder_subject}/{session}".encode()
        payload[file_id] = data
        rows.append(
            (
                file_id,
                group,
                folder_subject,
                session,
                len(data),
                hashlib.md5(data).hexdigest(),
            )
        )
    rows = tuple(sorted(rows, key=lambda row: (row[1], row[2], row[3])))
    monkeypatch.setattr(hefmi_ich2025, "_EPOCH_FILES", rows)

    calls = {"files": []}

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None, fname=None):
        file_id = int(url.rsplit("/", 1)[1])
        calls["files"].append(file_id)
        destination = Path(path) / "MNE-hefmiich2025-data" / fname
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload[file_id])
        return str(destination)

    monkeypatch.setattr(dl, "data_dl", fake_data_dl)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    return calls


def test_data_path_uses_the_depositor_layout(mne_data, synthetic):
    base = mne_data / "MNE-hefmiich2025-data"
    dataset = HefmiIch2025()

    # Patients/subject9 is subject 26; its files come in session order.
    assert dataset.data_path(26) == [
        str(base / "epo_data/Patients/EEG/subject9/1_epo.mat"),
        str(base / "epo_data/Patients/EEG/subject9/2_epo.mat"),
    ]
    assert synthetic["files"] == [20, 11]
    assert dataset.data_path(1) == [
        str(base / "epo_data/Normals/EEG/subject1/1_epo.mat"),
        str(base / "epo_data/Normals/EEG/subject1/2_epo.mat"),
    ]
    assert synthetic["files"] == [20, 11, 12, 31]
    # Present and intact files are not downloaded again.
    dataset.data_path(26)
    assert synthetic["files"] == [20, 11, 12, 31]


def test_data_path_replaces_a_corrupt_file(mne_data, synthetic):
    base = mne_data / "MNE-hefmiich2025-data"
    target = base / "epo_data/Normals/EEG/subject2/1_epo.mat"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")
    assert HefmiIch2025().data_path(2) == [str(target)]
    assert synthetic["files"] == [40]
    assert target.read_bytes() == b"Normals/2/1"


def test_data_path_rejects_a_file_that_does_not_match(mne_data, synthetic, monkeypatch):
    def wrong_bytes(url, sign, path=None, force_update=False, verbose=None, fname=None):
        destination = Path(path) / "MNE-hefmiich2025-data" / fname
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"something else")
        return str(destination)

    monkeypatch.setattr(dl, "data_dl", wrong_bytes)
    with pytest.raises(OSError, match="does not match figshare file 40"):
        HefmiIch2025().data_path(2)


def test_sessions_are_the_files_of_the_folder(mne_data, monkeypatch, tmp_path):
    """Session ``str(k - 1)`` holds ``<k>_epo.mat``, first and last trials kept."""
    fs, n_trials = 32.0, 4
    n_times = int(27 * fs)
    files = []
    for session in (1, 2, 3):
        X = np.zeros((n_times, 32, n_trials))
        # Session k, trial t: channel 0 alternates +-(10 k + t) microvolts (zero
        # mean, so the per-trial de-meaning keeps it).
        alternating = np.where(np.arange(n_times) % 2 == 0, 1.0, -1.0)
        for trial in range(n_trials):
            X[:, 0, trial] = (session * 10 + trial) * alternating
        y = np.array([0, 1, 1, 0]) if session != 2 else np.array([1, 0, 0, 1])
        path = tmp_path / f"{session}_epo.mat"
        savemat(path, {"epo": {"X": X, "y": y, "fs": fs}})
        files.append(str(path))
    monkeypatch.setattr(HefmiIch2025, "data_path", lambda self, subject, **kwargs: files)

    sessions = HefmiIch2025()._get_single_subject_data(3)
    assert list(sessions) == ["0", "1", "2"]
    for key, codes in (("0", [1, 2, 2, 1]), ("1", [2, 1, 1, 2]), ("2", [1, 2, 2, 1])):
        raw = sessions[key]["0"]
        events = mne.find_events(raw, shortest_event=1, verbose=False)
        assert events[:, 2].tolist() == codes
        # Each event sits in its own trial of this session's file, in volts.
        values = np.abs(raw.get_data(picks=[0])[0, events[:, 0]]) * 1e6
        session = int(key) + 1
        np.testing.assert_allclose(values, [session * 10 + t for t in range(n_trials)])
