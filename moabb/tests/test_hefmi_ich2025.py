"""HefmiIch2025 subjects and sessions follow the depositor's folders (offline).

The figshare file names (``1_epo.mat``, ``2_epo.mat``, ``3_epo.mat``) carry no
subject; the record of article 28955456 version 4 gives every file's folder in
``folder_structure``. These tests mock the record and the downloads with
synthetic data.
"""

import hashlib
import json
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
from moabb.datasets.hefmi_ich2025 import (
    _EPOCH_FILES,
    HefmiIch2025,
    _epoch_files_from_record,
    _subject,
)


# Files of the record that are not EEG epochs: fNIRS epochs with the same
# names, raw recordings of the same folders, the code archive.
_OTHER_FILES = (
    (54450017, "source_data/Patients/NIRS/subject5", "1.hdf5"),
    (54460000, "epo_data/Patients/NIRS/subject9", "2_epo.mat"),
    (54460001, "source_data/Patients/EEG/subject9", "2.hdf5"),
    (54460002, "", "code.zip"),
)


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


def _record(rows, other=_OTHER_FILES):
    """A figshare article record listing ``rows`` (``_EPOCH_FILES`` layout)."""
    files, folders = [], {}
    for file_id, group, folder_subject, session, size, md5 in rows:
        files.append(
            {
                "id": file_id,
                "name": f"{session}_epo.mat",
                "size": size,
                "computed_md5": md5,
            }
        )
        folders[str(file_id)] = f"epo_data/{group}/EEG/subject{folder_subject}"
    for file_id, folder, name in other:
        files.append(
            {"id": file_id, "name": name, "size": 28_000_000, "computed_md5": "0" * 32}
        )
        folders[str(file_id)] = folder
    # The API lists the files in upload order (by id), not by folder.
    files.sort(key=lambda entry: entry["id"])
    return {"id": 28955456, "version": 4, "files": files, "folder_structure": folders}


def _subjects_sessions(rows):
    """{subject: sorted session keys} of a file map."""
    sessions = {}
    for _id, group, folder_subject, session, _size, _md5 in rows:
        subject = _subject(group, folder_subject)
        sessions.setdefault(subject, []).append(str(session - 1))
    return {subject: sorted(keys) for subject, keys in sessions.items()}


def test_record_mapping_is_the_shipped_map():
    assert _epoch_files_from_record(_record(_EPOCH_FILES)) == _EPOCH_FILES


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
    rows = {row[0]: row for row in _epoch_files_from_record(_record(_EPOCH_FILES))}
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


def test_upload_order_grouping_is_gone():
    assert not hasattr(hefmi_ich2025, "_MANIFEST")
    assert not hasattr(HefmiIch2025, "_get_manifest")

    record = _record(_EPOCH_FILES)
    folders = record["folder_structure"]
    # The former grouping: EEG epochs (_epo.mat above 40 MB) by figshare id, a
    # new subject at every 1_epo.mat. On this record 23 of its 37 subjects mix
    # files of two to four folders.
    old, subject = {}, 0
    for entry in record["files"]:
        if not entry["name"].endswith("_epo.mat") or entry["size"] <= 40_000_000:
            continue
        if entry["name"] == "1_epo.mat":
            subject += 1
        old.setdefault(subject, set()).add(folders[str(entry["id"])])
    assert len(old) == 37
    assert sum(len(f) > 1 for f in old.values()) == 23

    # The loader's mapping never mixes folders and keeps each folder whole.
    new = {}
    for _id, group, folder_subject, *_ in _epoch_files_from_record(record):
        new.setdefault(_subject(group, folder_subject), set()).add(
            (group, folder_subject)
        )
    assert all(len(f) == 1 for f in new.values())
    assert len(new) == 37


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

    calls = {"record": 0, "files": []}

    def fake_record(method, url, headers, data=None, binary=False):
        assert url == "https://api.figshare.com/v2/articles/28955456/versions/4"
        calls["record"] += 1
        return _record(rows)

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None, fname=None):
        file_id = int(url.rsplit("/", 1)[1])
        calls["files"].append(file_id)
        destination = Path(path) / "MNE-hefmiich2025-data" / fname
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload[file_id])
        return str(destination)

    monkeypatch.setattr(dl, "fs_issue_request", fake_record)
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
    # Present and intact files are not downloaded again; the record is cached.
    dataset.data_path(26)
    assert synthetic["files"] == [20, 11, 12, 31]
    assert synthetic["record"] == 1
    assert json.loads((base / "figshare_article_28955456_v4.json").read_text())


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


def test_without_the_record_the_shipped_map_is_used(mne_data, synthetic, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr(dl, "fs_issue_request", offline)
    base = mne_data / "MNE-hefmiich2025-data"
    assert HefmiIch2025().data_path(26) == [
        str(base / "epo_data/Patients/EEG/subject9/1_epo.mat"),
        str(base / "epo_data/Patients/EEG/subject9/2_epo.mat"),
    ]
    assert not (base / "figshare_article_28955456_v4.json").exists()


def test_a_record_that_disagrees_warns_and_keeps_the_shipped_map(
    mne_data, synthetic, monkeypatch
):
    record = _record(hefmi_ich2025._EPOCH_FILES)
    record["folder_structure"]["11"] = "epo_data/Normals/EEG/subject2"
    monkeypatch.setattr(dl, "fs_issue_request", lambda *args, **kwargs: record)
    base = mne_data / "MNE-hefmiich2025-data"
    with pytest.warns(RuntimeWarning, match="no longer matches the file map"):
        paths = HefmiIch2025().data_path(26)
    assert paths[1] == str(base / "epo_data/Patients/EEG/subject9/2_epo.mat")


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
