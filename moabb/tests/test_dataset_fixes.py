"""Regression tests for dataset-loader fixes."""

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
import requests

from moabb.datasets import download as _dl
from moabb.datasets.bnci.bnci_2020 import _convert_attention_shift
from moabb.datasets.ding2025 import _KNOWN_NONFINITE_RUNS, Ding2025
from moabb.datasets.garro2025 import _ANNOT_RENAME as GARRO_ANNOT_RENAME
from moabb.datasets.garro2025 import _ROOT_FILE_IDS as GARRO_ROOT_FILE_IDS
from moabb.datasets.garro2025 import _SUBJECT_FILE_IDS as GARRO_SUBJECT_FILE_IDS
from moabb.datasets.garro2025 import Garro2025
from moabb.datasets.jia2019 import _FILE_IDS as JIA2019_FILE_IDS
from moabb.datasets.jia2019 import Jia2019
from moabb.datasets.ma2022 import MA2022_CH_NAMES, SFREQ, Ma2022
from moabb.datasets.perezblanco2026 import PerezBlanco2026
from moabb.datasets.sensoryguidedmi2026 import SUBJECT_MAP, SensoryGuidedMI2026
from moabb.datasets.shin2022 import Shin2022
from moabb.datasets.ssvep_mamem import MAMEM1
from moabb.datasets.sun2026 import Sun2026


def _fake_bciexp(n_channels=3, n_samples=4, n_trials=5):
    rng = np.random.default_rng(0)
    data = np.asfortranarray(
        rng.standard_normal((n_channels, n_samples, n_trials)).astype(np.float64)
    )
    return SimpleNamespace(
        data=data,
        heog=rng.standard_normal((n_samples, n_trials)).astype(np.float64),
        veog=rng.standard_normal((n_samples, n_trials)).astype(np.float64),
        srate=250.0,
        label=[f"E{i + 1}" for i in range(n_channels)],
        intention=np.array(["yes", "no", "yes", "no", "yes"], dtype=object),
    )


def test_bnci2020_002_layout():
    bciexp = _fake_bciexp()
    with patch("moabb.datasets.bnci.bnci_2020.loadmat", return_value={"bciexp": bciexp}):
        raw, event_id = _convert_attention_shift("dummy.mat")

    n_channels, n_samples, n_trials = bciexp.data.shape
    expected = bciexp.data.transpose(0, 2, 1).reshape(n_channels, -1) * 1e-6
    np.testing.assert_allclose(
        raw.copy().pick("eeg").get_data(), expected, rtol=0, atol=1e-15
    )

    stim = raw.copy().pick("stim").get_data()[0]
    nonzero_idx = np.flatnonzero(stim)
    np.testing.assert_array_equal(nonzero_idx, np.arange(n_trials) * n_samples)
    assert event_id == {"NonTarget": 1, "Target": 2}
    np.testing.assert_array_equal(stim[nonzero_idx], [2, 1, 2, 1, 2])


def test_fs_get_file_list_caches():
    _dl.fs_get_file_list.cache_clear()
    side_effect = [
        [{"id": 1, "name": "a.mat", "supplied_md5": "0" * 32}],
        [{"id": 2, "name": "b.mat", "supplied_md5": "1" * 32}],
    ]
    with patch.object(_dl, "_fs_paginated_file_list", side_effect=side_effect) as page:
        first = _dl.fs_get_file_list(123456)
        second = _dl.fs_get_file_list(123456)
        third = _dl.fs_get_file_list(123456, version=2)
    assert page.call_count == 2
    assert first is second
    assert third is not first
    _dl.fs_get_file_list.cache_clear()


def test_mamem_filelist_disk_cache(tmp_path: Path):
    _dl.fs_get_file_list.cache_clear()
    ds = MAMEM1()
    sentinel = [{"id": 99, "name": "U001ai.mat", "supplied_md5": "0" * 32}]
    cache_path = Path(ds._filelist_cache_path(str(tmp_path)))

    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list", return_value=sentinel
    ) as api:
        assert ds._load_or_fetch_filelist(str(tmp_path)) == sentinel
    assert api.call_count == 1
    assert json.loads(cache_path.read_text()) == sentinel

    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list",
        side_effect=AssertionError("API hit when cache present"),
    ):
        assert ds._load_or_fetch_filelist(str(tmp_path)) == sentinel

    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list", return_value=sentinel
    ) as api:
        ds._load_or_fetch_filelist(str(tmp_path), force_update=True)
    assert api.call_count == 1
    _dl.fs_get_file_list.cache_clear()


def test_mamem_already_downloaded_does_not_ping_figshare(tmp_path: Path):
    _dl.fs_get_file_list.cache_clear()

    ds = MAMEM1()

    filelist = []
    for sub_id in (1, 2, 3):
        for run_letter in "ai":
            payload = f"S{sub_id:02d}{run_letter}".encode()
            file_id = sub_id * 100 + ord(run_letter)
            filelist.append(
                {
                    "id": file_id,
                    "name": f"U0{sub_id:02d}{run_letter}i.mat",
                    "supplied_md5": hashlib.md5(payload).hexdigest(),
                }
            )
            (tmp_path / str(file_id)).write_bytes(payload)

    Path(ds._filelist_cache_path(str(tmp_path))).write_text(json.dumps(filelist))

    with (
        patch.object(ds, "_dataset_root", return_value=str(tmp_path)),
        patch(
            "moabb.datasets.ssvep_mamem.fs_get_file_list",
            side_effect=AssertionError("API hit when dataset already downloaded"),
        ),
    ):
        for sub_id in (1, 2, 3):
            paths = ds.data_path(sub_id)
            assert paths
            assert all(Path(p).exists() for p in paths)
    _dl.fs_get_file_list.cache_clear()


def test_ma2022_prefers_complete_local_edf_set(tmp_path: Path):
    ds = Ma2022()
    edf_dir = tmp_path / "SHU_edf" / "edf"
    edf_dir.mkdir(parents=True)
    expected = []
    for filename in ds._session_filenames(1, "edf"):
        path = edf_dir / filename
        path.touch()
        expected.append(str(path))

    with (
        patch("moabb.datasets.ma2022.dl.get_dataset_path", return_value=str(tmp_path)),
        patch.object(
            ds,
            "_mat_session_files",
            side_effect=AssertionError("MAT fallback used despite complete EDF set"),
        ),
    ):
        assert ds.data_path(1) == expected


def test_ding2025_session_and_run_names_follow_moabb_contract(tmp_path: Path):
    ds = Ding2025()
    offline = tmp_path / "OfflineImagery" / "S01_OfflineImagery_R01.mat"
    online = (
        tmp_path
        / "OnlineImagery_Sess01_2class_Base"
        / "S01_OnlineImagery_Sess01_2class_Base_R01.mat"
    )
    for path in (offline, online):
        path.parent.mkdir(parents=True)
        path.touch()

    with (
        patch.object(ds, "data_path", return_value=[str(tmp_path)]),
        patch.object(ds, "_load_run", return_value=object()),
    ):
        sessions = ds._get_single_subject_data(1)

    assert list(sessions) == ["0SessionOfflineImagery", "1SessionOnlineImagerySess01"]
    assert list(sessions["0SessionOfflineImagery"]) == ["0RunR01"]
    assert list(sessions["1SessionOnlineImagerySess01"]) == ["0Run2classBaseR01"]


def test_ding2025_rejects_nonfinite_raw_data_before_scaling(tmp_path: Path):
    """A corrupt MATLAB run must not enter MNE/preprocessing unchanged."""
    ds = Ding2025()
    mat_path = tmp_path / "S07_OnlineImagery_Sess05_3class_Base_R01.mat"
    mat_path.touch()
    data = np.ones((128, 16))
    data[0, 0] = np.nan
    data[1, 1] = np.inf
    data[2, 2] = -np.inf

    with patch(
        "moabb.datasets.ding2025.read_mat",
        return_value={"eeg": {"data": data, "fsample": 1024.0}},
    ):
        try:
            ds._load_run(mat_path)
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("Ding2025 accepted a raw run with non-finite EEG data")

    assert "non-finite" in message
    assert "nan=1" in message
    assert "posinf=1" in message
    assert "neginf=1" in message


def test_ding2025_skips_only_nonfinite_runs_with_counts(tmp_path: Path, caplog):
    ds = Ding2025()
    run_dir = tmp_path / "OnlineImagery_Sess05_3class_Base"
    bad_run = run_dir / "S07_OnlineImagery_Sess05_3class_Base_R01.mat"
    good_run = run_dir / "S07_OnlineImagery_Sess05_3class_Base_R09.mat"
    run_dir.mkdir()
    bad_run.touch()
    good_run.touch()

    bad_data = np.ones((128, 16))
    bad_data[0, 0] = np.nan
    good_data = np.ones((128, 16))

    def read_run(path):
        data = bad_data if Path(path).name == bad_run.name else good_data
        return {"eeg": {"data": data, "fsample": 1024.0}}

    with (
        patch.object(ds, "data_path", return_value=[str(tmp_path)]),
        patch("moabb.datasets.ding2025.read_mat", side_effect=read_run),
        caplog.at_level(logging.WARNING, logger="moabb.datasets.ding2025"),
    ):
        sessions = ds._get_single_subject_data(7)

    runs = sessions["0SessionOnlineImagerySess05"]
    assert list(runs) == ["1Run3classBaseR09"]
    assert (
        "Skipping known corrupt run "
        "OnlineImagery_Sess05_3class_Base/"
        "S07_OnlineImagery_Sess05_3class_Base_R01.mat"
    ) in caplog.text
    assert "non-finite eeg.data (nan=1, posinf=0, neginf=0)" in caplog.text


def test_ding2025_known_nonfinite_guard_is_exact_and_unexpected_corruption_raises(
    tmp_path: Path,
):
    expected = {
        "OnlineImagery_Sess05_3class_Base/"
        f"S07_OnlineImagery_Sess05_3class_Base_R{run:02d}.mat"
        for run in range(1, 9)
    }
    assert _KNOWN_NONFINITE_RUNS == expected

    ds = Ding2025()
    run_dir = tmp_path / "OnlineImagery_Sess05_3class_Base"
    unexpected = run_dir / "S07_OnlineImagery_Sess05_3class_Base_R09.mat"
    run_dir.mkdir()
    unexpected.touch()
    data = np.ones((128, 16))
    data[0, 0] = np.nan

    with (
        patch.object(ds, "data_path", return_value=[str(tmp_path)]),
        patch(
            "moabb.datasets.ding2025.read_mat",
            return_value={"eeg": {"data": data, "fsample": 1024.0}},
        ),
    ):
        try:
            ds._get_single_subject_data(7)
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("Ding2025 silently skipped unexpected corruption")

    assert "non-finite eeg.data (nan=1, posinf=0, neginf=0)" in message


def test_ma2022_edf_reader_attaches_open_v1_labels():
    ds = Ma2022()
    edf_names = [name.upper() for name in MA2022_CH_NAMES]
    info = mne.create_info(edf_names, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(
        np.zeros((len(MA2022_CH_NAMES), int(8 * SFREQ))), info, verbose=False
    )

    with (
        patch("moabb.datasets.ma2022.read_raw_edf", return_value=raw) as reader,
        patch(
            "moabb.datasets.ma2022.sio.loadmat",
            return_value={"labels": np.array([[1, 2]])},
        ) as loadmat,
    ):
        loaded = ds._edf_to_raw("session.edf", "session.mat")

    reader.assert_called_once_with("session.edf", preload=True, verbose=False)
    loadmat.assert_called_once_with("session.mat", variable_names=["labels"])
    assert loaded is raw
    assert loaded.ch_names == MA2022_CH_NAMES
    assert list(loaded.annotations.description) == ["left_hand", "right_hand"]
    np.testing.assert_allclose(loaded.annotations.onset, [0.0, 4.0])


def test_ma2022_bad_edf_numeric_header_uses_mat_session_fallback():
    ds = Ma2022()
    good_raw = object()
    fallback_raw = object()
    unreadable_header = ValueError("could not convert string to float: '        '")

    with (
        patch.object(ds, "data_path", return_value=["good.edf", "bad.edf"]),
        patch.object(
            ds,
            "_mat_session_files",
            return_value=["good.mat", "bad.mat"],
        ),
        patch.object(
            ds,
            "_edf_to_raw",
            side_effect=[good_raw, unreadable_header],
        ) as read_edf,
        patch.object(ds, "_mat_to_raw", return_value=fallback_raw) as read_mat,
    ):
        sessions = ds._get_single_subject_data(1)

    assert sessions == {
        "0": {"0": good_raw},
        "1": {"0": fallback_raw},
    }
    assert read_edf.call_count == 2
    read_mat.assert_called_once_with("bad.mat")


def test_jia2019_local_files_skip_figshare_api(tmp_path: Path):
    ds = Jia2019()
    files_dir = tmp_path / "MNE-jia2019-data" / "files"
    files_dir.mkdir(parents=True)
    expected = []
    for side in ("left", "right"):
        path = files_dir / JIA2019_FILE_IDS[1][side]
        path.touch()
        expected.append(str(path))

    with (
        patch("moabb.datasets.jia2019.dl.get_dataset_path", return_value=str(tmp_path)),
        patch(
            "moabb.datasets.jia2019.dl.fs_get_file_list",
            side_effect=AssertionError("Figshare API called for local files"),
        ),
    ):
        assert ds.data_path(1) == expected


def test_garro2025_cached_archives_skip_figshare_api(tmp_path: Path):
    ds = Garro2025()
    root = tmp_path / "MNE-garro2025-data"
    files_dir = root / "files"
    files_dir.mkdir(parents=True)

    for filename, file_id in GARRO_ROOT_FILE_IDS.items():
        (files_dir / str(file_id)).write_text(filename)

    archive = files_dir / str(GARRO_SUBJECT_FILE_IDS[1])
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "sub-01/eeg/sub-01_task-free_eeg.vhdr",
            "Brain Vision Data Exchange Header File Version 1.0",
        )

    with (
        patch("moabb.datasets.garro2025.dl.get_dataset_path", return_value=str(tmp_path)),
        patch(
            "moabb.datasets.garro2025.dl.fs_get_file_list",
            side_effect=AssertionError("Figshare API called for cached archives"),
        ),
        patch(
            "moabb.datasets.garro2025.dl.data_dl",
            side_effect=AssertionError("Network download attempted for cached archives"),
        ),
    ):
        downloaded_root = ds._download_root(1)

    assert downloaded_root == root
    assert (root / "dataset_description.json").exists()
    assert (root / "participants.tsv").exists()
    assert (root / "README.txt").exists()
    assert (root / "sub-01/eeg/sub-01_task-free_eeg.vhdr").exists()


def test_garro2025_epochs_are_locked_to_go_cues():
    ds = Garro2025()

    assert GARRO_ANNOT_RENAME == {
        "Stimulus/G 1": "reach_1",
        "Stimulus/G 2": "reach_2",
        "Stimulus/G 3": "reach_3",
    }
    assert ds.interval == [-0.5, 2]


def test_perezblanco2026_extracted_subject_skips_figshare_api(tmp_path: Path):
    ds = PerezBlanco2026()
    eeg_dir = tmp_path / "MNE-perezblanco2026-data" / "files" / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    edf = eeg_dir / "sub-01_task-WristPointingTask_run-01_eeg.edf"
    edf.touch()

    with (
        patch(
            "moabb.datasets.perezblanco2026.dl.get_dataset_path",
            return_value=str(tmp_path),
        ),
        patch(
            "moabb.datasets.perezblanco2026.dl.fs_get_file_list",
            side_effect=AssertionError("Figshare API called for extracted subject"),
        ),
    ):
        assert ds.data_path(1) == [str(edf)]


def test_sun2026_offline_probe_stops_after_local_run(tmp_path: Path):
    ds = Sun2026()

    def fake_download(_root, rel_path, _force_update):
        if "_run-02_eeg.vhdr" in rel_path:
            raise requests.RequestException("offline")
        return True

    with (
        patch("moabb.datasets.sun2026.get_dataset_path", return_value=str(tmp_path)),
        patch.object(ds, "_download_file", side_effect=fake_download),
    ):
        paths = ds.data_path(1)

    assert [(path.session, path.run) for path in paths] == [("01", "01"), ("02", "01")]


def test_sun2026_complete_local_runs_do_not_probe_openneuro(tmp_path: Path):
    ds = Sun2026()
    bids_root = tmp_path / "MNE-sun2026-data"
    for filename in ("dataset_description.json", "participants.tsv", "participants.json"):
        (bids_root / filename).parent.mkdir(parents=True, exist_ok=True)
        (bids_root / filename).touch()

    for session in ("01", "02"):
        eeg_dir = bids_root / "sub-01" / f"ses-{session}" / "eeg"
        eeg_dir.mkdir(parents=True)
        for suffix in ("electrodes.tsv", "coordsystem.json"):
            (eeg_dir / f"sub-01_ses-{session}_space-CapTrak_{suffix}").touch()
        stem = f"sub-01_ses-{session}_task-graz_run-01"
        for suffix in (
            "eeg.vhdr",
            "eeg.vmrk",
            "eeg.json",
            "channels.tsv",
            "events.tsv",
            "events.json",
        ):
            (eeg_dir / f"{stem}_{suffix}").touch()
        (eeg_dir / f"{stem}_eeg.eeg").touch()

    with (
        patch("moabb.datasets.sun2026.get_dataset_path", return_value=str(tmp_path)),
        patch(
            "moabb.datasets.sun2026.requests.get",
            side_effect=AssertionError("OpenNeuro probed for complete local runs"),
        ),
    ):
        paths = ds.data_path(1)

    assert [(path.session, path.run) for path in paths] == [("01", "01"), ("02", "01")]


def test_shin2022_run_names_start_with_integer(tmp_path: Path):
    ds = Shin2022()
    run_dir = tmp_path / "BW"
    run_dir.mkdir()
    (run_dir / "BW120.dat").touch()

    with (
        patch.object(ds, "data_path", return_value=str(tmp_path)),
        patch.object(ds, "_read_bci2000_dat", return_value=object()),
    ):
        runs = ds._get_single_subject_data(1)["0"]

    assert list(runs) == ["0BW120"]


def test_sensoryguided_subject_ids_are_group_scoped():
    assert len(SUBJECT_MAP) == 39
    assert SUBJECT_MAP[1] == ("JointLearning", 1)
    assert SUBJECT_MAP[16] == ("BCI2000Control", 1)
    assert SUBJECT_MAP[24] == ("TactileControl", 1)
    assert SUBJECT_MAP[32] == ("EEGNetControl", 1)


def test_sensoryguided_bci2000_codebook():
    ds = SensoryGuidedMI2026()
    two_dimensional = {
        "trialTargetCode": [
            np.array([1, 1]),
            np.array([2, 2]),
            np.array([3, 3]),
            np.array([4, 4]),
        ]
    }
    up_down = {"trialTargetCode": [np.array([1, 1]), np.array([2, 2])]}

    np.testing.assert_array_equal(
        ds._trial_label_events(two_dimensional, 4, "S001_sess04_run01.mat"), [2, 1, 3, 4]
    )
    np.testing.assert_array_equal(
        ds._trial_label_events(up_down, 2, "S001_sess03_run01UD.mat"), [3, 4]
    )


def test_sensoryguided_read_run_keeps_first_trial_event():
    ds = SensoryGuidedMI2026()
    run_data = {
        "meta": {"sampling_rate_hz": 1000.0},
        "trialSignal": np.zeros((5000, 2, 2)),
        "trialTargetClass": np.column_stack(
            [np.zeros(5000, dtype=int), np.ones(5000, dtype=int)]
        ),
    }

    with patch.object(ds, "_read_mat", return_value=run_data):
        raw = ds._read_run("S001_sess01_run01.mat")

    assert raw.get_channel_types() == ["eeg", "eeg"]
    assert list(raw.annotations.description) == ["left_hand", "right_hand"]
    np.testing.assert_allclose(raw.annotations.onset, [0.0, 5.0])
