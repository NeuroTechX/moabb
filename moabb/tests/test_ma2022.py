"""Offline regressions for the authors' EDF/BIDS Ma2022 release."""

import hashlib
import json
from unittest.mock import Mock

import httpx
import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from moabb.datasets import Ma2022
from moabb.datasets.bids_interface import get_bids_root
from moabb.datasets.download import NemarDownloadError
from moabb.datasets.ma2022 import MA2022_CH_NAMES
from moabb.paradigms import MotorImagery


@pytest.fixture
def bids_root(tmp_path):
    """Small synthetic EDFs, never downloads or reads real recordings."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Synthetic Ma2022", "BIDSVersion": "1.7.0"})
    )
    (root / "task-motorimagery_eeg.json").write_text(
        json.dumps(
            {
                "TaskName": "motorimagery",
                "SamplingFrequency": 250,
                "PowerLineFrequency": 50,
                "EEGReference": "M1",
                "EEGGround": "AFz",
            }
        )
    )
    (root / "participants.tsv").write_text("participant_id\tsex\tage\nsub-003\tF\t23\n")
    times = np.arange(3000) / 250
    signal = 20e-6 * np.sin(2 * np.pi * 10 * times)
    data = np.tile(signal, (32, 1))
    for session in range(1, 6):
        folder = root / "sub-003" / f"ses-{session:02d}" / "eeg"
        folder.mkdir(parents=True)
        stem = f"sub-003_ses-{session:02d}_task-motorimagery"
        raw = mne.io.RawArray(
            data.copy(), mne.create_info(MA2022_CH_NAMES, 250, "eeg"), verbose=False
        )
        if session == 2:
            raw._data[MA2022_CH_NAMES.index("F3")] = 0
        mne.export.export_raw(folder / f"{stem}_eeg.edf", raw, verbose=False)
        channels = "name\ttype\tunits\tstatus\tstatus_description\n"
        for name in MA2022_CH_NAMES:
            bad = name == "F3" and session == 2
            channels += (
                f"{name}\tEEG\tµV\t{'bad' if bad else 'good'}\t"
                f"{'flat: zeroed by the authors' if bad else 'n/a'}\n"
            )
        (folder / f"{stem}_channels.tsv").write_text(channels)
        (folder / f"{stem}_events.tsv").write_text(
            "onset\tduration\ttrial_type\tvalue\tsample\n"
            "0\t4\tright_hand\t2\t0\n"
            "4\t4\tleft_hand\t1\t1000\n"
            "8\t4\tright_hand\t2\t2000\n"
        )
    return root


@pytest.fixture
def offline_dataset(bids_root, monkeypatch):
    dataset = Ma2022(subjects=[3])
    monkeypatch.setattr(dataset, "_download_nemar", lambda *a, **k: str(bids_root))
    monkeypatch.setattr(
        dataset,
        "sourcedata_path",
        lambda *a, **k: pytest.fail("unused sourcedata fetched"),
    )
    return dataset


@pytest.mark.parametrize("provider", ["auto", "nemar"])
def test_download_uses_bids_and_forwards_flags(bids_root, monkeypatch, provider):
    # ``upstream`` deliberately skips NEMAR -- its coverage lives in
    # ``test_upstream_provider_uses_figshare_mirror`` below.
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", provider)
    download = Mock(return_value=str(bids_root))
    monkeypatch.setattr("moabb.datasets.base.nemar_dl", download)
    monkeypatch.setattr(
        "moabb.datasets.base.nemar_sourcedata_dl",
        lambda *a, **k: pytest.fail("unused sourcedata fetched"),
    )
    dataset = Ma2022(subjects=[3])
    dataset.download(path="custom-root", force_update=True, verbose=False)
    download.assert_called_once_with(
        "nm000288",
        "Ma-edf2022",
        path="custom-root",
        force_update=True,
        subject="003",
        verbose=False,
        scope="raw",
    )
    paths = dataset.data_path(3)
    assert len(paths) == 5
    assert all(p.suffix == ".edf" for p in paths)
    assert [p.parent.parent.name for p in paths] == [f"ses-{i:02d}" for i in range(1, 6)]
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(2)


def test_nemar_pin_never_falls_back_on_failure(monkeypatch, tmp_path):
    """``MOABB_DOWNLOAD_PROVIDER=nemar`` keeps NEMAR failures fatal.

    The figshare mirror only activates when the user has not explicitly
    opted into NEMAR-only; the mirror must never be reached from an
    ``nemar``-pinned run (silent routing would defeat the pin's purpose).
    """
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    legacy = tmp_path / "MNE-ma2022-data" / "mat"
    legacy.mkdir(parents=True)
    (legacy / "sub-003_ses-01_task_motorimagery_eeg.mat").write_bytes(b"legacy")
    dataset = Ma2022(subjects=[3])
    monkeypatch.setattr(
        dataset, "_download_nemar", Mock(side_effect=NemarDownloadError("unavailable"))
    )
    monkeypatch.setattr(
        "moabb.datasets.download.data_dl",
        lambda *a, **k: pytest.fail("upstream MAT fallback"),
    )
    monkeypatch.setattr(
        "moabb.datasets._ma2022_figshare.ensure_bids_mirror",
        lambda *a, **k: pytest.fail("figshare fallback used from nemar-pinned run"),
    )
    with pytest.raises(NemarDownloadError, match="unavailable"):
        dataset.download(path=tmp_path)
    with pytest.raises(NemarDownloadError, match="unavailable"):
        dataset.get_data([3], cache_config={"use": False})


@pytest.mark.parametrize("provider", ["auto", "upstream"])
def test_nemar_outage_falls_back_to_figshare(bids_root, monkeypatch, tmp_path, provider):
    """NEMAR outage (or ``provider=upstream``) routes to the figshare mirror.

    Mock-only: ``ensure_bids_mirror`` is intercepted so no file leaves the
    laptop. Only the fallback *selection* is under test here.
    """
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", provider)
    dataset = Ma2022(subjects=[3])
    # provider=auto still goes NEMAR first; simulate an outage so the
    # fallback branch gets exercised. provider=upstream skips NEMAR by
    # policy, so the Mock is never consulted there.
    nemar_mock = Mock(side_effect=NemarDownloadError("NEMAR down"))
    monkeypatch.setattr(dataset, "_download_nemar", nemar_mock)
    captured: dict[str, object] = {}

    def fake_ensure(root, subjects, *, force_update=False, verbose=None):
        captured["root"] = root
        captured["subjects"] = list(subjects)
        captured["force_update"] = force_update

    monkeypatch.setattr("moabb.datasets._ma2022_figshare.ensure_bids_mirror", fake_ensure)
    monkeypatch.setattr(
        "moabb.datasets.download.data_dl",
        lambda *a, **k: pytest.fail("upstream MAT fallback"),
    )
    root = dataset._download_subject(3, str(tmp_path), False, None, None)
    assert captured["subjects"] == [3]
    assert captured["force_update"] is False
    assert str(captured["root"]).endswith("MNE-BIDS-ma-edf2022-figshare")
    assert root == str(captured["root"])
    if provider == "auto":
        nemar_mock.assert_called_once()
    else:
        nemar_mock.assert_not_called()


def test_missing_session_is_not_silently_skipped(bids_root, offline_dataset):
    next(bids_root.glob("sub-003/ses-05/eeg/*.edf")).unlink()
    with pytest.raises(FileNotFoundError, match="five EDF sessions"):
        offline_dataset.data_path(3)


def test_fresh_download_keeps_sidecars(bids_root, tmp_path, monkeypatch):
    """Exercise NEMAR selection and transfer, mocking only the HTTP server."""
    files = {
        p.relative_to(bids_root).as_posix(): p.read_bytes()
        for p in bids_root.rglob("*")
        if p.is_file()
    }
    # Session metadata has no task entity; original sourcedata and other
    # subjects must not be fetched with the selected subject's BIDS files.
    scans = "sub-003/ses-01/sub-003_ses-01_scans.tsv"
    files[scans] = (
        b"filename\tacq_time\neeg/sub-003_ses-01_task-motorimagery_eeg.edf\tn/a\n"
    )
    unwanted = {
        "sourcedata/sub-003/ses-01/eeg/sub-003_ses-01_task-motorimagery_eeg.edf",
        "sub-004/ses-01/eeg/sub-004_ses-01_task-motorimagery_eeg.edf",
    }
    files.update(dict.fromkeys(unwanted, b"unused"))
    manifest = [
        {
            "path": name,
            "url": f"https://data.nemar.org/files/{name}",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in files.items()
    ]
    fetched = set()

    def respond(request):
        path = request.url.path
        if path == "/nm000288/":
            return httpx.Response(
                200,
                json={
                    "dataset_id": "nm000288",
                    "latest": "1.0.0",
                    "versions": [
                        {"version": "1.0.0", "manifest_url": "/nm000288/manifest.json"}
                    ],
                },
            )
        if path == "/nm000288/manifest.json":
            return httpx.Response(200, json=manifest)
        assert path.startswith("/files/"), path
        name = path.removeprefix("/files/")
        fetched.add(name)
        return httpx.Response(200, content=files[name])

    client_init = httpx.Client.__init__

    def init_client(self, *args, **kwargs):
        kwargs.update(transport=httpx.MockTransport(respond), trust_env=False)
        client_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", init_client)
    (tmp_path / "download").mkdir()
    monkeypatch.setenv("MNE_DATASETS_MA-EDF2022_PATH", str(tmp_path / "download"))
    dataset = Ma2022(subjects=[3])
    # Keep the real selection/verification pipeline, using HTTPS rather
    # than nemar-py's optional S3 backend for this in-memory HTTP fixture.
    dataset.nemar_bids_filters = {**dataset.nemar_bids_filters, "downloader": "python"}
    sessions = dataset.get_data([3], cache_config={"use": False})[3]
    assert fetched == set(files) - unwanted
    for session, runs in sessions.items():
        raw = runs["0"]
        assert raw.info["bads"] == (["F3"] if session == "1" else [])
        events, _ = mne.events_from_annotations(raw, event_id=dataset.event_id)
        assert_array_equal(events[:, 0], [0, 1000, 2000])
        assert_array_equal(events[:, 2], [2, 1, 2])


def test_edf_units_annotations_montage_and_bads(offline_dataset):
    sessions = offline_dataset.get_data([3], cache_config={"use": False})[3]
    assert list(sessions) == [str(i) for i in range(5)]
    for session, runs in sessions.items():
        assert list(runs) == ["0"]
        raw = runs["0"]
        assert raw.ch_names == MA2022_CH_NAMES
        assert raw.info["sfreq"] == 250
        assert raw.n_times == 3000
        assert raw.info["bads"] == (["F3"] if session == "1" else [])
        assert all(np.isfinite(ch["loc"][:3]).all() for ch in raw.info["chs"])
        assert_allclose(np.max(np.abs(raw.get_data(picks=["C3"]))), 20e-6, rtol=0.003)
        events, _ = mne.events_from_annotations(raw, event_id=offline_dataset.event_id)
        assert_array_equal(events[:, 0], [0, 1000, 2000])
        assert_array_equal(events[:, 2], [2, 1, 2])
    assert_allclose(sessions["1"]["0"].get_data(picks=["F3"]), 0, atol=1e-9)


def test_paradigm_keeps_first_last_trials_and_handles_common_channels(offline_dataset):
    # A common good-channel set avoids session-dependent 31/32-channel arrays.
    channels = [ch for ch in MA2022_CH_NAMES if ch not in {"F3", "T6", "A2"}]
    paradigm = MotorImagery(n_classes=2, channels=channels)
    X, y, metadata = paradigm.get_data(offline_dataset, [3], cache_config={"use": False})
    assert X.shape == (15, 29, 1000)
    assert_array_equal(y, ["right_hand", "left_hand", "right_hand"] * 5)
    assert metadata.groupby("session").size().to_dict() == {str(i): 3 for i in range(5)}
    assert 5 < np.median(np.abs(X)) < 25  # paradigm output is microvolts
    # The ordinary epoch picker excludes flagged channels, without clearing bads.
    raw = offline_dataset.get_data([3], cache_config={"use": False})[3]["1"]["0"]
    pipeline = MotorImagery(n_classes=2)
    pipeline.prepare_process(offline_dataset)
    epochs = pipeline._get_epochs_pipeline(True, False, offline_dataset).transform(raw)
    assert "F3" not in epochs.ch_names
    assert len(epochs) == 3


def test_cache_identity_and_metadata(tmp_path):
    dataset = Ma2022()
    assert dataset.code == "Ma-edf2022"
    assert get_bids_root(dataset.code, tmp_path) != get_bids_root("Ma2022", tmp_path)
    assert dataset.METADATA.file_format == "EDF"
    assert dataset.METADATA.data_processed is True
    assert dataset.METADATA.preprocessing.preprocessing_applied is True
    assert dataset.interval == [0, 3.996]


def test_raw_cache_roundtrip_preserves_bads(offline_dataset, tmp_path, monkeypatch):
    config = {"path": tmp_path / "cache", "save_raw": True, "use": True}
    first = offline_dataset.get_data([3], cache_config=config)[3]
    monkeypatch.setattr(
        offline_dataset,
        "_download_nemar",
        lambda *a, **k: pytest.fail("cache hit must not download"),
    )
    second = offline_dataset.get_data([3], cache_config=config)[3]
    for session in first:
        assert second[session]["0"].info["bads"] == first[session]["0"].info["bads"]
        assert_allclose(second[session]["0"].get_data(), first[session]["0"].get_data())


def test_additional_metadata_maps_sessions_without_run_entity(offline_dataset):
    metadata = offline_dataset.get_additional_metadata(3, "1", "0")
    assert metadata["onset"].tolist() == [0, 4, 8]
    assert metadata["trial_type"].tolist() == ["right_hand", "left_hand", "right_hand"]
    assert metadata["session"].tolist() == ["1"] * 3


def test_duplicate_session_recordings_are_rejected(bids_root, offline_dataset):
    original = next(bids_root.glob("sub-003/ses-01/eeg/*.edf"))
    duplicate = original.with_name(original.name.replace("_eeg.edf", "_run-02_eeg.edf"))
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(FileNotFoundError, match="five EDF sessions"):
        offline_dataset.data_path(3)


def test_selected_sessions_preserve_zero_based_labels(offline_dataset):
    offline_dataset._selected_sessions = [1]
    sessions = offline_dataset.get_data([3], cache_config={"use": False})[3]
    assert list(sessions) == ["1"]
    assert sessions["1"]["0"].info["bads"] == ["F3"]


def test_filtering_respects_discontinuous_trial_boundaries(offline_dataset):
    raw = offline_dataset.get_data([3], cache_config={"use": False})[3]["0"]["0"]
    edges = raw.annotations.description == "EDGE boundary"
    assert_array_equal(raw.annotations.onset[edges], [4, 8])
    assert_array_equal(raw.annotations.duration[edges], [0, 0])
    # Deliberately dissimilar neighboring trials expose cross-window filtering.
    rng = np.random.default_rng(7)
    raw._data[:] = rng.normal(scale=10e-6, size=raw._data.shape)
    raw._data[:, 1000:2000] += 100e-6
    filtered = raw.copy().filter(8, 30, verbose=False)
    for start in (0, 1000, 2000):
        independent = mne.io.RawArray(
            raw.get_data(start=start, stop=start + 1000), raw.info.copy(), verbose=False
        ).filter(8, 30, verbose=False)
        assert_allclose(
            filtered.get_data(start=start, stop=start + 1000),
            independent.get_data(),
            rtol=1e-12,
            atol=1e-15,
        )
