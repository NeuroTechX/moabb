"""Regression tests for dataset-loader fixes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from moabb.datasets import download as _dl
from moabb.datasets import schirrmeister2017
from moabb.datasets.bnci.bnci_2020 import _convert_attention_shift
from moabb.datasets.braininvaders import BI2015b
from moabb.datasets.fake import FakeDataset
from moabb.datasets.hefmi_ich2025 import HefmiIch2025
from moabb.datasets.kaneshiro2015 import Kaneshiro2015
from moabb.datasets.kojima2024a import Kojima2024A
from moabb.datasets.lenaig2026 import Lenaig2026
from moabb.datasets.mainsah2025 import _parse_manifest
from moabb.datasets.schirrmeister2017 import Schirrmeister2017
from moabb.datasets.ssvep_chen2017 import Chen2017SingleFlicker
from moabb.datasets.ssvep_mamem import MAMEM1
from moabb.datasets.yang2025 import Yang2025


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


def test_mainsah_manifest_includes_hyphenated_path_components(tmp_path: Path):
    manifest = tmp_path / "SHA256SUMS.txt"
    paths = [
        "bigP3BCI-data/StudyQ/Q_01/SE001/Train/Grey-to-White/run-1.edf",
        "bigP3BCI-data/StudyQ/Q_01/SE001/Train/Grey-to-Color/run-2.edf",
        "bigP3BCI-data/StudyQ/Q_01/SE001/Train/ColorIntensification/run-3.edf",
    ]
    manifest.write_text("\n".join(f"deadbeef  {path}" for path in reversed(paths)))

    parsed = _parse_manifest(manifest)

    assert parsed["Q"]["Q_01"][1] == sorted(paths)


def test_chen2017_data_path_finds_prefixed_xdf_files(tmp_path: Path, monkeypatch):
    dataset = Chen2017SingleFlicker()
    xdf_file = (
        tmp_path
        / "MNE-chen2017singleflicker-data"
        / "3.raw_data"
        / "1.training_data"
        / "sub_1_1.xdf"
    )
    xdf_file.parent.mkdir(parents=True)
    xdf_file.touch()
    monkeypatch.setattr(
        "moabb.datasets.ssvep_chen2017.dl.get_dataset_path", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "moabb.datasets.ssvep_chen2017.dl.data_dl",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    paths = dataset.data_path(1)

    assert paths == {"mat": [], "xdf": [str(xdf_file)]}


def test_yang2025_recognizes_extracted_bdf_files(tmp_path: Path, monkeypatch):
    data_file = (
        tmp_path
        / "MNE-yang2025-data"
        / "WBCIC_SHU Motor Imagery dataset"
        / "sourcedata"
        / "2C dataset"
        / "sub-001"
        / "ses-01"
        / "eeg"
        / "data.bdf"
    )
    data_file.parent.mkdir(parents=True)
    data_file.touch()
    monkeypatch.setattr(
        "moabb.datasets.yang2025.dl.get_dataset_path", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "moabb.datasets.yang2025.dl.data_dl",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert Yang2025().data_path(1) == str(tmp_path / "MNE-yang2025-data")


def test_hefmi_ich_data_path_downloads_subject_files(tmp_path: Path, monkeypatch):
    dataset = HefmiIch2025()
    monkeypatch.setattr(
        "moabb.datasets.hefmi_ich2025.dl.get_dataset_path", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "moabb.datasets.hefmi_ich2025._MANIFEST", {1: [(42, "sub01_1_epo.mat")]}
    )

    def fake_download(url, sign, root, force_update, verbose):
        downloaded = tmp_path / "MNE-hefmiich2025-data" / "42"
        downloaded.write_bytes(b"data")
        return downloaded

    monkeypatch.setattr("moabb.datasets.hefmi_ich2025.dl.data_dl", fake_download)

    assert dataset.data_path(1) == [
        str(tmp_path / "MNE-hefmiich2025-data" / "sub01_1_epo.mat")
    ]


def test_bi2015b_data_path_returns_existing_mat_paths(tmp_path: Path, monkeypatch):
    group_dir = tmp_path / "zenodo" / "3268762" / "group_01"
    group_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "moabb.datasets.braininvaders.dl.data_dl",
        lambda *args, **kwargs: str(tmp_path / "zenodo" / "3268762" / "group_01_mat.zip"),
    )

    paths = BI2015b().data_path(1)

    assert all(path.endswith(".mat") for path in paths)


def test_kojima2024_removes_invalid_cached_manifest(tmp_path: Path, monkeypatch):
    dataset = Kojima2024A()
    manifest_path = tmp_path / f"MNE-{dataset.code}-data" / "kojima2024_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("<html>upstream error</html>")
    monkeypatch.setattr(
        "moabb.datasets.kojima2024a.dl.download_if_missing", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError, match="invalid and has been removed"):
        dataset.download_by_subject(1, tmp_path)

    assert not manifest_path.exists()


def _recording_data_dl(calls):
    """Stand-in for ``data_dl`` that records every actual download.

    It reproduces the caching contract of the real function: the file is
    written to the location derived from the URL and reused on later calls.
    """

    def data_dl(url, sign, path=None, force_update=False, verbose=None):
        root = Path(_dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        destination = _dl._sanitize_path(_dl._normalize_destination(url, root))
        if destination.is_file() and not force_update:
            return str(destination)
        calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"edf")
        return str(destination)

    return data_dl


def test_schirrmeister2017_does_not_redownload(tmp_path: Path, monkeypatch):
    """``data_path`` must fetch each recording only once (issue #851)."""
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    calls = []
    monkeypatch.setattr(schirrmeister2017.dl, "data_dl", _recording_data_dl(calls))

    dataset = Schirrmeister2017()
    first = dataset.data_path(1, path=str(tmp_path))
    assert len(calls) == 2

    for _ in range(2):
        assert dataset.data_path(1, path=str(tmp_path)) == first
    assert len(calls) == 2, "the recordings were downloaded more than once"

    # A single copy of each recording, not one per storage layout.
    assert len(list(tmp_path.rglob("1.edf"))) == 2


def test_schirrmeister2017_reuses_relocated_files(tmp_path: Path, monkeypatch):
    """A copy left by the old layout is reused instead of downloaded again."""
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    dataset_folder = tmp_path / "MNE-schirrmeister2017-data"
    relocated = []
    for split in ("train", "test"):
        path = dataset_folder / split / "1.edf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"edf")
        relocated.append(str(path))

    monkeypatch.setattr(
        schirrmeister2017.dl,
        "data_dl",
        lambda *args, **kwargs: pytest.fail("downloaded an already available file"),
    )

    dataset = Schirrmeister2017()
    assert dataset.data_path(1, path=str(tmp_path)) == relocated


def test_kaneshiro2015_valid_for_declared_paradigm():
    """Kaneshiro2015 is accepted by its declared paradigm (gh-1143): six object
    categories, no Target/NonTarget, so "imagery" routes it to n-class paradigms."""
    from moabb.paradigms import P300, Imagery, MotorImagery

    dataset = Kaneshiro2015()
    assert dataset.paradigm == "imagery"
    for paradigm in (MotorImagery(), Imagery(), MotorImagery(n_classes=6)):
        assert paradigm.is_valid(dataset)
        assert paradigm.used_events(dataset) == dataset.event_id
    # The old declaration was broken: P300 requires Target/NonTarget.
    assert not P300().is_valid(dataset)


def test_lenaig2026_data_path_accepts_wrapped_and_flat_layouts(tmp_path, monkeypatch):
    """data_path() assumed the extracted RAR always sits under an
    ``EEG_24Chan_AudioStim/`` wrapper directory, but the current Zenodo v2
    archive (record 21156618) extracts ``EXP1/``/``EXP2/`` directly at the
    root -- confirmed by direct inspection on Voyager. The loader must find
    the files either way, without touching events/labels."""
    import moabb.datasets.lenaig2026 as lenaig2026_mod

    def make_tree(root):
        for run in (1, 2):
            run_dir = root / "EXP1" / "S1" / f"R{run}"
            run_dir.mkdir(parents=True)
            (run_dir / f"01_R{run}.gdf").write_bytes(b"")

    # Flat layout: EXP1/ directly at the extraction root (current archive).
    flat_root = tmp_path / "flat" / "MNE-Lenaig2026-data"
    make_tree(flat_root)
    monkeypatch.setattr(
        lenaig2026_mod.dl, "get_dataset_path", lambda *a, **k: str(tmp_path / "flat")
    )
    paths = Lenaig2026(exp=1, run="both").data_path(1)
    assert [Path(p).name for p in paths] == ["01_R1.gdf", "01_R2.gdf"]
    assert all(Path(p).is_file() for p in paths)

    # Wrapped layout: EXP1/ under EEG_24Chan_AudioStim/ (what the loader
    # originally -- and exclusively -- assumed).
    wrapped_root = tmp_path / "wrapped" / "MNE-Lenaig2026-data"
    make_tree(wrapped_root / "EEG_24Chan_AudioStim")
    monkeypatch.setattr(
        lenaig2026_mod.dl, "get_dataset_path", lambda *a, **k: str(tmp_path / "wrapped")
    )
    paths2 = Lenaig2026(exp=1, run="both").data_path(1)
    assert [Path(p).name for p in paths2] == ["01_R1.gdf", "01_R2.gdf"]
    assert all(Path(p).is_file() for p in paths2)


def test_convert_to_bids_skips_subject_with_no_usable_data(tmp_path, monkeypatch, caplog):
    """gh: Schrag2026Pediatric subject 16 has a single game recording whose
    only run is dropped by the >10%% drift policy, leaving zero usable runs;
    _get_single_subject_data correctly raises FileNotFoundError for that one
    subject (a loader's own "nothing to write" signal). Before this fix,
    convert_to_bids's per-subject loop had no try/except around
    self.get_data(), so that single subject's FileNotFoundError aborted the
    whole multi-subject convert, losing every subject not yet processed.
    convert_to_bids must now skip that subject (with a warning) and keep
    converting the rest."""
    from moabb.datasets.base import BaseDataset

    dataset = FakeDataset(event_list=["fake1", "fake2"], n_sessions=1, n_subjects=3)
    real_get_data = BaseDataset.get_data

    def flaky_get_data(self, subjects=None, **kwargs):
        if subjects == [2]:
            raise FileNotFoundError("subject 2: no usable runs after drift policy")
        return real_get_data(self, subjects=subjects, **kwargs)

    monkeypatch.setattr(FakeDataset, "get_data", flaky_get_data)

    with caplog.at_level("WARNING"):
        bids_root = dataset.convert_to_bids(path=tmp_path, subjects=[1, 2, 3])

    assert any("skipping subject" in record.message for record in caplog.records), (
        "a warning naming the skipped subject must be logged"
    )
    subjects_found = {f.parent.parent.parent.name for f in bids_root.rglob("*.edf")}
    assert subjects_found == {"sub-1", "sub-3"}, "subject 2 must be skipped, not abort"


def test_lenaig2026_trials_per_class_builds_a_readme(tmp_path):
    """gh: METADATA.experiment.trials_per_class was a bare int (``10``),
    violating the schema's declared ``Dict[str, int]`` type. A real
    end-to-end convert crashed in ``bids_interface._build_readme`` with
    ``AttributeError: 'int' object has no attribute 'items'`` because
    ``_format_dict`` assumes a mapping. It is now a per-class dict; the
    README builder must run without crashing."""
    from moabb.datasets.bids_interface import _build_readme

    dataset = Lenaig2026(exp=1, run="both")
    assert isinstance(dataset.METADATA.experiment.trials_per_class, dict)
    readme = _build_readme(dataset)
    assert "Trials per class" in readme


def test_schrag2026_skips_run_with_zero_annotations_after_high_drift(
    tmp_path, monkeypatch, caplog
):
    """gh: subject 1's personal-stimulus game run has >10%% Trial/CSV drift,
    so ``_load_game_run``'s documented policy drops every label, returning a
    zero-annotation Raw. ``bids_interface._write_file`` hard-requires every
    Raw to carry annotations, so writing that Raw crashes the whole convert.
    ``_get_single_subject_data`` must skip (not return) a zero-annotation
    run, with a warning naming the subject and run, instead of crashing
    downstream."""
    import mne
    import numpy as np

    from moabb.datasets import schrag2026
    from moabb.datasets.schrag2026 import Schrag2026Pediatric

    eeg_dir = tmp_path / "P001" / "EEG"
    eeg_dir.mkdir(parents=True)
    std_name = "sub-P001_ses-S001_task-T2_acq-BW_M1_run-001_eeg.xdf"
    pers_name = "sub-P001_ses-S001_task-T3_acq-C3S1_M2_run-001_eeg.xdf"
    (eeg_dir / std_name).write_bytes(b"")
    (eeg_dir / pers_name).write_bytes(b"")

    info = mne.create_info(["Fz"], 256.0, "eeg")

    def fake_load_game_run(path):
        raw = mne.io.RawArray(np.zeros((1, 10)), info, verbose=False)
        if path.name == pers_name:
            return raw  # the >10%% drift run: zero annotations
        raw.set_annotations(
            mne.Annotations(onset=[0.0], duration=[5.0], description=["6.25"])
        )
        return raw

    monkeypatch.setattr(schrag2026, "_load_game_run", fake_load_game_run)
    dataset = Schrag2026Pediatric()
    monkeypatch.setattr(dataset, "data_path", lambda *a, **k: str(tmp_path / "P001"))

    with caplog.at_level("WARNING"):
        session = dataset._get_single_subject_data(1)

    runs = session["0"]
    assert set(runs) == {"0standard"}, "the zero-annotation run must be skipped"
    assert len(runs["0standard"].annotations) == 1
    assert any(
        "zero labelled trials" in record.message and "1personal" in record.message
        for record in caplog.records
    ), "a warning naming the subject/run must be logged"
