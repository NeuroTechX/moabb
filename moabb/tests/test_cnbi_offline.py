"""Synthetic-only CNBI transport, annotation and participant regressions."""

from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import Perdikis2018, SpinalStim2025
from moabb.datasets import download as dl
from moabb.datasets import perdikis2018 as per
from moabb.datasets import spinalstim2025 as spinal


def test_spinal_exact_roots_deduplicate_subject_identity(tmp_path, monkeypatch):
    root = tmp_path / "MNE-spinalstim2025-data" / "d3_SinglePulse_n5"
    # The following subject's nested folder mistakenly uses the previous token.
    for owner in (501, 502):
        folder = root / f"Subject_{owner}_REST_Offline" / "Subject_501_session"
        folder.mkdir(parents=True)
        (folder / "run.gdf").touch()
    monkeypatch.setattr(dl, "get_dataset_path", Mock(return_value=str(tmp_path)))
    data_dl = Mock()
    monkeypatch.setattr(dl, "data_dl", data_dl)
    paths = SpinalStim2025().data_path(21)
    assert len(paths) == 1
    assert "Subject_501_REST_Offline" in paths[0]
    assert set(paths).isdisjoint(SpinalStim2025().data_path(22))
    with pytest.raises(FileNotFoundError, match="No offline roots"):
        SpinalStim2025().data_path(23)
    data_dl.assert_not_called()  # already extracted


@pytest.mark.parametrize("dataset", [Perdikis2018(), SpinalStim2025()])
def test_invalid_subject_before_transport(dataset):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(0)


def test_spinal_units_bad_flags_first_last_events_channel_order(monkeypatch, tmp_path):
    # d3/d4 GDFs store O2 before OZ; the reader restores the documented order.
    names = [*spinal._EEG_CHANNELS[:-2], "O2", "OZ", *spinal._AUX_CHANNELS, "trigger"]
    raw = mne.io.RawArray(
        np.full((len(names), 1024), 12.0),
        mne.create_info(names, 512, ["eeg"] * 32 + ["misc"] * 3 + ["stim"]),
        verbose=False,
    )
    raw.info["bads"] = ["C3"]
    raw.set_annotations(mne.Annotations([0, 1023 / 512], [0, 0], ["769", "770"]))
    monkeypatch.setattr(mne.io, "read_raw_gdf", Mock(return_value=raw))
    loaded = SpinalStim2025._read_run(tmp_path / "run.gdf")
    np.testing.assert_allclose(loaded.get_data(), 12e-6)
    assert loaded.info["bads"] == ["C3"]
    events, _ = mne.events_from_annotations(
        loaded, event_id={"left_hand": 1, "right_hand": 2}
    )
    assert events[:, 0].tolist() == [0, 1023]
    assert loaded.ch_names == [*spinal._EEG_CHANNELS, *spinal._AUX_CHANNELS]
    assert loaded.get_channel_types() == ["eeg"] * 32 + ["eog"] * 3


def test_spinal_missing_eeg_channel_raises_useful_error(tmp_path, monkeypatch):
    channels = spinal._EEG_CHANNELS[:-1]
    raw = mne.io.RawArray(
        np.zeros((len(channels), 16)),
        mne.create_info(channels, sfreq=512, ch_types="eeg"),
        verbose=False,
    )
    monkeypatch.setattr(mne.io, "read_raw_gdf", Mock(return_value=raw))
    with pytest.raises(ValueError, match="missing required EEG channels.*O2"):
        SpinalStim2025._read_run(tmp_path / "incomplete.gdf")


@pytest.mark.parametrize("codes", [["769", "771", "773"], ["771"]])
def test_perdikis_excludes_exploratory_and_single_class_runs(codes, monkeypatch):
    raw = mne.io.RawArray(
        np.zeros((16, 4096)),
        mne.create_info([f"eeg:{i}" for i in range(1, 17)], 512, "eeg"),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(np.arange(len(codes)), np.zeros(len(codes)), codes)
    )
    monkeypatch.setattr(mne.io, "read_raw_gdf", Mock(return_value=raw))
    assert Perdikis2018()._read_calibration_run(Path("run.gdf"), None) is None


def test_perdikis_units_first_last_epoch_bad_channel_and_labels(monkeypatch):
    names = [f"eeg:{i}" for i in range(1, 17)] + ["trigger:1"]
    raw = mne.io.RawArray(
        np.full((17, 8193), 9.0),
        mne.create_info(names, 512, ["eeg"] * 16 + ["stim"]),
        verbose=False,
    )
    raw.info["bads"] = ["eeg:1"]
    raw.set_annotations(mne.Annotations([0, 11, 12], [0, 0, 0], ["771", "773", "783"]))
    monkeypatch.setattr(mne.io, "read_raw_gdf", Mock(return_value=raw))
    ds = Perdikis2018()
    loaded = ds._read_calibration_run(
        Path("run.gdf"), mne.channels.make_standard_montage("colin27_1005")
    )
    events, _ = mne.events_from_annotations(loaded, event_id=ds.event_id)
    epochs = mne.Epochs(
        loaded,
        events,
        event_id={"both_feet": 771, "both_hands": 773},
        tmin=1,
        tmax=5,
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert len(epochs) == 2
    assert loaded.info["bads"] == ["Fz"]
    assert loaded.ch_names == per._EEG_CHANNELS
    assert loaded.annotations.description.tolist() == ["both_feet", "both_hands", "rest"]
    np.testing.assert_allclose(loaded.get_data(), 9e-6)


@pytest.mark.parametrize("kind", ["perdikis", "spinal"])
def test_download_flags_and_force_update_reextracts_real_archive(
    kind, tmp_path, monkeypatch
):
    """Mock only transport; exercise actual archive extraction, refresh and caching."""
    import io
    import tarfile
    import zipfile

    if kind == "perdikis":
        dataset = Perdikis2018()
        archive = tmp_path / "MA25VE.tar.gz"
        relative = "MA25VE/run.offline.mi.test.gdf"
        destination = tmp_path / relative
        with tarfile.open(archive, "w:gz") as handle:
            member = tarfile.TarInfo(relative)
            member.size = 3
            handle.addfile(member, io.BytesIO(b"new"))
    else:
        dataset = SpinalStim2025()
        archive = tmp_path / "d1.zip"
        relative = "d1_Main_Group_n20/Subject_004_REST_Offline/run.gdf"
        destination = tmp_path / "MNE-spinalstim2025-data" / relative
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(relative, b"new")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    transport = Mock(return_value=archive)
    monkeypatch.setattr(per.dl, "data_dl", transport)
    dataset.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert transport.call_args.args[2:] == (tmp_path, True, "ERROR")
    assert destination.read_bytes() == b"new"
    # Once extracted, a plain call does not re-extract (SpinalStim2025 does not
    # even look the archive up again).
    destination.write_bytes(b"cached")
    dataset.data_path(1, path=tmp_path, verbose="ERROR")
    assert destination.read_bytes() == b"cached"
    assert transport.call_count == (2 if kind == "perdikis" else 1)
