"""Offline signal, event and transport contracts for the archive B loaders."""

import bz2
import importlib
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from moabb.datasets import (
    Leeuwis2021,
    Li2026,
    MartinezPeon2024,
    OpenViBE,
    PardoGarcia2026,
    li2026,
    openvibe,
)
from moabb.datasets.leeuwis2021 import LEEUWIS2021_EEG_CHANNELS
from moabb.datasets.preprocessing import SetRawAnnotations


def _capture_transport(monkeypatch, module, local_path_for):
    """Replace ``module.dl.data_dl`` with a recorder returning ``local_path_for(url)``."""
    calls = []

    def transport(url, sign, path=None, force_update=False, verbose=None):
        calls.append((path, force_update, verbose))
        return str(local_path_for(url))

    monkeypatch.setattr(module.dl, "data_dl", transport)
    return calls


@pytest.mark.parametrize(
    "cls,count",
    [(Leeuwis2021, 4), (MartinezPeon2024, 6), (OpenViBE, 1), (PardoGarcia2026, 6)],
)
def test_transport_flags(cls, count, tmp_path, monkeypatch):
    module = importlib.import_module(cls.__module__)
    calls = _capture_transport(
        monkeypatch, module, lambda url: tmp_path / url.rsplit("/", 1)[-1]
    )
    dataset = cls()
    dataset.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert calls == [(tmp_path, True, "ERROR")] * count
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(0)


def test_li2026_archive_transport_and_missing_task(tmp_path, monkeypatch):
    archive = tmp_path / "MI_A_Dataset.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for task in li2026._TASKS:
            for subject in ("01", "50", "100", "150", "200"):
                stream.writestr(
                    f"MI_A_Dataset/MI_A_Dataset/Raw_data/{task}/Sub_{subject}.cdt", b""
                )
    calls = _capture_transport(monkeypatch, li2026, lambda url: archive)
    paths = Li2026().data_path(5, path=tmp_path, force_update=True, verbose="ERROR")
    assert len(paths) == 5
    assert all(path.endswith("Sub_200.cdt") for path in paths)
    assert calls == [(tmp_path, True, "ERROR")]

    Path(paths[-1]).unlink()
    with pytest.raises(FileNotFoundError, match="Expected at least"):
        Li2026().data_path(5)


def test_leeuwis_first_last_si_and_discontinuities(tmp_path):
    frame = pd.DataFrame(
        {ch: np.r_[np.zeros(2000), np.ones(2000) * 20] for ch in LEEUWIS2021_EEG_CHANNELS}
    )
    frame["trial"] = np.repeat([1, 2], 2000)
    frame["class"] = np.repeat([-1, 1], 2000)
    frame["TimeStamp"] = np.tile(np.arange(2000) / 250 - 3, 2)
    path = tmp_path / "run.csv"
    frame.to_csv(path, index=False)
    dataset = Leeuwis2021()
    raw = dataset._csv_to_raw(path)
    np.testing.assert_allclose(raw.get_data(picks="eeg")[:, -1], 20e-6)
    transformed = SetRawAnnotations(dataset.event_id, dataset.interval).transform(raw)
    assert "EDGE boundary" in transformed.annotations.description
    events, _ = mne.events_from_annotations(transformed, event_id=dataset.event_id)
    assert events[:, 0].tolist() == [750, 2750]
    epochs = mne.Epochs(
        transformed,
        events,
        dataset.event_id,
        tmin=0,
        tmax=dataset.interval[1],
        baseline=None,
        preload=True,
    )
    assert len(epochs) == 2
    assert epochs.get_data().shape[-1] == 1250


def test_martinez_units_and_first_last_cue(tmp_path):
    path = tmp_path / "record.txt"
    np.savetxt(path, np.full((5120, 20), 15.0))
    raw = MartinezPeon2024()._read_run(path, "level_70")
    np.testing.assert_allclose(raw.get_data(), 15e-6)
    np.testing.assert_allclose(raw.annotations.onset, [2.9, 10.9, 18.9, 26.9, 34.9])
    assert raw.annotations.description.tolist() == ["level_70"] * 5


@pytest.mark.parametrize("reference", ["Ref_Nose", "Nz"])
def test_openvibe_reference_header_variants(reference, tmp_path, monkeypatch):
    """Records 01-04 name the nasion ``Ref_Nose``, records 05-14 ``Nz``."""
    columns = [reference if ch == "Nz" else ch for ch in openvibe._CHANNELS]
    frame = pd.DataFrame(
        {column: np.arange(3, dtype=float) for column in columns}
        | {"Event Id": [np.nan, str(openvibe.CODE_LEFT), str(openvibe.CODE_RIGHT)]}
    )
    path = tmp_path / "signal.csv.bz2"
    with bz2.open(path, "wt") as fout:
        frame.to_csv(fout, index=False)

    dataset = OpenViBE()
    monkeypatch.setattr(dataset, "data_path", lambda subject: [str(path)])
    raw = dataset._get_single_subject_data(5)["0"]["0"]

    assert raw.ch_names == openvibe._CHANNELS
    np.testing.assert_allclose(raw.get_data()[2], np.arange(3) * 1e-6)
    assert list(raw.annotations.description) == ["left_hand", "right_hand"]


def test_li2026_legacy_curry_fallback_reads_float32_sidecars(tmp_path):
    cdt = tmp_path / "recording.cdt"
    np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4").tofile(cdt)
    cdt.with_suffix(".cdt.dpa").write_text(
        """NumSamples = 2\nNumChannels = 3\nSampleFreqHz = 1000\n
LABELS START_LIST
Cz
C3
C4
LABELS END_LIST
LABELS_OTHERS START_LIST
LABELS_OTHERS END_LIST
""",
        encoding="utf-8",
    )
    cdt.with_suffix(".cdt.ceo").write_text(
        """NUMBER_LIST START_LIST
0 0 1 -1
1 0 2 -1
NUMBER_LIST END_LIST
""",
        encoding="utf-8",
    )

    raw = Li2026._read_legacy_curry(cdt)

    assert raw.ch_names == ["Cz", "C3", "C4"]
    np.testing.assert_allclose(raw.get_data()[:, 0], [1e-6, 2e-6, 3e-6])
    assert raw.annotations.description.tolist() == ["1", "2"]
    np.testing.assert_allclose(raw.annotations.onset, [0.0, 0.001])


@pytest.mark.parametrize("subject,expected", [(1, ["0pre", "1post"]), (2, ["0pre"])])
def test_pardo_missing_post_is_not_duplicated(subject, expected, monkeypatch):
    dataset = PardoGarcia2026()
    monkeypatch.setattr(
        dataset, "data_path", lambda subject: [str(i) for i in range(len(expected))]
    )
    monkeypatch.setattr(dataset, "_load_raw", lambda path: path)
    sessions = dataset._get_single_subject_data(subject)
    assert list(sessions) == expected
    assert [session["0"] for session in sessions.values()] == [
        Path(str(i)) for i in range(len(expected))
    ]


def test_pardo_stale_marker_reference_uses_unambiguous_sibling(tmp_path, monkeypatch):
    vhdr = tmp_path / "PAC03-POST.vhdr"
    original = "[Common Infos]\nDataFile=PAC03-POST.eeg\nMarkerFile=PAC02-POST.vmrk\n"
    vhdr.write_text(original, encoding="utf-8")
    vhdr.with_suffix(".eeg").touch()
    vhdr.with_suffix(".vmrk").touch()
    seen = {}

    def fake_reader(path, *, preload, verbose):
        temporary = Path(path)
        seen["path"] = temporary
        seen["header"] = temporary.read_text(encoding="utf-8")
        return object()

    monkeypatch.setattr(
        "moabb.datasets.pardogarcia2026.mne.io.read_raw_brainvision", fake_reader
    )

    assert PardoGarcia2026._read_brainvision(vhdr) is not None
    assert seen["path"] != vhdr
    assert "MarkerFile=PAC03-POST.vmrk" in seen["header"]
    assert vhdr.read_text(encoding="utf-8") == original
    assert not seen["path"].exists()
