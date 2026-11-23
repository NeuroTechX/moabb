"""Synthetic, no-download tests for :class:`moabb.datasets.Dzianok2022`."""

from unittest.mock import Mock

import mne
import numpy as np

from moabb.datasets import Dzianok2022


def _write_recording(tmp_path):
    """BrainVision file with the deposit's markers and an EEGLAB join at 7 s."""
    raw = mne.io.RawArray(
        np.zeros((3, 10000)),
        mne.create_info(["Pz", "Cz", "TP9 LEFT EAR"], 1000.0, "eeg"),
        verbose=False,
    )
    codes = ["S 12", "S  5", "S  6", "S  1", "S  7", "S  5", "S 10"]
    raw.set_annotations(
        mne.Annotations(
            np.arange(1, 8, dtype=float), 0.001, [f"Stimulus/{c}" for c in codes]
        )
    )
    vhdr = tmp_path / "sub-01_task-oddball_eeg.vhdr"
    mne.export.export_raw(vhdr, raw, fmt="brainvision", verbose=False)
    # pybv cannot write these marker types; append them as the deposit has them.
    with open(vhdr.with_suffix(".vmrk"), "a") as vmrk:
        vmrk.write("Mk8=New Segment,,1,0,0,0\n")
        vmrk.write("Mk9=New Segment,,7501,1,0,0\n")
        vmrk.write("Mk10=Stimulus,boundary,7501,NaN,0,0\n")
    return str(vhdr)


def test_markers_become_target_nontarget_and_joins_edges(tmp_path, monkeypatch):
    ds = Dzianok2022()
    monkeypatch.setattr(ds, "data_path", Mock(return_value=[_write_recording(tmp_path)]))

    raw = ds._get_single_subject_data(1)["0"]["0"]

    events = mne.find_events(raw, "STIM", shortest_event=1, verbose=False)
    # standard (S 5) and distractor (S 7) are NonTarget = 1, target (S 6) is 2;
    # the key press, break and run-start markers are not trials.
    assert events[:, 0].tolist() == [2000, 3000, 5000, 6000]
    assert events[:, 2].tolist() == [1, 2, 1, 1]
    edges = raw.annotations[raw.annotations.description == "EDGE boundary"]
    assert edges.onset.tolist() == [7.5]
    assert np.isnan(raw.info["chs"][raw.ch_names.index("TP9 LEFT EAR")]["loc"][0])
    assert not np.isnan(raw.info["chs"][raw.ch_names.index("Pz")]["loc"][0])


def test_mirror_downloads_only_the_oddball_task(monkeypatch, tmp_path):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    transport = Mock()
    monkeypatch.setattr("moabb.datasets.download.nemar.download", transport)
    ds = Dzianok2022()

    path = ds.data_path(7, tmp_path, True, None, "ERROR")

    root = tmp_path / "MNE-dzianok2022-data" / "on004621"
    transport.assert_called_once_with(
        dataset="on004621",
        target_dir=root,
        subject="07",
        trust_existing=False,
        scope="raw",
        datatype="eeg",
        task="oddball",
    )
    assert path == [str(root / "sub-07/eeg/sub-07_task-oddball_eeg.vhdr")]
