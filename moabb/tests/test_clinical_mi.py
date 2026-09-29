"""Synthetic clinical/graded-force source-format checks."""

import json
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from moabb.datasets import KMIHandGrip2025, MILimbEEG
from moabb.datasets.preprocessing import SetRawAnnotations


@pytest.mark.parametrize("indexed", [True, False])
def test_milimb_si_and_last_window(tmp_path, indexed):
    ds = MILimbEEG()
    for code, value in [(2, 12), (3, -8)]:
        values = np.full((500, 16), value)
        if indexed:
            values = np.column_stack([np.arange(500), values])
        pd.DataFrame(values).to_csv(
            tmp_path / f"S1R1I{code}_1.csv", index=False, header=False
        )
    # Executed movement files must never enter the imagery set.
    (tmp_path / "S1R1M2_1.csv").write_text("not imagery")
    with patch.object(ds, "data_path", return_value=[str(tmp_path)]):
        raw = ds._get_single_subject_data(1)["0"]["0"]
    assert raw.n_times == 1000
    np.testing.assert_allclose(raw.get_data(picks="eeg")[:, 0], 12e-6)
    np.testing.assert_allclose(raw.get_data(picks="eeg")[:, -1], -8e-6)
    raw = SetRawAnnotations(ds.event_id, ds.interval).transform(raw)
    events = mne.find_events(raw, initial_event=True, verbose=False)
    np.testing.assert_array_equal(events[:, 0], [0, 500])
    epochs = mne.Epochs(
        raw,
        events,
        {"clh": 2, "crh": 3},
        tmin=0,
        tmax=ds.interval[1],
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert epochs.get_data().shape == (2, 17, 500)
    # The non-rejecting boundary must sit exactly at the stored-trial join.
    edges = raw.annotations[raw.annotations.description == "EDGE boundary"]
    np.testing.assert_allclose(edges.onset, [500 / 125])


def test_milimb_rejects_short_trial(tmp_path):
    path = tmp_path / "short.csv"
    pd.DataFrame(np.zeros((499, 17))).to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="four-second"):
        MILimbEEG()._read_trial(path)


def test_kmi_si_and_protocol(tmp_path):
    path = tmp_path / "grip.csv"
    pd.DataFrame({"C3": np.full(42000, 25), "C4": np.full(42000, -12)}).to_csv(
        path, index=False
    )
    ds = KMIHandGrip2025()
    raw = ds._read_run(path, 1)
    np.testing.assert_allclose(raw.get_data()[0], 25e-6)
    np.testing.assert_allclose(raw.annotations.onset, np.arange(4, 84, 8))
    events, _ = mne.events_from_annotations(raw, event_id=ds.event_id, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        {"grip_10": 1},
        tmin=0,
        tmax=ds.interval[1],
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert epochs.get_data().shape == (10, 2, 2000)


def test_kmi_metadata_and_signal_transport_flags(tmp_path):
    path = tmp_path / "record.json"
    path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "filename": f"S01_{level}_KMI.csv",
                        "content_details": {
                            "download_url": f"https://example.test/{level}"
                        },
                    }
                    for level in [10, 40, 70, 100]
                ]
            }
        )
    )
    with patch(
        "moabb.datasets.kmi_handgrip2025.dl.data_dl",
        side_effect=[str(path)] + ["synthetic"] * 4,
    ) as download:
        paths = KMIHandGrip2025().data_path(
            1, path="custom", force_update=True, verbose=False
        )
    assert len(paths) == 4
    assert len(download.call_args_list) == 5
    for call in download.call_args_list:
        assert call.kwargs == {"path": "custom", "force_update": True, "verbose": False}


def test_milimb_transport_flags(tmp_path):
    import zipfile

    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("S1/synthetic.csv", "fixture")
    with patch(
        "moabb.datasets.milimbeeg.dl.data_dl", return_value=str(archive_path)
    ) as download:
        assert MILimbEEG().data_path(
            1, path="custom", force_update=True, verbose=False
        ) == [str(tmp_path / "MILimbEEG" / "S1")]
    assert download.call_args.kwargs == {
        "path": "custom",
        "force_update": True,
        "verbose": False,
    }
