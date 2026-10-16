"""Offline tests for the Peters2026 (BCI-FIT, OpenNeuro ds007720) loader.

A synthetic BIDS tree with the release's layout (BrainVision files, per-file
``value`` codes, every flash listed twice, inquiry-preview and key-press rows)
replaces the download; transport tests mock ``data_dl`` and ``nemar.download``.
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, write_raw_bids

import moabb.datasets.peters2026 as peters2026
from moabb.datasets import Peters2026
from moabb.datasets.base import is_abbrev, is_camel_kebab_case
from moabb.paradigms import P300


# The synthetic recordings carry no annotations for write_raw_bids.
pytestmark = pytest.mark.filterwarnings("ignore:No events found or provided")

SFREQ = 300.0
N_TIMES = 6000  # 20 s
# Synthetic inventory: subject 1 visit 1 on the DSI-Flex, visit 2 on the
# DSI-24; subject 3 only on the DSI-24.
INVENTORY = {
    1: {
        1: (("MatrixCalibration", 1), ("RSVPCalibration", 2)),
        2: (("RSVPCalibration", 4), ("MatrixCalibration", 3)),
    },
    3: {1: (("MatrixCalibration", 1),)},
}
DSI24 = (
    peters2026._DSI24_EEG[:7] + ["A2"] + peters2026._DSI24_EEG[7:]
)  # file order varies
FLEX = ["Cz", "Oz", "P4", "P3", "PO8", "Pz", "PO7"]


def _flash_samples(offset):
    # First flash at sample 0 and last one ending on the final sample, so
    # both edge epochs of the [0, 0.8] s window must survive.
    middle = [600 + offset + 70 * k for k in range(14)]
    return [0, *middle, N_TIMES - 1 - int(0.8 * SFREQ)]


def _write_run(root, visit, task, run, ch_names, targets, subject=1):
    info = mne.create_info(ch_names, SFREQ, "eeg")
    data = np.random.default_rng(run).standard_normal((len(ch_names), N_TIMES)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_meas_date(datetime(2024, 1, 1, tzinfo=timezone.utc))
    bids_path = BIDSPath(
        subject=f"P{subject}",
        session=str(visit),
        task=task,
        run=str(run),
        datatype="eeg",
        root=root,
    )
    write_raw_bids(
        raw,
        bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose=False,
    )
    samples = _flash_samples(run)
    rows = [(s / SFREQ, 0.5, "preview", 33, s) for s in [samples[1] - 300]]
    rows += [(s / SFREQ, 0.5, "bcipy_key_press_space", 31, s) for s in [samples[1] - 150]]
    for k, s in enumerate(samples):
        label = "target" if k in targets else "nontarget"
        # ``value`` codes differ between files in the release.
        rows.append((s / SFREQ, 0.5, label, 29 + run + (label == "target"), s))
        rows.append((s / SFREQ, 0.5, "ABCDEFGHIJKLMNOPQRSTUVWXYZ<_"[k], k + 1, s))
    events = pd.DataFrame(
        rows, columns=["onset", "duration", "trial_type", "value", "sample"]
    )
    events.to_csv(
        bids_path.copy().update(suffix="events", extension=".tsv").fpath,
        sep="\t",
        index=False,
    )
    return {"targets": sorted(targets), "samples": samples}


@pytest.fixture
def bids_root(tmp_path, monkeypatch):
    monkeypatch.setattr(peters2026, "_CALIBRATION_RUNS", INVENTORY)
    root = tmp_path / "ds007720"
    truth = {
        (1, 1): _write_run(root, 1, "MatrixCalibration", 1, FLEX, {3}),
        (1, 2): _write_run(root, 1, "RSVPCalibration", 2, FLEX, {0, 5}),
        (2, 3): _write_run(root, 2, "MatrixCalibration", 3, DSI24, {15}),
        (2, 4): _write_run(root, 2, "RSVPCalibration", 4, DSI24, set()),
    }
    _write_run(root, 1, "MatrixCalibration", 1, DSI24, {2, 9}, subject=3)
    monkeypatch.setattr(
        Peters2026, "_download_subject", lambda self, *args, **kwargs: str(root)
    )
    return root, truth


def _labels(raw):
    return list(raw.annotations.description)


def test_sessions_runs_events_and_common_channels(bids_root):
    _, truth = bids_root
    data = Peters2026().get_data([1])[1]
    assert data.keys() == {"0", "1"}
    # Runs keep a unique index (BIDS run order) and the layout.
    assert list(data["0"]) == ["0matrix", "1rsvp"]
    assert list(data["1"]) == ["0matrix", "1rsvp"]
    for session, run, bids_run in [
        ("0", "0matrix", 1),
        ("0", "1rsvp", 2),
        ("1", "0matrix", 3),
        ("1", "1rsvp", 4),
    ]:
        raw = data[session][run]
        expected = truth[(int(session) + 1, bids_run)]
        # Mixed caps: only the four channels both caps share, in one order.
        assert raw.ch_names == ["Cz", "P3", "Pz", "P4"]
        assert raw.info["sfreq"] == SFREQ
        labels = _labels(raw)
        assert len(labels) == len(expected["samples"])  # letters/preview dropped
        assert [i for i, x in enumerate(labels) if x == "Target"] == expected["targets"]
        assert set(labels) <= {"Target", "NonTarget"}
        events, _ = mne.events_from_annotations(raw, verbose=False)
        assert events[:, 0].tolist() == expected["samples"]


def test_cap_variants_select_visits_and_full_montages(bids_root):
    dsi24 = Peters2026(cap="DSI-24")
    assert dsi24.code == "Peters-dsi24-2026"
    data = dsi24.get_data([1])[1]
    assert data.keys() == {"1"}
    raw = data["1"]["0matrix"]
    assert raw.ch_names == peters2026._DSI24_EEG + ["A2"]
    assert raw.get_channel_types(picks=["A2"]) == ["misc"]
    assert raw.get_channel_types(picks=peters2026._DSI24_EEG) == ["eeg"] * 19

    flex = Peters2026(cap="DSI-Flex")
    assert flex.code == "Peters-dsiflex-2026"
    data = flex.get_data([1])[1]
    assert data.keys() == {"0"}
    assert data["0"]["1rsvp"].ch_names == ["Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]


def test_p300_epochs_keep_first_and_last_flash(bids_root):
    _, truth = bids_root
    # Evaluations load every subject into one array: a DSI-24-only subject
    # must come back with the same four channels as a subject that switched.
    X, y, meta = P300().get_data(Peters2026(), [1, 3])
    n_flashes = sum(len(t["samples"]) for t in truth.values()) + 16
    n_targets = sum(len(t["targets"]) for t in truth.values()) + 2
    assert X.shape == (n_flashes, 4, int(0.8 * SFREQ) + 1)
    assert (y == "Target").sum() == n_targets
    assert set(meta["run"]) == {"0matrix", "1rsvp"}
    sizes = meta.groupby(["subject", "session"]).size().to_dict()
    assert sizes == {(1, "0"): 32, (1, "1"): 32, (3, "0"): 16}

    X24, _, meta24 = P300().get_data(Peters2026(cap="DSI-24"), [1, 3])
    assert X24.shape[1] == 19
    assert meta24.groupby("subject")["session"].unique().apply(list).to_dict() == {
        1: ["1"],
        3: ["0"],
    }


def test_additional_metadata_lists_character_and_layout(bids_root):
    ds = Peters2026()
    meta = ds.get_additional_metadata(subject=1, session="1", run="0matrix")
    assert len(meta) == 16
    assert meta["trial_type"].iloc[15] == "Target"
    assert list(meta["character"].iloc[:3]) == ["A", "B", "C"]
    assert set(meta["layout"]) == {"matrix"}
    assert set(meta["bids_run"]) == {3}
    assert set(meta["cap"]) == {"DSI-24"}
    flex_meta = ds.get_additional_metadata(subject=1, session="0", run="1rsvp")
    assert set(flex_meta["cap"]) == {"DSI-Flex"}
    assert set(flex_meta["task"]) == {"RSVPCalibration"}


def test_missing_calibration_run_is_an_error(bids_root):
    root, _ = bids_root
    for f in root.glob("sub-P1/ses-2/eeg/*run-4_eeg.*"):
        f.unlink()
    with pytest.raises(FileNotFoundError, match=r"\(2, 'RSVPCalibration', 4\)"):
        Peters2026().get_data([1])


def test_release_inventory():
    runs = peters2026._CALIBRATION_RUNS
    assert sorted(runs) == [1, 2, 3, 4, 5]
    assert [len(runs[s]) for s in sorted(runs)] == [9, 10, 9, 6, 6]
    tasks = [task for visits in runs.values() for r in visits.values() for task, _ in r]
    assert len(tasks) == 88
    assert tasks.count("MatrixCalibration") == 70
    assert Peters2026().subject_list == [1, 2, 3, 4, 5]
    assert Peters2026(cap="DSI-24").subject_list == [1, 2, 3, 4, 5]
    assert Peters2026(cap="DSI-Flex").subject_list == [1, 2]
    with pytest.raises(ValueError, match="cap must be"):
        Peters2026(cap="DSI-7")


@pytest.mark.parametrize(
    "cap, code",
    [
        (None, "Peters2026"),
        ("DSI-24", "Peters-dsi24-2026"),
        ("DSI-Flex", "Peters-dsiflex-2026"),
    ],
)
def test_variant_codes_follow_moabb_naming(cap, code):
    # One code per cap variant, so cached epochs and results never mix.
    assert Peters2026(cap=cap).code == code
    assert is_camel_kebab_case(code)
    assert is_abbrev("Peters2026", code)


@pytest.mark.parametrize("cap", [None, "DSI-Flex"])
def test_upstream_download_fetches_only_calibration_files(cap, tmp_path, monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        peters2026,
        "data_dl",
        lambda url, sign, fname=None, **kw: calls.append((url, sign, fname)),
    )
    Peters2026(cap=cap).download([1], path=str(tmp_path))
    root = Peters2026(cap=cap)._download_subject(1, str(tmp_path), False, None, None)
    assert root == str(tmp_path / "MNE-peters2026-data")
    calls = calls[: len(calls) // 2]
    fnames = [fname for _, _, fname in calls]
    assert {sign for _, sign, _ in calls} == {"Peters2026"}
    assert all(url.endswith(fname) for url, _, fname in calls)
    assert all(url.startswith(peters2026._S3_BASE + "/") for url, _, _ in calls)
    assert fnames[:3] == list(peters2026._ROOT_FILES)
    assert len(fnames) == 3 + 19 * 7  # all 19 calibration runs of sub-P1
    assert not [f for f in fnames if "CopyPhrase" in f]
    assert "sub-P1/ses-8/eeg/sub-P1_ses-8_task-MatrixCalibration_run-2_eeg.eeg" in fnames


@pytest.mark.parametrize("cap", [None, "DSI-24"])
def test_nemar_download_filters_calibration_tasks(cap, tmp_path, monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    transport = Mock()
    monkeypatch.setattr("moabb.datasets.download.nemar.download", transport)
    ds = Peters2026(cap=cap)
    ds.download([3], tmp_path, True, False, verbose="ERROR")
    transport.assert_called_once_with(
        dataset="on007720",
        # One folder for every cap variant.
        target_dir=tmp_path / "MNE-peters2026-data" / "on007720",
        subject="P3",
        trust_existing=False,
        scope="raw",
        datatype="eeg",
        task=["MatrixCalibration", "RSVPCalibration"],
    )
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    ds._prefetch_nemar_sourcedata([3])
