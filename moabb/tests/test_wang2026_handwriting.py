"""Offline tests for the Wang2026Handwriting loader on a synthetic BIDS tree."""

import json

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from moabb.datasets import Wang2026Handwriting
from moabb.datasets.wang2026_handwriting import _SFREQ, WANG2026_CH_NAMES, WANG2026_EVENTS
from moabb.paradigms import MotorImagery


# Both tasks of one session, at their released trial counts (200 / 240) but
# with 0.5 s trials instead of 4 s so the synthetic recordings stay small.
_CCS_LABELS = [
    "horizontal('一')",
    "vertical('丨')",
    "left-falling ('丿')",
    "right-falling('㇏')",
    "turning('𠃌')",
]
_SV_LABELS = ["a(/a/)", "o(/o/)", "e(/e/)", "i(/i/)", "u(/u/)", "v(/ü/)"]


@pytest.fixture
def bids_root(tmp_path):
    """One subject with two sessions x two tasks, 40 trials per class."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Synthetic Wang2026Handwriting", "BIDSVersion": "1.1.1"})
    )
    (root / "participants.tsv").write_text(
        "participant_id\tgender\tage\tgroup\thandedness\nsub-01\tM\t28\tnovice\tR\n"
    )
    # 20 s of signal: long enough for the paradigm's 8-32 Hz filter and the
    # resampling to 100 Hz to run on the whole recording.
    signal = 20e-6 * np.sin(2 * np.pi * 10 * np.arange(20 * int(_SFREQ)) / _SFREQ)
    for session in (1, 2):
        for task, labels in (("CCSHI", _CCS_LABELS), ("SVHI", _SV_LABELS)):
            folder = root / "sub-01" / f"ses-{session:02d}" / "eeg"
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"sub-01_ses-{session:02d}_task-{task}"
            raw = mne.io.RawArray(
                np.tile(signal, (len(WANG2026_CH_NAMES), 1)),
                mne.create_info(WANG2026_CH_NAMES, _SFREQ, "eeg"),
                verbose=False,
            )
            mne.export.export_raw(folder / f"{stem}_eeg.bdf", raw, verbose=False)
            rows = "".join(f"{ch}\tEEG\n" for ch in WANG2026_CH_NAMES)
            (folder / f"{stem}_channels.tsv").write_text("name\ttype\n" + rows)
            onsets = np.arange(len(labels)) * 2.0
            body = "".join(
                f"{onset:.3f}\t0.5\t{label}\tn/a\t{int(onset * _SFREQ)}\n"
                for onset, label in zip(onsets, labels)
            )
            # The trial_type labels embed Chinese characters; write with an
            # explicit encoding so Windows' charmap default does not choke.
            (folder / f"{stem}_events.tsv").write_text(
                "onset\tduration\ttrial_type\tresponse_time\tsample\n" + body,
                encoding="utf-8",
            )
    return root


@pytest.fixture
def dataset(bids_root, monkeypatch):
    ds = Wang2026Handwriting(subjects=[1])
    monkeypatch.setattr(ds, "_download_subject", lambda *a, **k: str(bids_root))
    return ds


def test_metadata():
    ds = Wang2026Handwriting()
    assert ds.code == "Wang2026Handwriting"
    assert ds.interval == [0, 4.0 - 1.0 / _SFREQ]
    assert ds.METADATA.file_format == "BDF"
    assert ds.METADATA.participants.n_subjects == 21
    assert ds.METADATA.acquisition.sampling_rate == 1000.0
    # MOABB's metadata catalog rewrites detected_paradigm from the dataset's
    # paradigm ("imagery" -> "motor_imagery") in place the first time the
    # catalog is touched, for every dataset (BNCI2024_001 included); the
    # handwriting-specific value survives until then.
    assert ds.METADATA.paradigm_specific.detected_paradigm in (
        "motor_imagery_handwriting",
        "motor_imagery",
    )
    assert ds.METADATA.paradigm_specific.imagery_tasks == [
        "horizontal",
        "vertical",
        "left_falling",
        "right_falling",
        "turning",
    ]
    # The BIDS value codes of both tasks must be unique across the dataset.
    assert len(set(WANG2026_EVENTS.values())) == len(WANG2026_EVENTS)


def test_subject_ids_are_non_contiguous():
    # sub-02, sub-04, sub-09 and sub-16 are absent from the release; MOABB
    # still numbers its own subjects 1..21.
    ds = Wang2026Handwriting()
    assert ds.subject_list == list(range(1, 22))
    from moabb.datasets.wang2026_handwriting import _SUBJECT_IDS

    assert _SUBJECT_IDS[1] == 3
    assert 2 not in _SUBJECT_IDS


def test_tasks_do_not_overwrite_each_other(dataset):
    sessions = dataset.get_data([1], cache_config={"use": False})[1]
    assert sorted(sessions) == ["0", "1"]
    for runs in sessions.values():
        # Both BIDS tasks survive, keyed by their unique run index.
        assert sorted(runs) == ["1CCSHI", "2SVHI"]
        for raw in runs.values():
            assert raw.ch_names == WANG2026_CH_NAMES
            assert raw.info["sfreq"] == _SFREQ
            assert all(np.isfinite(ch["loc"][:3]).all() for ch in raw.info["chs"])


def test_trial_type_labels_are_normalized(dataset):
    sessions = dataset.get_data([1], cache_config={"use": False})[1]
    for raw in sessions["0"].values():
        descriptions = set(raw.annotations.description)
        # Only the release's trial types plus the boundaries this loader adds;
        # MNE's own BAD_ACQ_SKIP from the BDF container is tolerated.
        assert descriptions - {"BAD_ACQ_SKIP", "EDGE boundary"} <= set(WANG2026_EVENTS)
    # The discontinuous trials are separated by EDGE annotations.
    ccs = sessions["0"]["1CCSHI"]
    n_edges = sum(1 for d in ccs.annotations.description if d == "EDGE boundary")
    assert n_edges == len(_CCS_LABELS) - 1


def test_paradigm_ccs(dataset):
    paradigm = MotorImagery(
        fmin=8,
        fmax=32,
        resample=100,
        events=["horizontal", "vertical", "left_falling", "right_falling", "turning"],
        n_classes=5,
    )
    X, y, metadata = paradigm.get_data(dataset, [1], cache_config={"use": False})
    assert X.shape == (2 * len(_CCS_LABELS), len(WANG2026_CH_NAMES), 400)
    # Every synthetic trial of the task reaches the epochs, once per session,
    # under the normalized class labels rather than the raw BIDS trial_type.
    assert_array_equal(
        sorted(set(y)),
        sorted({"horizontal", "vertical", "left_falling", "right_falling", "turning"}),
    )
    assert metadata["session"].nunique() == 2


def test_paradigm_sv(dataset):
    paradigm = MotorImagery(
        fmin=8, fmax=32, resample=100, events=["a", "o", "e", "i", "u", "v"], n_classes=6
    )
    X, y, metadata = paradigm.get_data(dataset, [1], cache_config={"use": False})
    assert X.shape == (2 * len(_SV_LABELS), len(WANG2026_CH_NAMES), 400)
    assert metadata["session"].nunique() == 2


def test_invalid_subject(dataset):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(22)
