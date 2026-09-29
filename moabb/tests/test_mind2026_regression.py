from collections import Counter

import mne
import numpy as np
import pytest

from moabb.datasets.mind2026 import MIND2026


_VALID_RUN_CODES = [4, 5] * 10 + [6, 7] * 10


def _write_events(path, coded_events):
    rows = ["onset\tduration\ttrial_type\tvalue"]
    rows.extend(f"{onset:.3f}\t0.0\tevent_{code}\t{code}" for onset, code in coded_events)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _raw_with_codes(codes):
    sfreq = 100.0
    raw = mne.io.RawArray(
        np.zeros((1, (len(codes) + 1) * int(sfreq))),
        mne.create_info(["Cz"], sfreq=sfreq, ch_types="eeg"),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(len(codes), dtype=float),
            duration=np.zeros(len(codes)),
            description=[f"Comment/{code}" for code in codes],
        )
    )
    return raw


def _assert_complete_run(annotations):
    descriptions = annotations.description.tolist()
    assert len(descriptions) == 40
    assert Counter(descriptions) == {
        "left_to_right": 10,
        "up_to_down": 10,
        "upperleft_to_lowerright": 10,
        "upperright_to_lowerleft": 10,
    }


def test_mi_events_anchor_ten_second_execution_window(tmp_path):
    """Codes 4-7 start MI; code 3 is the preceding two-second cue."""
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    events = tmp_path / "sub-01_task-mi2d_run-01_events.tsv"
    coded_events = [(100.0, 3), (102.016, _VALID_RUN_CODES[0])]
    coded_events.extend(
        (103.016 + index, code) for index, code in enumerate(_VALID_RUN_CODES[1:])
    )
    _write_events(events, coded_events)
    raw = mne.io.RawArray(
        np.zeros((1, 150_000)),
        mne.create_info(["Cz"], sfreq=1000.0, ch_types="eeg"),
        verbose=False,
    )

    annotations = MIND2026._mi_annotations(vhdr, raw)
    dataset = MIND2026()

    _assert_complete_run(annotations)
    assert annotations.description[:2].tolist() == ["left_to_right", "up_to_down"]
    np.testing.assert_allclose(annotations.onset[:2], [102.016, 103.016])
    assert dataset.interval == [0, 10]
    np.testing.assert_allclose(
        annotations.onset[0] + np.asarray(dataset.interval), [102.016, 112.016]
    )


def test_exact_balanced_run_does_not_require_restart_layout(tmp_path):
    """A normal 40-event run is accepted solely by its balanced class counts."""
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    events = tmp_path / "sub-01_task-mi2d_run-01_events.tsv"
    codes = [4, 5, 6, 7] * 10
    _write_events(events, list(enumerate(codes)))

    annotations = MIND2026._mi_annotations(vhdr, _raw_with_codes([]))

    _assert_complete_run(annotations)
    assert annotations.description[:4].tolist() == [
        "left_to_right",
        "up_to_down",
        "upperleft_to_lowerright",
        "upperright_to_lowerleft",
    ]


def test_subject_4_overfull_tsv_keeps_complete_post_restart_run(tmp_path):
    """Subject 4 has three aborted MI markers before its final code-1 restart."""
    vhdr = tmp_path / "sub-04_task-mi2d_run-01_eeg.vhdr"
    events = tmp_path / "sub-04_task-mi2d_run-01_events.tsv"
    prefix = [1, 1, 3, 5, 7, 15, 63, 255, 7, 1, 3]
    codes = prefix + _VALID_RUN_CODES
    _write_events(events, list(enumerate(codes)))

    annotations = MIND2026._mi_annotations(vhdr, _raw_with_codes([]))

    _assert_complete_run(annotations)
    np.testing.assert_allclose(annotations.onset, np.arange(11, 51))


def test_subject_19_overfull_raw_fallback_keeps_complete_post_restart_run(tmp_path):
    """The raw-marker fallback applies the same restart guard as the TSV path."""
    vhdr = tmp_path / "sub-19_task-mi2d_run-01_eeg.vhdr"
    prefix = [1, 3] + [4, 5] * 9 + [800000, 800001, 1, 2, 3]
    codes = prefix + _VALID_RUN_CODES

    annotations = MIND2026._mi_annotations(vhdr, _raw_with_codes(codes))

    _assert_complete_run(annotations)
    np.testing.assert_allclose(
        annotations.onset, np.arange(len(prefix), len(prefix) + 40)
    )


@pytest.mark.parametrize("source", ["tsv", "raw"])
def test_overfull_run_without_code_1_restart_fails_closed(tmp_path, source):
    """Extra MI markers are never trimmed without the acquisition restart marker."""
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    codes = _VALID_RUN_CODES + [4]
    raw = _raw_with_codes(codes if source == "raw" else [])
    if source == "tsv":
        events = tmp_path / "sub-01_task-mi2d_run-01_events.tsv"
        _write_events(events, list(enumerate(codes)))

    with pytest.raises(RuntimeError, match="final code 1 restart marker"):
        MIND2026._mi_annotations(vhdr, raw)


def test_overfull_run_requires_expected_post_restart_half_order(tmp_path):
    """A balanced suffix must still match the released run's two task halves."""
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    events = tmp_path / "sub-01_task-mi2d_run-01_events.tsv"
    codes = [4, 1] + [4, 5, 6, 7] * 10
    _write_events(events, list(enumerate(codes)))

    with pytest.raises(RuntimeError, match="first 20.*codes 4/5"):
        MIND2026._mi_annotations(vhdr, _raw_with_codes([]))
