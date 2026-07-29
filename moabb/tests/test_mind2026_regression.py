import mne
import numpy as np

from moabb.datasets.mind2026 import MIND2026


def test_mi_events_anchor_ten_second_execution_window(tmp_path):
    """Codes 4-7 start MI; code 3 is the preceding two-second cue."""
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    events = tmp_path / "sub-01_task-mi2d_run-01_events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\tvalue\n"
        "100.000\t2.0\tmi_start\t3\n"
        "102.016\t10.0\tmi_l2r\t4\n"
        "129.016\t10.0\tmi_t2b\t5\n",
        encoding="utf-8",
    )
    raw = mne.io.RawArray(
        np.zeros((1, 150_000)),
        mne.create_info(["Cz"], sfreq=1000.0, ch_types="eeg"),
        verbose=False,
    )

    annotations = MIND2026._mi_annotations(vhdr, raw)
    dataset = MIND2026()

    assert annotations.description.tolist() == ["left_to_right", "up_to_down"]
    np.testing.assert_allclose(annotations.onset, [102.016, 129.016])
    assert dataset.interval == [0, 10]
    np.testing.assert_allclose(
        annotations.onset[0] + np.asarray(dataset.interval), [102.016, 112.016]
    )
