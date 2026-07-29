import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets.wrcc2023_mi_a import WRCC2023_MI_A_CHANNELS
from moabb.datasets.wrcc2023_mi_b import WRCC2023_MI_B
from moabb.datasets.wrcc2023_mi_c import WRCC2023_MI_C


@pytest.mark.parametrize(
    ("dataset_class", "include_sfreq"),
    [(WRCC2023_MI_B, False), (WRCC2023_MI_C, True)],
)
def test_all_wrcc_trials_survive_epoch_boundaries(
    tmp_path, dataset_class, include_sfreq
):
    """Leading and trailing pads retain every stored trial."""
    n_trials, n_channels, n_samples = 3, len(WRCC2023_MI_A_CHANNELS), 4
    contents = {
        "data": np.ones((n_trials, n_channels, n_samples)),
        "label": np.array([1, 2, 3]),
    }
    if include_sfreq:
        contents["fs"] = np.array([[1000.0]])
    path = tmp_path / "subject.mat"
    savemat(path, contents)

    raw = dataset_class._mat_to_raw(path)
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left_hand": 1, "right_hand": 2, "feet": 3},
        tmin=0,
        tmax=n_samples / 1000.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    assert len(events) == n_trials
    assert len(epochs) == n_trials
    assert epochs.drop_log == ((), (), ())


def test_wrcc_mi_b_uses_companion_neuracle_montage(tmp_path):
    """MI-B shares the documented 59-channel WRCC acquisition order."""
    path = tmp_path / "subject.mat"
    savemat(
        path,
        {
            "data": np.ones((2, len(WRCC2023_MI_A_CHANNELS), 4)),
            "label": np.array([1, 2]),
        },
    )

    raw = WRCC2023_MI_B._mat_to_raw(path)
    eeg = raw.copy().pick("eeg")
    positioned = [
        channel
        for channel in eeg.ch_names
        if np.any(eeg.info["chs"][eeg.ch_names.index(channel)]["loc"][:3])
    ]

    assert eeg.ch_names == WRCC2023_MI_A_CHANNELS
    assert {"C3", "Cz", "C4"}.issubset(eeg.ch_names)
    assert len(positioned) == len(WRCC2023_MI_A_CHANNELS)
