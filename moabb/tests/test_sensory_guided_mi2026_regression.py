from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from moabb.datasets.base import check_run_names
from moabb.datasets.sensoryguidedmi2026 import SensoryGuidedMI2026


def test_bci2000_paired_lr_ud_runs_receive_unique_order_indices(tmp_path: Path):
    """Paired run01/run01UD must not both expose MOABB index ``1``."""
    ds = SensoryGuidedMI2026()
    files = [
        tmp_path / "S016_sess01_run01.mat",
        tmp_path / "S016_sess01_run01UD.mat",
        tmp_path / "S016_sess01_run02.mat",
        tmp_path / "S016_sess01_run02UD.mat",
    ]
    for path in files:
        path.touch()

    with (
        patch.object(ds, "data_path", return_value=[str(path) for path in files]),
        patch.object(ds, "_read_run", return_value=object()),
    ):
        sessions = ds._get_single_subject_data(16)

    assert list(sessions) == ["1"]
    assert list(sessions["1"]) == [
        "0Run1",
        "1Run1UD",
        "2Run2",
        "3Run2UD",
    ]
    indices = [key.split("Run", maxsplit=1)[0] for key in sessions["1"]]
    assert len(indices) == len(set(indices))
    check_run_names({16: sessions})


def test_read_run_rejects_nonfinite_source_eeg_before_filtering():
    """One corrupt trial must reject its run instead of poisoning filtering."""
    ds = SensoryGuidedMI2026()
    signal = np.zeros((5, 2, 2))
    signal[-1, :, 1] = np.nan
    run_data = {
        "meta": {"sampling_rate_hz": 1000.0},
        "trialSignal": signal,
        "trialTargetClass": np.column_stack(
            [np.zeros(5, dtype=int), np.ones(5, dtype=int)]
        ),
    }

    with (
        patch.object(ds, "_read_mat", return_value=run_data),
        pytest.warns(
            RuntimeWarning,
            match=r"corrupt\.mat.*2 non-finite EEG values.*1 of 2 trials",
        ),
    ):
        raw = ds._read_run("corrupt.mat")

    assert raw is None
