from pathlib import Path
from unittest.mock import patch

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
