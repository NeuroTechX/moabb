"""BNCI 2019 datasets."""

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_gdf

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


BNCI_URL_001_2019 = "http://bnci-horizon-2020.eu/database/data-sets/001-2019/"


class BNCI2019_001(BaseDataset):
    """BNCI 2019-001 Motor Imagery dataset for Spinal Cord Injury patients.

    .. admonition:: Dataset summary

        ============= ======= ======= ================= =============== =============== ============
        Name          #Subj   #Chan   #Trials/class     Trials length   Sampling Rate   #Sessions
        ============= ======= ======= ================= =============== =============== ============
        BNCI2019_001  10      61+3EOG 72 per class      3s              256Hz           1
        ============= ======= ======= ================= =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset consists of EEG recordings from 10 participants with cervical
    spinal cord injury (SCI) performing attempted hand and arm movements.

    Participants attempted five movement types: supination, pronation, hand open,
    palmar grasp, and lateral grasp.

    **Participants**

    - 10 participants with cervical spinal cord injury
    - Age range: 20-78 years (mean 49.8, SD 17.6)
    - Gender: 9 male, 1 female
    - Handedness: All right-handed

    **Recording Details**

    - Channels: 61 EEG + 3 EOG electrodes
    - Sampling rate: 256 Hz
    - Reference: Left earlobe

    **Motor Imagery Classes**

    - supination (776): Forearm supination
    - pronation (777): Forearm pronation
    - hand_open (779): Hand opening movement
    - palmar_grasp (925): Palmar (power) grasp
    - lateral_grasp (926): Lateral (key) grasp

    References
    ----------
    .. [1] Ofner, P. et al. (2019). Attempted arm and hand movements can be
           decoded from low-frequency EEG from persons with spinal cord injury.
           Scientific Reports, 9(1), 7134.
           https://doi.org/10.1038/s41598-019-43594-9

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    _EVENTS = {
        "supination": 776,
        "pronation": 777,
        "hand_open": 779,
        "palmar_grasp": 925,
        "lateral_grasp": 926,
    }

    _MOVEMENT_RUNS = [3, 4, 5, 6, 7, 10, 11, 12, 13]

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=self._EVENTS,
            code="BNCI2019-001",
            interval=[2, 5],
            paradigm="imagery",
            doi="10.1038/s41598-019-43594-9",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        eog_channels = ["eog-l", "eog-m", "eog-r"]
        data = {}
        for run_idx, path in enumerate(file_paths):
            raw = read_raw_gdf(path, eog=eog_channels, preload=True, verbose="ERROR")
            raw.set_montage(montage, on_missing="ignore")
            raw._data[np.isnan(raw._data)] = 0
            # Convert EEG channels (excluding last 3 EOG channels) from uV to V
            raw._data[:-3] *= 1e-6
            stim = raw.annotations.description.astype(np.dtype("<21U"))
            stim[stim == "776"] = "supination"
            stim[stim == "777"] = "pronation"
            stim[stim == "779"] = "hand_open"
            stim[stim == "925"] = "palmar_grasp"
            stim[stim == "926"] = "lateral_grasp"
            raw.annotations.description = stim
            raw.info["line_freq"] = 50.0
            raw.set_meas_date(datetime(2019, 1, 1, tzinfo=timezone.utc))
            data[str(run_idx)] = raw
        return {"0": data}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return paths to data files for a given subject."""
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject number {subject}. Valid: {self.subject_list}"
            )
        url = f"{BNCI_URL_001_2019}P{subject:02d}.zip"
        zip_path = dl.data_dl(url, "BNCI", path, force_update, verbose)
        zip_dir = Path(zip_path).parent
        if not (zip_dir / f"P{subject:02d} Run 3.gdf").exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_dir)
        paths = []
        for run in self._MOVEMENT_RUNS:
            gdf_path = zip_dir / f"P{subject:02d} Run {run}.gdf"
            if gdf_path.exists():
                paths.append(str(gdf_path))
        if not paths:
            raise FileNotFoundError(f"No GDF files found for subject {subject}")
        return paths
