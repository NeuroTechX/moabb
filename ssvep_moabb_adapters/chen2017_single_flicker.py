"""Single-flicker online SSVEP BCI for spatial navigation.

Chen et al. (2017), PLOS ONE.
DOI: 10.1371/journal.pone.0178385
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/580485/files/single%20flicker%20SSVEP%20BCI%20raw%20data.zip"


class Chen2017SingleFlicker(BaseDataset):
    """Single-flicker online SSVEP BCI dataset.

    Dataset from [1]_.

    This dataset uses a spatially coded SSVEP paradigm where a single white
    square flickers at 15 Hz in the center of the screen. Four non-flickering
    target squares are placed at the cardinal directions (N, E, W, S). The
    user gazes at one target, producing a distinct spatial topography of the
    15 Hz SSVEP response for each direction.

    The dataset contains 32-channel EEG recorded from 12 healthy subjects
    (7 female, 5 male, mean age 23.5, range 19-32) using a BioSemi ActiveTwo
    system at 512 Hz.

    Only the online .mat files are loaded (training .xdf files are skipped
    as pyxdf is not a moabb dependency). Each .mat file corresponds to one
    game round of an online spatial navigation task. Data contains
    variable-length trials from the adaptive online BCI.

    Each subject completed approximately 16 game rounds. Trial durations
    vary as the online classifier made decisions at different speeds.

    Warnings
    --------
    This paradigm uses a SINGLE flicker frequency (15 Hz) with spatially-coded
    directions. Standard frequency-based SSVEP analysis (CCA, FBCCA) will NOT
    work. Use broadband spatial features or classification approaches instead.

    The .xdf training files are not loaded. If needed, install pyxdf separately.

    References
    ----------
    .. [1] J. Chen, D. Zhang, A. K. Engel, Q. Gong, and A. Maye,
       "Application of a single-flicker online SSVEP BCI for spatial
       navigation," PLoS ONE, vol. 12, no. 5, e0178385, 2017.
       DOI: 10.1371/journal.pone.0178385
    """

    # BioSemi 32-channel layout
    # fmt: off
    _ch_names = [
        "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
        "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
        "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
        "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
        "stim",
    ]
    # fmt: on

    _events = {"north": 1, "east": 2, "west": 3, "south": 4}

    # ASCII class codes in .mat files → event IDs
    _CLASS_MAP = {78: 1, 69: 2, 87: 3, 83: 4}  # N→north, E→east, W→west, S→south

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events=self._events,
            code="Chen2017-SingleFlicker",
            interval=[0.0, 4.0],
            paradigm="ssvep",
            doi="10.1371/journal.pone.0178385",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject from all online game rounds.

        Raw .mat files contain 57-channel BioSemi data: row 0 is trigger
        (Trig1), rows 1-32 are EEG (A1-A32), rows 33-56 are external
        channels.  Only the 32 EEG channels are kept.

        Class labels are ASCII codes: N=78, E=69, S=83, W=87.
        """
        n_channels = 32
        sfreq = 512

        mat_files = self._get_mat_files(subject)
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found for subject {subject}")

        all_trials = []
        for mat_file in mat_files:
            mat = loadmat(str(mat_file), squeeze_me=True)
            data_struct = mat["data"]

            # .item() needed to unwrap 0-d structured array
            trials = data_struct["trial"].item()
            classes = data_struct["class"].item()

            if not hasattr(trials, "__len__"):
                trials = [trials]
                classes = [classes]

            for trial_data, trial_class in zip(trials, classes):
                if trial_data.ndim == 1:
                    continue
                # trial_data shape: (57, n_samples) — select rows 1:33 (A1-A32)
                eeg = trial_data[1:33, :]
                n_samples = eeg.shape[1]

                # De-mean
                eeg = eeg - eeg.mean(axis=1, keepdims=True)

                # Map ASCII class code to event ID
                event_id = self._CLASS_MAP.get(int(trial_class), int(trial_class))

                # Build stim channel
                stim = np.zeros((1, n_samples))
                stim[0, 0] = event_id

                # Concatenate EEG (scaled to V) + stim
                trial_with_stim = np.concatenate([1e-6 * eeg, stim], axis=0)

                # Add buffer
                buff = np.zeros((n_channels + 1, 50))
                trial_with_stim = np.concatenate([buff, trial_with_stim, buff], axis=1)
                all_trials.append(trial_with_stim)

        if not all_trials:
            raise ValueError(f"No valid trials found for subject {subject}")

        # Concatenate all trials into continuous data
        log.warning(
            "Trial data de-meaned and concatenated with a buffer"
            " to create continuous data"
        )
        continuous = np.concatenate(all_trials, axis=1)

        ch_types = ["eeg"] * n_channels + ["stim"]
        info = create_info(self._ch_names, sfreq, ch_types)
        raw = RawArray(data=continuous, info=info, verbose=False)
        montage = make_standard_montage("biosemi32")
        raw.set_montage(montage, on_missing="ignore")
        return {"0": {"0": raw}}

    def _get_mat_files(self, subject):
        """Get all .mat files for a given subject."""
        data_dir = self._get_extract_dir(subject)
        # Online .mat files named: {subject}_{repeat}_{random}.mat
        pattern = f"{subject}_*.mat"
        files = sorted(data_dir.rglob(pattern))
        return files

    def _get_extract_dir(self, subject):
        """Get the extraction directory, downloading if needed."""
        sign = "CHEN2017SINGLEFLICKER"
        data_dir = Path(dl.get_dataset_path(sign, None)) / f"MNE-{sign.lower()}-data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = "CHEN2017SINGLEFLICKER"
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if already extracted
        mat_files = sorted(data_dir.rglob(f"{subject}_*.mat"))
        if mat_files and not force_update:
            return [str(f) for f in mat_files]

        # Download the zip
        zip_path = dl.data_dl(ZENODO_URL, sign, path, force_update, verbose)

        # Extract only .mat files (skip .xdf)
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".mat"):
                    zf.extract(member, data_dir)

        mat_files = sorted(data_dir.rglob(f"{subject}_*.mat"))
        return [str(f) for f in mat_files]
