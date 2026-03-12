"""BCIAUT-P300 dataset for autism P300 BCI.

Simoes, Borra, Santamaria-Vazquez, et al. (2020), Frontiers in Neuroscience.
DOI: 10.3389/fnins.2020.568104
Data: https://www.kaggle.com/datasets/disbeat/bciaut-p300
"""

import logging
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_DOI = "10.3389/fnins.2020.568104"
_SIGN = "simoes2020"

# 8 EEG channels.
_CH_NAMES = ["C3", "Cz", "C4", "CPz", "P3", "Pz", "P4", "POz"]


class Simoes2020(BaseDataset):
    """BCIAUT-P300 dataset for autism from Simoes et al 2020.

    Dataset from the paper [1]_.

    **Dataset Description**

    Fifteen subjects with autism spectrum disorder (ASD) performed
    a P300-based BCI task across 7 sessions. EEG was recorded at
    250 Hz from 8 channels (C3, Cz, C4, CPz, P3, Pz, P4, POz)
    using a g.Nautilus system.

    The data is pre-epoched (8 channels x 300 samples x N trials).
    Each epoch spans -200 to +1000 ms relative to stimulus onset.
    Target/NonTarget labels are provided in text files.

    **Data must be downloaded manually** from Kaggle (requires
    account): https://www.kaggle.com/datasets/disbeat/bciaut-p300

    After downloading, extract the archive and set the path::

        # Option 1: Set MNE data path
        mne.set_config('MNE_DATA', '/path/to/data')
        # Then place data in: /path/to/data/MNE-simoes2020-data/BCIAUT_P300/

        # Option 2: Use kagglehub (if installed)
        import kagglehub
        kagglehub.dataset_download("disbeat/bciaut-p300")

    References
    ----------
    .. [1] Simoes, M., Borra, D., Santamaria-Vazquez, E., et al.
           (2020). BCIAUT-P300: A Multi-Session and Multi-Subject
           Benchmark Dataset on Autism for P300-Based Brain-Computer-
           Interfaces. Frontiers in Neuroscience, 14, 568104.
           https://doi.org/10.3389/fnins.2020.568104
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="standard_1020",
            hardware="g.Nautilus (g.tec)",
            reference="right ear",
            ground="AFz",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="patients",
            clinical_population="autism spectrum disorder (ASD)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=1.0,
            study_design=(
                "P300 BCI in virtual environment; 8 flashing objects; "
                "15 ASD subjects across 7 sessions"
            ),
            feedback_type="visual",
            stimulus_type="object flash",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Marco Simoes",
                "Davide Borra",
                "Eduardo Santamaria-Vazquez",
                "Miguel Castelo-Branco",
            ],
            institution="University of Coimbra",
            country="PT",
            publication_year=2020,
            data_url="https://www.kaggle.com/datasets/disbeat/bciaut-p300",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=7,
        tags=Tags(
            pathology=["Autism"],
            modality=["ERP"],
            type=["P300"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            soa_ms=300.0,
            isi_ms=200.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="~1600 training + varies testing per session",
            trials_context="per_session",
        ),
        data_processed=True,
        file_format="MATLAB (epoched)",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=7,
            events={"Target": 2, "NonTarget": 1},
            code="Simoes2020",
            interval=[0, 1],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        base = self._find_data_path()
        subj_dir = base / f"SBJ{subject:02d}"

        if not subj_dir.exists():
            raise FileNotFoundError(
                f"Data not found at {subj_dir}. Please download the dataset "
                "from https://www.kaggle.com/datasets/disbeat/bciaut-p300 "
                f"and extract to {base}"
            )

        sessions = {}
        for ses_idx in range(1, 8):
            ses_dir = subj_dir / f"S{ses_idx:02d}"
            if not ses_dir.exists():
                continue

            runs = {}
            for phase in ["Train", "Test"]:
                mat_file = ses_dir / phase / "trainData.mat"
                targets_file = ses_dir / phase / "trainTargets.txt"
                if phase == "Test":
                    mat_file = ses_dir / phase / "testData.mat"
                    targets_file = ses_dir / phase / "testTargets.txt"

                if not mat_file.exists() or not targets_file.exists():
                    continue

                try:
                    raw = self._load_epoched(mat_file, targets_file)
                    if raw is not None:
                        runs[str(len(runs))] = raw
                except Exception:
                    log.warning("Failed to load %s, skipping.", mat_file)

            if runs:
                sessions[str(ses_idx - 1)] = runs

        return sessions

    @staticmethod
    def _load_epoched(mat_path, targets_path):
        """Load epoched .mat and reconstruct continuous Raw."""
        data = loadmat(str(mat_path))

        # Find the data variable (could be 'data', 'trainData', etc.).
        mat_key = None
        for key in data:
            if not key.startswith("_"):
                arr = data[key]
                if hasattr(arr, "ndim") and arr.ndim == 3:
                    mat_key = key
                    break

        if mat_key is None:
            log.warning("No 3D array found in %s", mat_path)
            return None

        # Shape: (n_channels, n_samples_per_epoch, n_trials).
        epochs = data[mat_key].astype(np.float64)
        n_ch, n_time, n_trials = epochs.shape

        # Load target labels.
        targets = np.loadtxt(str(targets_path), dtype=int).ravel()
        if len(targets) != n_trials:
            # Truncate to match.
            n_trials = min(n_trials, len(targets))
            epochs = epochs[:, :, :n_trials]
            targets = targets[:n_trials]

        # Scale to Volts (data is in uV).
        epochs = epochs * 1e-6

        sfreq = 250.0
        buffer_samples = max(1, int(sfreq * 0.05))  # 50 ms buffer
        total_len = n_trials * (n_time + buffer_samples)

        continuous = np.zeros((n_ch, total_len))
        stim = np.zeros(total_len)

        # Skip first 50 samples (baseline, -200 ms) for event placement.
        # Event at sample 50 within each epoch (stimulus onset at 0 ms).
        onset_offset = int(sfreq * 0.2)  # 200 ms = 50 samples

        for i in range(n_trials):
            start = i * (n_time + buffer_samples)
            continuous[:, start : start + n_time] = epochs[:, :, i]
            event_sample = start + onset_offset
            if event_sample < total_len:
                stim[event_sample] = 2 if targets[i] == 1 else 1

        ch_names = list(_CH_NAMES) + ["STI"]
        ch_types = ["eeg"] * n_ch + ["stim"]
        all_data = np.vstack([continuous, stim[np.newaxis]])

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)
        raw.set_montage("standard_1020", on_missing="warn")

        return raw

    def _find_data_path(self):
        """Find the BCIAUT_P300 directory."""
        path = dl.get_dataset_path(_SIGN, None)
        base = Path(path) / f"MNE-{_SIGN}-data" / "BCIAUT_P300"

        if base.exists():
            return base

        # Try kagglehub cache.
        try:
            import kagglehub

            cache_path = Path(kagglehub.dataset_download("disbeat/bciaut-p300"))
            if cache_path.exists():
                return cache_path
        except (ImportError, Exception):
            pass

        return base

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = self._find_data_path()
        subj_dir = base / f"SBJ{subject:02d}"

        if not subj_dir.exists():
            raise FileNotFoundError(
                f"Data not found at {subj_dir}. Please download manually from "
                "https://www.kaggle.com/datasets/disbeat/bciaut-p300 "
                "and extract the archive. See the class docstring for details."
            )

        return str(subj_dir)
