"""BCIAUT-P300 dataset for autism P300 BCI.

Simoes, Borra, Santamaria-Vazquez, et al. (2020), Frontiers in Neuroscience.
DOI: 10.3389/fnins.2020.568104
Original data: https://www.kaggle.com/datasets/disbeat/bciaut-p300
Re-hosted: Zenodo (per-subject ZIPs for programmatic access)
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
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import download_and_extract_subject_zip


log = logging.getLogger(__name__)

_DOI = "10.3389/fnins.2020.568104"
_SIGN = "Simoes2020"

# 8 EEG channels (central + parietal).
_CH_NAMES = ["C3", "Cz", "C4", "CPz", "P3", "Pz", "P4", "POz"]

# Zenodo re-hosted record (per-subject ZIPs).
_ZENODO_RECORD = "19005186"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"


class Simoes2020(BaseDataset):
    """BCIAUT-P300 dataset for autism from Simoes et al 2020.

    Dataset from the paper [1]_.

    **Dataset Description**

    Fifteen subjects with autism spectrum disorder (ASD) performed
    a P300-based BCI joint-attention training task across 7 sessions
    (105 total sessions). EEG was recorded at 250 Hz from 8 channels
    (C3, Cz, C4, CPz, P3, Pz, P4, POz) using a g.Nautilus wireless
    amplifier (g.tec).

    The BCI used a virtual environment with 8 objects. One object per
    block was the target; the 8 objects flashed in rapid succession
    (10 runs per block in training, 3-10 in testing). Each flash
    produces a P300 response if it is the target object.

    Data is pre-epoched: (8 channels x 350 samples x N trials).
    Each epoch spans -200 to +1200 ms relative to stimulus onset
    (1400 ms total at 250 Hz = 350 samples).

    - Training: 1600 epochs per session (8 objects x 10 runs x 20 blocks)
    - Testing: 400 x K epochs per session (K = runs_per_block, 3-10)

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
            hardware="g.Nautilus (g.tec, wireless)",
            reference="right ear",
            ground="AFz",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="patients",
            clinical_population="autism spectrum disorder (ASD)",
            gender={"male": 15},
            age_mean=22.17,
            age_std=5.5,
            age_min=16,
            age_max=38,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=1.2,
            study_design=(
                "P300 BCI joint-attention training in virtual environment; "
                "8 flashing objects; 15 ASD subjects across 7 sessions "
                "(clinical trial NCT02445625)"
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
                "Mayra Bittencourt-Villalpando",
                "Dominik Krzeminski",
                "Aleksandar Miladinovic",
                "Carlos Amaral",
                "Bruno Direito",
                "Miguel Castelo-Branco",
            ],
            institution="University of Coimbra",
            country="PT",
            publication_year=2020,
            data_url="https://zenodo.org/records/19005186",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=7,
        runs_per_session=2,
        tags=Tags(pathology=["Autism"], modality=["ERP"], type=["P300"]),
        paradigm_specific=ParadigmSpecificMetadata(detected_paradigm="p300"),
        data_structure=DataStructureMetadata(
            n_trials="1600 train + 400*K test per session (K=3-10)",
            trials_context="per_session",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["EEGNet", "LDA", "SVM", "MLP"],
            feature_extraction=["temporal_features", "deep_learning"],
            frequency_bands={"bandpass": [2.0, 30.0]},
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="calibration_vs_online",
            evaluation_type=["within_subject", "cross_session", "cross_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["joint_attention_training"],
            environment="clinical",
            online_feedback=True,
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
            interval=[0, 1.2],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        self.data_path(subject)
        base = (
            Path(dl.get_dataset_path(_SIGN, None))
            / f"MNE-{_SIGN}-data"
            / "BCIAUT_P300"
            / f"SBJ{subject:02d}"
        )

        sessions = {}
        for ses_idx in range(1, 8):
            ses_dir = base / f"S{ses_idx:02d}"
            if not ses_dir.exists():
                continue

            runs = {}
            for phase in ["Train", "Test"]:
                prefix = phase.lower()
                mat_file = ses_dir / phase / f"{prefix}Data.mat"
                targets_file = ses_dir / phase / f"{prefix}Targets.txt"

                if not mat_file.exists() or not targets_file.exists():
                    continue

                try:
                    raw = self._load_epoched(mat_file, targets_file)
                    if raw is not None:
                        runs[str(len(runs))] = raw
                except Exception:
                    log.warning("Failed to load %s, skipping.", mat_file, exc_info=True)

            if runs:
                sessions[str(ses_idx - 1)] = runs

        return sessions

    @staticmethod
    def _load_epoched(mat_path, targets_path):
        """Load epoched .mat and reconstruct continuous Raw.

        Data shape: (8 channels, 350 samples, N trials).
        Epoch window: -200 to +1200 ms at 250 Hz.
        Stimulus onset is at sample 50 (200 ms into the epoch).
        """
        data = loadmat(str(mat_path))

        # Find the 3D data variable.
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
        epochs = data[mat_key].astype(np.float64, copy=False)
        n_ch, n_time, n_trials = epochs.shape

        if n_ch != len(_CH_NAMES):
            log.warning(
                "Expected %d channels, got %d in %s", len(_CH_NAMES), n_ch, mat_path
            )
            return None

        # Load target labels (1 = target flash, 0 = non-target flash).
        targets = np.loadtxt(str(targets_path), dtype=int).ravel()
        if len(targets) != n_trials:
            n_trials = min(n_trials, len(targets))
            epochs = epochs[:, :, :n_trials]
            targets = targets[:n_trials]

        # Scale to Volts (data is in uV).
        epochs *= 1e-6

        sfreq = 250.0
        buffer_samples = max(1, int(sfreq * 0.05))  # 50 ms gap between epochs
        total_len = n_trials * (n_time + buffer_samples)

        # Pre-allocate EEG + stim channel.
        all_data = np.zeros((n_ch + 1, total_len))
        continuous = all_data[:n_ch]
        stim = all_data[n_ch]

        # Baseline is 200 ms = 50 samples. Event at stimulus onset (sample 50).
        onset_offset = int(sfreq * 0.2)

        for i in range(n_trials):
            start = i * (n_time + buffer_samples)
            continuous[:, start : start + n_time] = epochs[:, :, i]
            event_sample = start + onset_offset
            if event_sample < total_len:
                stim[event_sample] = 2 if targets[i] == 1 else 1

        ch_names = list(_CH_NAMES) + ["STI"]
        ch_types = ["eeg"] * n_ch + ["stim"]
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)
        raw.set_montage("standard_1020", on_missing="warn")

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = (
            Path(dl.get_dataset_path(_SIGN, path))
            / f"MNE-{_SIGN}-data"
            / "BCIAUT_P300"
            / f"SBJ{subject:02d}"
        )

        if base.exists() and not force_update:
            return str(base)

        # Download per-subject ZIP from Zenodo and extract.
        url = f"{_ZENODO_BASE}/SBJ{subject:02d}.zip"
        parent = base.parent
        download_and_extract_subject_zip(url, _SIGN, parent, path, force_update, verbose)

        if not base.exists():
            raise FileNotFoundError(
                f"Data not found at {base}. Check the Zenodo record "
                f"https://zenodo.org/records/{_ZENODO_RECORD}"
            )

        return str(base)
