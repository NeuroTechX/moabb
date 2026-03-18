"""Visual object ERP dataset.

Kaneshiro, Perreau Guimaraes, Kim, Norcia, and Suppes (2015), PLoS ONE.
DOI: 10.1371/journal.pone.0135697
Data: https://purl.stanford.edu/bq914sc3730
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
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_DOI = "10.1371/journal.pone.0135697"
_SIGN = "kaneshiro2015"
_BASE_URL = "https://stacks.stanford.edu/file/bq914sc3730"

# 6 object categories.
_EVENTS = {
    "human_body": 1,
    "human_face": 2,
    "animal_body": 3,
    "animal_face": 4,
    "fruit_vegetable": 5,
    "inanimate_object": 6,
}


class Kaneshiro2015(BaseDataset):
    """Visual object ERP dataset from Kaneshiro et al 2015.

    Dataset from the paper [1]_.

    **Dataset Description**

    Ten subjects viewed 72 images from 6 object categories (human
    body, human face, animal body, animal face, fruit/vegetable,
    inanimate object) presented for 500 ms each with 750 ms ISI.
    Each image was shown 72 times across 6 blocks (2 sessions x 3
    blocks), yielding ~5184 trials per subject.

    EEG was recorded at 1000 Hz from 128 EGI HydroCel channels
    (124 retained after preprocessing). Data was bandpass filtered
    (1-25 Hz), ICA-cleaned, re-referenced to average, and
    downsampled to 62.5 Hz. Epochs span 0-496 ms post-stimulus
    (32 time samples).

    The adapter reconstructs continuous Raw objects from the epoched
    data for compatibility with MOABB's paradigm pipeline.

    References
    ----------
    .. [1] Kaneshiro, B., Perreau Guimaraes, M., Kim, H.-S.,
           Norcia, A. M., & Suppes, P. (2015). A Representational
           Similarity Analysis of the Dynamics of Object Processing
           Using Single-Trial EEG Classification. PLoS ONE, 10(8),
           e0135697.
           https://doi.org/10.1371/journal.pone.0135697
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=62.5,
            n_channels=124,
            channel_types={"eeg": 124},
            montage="GSN-HydroCel-128",
            hardware="EGI Net Amps 300",
            sensor_type="HydroCel Geodesic Sensor Net",
            reference="average",
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"female": 3, "male": 7},
            age_min=21,
            age_max=57,
            handedness={"right": 9, "left": 1},
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="p300",
            n_classes=6,
            class_labels=list(_EVENTS.keys()),
            trial_duration=0.496,
            study_design=(
                "Visual object recognition; 72 images from 6 categories; "
                "500 ms presentation, 750 ms ISI"
            ),
            feedback_type="none",
            stimulus_type="photograph",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Blair Kaneshiro",
                "Marcos Perreau Guimaraes",
                "Hyung-Suk Kim",
                "Anthony M. Norcia",
                "Patrick Suppes",
            ],
            institution="Stanford University",
            country="US",
            publication_year=2015,
            data_url="https://purl.stanford.edu/bq914sc3730",
            license="CC-BY-3.0",
        ),
        sessions_per_subject=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["ERP"],
            type=["Visual ERP"],
        ),
        data_structure=DataStructureMetadata(
            n_trials=5184,
            trials_context="per subject (72 images x 72 repetitions)",
        ),
        data_processed=True,
        file_format="MATLAB",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Kaneshiro2015",
            interval=[0, 0.496],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        mat_path = self.data_path(subject)
        data = loadmat(mat_path, squeeze_me=True)

        # X_3D: (124, 32, T) - electrodes x time x trials
        X = data["X_3D"].astype(np.float64)
        labels = data["categoryLabels"].astype(int).ravel()
        sfreq = float(data["Fs"])

        n_ch, n_times, n_trials = X.shape

        # Scale to Volts (data is in uV).
        X = X * 1e-6

        # Reconstruct continuous signal by concatenating trials with
        # a short buffer between them to prevent event overlap.
        buffer_samples = max(1, int(sfreq * 0.1))  # 100 ms buffer
        total_len = n_trials * (n_times + buffer_samples)

        continuous = np.zeros((n_ch, total_len))
        stim = np.zeros(total_len)

        for i in range(n_trials):
            start = i * (n_times + buffer_samples)
            continuous[:, start : start + n_times] = X[:, :, i]
            stim[start] = labels[i]

        # Channel names: E1 through E124 (EGI HydroCel).
        ch_names = [f"E{i}" for i in range(1, n_ch + 1)] + ["STI"]
        ch_types = ["eeg"] * n_ch + ["stim"]

        all_data = np.vstack([continuous, stim[np.newaxis]])
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)

        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
        # Only keep channels E1-E124.
        raw.set_montage(montage, on_missing="warn")

        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = Path(dl.get_dataset_path(_SIGN, None)) / f"MNE-{_SIGN}-data"
        base.mkdir(parents=True, exist_ok=True)

        fname = f"S{subject}.mat"
        local = base / fname

        if not local.exists() or force_update:
            url = f"{_BASE_URL}/{fname}"
            downloaded = dl.data_dl(url, _SIGN)
            downloaded = Path(downloaded)
            if downloaded != local:
                downloaded.rename(local)

        return str(local)
