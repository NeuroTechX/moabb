"""Pan2025 cross-session motor imagery dataset."""

import numpy as np
import scipy.io as sio

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.pan2023 import PAN2023_CHANNELS, _PanDataverse, _trials_to_raw


# Harvard Dataverse "Cross-Session Motor Imagery EEG dataset" (doi:10.7910/DVN/GH74ZG),
# subject -> {session_index: datafile_id}, resolved from the Dataverse files API.
PAN2025_FILE_IDS = {
    1: {0: 11704410, 1: 11704404},
    2: {0: 11704420, 1: 11704417},
    3: {0: 11704412, 1: 11704421},
    4: {0: 11704402, 1: 11704418},
    5: {0: 11704405, 1: 11704407},
    6: {0: 11704419, 1: 11704411},
    7: {0: 11704413, 1: 11704409},
    8: {0: 11704415, 1: 11704406},
    9: {0: 11704414, 1: 11704416},
    10: {0: 11704403, 1: 11704408},
}


class Pan2025(_PanDataverse):
    """Cross-session motor imagery dataset from Pan et al. 2025.

    EEG from 10 healthy subjects performing cued left- vs right-hand motor imagery
    on two separate days (D1, D2 -> sessions ``"0"``, ``"1"``), 180 trials per
    session, 28 sensorimotor channels at 250 Hz. The MATLAB ``.mat`` files store
    epochs from -1.5 s to 4 s around the cue plus an ``Info`` struct (channel
    names, sampling rate, epoch period); the loader concatenates the epochs,
    converts microvolts to volts, places each class event at the cue and marks
    every trial join with a non-rejecting ``EDGE boundary`` annotation.

    References
    ----------

    .. [1] Pan, Lincong (2025). Cross-Session Motor Imagery EEG dataset.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/GH74ZG

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    _file_ids = PAN2025_FILE_IDS

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            channel_types={"eeg": 28},
            montage="10-10",
            reference=None,
            ground=None,
            sensors=PAN2023_CHANNELS,
        ),
        participants=ParticipantMetadata(n_subjects=10, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 90, "right_hand": 90},
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/GH74ZG",
            description="Cross-session motor imagery EEG dataset from 10 subjects "
            "performing cued left- vs right-hand motor imagery across two sessions.",
            investigators=["Lincong Pan"],
            institution="Tianjin University",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/GH74ZG",
            publication_year=2025,
            license="CC0-1.0",
            repository="Harvard Dataverse",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10 + 1)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2},
            code="Pan2025",
            interval=[0, 4 - 1 / 250],
            paradigm="imagery",
            doi="10.7910/DVN/GH74ZG",
        )

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one ``.mat`` session file into a continuous Raw."""
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        info = mat["Info"]
        sfreq = float(info.fs)
        # (n_channels, n_samples, n_trials) -> (n_channels, n_trials, n_samples)
        data = np.asarray(mat["data"], dtype=float).transpose(0, 2, 1)
        labels = np.atleast_1d(np.asarray(mat["label"]).ravel()).astype(int)
        cue_offset = int(round((0.0 - np.asarray(info.period, dtype=float)[0]) * sfreq))
        ch_names = [str(c) for c in np.atleast_1d(info.chaninfo)]
        return _trials_to_raw(data, labels, ch_names, sfreq, cue_offset)
