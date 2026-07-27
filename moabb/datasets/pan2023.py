"""Pan2023 cross-session motor imagery dataset."""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


# Harvard Dataverse "A cross-session motor imagery EEG dataset" (doi:10.7910/DVN/251NOW).
# Distinct from Pan2025 (doi:10.7910/DVN/GH74ZG): 14 subjects vs 10, 120 trials per
# session vs 180, and a different on-disk layout. Files carry no persistent id, so
# they are addressed by their numeric Dataverse datafile id. The published dataset
# (versions 1.0-1.3, all RELEASED) distributes the recordings as MATLAB v7.3 (HDF5)
# ".mat" files, which are read here with h5py.
PAN2023_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# subject -> {session_index: datafile_id}, resolved from the Dataverse files API
# (file names S<subject>D<day>.mat; D1 -> session 0, D2 -> session 1).
PAN2023_FILE_IDS = {
    1: {0: 7574475, 1: 7574479},
    2: {0: 7574481, 1: 7574472},
    3: {0: 7574469, 1: 7574484},
    4: {0: 7574476, 1: 7574467},
    5: {0: 7574489, 1: 7574491},
    6: {0: 7574464, 1: 7574466},
    7: {0: 7574483, 1: 7574470},
    8: {0: 7574474, 1: 7574488},
    9: {0: 7574486, 1: 7574482},
    10: {0: 7574471, 1: 7574487},
    11: {0: 7574473, 1: 7574485},
    12: {0: 7574477, 1: 7574480},
    13: {0: 7574468, 1: 7574478},
    14: {0: 7574490, 1: 7574465},
}

# 28 sensorimotor electrodes (FC, C, CP and P rows of the 10-10 system), as reported
# by the dataset author's own loader (github.com/PLC-TJU/moabb, pan2023.py). The
# stored ".mat" files do not carry channel names, so this fixed order is used.
PAN2023_CHANNELS = [
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
]

# Each stored epoch spans -3 s to +4 s (7 s, 1750 samples at 250 Hz) around the
# motor-imagery task onset: 2 s rest, 1 s preparation, then 4 s task. The task cue
# (t = 0) therefore sits 3 s into every epoch.
PAN2023_TASK_ONSET_S = 3.0


class Pan2023(BaseDataset):
    """Cross-session motor imagery dataset from Pan et al. 2023.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Pan2023         14       28           2                 60            4s            250 Hz            2
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings from 14 healthy subjects (five female,
    aged 22-25, two left-handed) performing a cued left- vs right-hand motor imagery
    task across two experimental sessions (recorded on two separate days, D1 and D2),
    designed to study cross-session variability in brain-computer interface
    applications.

    Each session provides 120 trials (60 left-hand, 60 right-hand). Signals were
    recorded with a Neuroscan SynAmps2 amplifier from 28 EEG channels covering the
    sensorimotor cortex (FC, C, CP and P rows of the 10-10 system), acquired at
    1000 Hz with a 0.01-200 Hz band-pass and a 50 Hz notch, then downsampled to
    250 Hz. Each stored epoch spans -3 s to +4 s around the imagery cue (1750
    samples); the 2 s rest and 1 s preparation periods precede the cue and the 4 s
    motor-imagery task window runs from 0 to 4 s.

    The data are distributed as MATLAB v7.3 (HDF5) ``.mat`` files (one per subject
    and session) on Harvard Dataverse, each containing a ``data`` array of shape
    ``(n_channels, n_samples, n_trials)`` (MATLAB order), an integer ``label``
    vector (1 = left hand, 2 = right hand) and the sampling rate ``fs``. Per-trial
    class labels are therefore data-borne in the ``label`` array.

    This dataset is the companion of, but distinct from, :class:`Pan2025`
    (doi:10.7910/DVN/GH74ZG, 10 subjects, 180 trials per session).

    References
    ----------

    .. [1] Pan, Lincong (2023). A cross-session motor imagery EEG dataset.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/251NOW
    .. [2] Pan, L. et al. (2023). Riemannian geometric and ensemble learning for
       decoding cross-session motor imagery electroencephalography signals.
       Journal of Neural Engineering, 20(6), 066011.
       DOI: https://doi.org/10.1088/1741-2552/ad0a01

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=28,
            channel_types={"eeg": 28},
            montage="10-10",
            hardware="Neuroscan SynAmps2",
            reference=None,
            ground=None,
            sensors=PAN2023_CHANNELS,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            age_min=22,
            age_max=25,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 60, "right_hand": 60},
            trial_duration=4.0,
            study_design="Cued left- vs right-hand motor imagery. Each of 120 trials "
            "per session has a 2 s rest, a 1 s 'Ready' preparation, then a 4 s motor "
            "imagery task period.",
            feedback_type="none",
            stimulus_type="cue",
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/251NOW",
            description="Cross-session motor imagery EEG dataset from 14 subjects "
            "performing cued left- vs right-hand motor imagery across two sessions.",
            investigators=["Lincong Pan"],
            institution="Tianjin University",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/251NOW",
            associated_paper_doi="10.1088/1741-2552/ad0a01",
            publication_year=2023,
            license="CC0-1.0",
            repository="Harvard Dataverse",
            keywords=[
                "motor imagery",
                "cross-session",
                "BCI",
                "brain-computer interface",
                "EEG",
                "left hand",
                "right hand",
            ],
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        file_format="MAT (v7.3/HDF5)",
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14 + 1)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2},
            code="Pan2023",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.7910/DVN/251NOW",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of a single subject's two session files."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for session in (0, 1):
            file_id = PAN2023_FILE_IDS[subject][session]
            url = f"{PAN2023_BASE_URL}{file_id}"
            local = dl.data_dl(url, self.code, path, force_update, verbose)
            paths.append(local)
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        file_paths = self.data_path(subject)

        sessions = {}
        for session_idx, file_path in enumerate(file_paths):
            raw = self._mat_to_raw(file_path)
            sessions[str(session_idx)] = {"0": raw}
        return sessions

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one v7.3 (HDF5) ``.mat`` session file into a continuous Raw.

        Epoched trials are concatenated along time; a stim channel marks each
        trial's motor-imagery cue onset (t = 0, i.e. ``PAN2023_TASK_ONSET_S``
        seconds into the 7 s epoch) with its class code (1 = left hand,
        2 = right hand).
        """
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading the Pan2023 v7.3 .mat files requires h5py (`pip install h5py`)."
            ) from exc

        with h5py.File(file_path, "r") as f:
            sfreq = float(np.asarray(f["fs"]).ravel()[0])
            # MATLAB stores ``data`` as (n_channels, n_samples, n_trials); h5py
            # returns the transposed view (n_trials, n_samples, n_channels).
            data = np.asarray(f["data"], dtype=float)
            labels = np.asarray(f["label"]).ravel().astype(int)

        n_trials, n_samples, n_channels = data.shape
        if n_channels != len(PAN2023_CHANNELS):
            raise ValueError(
                f"Pan2023: expected {len(PAN2023_CHANNELS)} channels, "
                f"got {n_channels} in {file_path}"
            )

        # (n_trials, n_samples, n_channels) -> (n_channels, n_trials, n_samples)
        data = np.transpose(data, (2, 0, 1))
        # Concatenate trials along time and convert microvolts to volts.
        cont = data.reshape(n_channels, n_trials * n_samples) * 1e-6

        # Sample offset of the motor-imagery cue (t = 0) within each epoch.
        cue_offset = int(round(PAN2023_TASK_ONSET_S * sfreq))

        stim = np.zeros((1, cont.shape[1]))
        for trial_idx, label in enumerate(labels[:n_trials]):
            onset = trial_idx * n_samples + cue_offset
            stim[0, onset] = label

        full = np.vstack([cont, stim])
        mne_info = create_info(
            ch_names=list(PAN2023_CHANNELS) + ["STI 014"],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels + ["stim"],
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
