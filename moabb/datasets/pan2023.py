"""Pan2023 cross-session motor imagery dataset."""

import h5py
import numpy as np
from mne import Annotations, create_info
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
# Files carry no persistent id, so they are addressed by their numeric datafile id.
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

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


# 28 sensorimotor electrodes (FC, C, CP and P rows of the 10-10 system), in the order
# of the dataset author's own loader; the v7.3 ".mat" files carry no channel names.
PAN2023_CHANNELS = (
    "FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 "
    "CP5 CP3 CP1 CPz CP2 CP4 CP6 P5 P3 P1 Pz P2 P4 P6"
).split()


def _trials_to_raw(data, labels, ch_names, sfreq, cue_offset):
    """Concatenate ``(n_channels, n_trials, n_samples)`` microvolt trials into a Raw.

    A stim channel marks each trial's cue (t = 0, ``cue_offset`` samples into the
    stored epoch) with its class code (1 = left hand, 2 = right hand), and a
    non-rejecting ``EDGE boundary`` annotation marks every stored-trial join.
    """
    n_channels, n_trials, n_samples = data.shape
    if len(ch_names) != n_channels:
        raise ValueError(f"Expected {len(ch_names)} channels, got {n_channels}")
    if len(labels) != n_trials or not np.isin(labels, [1, 2]).all():
        raise ValueError("Expected one valid class label per stored trial")
    if cue_offset < 0 or cue_offset + round(4 * sfreq) > n_samples:
        raise ValueError("Stored trial does not contain the imagery window")

    stim = np.zeros((1, n_trials * n_samples))
    stim[0, np.arange(n_trials) * n_samples + cue_offset] = labels
    info = create_info(
        list(ch_names) + ["STI 014"], sfreq, ["eeg"] * n_channels + ["stim"]
    )
    cont = data.reshape(n_channels, n_trials * n_samples) * 1e-6
    raw = RawArray(np.vstack([cont, stim]), info, verbose=False)
    montage = make_standard_montage("colin27_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    joins = np.arange(1, n_trials) * n_samples / sfreq
    raw.set_annotations(Annotations(joins, 0.0, "EDGE boundary"))
    return raw


class _PanDataverse(BaseDataset):
    """Shared download/session layout of the two Pan Dataverse deposits."""

    _file_ids = {}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of a single subject's two session files."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        return [
            dl.data_dl(
                f"{DATAVERSE_URL}{file_id}", self.code, path, force_update, verbose
            )
            for file_id in self._file_ids[subject].values()
        ]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        return {
            str(session): {"0": self._mat_to_raw(file_path)}
            for session, file_path in enumerate(self.data_path(subject))
        }


class Pan2023(_PanDataverse):
    """Cross-session motor imagery dataset from Pan et al. 2023.

    EEG from 14 healthy subjects (five females, two left-handed, aged 22-25)
    performing cued left- vs right-hand motor imagery on two separate days (D1,
    D2 -> sessions ``"0"``, ``"1"``), 120 trials per session (the deposit
    description gives no per-class breakdown; the 60/60 split assumes balanced
    classes). EEG was recorded with a Neuroscan SynAmps2 amplifier and 28 scalp
    electrodes at 250 Hz with a 0.01-200 Hz band-pass filter. The deposit stores
    epochs from -3 s to +4 s around the cue (2 s rest, 1 s preparation, 4 s
    imagery) as MATLAB v7.3 ``.mat`` files; the loader concatenates them,
    converts microvolts to volts, places each class event at the cue and marks
    every trial join with a non-rejecting ``EDGE boundary`` annotation. Distinct
    from :class:`Pan2025` (doi:10.7910/DVN/GH74ZG, 10 subjects, ~180 trials per
    session). The Dataverse deposit is registered as a supplement to [2]_ (closed
    access), so acquisition details here come from the deposit description.

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

    _file_ids = PAN2023_FILE_IDS

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            channel_types={"eeg": 28},
            montage="10-10",
            hardware="Neuroscan SynAmps2",
            reference=None,
            ground=None,
            sensors=PAN2023_CHANNELS,
            filters={"bandpass": [0.01, 200.0]},
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            gender={"female": 5, "male": 9},
            age_min=22,
            age_max=25,
            handedness={"right": 12, "left": 2},
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
            institution_department=(
                "School of Precision Instruments and Optoelectronics Engineering"
            ),
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
            interval=[0, 4 - 1 / 250],
            paradigm="imagery",
            doi="10.7910/DVN/251NOW",
        )

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one v7.3 (HDF5) ``.mat`` session file into a continuous Raw."""
        with h5py.File(file_path, "r") as f:
            sfreq = float(np.asarray(f["fs"]).ravel()[0])
            # h5py returns MATLAB's (n_channels, n_samples, n_trials) transposed.
            data = np.asarray(f["data"], dtype=float).transpose(2, 0, 1)
            labels = np.asarray(f["label"]).ravel().astype(int)
        # Each stored epoch starts 3 s before the motor-imagery cue.
        cue_offset = int(round(3.0 * sfreq))
        return _trials_to_raw(data, labels, PAN2023_CHANNELS, sfreq, cue_offset)
