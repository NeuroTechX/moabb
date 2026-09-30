"""WRCC2023 MI-A/B/C three-class motor imagery datasets (World Robot Contest)."""

import numpy as np
import scipy.io as sio
from mne import Annotations, create_info
from mne.io import RawArray

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


# Files carry no persistent id; they are addressed by Harvard Dataverse datafile id.
WRCC2023_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# EEG channels 1-59 of the Neuracle 64-channel 10-10 cap (60 ECG and 61-64 EOG
# are not distributed). The .mat files carry no names; this is the WRC/Neuracle
# order also used by :class:`Yang2025`.
# fmt: off
WRCC2023_CHANNELS = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]
# fmt: on

# Data-borne per-trial ``label`` codes.
WRCC2023_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3}
_LABELS = {code: name for name, code in WRCC2023_EVENTS.items()}


class _WRCC2023(BaseDataset):
    """Shared loader: one ``.mat`` per subject with ``data`` (90, 59, 4000) in volts,
    ``label`` (90,) and, for MI-A/MI-C, ``fs`` (1000 Hz)."""

    FILE_IDS: dict

    def _init(self, code, doi):
        BaseDataset.__init__(
            self,
            subjects=list(self.FILE_IDS),
            sessions_per_subject=1,
            events=dict(WRCC2023_EVENTS),
            code=code,
            interval=[0, 3.999],
            paradigm="imagery",
            doi=doi,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path of a single subject's ``.mat`` file."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        url = f"{WRCC2023_BASE_URL}{self.FILE_IDS[subject]}"
        return dl.data_dl(url, self.code, path, force_update, verbose)

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}, one run per stored 4 s trial.

        Separate runs avoid filtering across trial discontinuities; the cue is
        annotated at sample 0 of every trial.
        """
        mat = sio.loadmat(self.data_path(subject), squeeze_me=True)
        if "fs" in mat and float(np.ravel(mat["fs"])[0]) != 1000:
            raise ValueError("Expected WRCC sampling rate of 1000 Hz")
        data = np.asarray(mat["data"], dtype=float)
        labels = np.atleast_1d(mat["label"])
        if data.ndim != 3 or data.shape[1:] != (59, 4000):
            raise ValueError("Expected stored WRCC trials of shape (n_trials, 59, 4000)")
        if len(labels) != len(data) or not np.isin(labels, [1, 2, 3]).all():
            raise ValueError("WRCC labels must match trials and use codes 1, 2, 3")
        info = create_info(WRCC2023_CHANNELS, 1000.0, "eeg")
        info.set_montage("colin27_1005", on_missing="ignore", verbose=False)
        runs = {}
        for i, (trial, label) in enumerate(zip(data, labels)):
            run = RawArray(trial, info.copy(), verbose=False)
            run.set_annotations(Annotations([0], [0], [_LABELS[label]]))
            runs[str(i)] = run
        return {"0": runs}


class WRCC2023_MI_A(_WRCC2023):
    """WRCC2023 MI-A three-class motor imagery dataset.

    **Dataset description**

    EEG from 9 subjects (7 healthy, 2 stroke patients) performing three-class
    motor imagery (left-hand grasping, right-hand grasping, foot hooking),
    released as the "MI-A" dataset of the BCI competition of the 2023 World
    Robot Contest (WRCC2023). Each subject has 90 trials (30 per class) of 59
    EEG channels recorded with a Neuracle 64-channel system at 1000 Hz; each
    stored 4 s epoch starts at the imagery cue. Amplitudes are stored in volts
    and include DC offsets removed by the paradigm band-pass filter. It shares
    hardware and paradigm with, but is distinct from, :class:`Yang2025`.

    References
    ----------

    .. [1] WRCC2023 (2024). MI-A dataset of the BCI competition WRCC2023.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/J9JFES

    Notes
    -----

    Each stored trial is exposed as a separate run to avoid filtering across
    discontinuities. The inclusive epoch endpoint is 3.999 s (4000 samples),
    without synthetic zero padding. Channel order and physical calibration
    follow the reconciled source loaders and still require source verification.

    The Harvard Dataverse record (V1, released 2024-07-05, CC0 1.0) lists nine
    files ``subject1.mat``..``subject9.mat``; its description only states that
    two of the individuals are stroke patients ("data from stroke patients
    (number unknown) for two individuals and from healthy individuals for the
    others"). Channel count, sampling rate and trial structure are read from
    the files, not from the record; no paper is linked (paper audit,
    2026-09-30).

    .. versionadded:: 1.8

    """

    FILE_IDS = {
        1: 10358829,
        2: 10358821,
        3: 10358827,
        4: 10358828,
        5: 10358824,
        6: 10358822,
        7: 10358825,
        8: 10358823,
        9: 10358826,
    }

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 59},
            montage="10-10",
            hardware="Neuracle 64-channel wireless EEG",
            reference=None,
            ground=None,
            sensors=WRCC2023_CHANNELS,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="mixed",
            clinical_population="stroke (2 of 9 subjects); remainder healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trials_per_class={"left_hand": 30, "right_hand": 30, "feet": 30},
            trial_duration=4.0,
            study_design="Three-class motor imagery (left-hand grasping, "
            "right-hand grasping, foot hooking). 90 trials per subject, 30 per "
            "class, each stored as a 4 s imagery epoch.",
            feedback_type="none",
            stimulus_type="cue",
            synchronicity="cue-based",
            mode="offline",
            events=dict(WRCC2023_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/J9JFES",
            description="MI-A three-class (left hand, right hand, feet) motor "
            "imagery EEG dataset from the 2023 World Robot Contest BCI competition.",
            investigators=["WRCC2023"],
            institution="World Robot Contest (WRCC2023) BCI competition",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/J9JFES",
            publication_year=2024,
            license="CC0-1.0",
            repository="Harvard Dataverse",
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "World Robot Contest",
                "WRCC2023",
                "left hand",
                "right hand",
                "feet",
                "stroke",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=90,
        file_format="MAT (v5)",
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet"],
            imagery_duration_s=4.0,
        ),
    )

    def __init__(self):
        self._init("WRCC2023-MI-A", "10.7910/DVN/J9JFES")


class WRCC2023_MI_B(_WRCC2023):
    """Three-class motor imagery dataset (MI-B) of the BCI competition WRCC2023.

    **Dataset description**

    EEG from 9 subjects (7 healthy, 2 stroke patients) performing cued
    three-class motor imagery (left hand, right hand, both feet), released as
    the "MI-B" dataset of the 2023 World Robot Contest BCI competition. Each
    subject file holds 90 interleaved trials (30 per class) of 59-channel EEG,
    4000 samples each, stored in volts (DC-coupled).

    References
    ----------

    .. [1] WRCC2023 (2024). MI-B dataset of the BCI competition WRCC2023.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/HFNWRX

    Notes
    -----

    The ``.mat`` files store neither channel names nor the sampling rate. The
    loader applies the WRCC/Neuracle 59-channel order shared with MI-A and MI-C
    and their 1000 Hz rate. Each stored trial is exposed as a separate run; the
    inclusive epoch endpoint is 3.999 s (4000 samples), without zero padding.

    The Harvard Dataverse record (V1, released 2024-07-05, CC0 1.0) lists nine
    files ``subject1.mat``..``subject9.mat``; its description only states that
    two of the individuals are stroke patients. No paper is linked (paper
    audit, 2026-09-30).

    .. versionadded:: 1.8

    """

    FILE_IDS = {
        1: 10358839,
        2: 10358833,
        3: 10358840,
        4: 10358836,
        5: 10358838,
        6: 10358834,
        7: 10358835,
        8: 10358837,
        9: 10358832,
    }

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 59},
            montage="standard_1005",
            reference=None,
            ground=None,
            hardware="Neuracle 64-channel EEG (59 EEG channels retained)",
            sensors=list(WRCC2023_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="mixed",
            clinical_population="stroke (2 of 9 subjects); remaining subjects healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trials_per_class={"left_hand": 30, "right_hand": 30, "feet": 30},
            trial_duration=4.0,
            study_design="Cued three-class motor imagery (left hand, right hand, "
            "both feet). Each subject file holds 90 interleaved trials, 30 per class.",
            synchronicity="cue-based",
            mode="offline",
            events=dict(WRCC2023_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/HFNWRX",
            description="MI-B three-class (left hand, right hand, feet) motor imagery "
            "EEG dataset of the BCI competition at the 2023 World Robot Contest; "
            "9 subjects (2 stroke, 7 healthy).",
            investigators=["WRCC2023"],
            country="CN",
            data_url="https://doi.org/10.7910/DVN/HFNWRX",
            publication_year=2024,
            license="CC0-1.0",
            repository="Harvard Dataverse",
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "World Robot Contest",
                "left hand",
                "right hand",
                "feet",
                "stroke",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=90,
        file_format="MAT (v5)",
        tags=Tags(
            pathology=["stroke", "healthy"], modality=["Motor"], type=["Motor Imagery"]
        ),
    )

    def __init__(self):
        self._init("WRCC2023-MI-B", "10.7910/DVN/HFNWRX")


class WRCC2023_MI_C(_WRCC2023):
    """Three-class motor imagery dataset from the World Robot Contest 2023 (MI-C).

    **Dataset description**

    EEG from 8 participants (2 stroke patients, the rest healthy) performing
    cued three-class motor imagery (left hand, right hand, feet), released as
    the "MI-C" dataset of the 2023 World Robot Contest BCI competition. Each
    subject provides 90 randomised trials (30 per class) of 59 EEG channels,
    stored as raw DC-coupled 4 s windows in volts at 1000 Hz.

    References
    ----------

    .. [1] WRCC2023 (2024). MI-C dataset of the BCI competition WRCC2023.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/G8FBHH

    Notes
    -----

    Each stored trial is exposed as a separate run to avoid filtering across
    discontinuities. The inclusive epoch endpoint is 3.999 s (4000 samples),
    without synthetic zero padding. Channel order and physical calibration
    follow the reconciled source loaders and still require source verification.

    The Harvard Dataverse record (V1, released 2024-07-05, CC0 1.0) lists eight
    files ``subject1.mat``..``subject8.mat``; its description only states that
    two of the individuals are stroke patients. No paper is linked (paper
    audit, 2026-09-30).

    .. versionadded:: 1.8

    """

    FILE_IDS = {
        1: 10358845,
        2: 10358849,
        3: 10358843,
        4: 10358842,
        5: 10358844,
        6: 10358847,
        7: 10358846,
        8: 10358848,
    }

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 59},
            montage="standard_1005",
            reference=None,
            ground=None,
            hardware="Neuracle NeuSen W 64-channel wireless EEG (channels 1-59 EEG)",
            line_freq=50.0,
            sensors=list(WRCC2023_CHANNELS),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            species="homo sapiens",
            health_status="mixed: healthy individuals and stroke patients (2 stroke)",
            clinical_population="stroke",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trials_per_class={"left_hand": 30, "right_hand": 30, "feet": 30},
            synchronicity="cue-based",
            mode="offline",
            events=dict(WRCC2023_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/G8FBHH",
            description="Three-class (left hand, right hand, feet) motor imagery "
            "EEG dataset from the 2023 World Robot Contest BCI competition (MI-C "
            "track); 8 subjects including 2 stroke patients, 90 trials each.",
            investigators=["WRCC2023"],
            institution="World Robot Contest (BCI-Controlled Robot Contest)",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/G8FBHH",
            publication_year=2024,
            license="CC0-1.0",
            repository="Harvard Dataverse",
        ),
        sessions_per_subject=1,
        runs_per_session=90,
        file_format="mat",
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        self._init("WRCC2023-MI-C", "10.7910/DVN/G8FBHH")
