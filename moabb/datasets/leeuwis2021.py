"""Leeuwis2021 left- vs right-hand motor-imagery EEG dataset (DataverseNL)."""

import warnings

import mne
import numpy as np
import pandas as pd

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

from .utils import edge_boundary_annotations, resolve_montage_name


# DataverseNL access API: a single file is fetched by its numeric datafile id.
LEEUWIS2021_BASE_URL = "https://dataverse.nl/api/access/datafile/"

# The 16 EEG channels, in the exact column order of the raw CSV header.
LEEUWIS2021_EEG_CHANNELS = (
    "F3 Fz F4 FC5 FC1 FC2 FC6 T7 C3 C4 Cz T8 CP5 CP1 CP2 CP6".split()
)

# Ordered run labels for the four runs shared by every subject: one calibration
# run (no feedback) followed by three feedback runs.
LEEUWIS2021_RUN_LABELS = ("calibration", "feedback_1", "feedback_2", "feedback_3")

# Original subject id (7-67, non-contiguous, in acquisition order; MOABB subject
# n maps to the n-th key) -> (calibration, feedback_1, feedback_2, feedback_3)
# DataverseNL datafile ids, resolved from the dataset files API (doi:10.34894/Z7ZVOD).
# Subject 40 has irregular feedback-run file names
# ("Subject40_eeg_1 (seq of run2).csv", "..._2.csv", "..._3.csv"); the per-trial
# labels are read from each file's own "class" column, so run ordering does not
# affect the class labels.
_FILE_IDS = {
    7: (100063, 100065, 100033, 100008),
    8: (99895, 99887, 99971, 100000),
    9: (99897, 99884, 99977, 99899),
    10: (99926, 99939, 99981, 99950),
    11: (99960, 99921, 99872, 100009),
    12: (99987, 100022, 99935, 99879),
    13: (99850, 100042, 99868, 100028),
    14: (100021, 100034, 100006, 99964),
    15: (99972, 99978, 100025, 99965),
    16: (100039, 99994, 100010, 99963),
    17: (99882, 100040, 100046, 100047),
    18: (99944, 99871, 99901, 100032),
    19: (100053, 99849, 100003, 99855),
    21: (99984, 99998, 99888, 99934),
    22: (99902, 100045, 99867, 99923),
    23: (100035, 100055, 100062, 99967),
    24: (99949, 99915, 99880, 100061),
    25: (99900, 100038, 99917, 99995),
    26: (99851, 99982, 99974, 99878),
    27: (99912, 100064, 99932, 99985),
    28: (99854, 100002, 100044, 99874),
    29: (100036, 99920, 99991, 99940),
    30: (99916, 99910, 99865, 99999),
    31: (100016, 100017, 100066, 99876),
    32: (99924, 99979, 100027, 99857),
    33: (99869, 99989, 99957, 99929),
    34: (100067, 99870, 99873, 100024),
    36: (99936, 99858, 100068, 99914),
    37: (100050, 99861, 99975, 99962),
    38: (99881, 100029, 99959, 99973),
    40: (99904, 100043, 99925, 100051),
    41: (99906, 99891, 99866, 99952),
    42: (99961, 99892, 100026, 99966),
    43: (100057, 99883, 99953, 100020),
    44: (99943, 99976, 100030, 99894),
    45: (99993, 100058, 99992, 100041),
    46: (99948, 100018, 99852, 99968),
    47: (100004, 99911, 99919, 99903),
    48: (99996, 99951, 99958, 99890),
    49: (99889, 100060, 99877, 100052),
    50: (99875, 99955, 99908, 99942),
    51: (100014, 99907, 99913, 99860),
    52: (99988, 99946, 100048, 99918),
    54: (99896, 100031, 100037, 100023),
    55: (99898, 100019, 99862, 99937),
    56: (99886, 99893, 99990, 100015),
    57: (99864, 99938, 99997, 99927),
    58: (99980, 99941, 99863, 99930),
    59: (99928, 99856, 99859, 99922),
    60: (99885, 99969, 99909, 99983),
    61: (99905, 99956, 99970, 100012),
    63: (99848, 100005, 100059, 100056),
    65: (100011, 99986, 100069, 100001),
    66: (99853, 99954, 99947, 100049),
    67: (100007, 100054, 99945, 99931),
}

SUBJECTS = list(_FILE_IDS)

# Per-trial class code in the CSV "class" column -> MOABB event code.
_CLASS_TO_EVENT = {-1: 1, 1: 2}

# Sampling rate (Hz); each trial spans t = -3 s .. +5 s around the cue.
_SFREQ = 250.0


class Leeuwis2021(BaseDataset):
    """Left- vs right-hand motor-imagery EEG dataset [1]_.

    **Dataset description**

    Fifty-five BCI-naive, right-handed students at Tilburg University performed
    a two-class, left- versus right-hand motor-imagery task in a single
    session, as part of a study on psychological and cognitive predictors of
    MI-BCI performance. Each session comprises one calibration run (no
    feedback) and three feedback runs of 40 trials (20 per class), recorded
    from 16 channels with a g.Nautilus amplifier at 250 Hz.

    Each stored trial spans t = -3 s to +5 s around the cue and carries its
    ``class`` label (-1 = left, +1 = right) in the CSV. This loader
    concatenates the 40 trials of each run, writes one stimulus event at each
    cue (t = 0) and marks the joins between stored trials with zero-duration
    ``EDGE boundary`` annotations. The inclusive analysis interval ends at
    4.996 s, spanning the 5 s of imagery after the cue.

    References
    ----------

    .. [1] Leeuwis, N., Paas, A., and Alimardani, M. (2021). Psychological and
       Cognitive Factors in Motor Imagery Brain Computer Interfaces.
       DataverseNL, V1. DOI: https://doi.org/10.34894/Z7ZVOD
       See also: Leeuwis, N., Paas, A., & Alimardani, M. (2021). Vividness of
       Visual Imagery and Personality Impact Motor-Imagery Brain Computer
       Interfaces. Frontiers in Human Neuroscience, 15, 634748.

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": 16},
            montage="standard_1020",
            hardware="g.Nautilus (g.tec Medical Engineering, Austria)",
            software="g.BSanalyze (g.tec)",
            reference="right earlobe",
            ground="AFz",
            sensors=list(LEEUWIS2021_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=55,
            health_status="healthy",
            bci_experience="naive",
            handedness="right",
            age_mean=20.71,
            age_std=3.52,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 80, "right_hand": 80},
            trial_duration=5.0,
            study_design="Single-session, two-class left- versus right-hand motor "
            "imagery in 55 novice BCI users. One calibration run (no feedback) plus "
            "three feedback runs of 40 trials each (20 left, 20 right). Study "
            "examined psychological and cognitive predictors of MI-BCI performance.",
            feedback_type="visual",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="online",
            has_training_test_split=True,
            events={"left_hand": 1, "right_hand": 2},
            instructions="Imagine moving the left or right hand following the cue.",
        ),
        documentation=DocumentationMetadata(
            doi="10.34894/Z7ZVOD",
            related_paper_dois=["10.3389/fnhum.2021.634748"],
            description="Single-session left/right-hand motor-imagery EEG from 55 "
            "novice BCI users (16 channels, 250 Hz, g.Nautilus), collected to study "
            "psychological and cognitive factors in motor-imagery BCI performance.",
            investigators=["Nikki Leeuwis", "Alissa Paas", "Maryam Alimardani"],
            institution="Tilburg University, Tilburg School of Humanities and "
            "Digital Sciences",
            country="NL",
            data_url="https://doi.org/10.34894/Z7ZVOD",
            publication_year=2021,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "cognition",
                "personality",
            ],
            license="CC-BY-4.0",
            repository="DataverseNL",
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(pathology=["healthy"], modality=["Motor"], type=["Motor Imagery"]),
        file_format="CSV",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, len(SUBJECTS) + 1)),
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="Leeuwis2021",
            interval=[0, 5 - 1 / _SFREQ],
            paradigm="imagery",
            doi="10.34894/Z7ZVOD",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the four run CSV paths (calibration, feedback 1-3)."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        return [
            dl.data_dl(
                f"{LEEUWIS2021_BASE_URL}{file_id}", self.code, path, force_update, verbose
            )
            for file_id in _FILE_IDS[SUBJECTS[subject - 1]]
        ]

    def _get_single_subject_data(self, subject):
        # Run keys must start with an integer index followed by letters+digits:
        # "feedback_1" -> "1feedback1".
        runs = {
            f"{idx}{label.replace('_', '')}": self._csv_to_raw(file_path)
            for idx, (label, file_path) in enumerate(
                zip(LEEUWIS2021_RUN_LABELS, self.data_path(subject))
            )
        }
        return {"0": runs}

    def _csv_to_raw(self, file_path):
        """Read one run CSV as a continuous Raw with a cue-onset stim channel."""
        df = pd.read_csv(file_path)

        # EEG data in microvolts -> Volts, shape (n_channels, n_samples).
        eeg = df[LEEUWIS2021_EEG_CHANNELS].to_numpy(dtype=float).T / 1e6

        n_samples = eeg.shape[1]
        stim = np.zeros((1, n_samples), dtype=float)

        # Trial boundaries: rows are stored contiguously per trial, in order.
        trial = df["trial"].to_numpy()
        trial_starts = np.concatenate(([0], np.flatnonzero(np.diff(trial)) + 1))
        trial_ends = np.concatenate((trial_starts[1:], [n_samples]))

        timestamps = df["TimeStamp"].to_numpy()
        cls = df["class"].to_numpy()
        for start, end in zip(trial_starts, trial_ends):
            # Cue onset within this trial: the sample with timestamp closest to 0.
            cue = start + int(np.argmin(np.abs(timestamps[start:end])))
            stim[0, cue] = _CLASS_TO_EVENT[int(cls[start])]

        data = np.vstack([eeg, stim])
        ch_names = LEEUWIS2021_EEG_CHANNELS + ["STI 014"]
        ch_types = ["eeg"] * len(LEEUWIS2021_EEG_CHANNELS) + ["stim"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = mne.create_info(ch_names, _SFREQ, ch_types)
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.set_montage(
                resolve_montage_name("colin27_1020"), on_missing="ignore", verbose=False
            )

        if len(trial_starts) > 1:
            raw.set_annotations(edge_boundary_annotations(trial_starts[1:] / _SFREQ))
        return raw
