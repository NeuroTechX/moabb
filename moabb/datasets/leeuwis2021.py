"""Leeuwis2021 left- vs right-hand motor-imagery EEG dataset.

Leeuwis, N., Paas, A., and Alimardani, M. (2021). "Psychological and Cognitive
Factors in Motor Imagery Brain Computer Interfaces." DataverseNL, V1.
Data DOI: 10.34894/Z7ZVOD
"""

import logging
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


log = logging.getLogger(__name__)

# DataverseNL access API: a single file is fetched by its numeric datafile id.
LEEUWIS2021_BASE_URL = "https://dataverse.nl/api/access/datafile/"

# The 16 EEG channels, in the exact column order of the raw CSV header
# ("F3","Fz","F4","FC5","FC1","FC2","FC6","T7","C3","C4","Cz","T8",
#  "CP5","CP1","CP2","CP6").
LEEUWIS2021_EEG_CHANNELS = [
    "F3",
    "Fz",
    "F4",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "C4",
    "Cz",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
]

# Ordered run labels for the four runs shared by every subject: one calibration
# run (no feedback) followed by three feedback runs.
LEEUWIS2021_RUN_LABELS = ("calibration", "feedback_1", "feedback_2", "feedback_3")

# Original per-subject identifiers (7-67, non-contiguous), in acquisition order.
# MOABB subject n (1-55) maps to SUBJECTS[n - 1].
SUBJECTS = [
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    36,
    37,
    38,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    63,
    65,
    66,
    67,
]

# Original subject id -> (calibration, feedback_1, feedback_2, feedback_3)
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

# Per-trial class code in the CSV "class" column -> MOABB event code.
_CLASS_TO_EVENT = {-1: 1, 1: 2}

# Sampling rate (Hz); each trial spans t = -3 s .. +5 s around the cue.
_SFREQ = 250.0


class Leeuwis2021(BaseDataset):
    """Left- vs right-hand motor-imagery EEG dataset [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Leeuwis2021     55       16           2                 80          5 s          250 Hz               1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Fifty-five novice BCI users performed a two-class, left- versus right-hand
    motor-imagery task in a single session, as part of a study on the
    psychological and cognitive factors underlying motor-imagery BCI
    performance. Participants were BCI-naive students recruited at Tilburg
    University (recruited cohort: 36 female / 21 male, mean age 20.71,
    SD 3.52), all right-handed with (corrected-to-)normal vision.

    Each session comprises four runs of 40 trials each: one calibration run
    (used to train the online classifier, no feedback) followed by three
    feedback runs. Every run contains 20 left-hand and 20 right-hand trials, so
    each subject contributes 160 labelled trials (80 per class).

    EEG was recorded from 16 electrodes of the international 10-20 system (F3,
    Fz, F4, FC1, FC5, FC2, FC6, C3, Cz, C4, CP1, CP5, CP2, CP6, T7, T8) with a
    g.Nautilus amplifier (g.tec, Austria), referenced to the right earlobe and
    grounded at AFz, and sampled at 250 Hz. Each trial in the raw CSV files
    spans a fixed window from t = -3 s to t = +5 s relative to the cue (2000
    samples); the cue is presented at t = 0 s and motor imagery follows. Every
    CSV row carries a ``trial`` number (1-40) and a ``class`` code
    (-1 = left hand, +1 = right hand), so the per-trial labels are read directly
    from the data.

    This loader concatenates the 40 trials of each run into a continuous
    recording and inserts a stimulus channel with one event at each trial's cue
    onset (t = 0). The exposed analysis interval spans the 5 s of imagery
    following the cue.

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

    .. versionadded:: 1.2.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=16,
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
            interval=[0, 5],
            paradigm="imagery",
            doi="10.34894/Z7ZVOD",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the four raw-CSV file paths for a single subject.

        Downloads each of the subject's four runs (calibration + three feedback
        runs) from DataverseNL if not already present.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-55).
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Deprecated, unused.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list of str
            The four local CSV file paths, ordered
            (calibration, feedback_1, feedback_2, feedback_3).
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        original_id = SUBJECTS[subject - 1]
        file_ids = _FILE_IDS[original_id]

        paths = []
        for file_id in file_ids:
            url = f"{LEEUWIS2021_BASE_URL}{file_id}"
            paths.append(dl.data_dl(url, self.code, path, force_update, verbose))
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Returns
        -------
        dict
            ``{"0": {run_label: Raw}}`` with the four runs (calibration and the
            three feedback runs) under the single session ``"0"``.
        """
        file_paths = self.data_path(subject)

        runs = {}
        for idx, (run_label, file_path) in enumerate(
            zip(LEEUWIS2021_RUN_LABELS, file_paths)
        ):
            # MOABB requires the run key to start with an integer index
            # (optionally followed by a letters+digits description); strip the
            # underscore from labels like "feedback_1" -> "1feedback1".
            runs[f"{idx}{run_label.replace('_', '')}"] = self._csv_to_raw(file_path)

        return {"0": runs}

    def _csv_to_raw(self, file_path):
        """Read one run CSV and build a continuous Raw with a stim channel.

        The 40 fixed-length trials are concatenated into one continuous signal;
        an event marking the cue onset (t = 0 within each trial) is written to a
        ``STI 014`` stimulus channel, using the per-trial ``class`` label
        (-1 -> left hand -> code 1, +1 -> right hand -> code 2).
        """
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
        ch_names = list(LEEUWIS2021_EEG_CHANNELS) + ["STI 014"]
        ch_types = ["eeg"] * len(LEEUWIS2021_EEG_CHANNELS) + ["stim"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = mne.create_info(ch_names, _SFREQ, ch_types)
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.set_montage("standard_1020", on_missing="ignore", verbose=False)

        return raw
