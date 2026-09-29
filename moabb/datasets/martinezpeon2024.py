"""EEG Kinesthetic Motor Imagery force-level dataset (Martinez-Peon, 2024)."""

import mne
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

from .utils import resolve_montage_name


# Figshare article 25773342 hosts 60 plain-text files, one per
# (subject, force-level, attempt). File names encode the class:
# ``userNNN_<level>_<attempt>.txt`` (e.g. ``user004_70_2.txt`` = subject 4,
# 70% MVC, second attempt). The force level in the name is the data-borne
# class label. The Figshare per-file download IDs are pinned below (v1).
MARTINEZPEON2024_BASE_URL = "https://ndownloader.figshare.com/files/"

# Three graded kinesthetic-MI force levels (% of maximal voluntary contraction),
# each recorded in two attempts.
MARTINEZPEON2024_LEVELS = ["10", "40", "70"]
MARTINEZPEON2024_ATTEMPTS = ["1", "2"]

# Figshare file ids (article 25773342, v1) per subject, in the order
# (10, 1), (10, 2), (40, 1), (40, 2), (70, 1), (70, 2) of (level, attempt).
MARTINEZPEON2024_FILE_IDS = {
    1: (46186644, 46186650, 46186647, 46186653, 46186656, 46186659),
    2: (46186662, 46186665, 46186668, 46186671, 46186674, 46186677),
    3: (46186680, 46186683, 46186686, 46186689, 46186692, 46186695),
    4: (46186698, 46186701, 46186704, 46186707, 46186710, 46186713),
    5: (46186716, 46186719, 46186722, 46186725, 46186728, 46186731),
    6: (46186734, 46186737, 46186740, 46186743, 46186746, 46186749),
    7: (46186752, 46186755, 46186758, 46186761, 46186764, 46186767),
    8: (46186770, 46186773, 46186776, 46186779, 46186782, 46186785),
    9: (46186788, 46186791, 46186794, 46186797, 46186800, 46186803),
    10: (46186806, 46186809, 46186812, 46186815, 46186818, 46186821),
}

# Emotiv EPOC, 14 EEG channels, in file-column order (columns 3-16 of each row).
MARTINEZPEON2024_CHANNELS = "AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4".split()

# The 14 EEG channels occupy 0-based columns 2..15; col 0 = Time, col 1 = Sample,
# cols 16-17 = gyroscope (GX/GY), col 18 = time (s), cols 19+ = zeros.
MARTINEZPEON2024_EEG_COLS = list(range(2, 16))

MARTINEZPEON2024_SFREQ = 128.0

# Five kinesthetic-MI cues per file at fixed protocol times (s); each 5 s long.
MARTINEZPEON2024_ONSETS = [2.9, 10.9, 18.9, 26.9, 34.9]
MARTINEZPEON2024_TRIAL_DUR = 5.0


class MartinezPeon2024(BaseDataset):
    """Kinesthetic motor imagery at graded force levels [1]_.

    **Dataset description**

    EEG recorded while 10 healthy subjects performed kinesthetic motor imagery
    (KMI, imagining the somatosensory sensations of the movement) of squeezing a
    ball with the right hand at 10%, 40% and 70% of their maximal voluntary
    contraction (MVC), treated here as three classes. Signals were acquired
    with a 14-channel Emotiv EPOC at 128 Hz. Each ~40 s recording holds five
    5 s KMI cues at 2.9, 10.9, 18.9, 26.9 and 34.9 s; each level is recorded
    twice, giving six runs (``"<index>lvl<level>rep<attempt>"``) in session
    ``"0"`` and 10 trials per class.

    The class label is the force level in the file name
    (``userNNN_<level>_<attempt>.txt``); the files carry no trigger channel, so
    cue onsets follow the fixed acquisition protocol. The published fourth
    "basal" (rest) class is not separately marked and is not exposed. Stored
    amplitudes are raw microvolts (large DC offset) converted to volts; the
    gyroscope and time columns are discarded.

    References
    ----------

    .. [1] Martinez-Peon, D. (2024). EEG Kinesthetic motor imagery levels.
       figshare. Dataset. DOI: https://doi.org/10.6084/m9.figshare.25773342
       Associated article: https://doi.org/10.1088/1741-2552/ad5f27

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            channel_types={"eeg": 14},
            montage="standard_1020",
            hardware="Emotiv EPOC",
            sensor_type="wet",
            electrode_type="saline",
            line_freq=60.0,
            sensors=list(MARTINEZPEON2024_CHANNELS),
        ),
        participants=ParticipantMetadata(
            n_subjects=10, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["level_10", "level_40", "level_70"],
            trials_per_class={"level_10": 10, "level_40": 10, "level_70": 10},
            trial_duration=5.0,
            study_design=(
                "Kinesthetic motor imagery of a right-hand ball squeeze at "
                "10/40/70% of maximal voluntary contraction. Each 40 s "
                "recording carries five KMI cues at 2.9/10.9/18.9/26.9/34.9 s "
                "(5 s each); each force level is recorded twice per subject. "
                "The force level is encoded in the file name."
            ),
            feedback_type="none",
            synchronicity="cue-based",
            mode="offline",
            events={"level_10": 1, "level_40": 2, "level_70": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.25773342.v1",
            description=(
                "EEG kinesthetic motor imagery at three graded hand-grip force "
                "levels (10/40/70% MVC) from 10 healthy subjects, Emotiv EPOC, "
                "14 channels, 128 Hz."
            ),
            investigators=["Dulce Martinez-Peon"],
            country="MX",
            data_url="https://doi.org/10.6084/m9.figshare.25773342",
            associated_paper_doi="10.1088/1741-2552/ad5f27",
            publication_year=2024,
            keywords=[
                "motor imagery",
                "kinesthetic motor imagery",
                "force level",
                "hand grip",
                "EEG",
                "BCI",
            ],
            license="CC-BY-4.0",
            repository="Figshare",
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="TXT",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10 + 1)),
            sessions_per_subject=1,
            events={"level_10": 1, "level_40": 2, "level_70": 3},
            code="MartinezPeon2024",
            interval=(0, 5),
            paradigm="imagery",
            doi="10.6084/m9.figshare.25773342.v1",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the six .txt paths, ordered 10_1, 10_2, 40_1, 40_2, 70_1, 70_2."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        return [
            str(
                dl.data_dl(
                    MARTINEZPEON2024_BASE_URL + str(file_id),
                    self.code,
                    path=path,
                    force_update=force_update,
                    verbose=verbose,
                )
            )
            for file_id in MARTINEZPEON2024_FILE_IDS[subject]
        ]

    def _read_run(self, file_path, label):
        """Build one Raw (one force-level recording) with five KMI events."""
        # Whitespace-delimited, no header; keep only the 14 EEG columns (uV -> V).
        data = pd.read_csv(
            file_path, sep=r"\s+", header=None, usecols=MARTINEZPEON2024_EEG_COLS
        ).to_numpy(dtype=float)
        info = mne.create_info(
            MARTINEZPEON2024_CHANNELS, MARTINEZPEON2024_SFREQ, ch_types="eeg"
        )
        raw = mne.io.RawArray(data.T * 1e-6, info, verbose=False)
        # All 14 Emotiv EPOC channels are standard 10-20 sites; "raise" makes a
        # future name mismatch fail loudly instead of dropping locations.
        raw.set_montage(
            resolve_montage_name("colin27_1020"), on_missing="raise", verbose=False
        )

        duration = raw.n_times / MARTINEZPEON2024_SFREQ
        onsets = [o for o in MARTINEZPEON2024_ONSETS if o < duration]
        annotations = mne.Annotations(
            onset=onsets,
            duration=[MARTINEZPEON2024_TRIAL_DUR] * len(onsets),
            description=[label] * len(onsets),
        )
        raw.set_annotations(annotations, verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        # Run keys: unique 0-based recording index + letters/digits description.
        keys = [
            (lv, at) for lv in MARTINEZPEON2024_LEVELS for at in MARTINEZPEON2024_ATTEMPTS
        ]
        runs = {
            f"{idx}lvl{level}rep{attempt}": self._read_run(path, f"level_{level}")
            for idx, ((level, attempt), path) in enumerate(
                zip(keys, self.data_path(subject))
            )
        }
        return {"0": runs}
