"""Martinez Peon 2025 graded-force kinesthetic motor imagery (hand-grip) dataset."""

import json

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


# Mendeley Data public-api record; the ``files`` array of the returned JSON
# carries a (long-lived) signed ``download_url`` for every file.
MARTINEZPEON2025_RECORD = "https://data.mendeley.com/public-api/datasets/msgzn862ns"

# The four graded-force conditions, encoded in the file name (Sxx_<level>_KMI.csv).
MARTINEZPEON2025_LEVELS = ["10", "40", "70", "100"]
MARTINEZPEON2025_EVENTS = {"grip_10": 1, "grip_40": 2, "grip_70": 3, "grip_100": 4}

# 20 Cognionics Quick-20m electrodes. The CSV header uses the legacy T3/T4
# labels; MNE's standard montages use the modern T7/T8, so we rename on load.
MARTINEZPEON2025_RENAME = {"T3": "T7", "T4": "T8"}
MARTINEZPEON2025_SENSORS = (
    "A2 C3 C4 Cz F3 F4 F7 F8 Fp1 Fp2 Fz O1 O2 P3 P4 P7 P8 Pz T7 T8".split()
)

# Sampling rate: not stated in the Mendeley record; every recording is 84 s
# long with 42000 samples -> 500 Hz.
MARTINEZPEON2025_SFREQ = 500.0

# 4 s initial rest, then 10 repetitions of 4 s task + 4 s rest (= 84 s): onsets
# 4, 12, ..., 76 s. The CSVs carry no trigger/marker channel, so this
# initial-rest-first ordering is inferred from the source description ("with
# initial and final resting period") rather than confirmed from an in-file
# marker; a task-first layout (onsets 0, 8, ..., 72) is arithmetically
# equivalent at 84 s.
MARTINEZPEON2025_ONSETS = [4.0 + 8.0 * i for i in range(10)]
MARTINEZPEON2025_TASK = 4.0


class MartinezPeon2025(BaseDataset):
    """Graded-force kinesthetic motor imagery of hand-grip [1]_.

    **Dataset description**

    Raw EEG recorded during kinesthetic motor imagery (KMI) of a hand-grip at
    10%, 40%, 70% and 100% of the maximal voluntary contraction (MVC) from 50
    healthy right-handed university students (23 females, 27 males, 17-30
    years, mean 21.78 +/- 2.66); subjects S01-S25 used the left hand and
    S26-S50 the right hand. Each force level is one 84 s recording (10
    repetitions of 4 s imagery followed by 4 s rest, with initial and final
    resting periods; rest corresponds to the 0% level), loaded as one run with
    10 trials of its class. EEG was recorded with a 20-channel Cognionics
    Quick-20m headset (electrodes Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz,
    C4, T8, A2, P7, P3, Pz, P4, P8, O1, O2; the record does not state the
    reference, and A2 is one of the 20 recorded columns) without filtering or
    pre-processing. The sampling rate is not stated in the record and is
    inferred from the 42000 samples per 84 s file (500 Hz). The legacy T3/T4
    labels are renamed to T7/T8 and microvolts are converted to volts on load.

    The Mendeley record has no linked paper; its summary sentence lists the
    levels as 0%, 40%, 70% and 100% while its experimental-design section
    lists 0%, 10%, 40%, 70% and 100% and each subject folder holds four KMI
    files (10/40/70/100). The dataset has no official name, so the class
    follows the first-author-surname + record-year rule (Martinez Peon, 2025).

    References
    ----------

    .. [1] Martinez Peon, D. C., & Perez Espinoza, M. (2025). EEG signals of
       KMI levels of the right and left hands. Mendeley Data, V1.
       DOI: https://doi.org/10.17632/msgzn862ns.1

    Notes
    -----

    .. versionadded:: 1.1.1

    Each class comes from a single continuous 84 s recording per subject
    (class==recording-block confound): within-recording CV can exploit recording-level
    nonstationarity instead of imagery content; prefer cross-subject evaluation.

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types={"eeg": 20},
            montage="10-20",
            hardware="Cognionics Quick-20m dry-electrode EEG headset",
            sensor_type="dry",
            electrode_type="dry",
            reference=None,
            ground=None,
            sensors=MARTINEZPEON2025_SENSORS,
        ),
        participants=ParticipantMetadata(
            n_subjects=50,
            health_status="healthy",
            gender={"male": 27, "female": 23},
            age_mean=21.78,
            age_std=2.66,
            age_min=17.0,
            age_max=30.0,
            handedness={"right": 50},
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=list(MARTINEZPEON2025_EVENTS),
            trials_per_class=dict.fromkeys(MARTINEZPEON2025_EVENTS, 10),
            trial_duration=4.0,
            study_design=(
                "Kinesthetic motor imagery of a hand-grip at 10/40/70/100% of "
                "maximal voluntary contraction. One 84 s continuous recording "
                "per force level: 10 repetitions of 4 s task followed by 4 s "
                "rest, with initial and final resting periods (rest = 0% level). "
                "S01-S25 left hand, S26-S50 right hand."
            ),
            feedback_type="none",
            synchronicity="cue-based",
            mode="offline",
            events=MARTINEZPEON2025_EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.17632/msgzn862ns.1",
            description=(
                "Raw, unfiltered EEG during graded-force kinesthetic motor "
                "imagery of hand-grip (10/40/70/100% MVC) from 50 healthy "
                "right-handed students, Cognionics Quick-20m, 20 channels, "
                "500 Hz (inferred)."
            ),
            investigators=["Dulce Citlalli Martinez Peon", "Marcos Perez Espinoza"],
            country="MX",
            data_url="https://data.mendeley.com/datasets/msgzn862ns/1",
            publication_year=2025,
            keywords=[
                "motor imagery",
                "kinesthetic motor imagery",
                "hand grip",
                "force level",
                "EEG",
                "BCI",
            ],
            license="CC-BY-NC-SA-4.0",
            repository="Mendeley Data",
            how_to_acknowledge=(
                "Martinez Peon, D. C., & Perez Espinoza, M. (2025). EEG signals "
                "of KMI levels of the right and left hands. Mendeley Data, V1. "
                "doi:10.17632/msgzn862ns.1"
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="CSV",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events=dict(MARTINEZPEON2025_EVENTS),
            code="MartinezPeon2025",
            interval=(0, 4 - 1 / 500),
            paradigm="imagery",
            doi="10.17632/msgzn862ns.1",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the four CSV paths of a subject, ordered 10/40/70/100% MVC."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        kwargs = {"path": path, "force_update": force_update, "verbose": verbose}
        # The record JSON maps every file name to a signed ``download_url``.
        record = dl.data_dl(MARTINEZPEON2025_RECORD, self.code, **kwargs)
        with open(record, encoding="utf-8") as fid:
            files = json.load(fid)["files"]
        urls = {f["filename"]: f["content_details"]["download_url"] for f in files}
        return [
            str(dl.data_dl(urls[f"S{subject:02d}_{level}_KMI.csv"], self.code, **kwargs))
            for level in MARTINEZPEON2025_LEVELS
        ]

    def _read_run(self, file_path, label):
        """Build a single ``Raw`` (one force level) with 10 imagery events."""
        df = pd.read_csv(file_path)
        if len(df) < round(84 * MARTINEZPEON2025_SFREQ):
            raise ValueError("Expected a complete 84-second force-level recording")
        ch_names = [MARTINEZPEON2025_RENAME.get(c, c) for c in df.columns]
        info = mne.create_info(ch_names, MARTINEZPEON2025_SFREQ, ch_types="eeg")
        # microvolts -> volts
        raw = mne.io.RawArray(df.to_numpy(dtype=float).T * 1e-6, info, verbose=False)
        raw.set_montage("colin27_1020", on_missing="ignore", verbose=False)
        n_trials = len(MARTINEZPEON2025_ONSETS)
        annotations = mne.Annotations(
            onset=MARTINEZPEON2025_ONSETS,
            duration=[MARTINEZPEON2025_TASK] * n_trials,
            description=[label] * n_trials,
        )
        raw.set_annotations(annotations, verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        paths = self.data_path(subject)
        runs = {
            str(idx): self._read_run(path, f"grip_{level}")
            for idx, (path, level) in enumerate(zip(paths, MARTINEZPEON2025_LEVELS))
        }
        return {"0": runs}
