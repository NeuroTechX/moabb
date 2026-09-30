"""IMUMIA2026: the IMU-MI_A multi-paradigm motor-imagery EEG dataset (Li et al. 2026)."""

import logging
import re
from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)

from .utils import download_and_extract_subject_zip


log = logging.getLogger(__name__)

# Public Zenodo release "MI_A_Dataset.zip" (~2.0 GB). This is a 5-subject sample
# (Sub_01, Sub_50, Sub_100, Sub_150, Sub_200) of a larger on-request dataset of
# 244 participants (Zenodo description) collected at Inner Mongolia University.
IMUMIA2026_URL = "https://zenodo.org/records/20421767/files/MI_A_Dataset.zip"

# Archive task folder -> (MOABB run name, class for cue code 1, class for code 2).
# Run names start with the recording order. The two Classic-Arrow cue codes map
# to the two sides of each task's body part following the authors' consistently
# documented condition order ("Left ... MI" listed before "Right ... MI") in
# every task-Task*_events.json.
_TASKS = {
    "task1": ("0hand", "left_hand", "right_hand"),
    "task2": ("1foot", "left_foot", "right_foot"),
    "task3": ("2thumb", "left_thumb", "right_thumb"),
    "task4": ("3indexfinger", "left_index", "right_index"),
    "task5": ("4pinch", "left_pinch", "right_pinch"),
}

# Globally unique integer code per exposed class label (left_hand=1 ... right_pinch=10).
_EVENTS = {
    label: code
    for code, label in enumerate(
        (lab for _, *sides in _TASKS.values() for lab in sides), start=1
    )
}

# 64 EEG channel labels in acquisition order (from the Curry .dpa SensorLabels).
# fmt: off
_EEG_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
]
# fmt: on

# 5 non-EEG channels trailing the montage, with their MNE channel types. The
# Trigger line is kept as ``misc`` (not ``stim``) so that MOABB reads the
# per-trial labels from the Curry event annotations rather than from the analog
# trigger channel.
_MISC_TYPES = {"HEO": "eog", "VEO": "eog", "EKG": "ecg", "EMG": "emg", "Trigger": "misc"}


def _subject_number(cdt_path):
    return int(re.search(r"Sub_(\d+)", cdt_path.name).group(1))


def _dpa_parameter(dpa, name, cdt_path):
    match = re.search(rf"(?m)^\s*{name}\s*=\s*(\S+)", dpa)
    if match is None:
        raise ValueError(f"Missing {name} in Curry sidecar {cdt_path}.dpa")
    return match.group(1)


def _dpa_labels(dpa, section):
    match = re.search(rf"{section} START_LIST.*?\n(.*?){section} END_LIST", dpa, re.S)
    if match is None:
        return []
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


class IMUMIA2026(BaseDataset):
    """Multi-paradigm motor-imagery EEG dataset (IMU-MI_A) [1]_.

    **Dataset description**

    Motor-imagery EEG covering diverse cognitive states, recorded with a
    64-channel Neuroscan system at 1000 Hz plus HEO/VEO (EOG), EKG, EMG and
    Trigger channels. The public Zenodo release is a five-subject sample
    (Sub_01, Sub_50, Sub_100, Sub_150, Sub_200) of a 244-participant dataset
    (the record: "synchronized EEG and EMG data from 244 participants";
    named ``IMU-MI_A`` by its authors) available on request; ``n_subjects``
    counts the released sample. Each subject performed five left/right tasks (hand,
    foot, thumb, index finger, index-thumb pinch), exposed as five runs of one
    session; trials are labelled by body part and side (ten classes).

    Each recording holds two paradigms separated by a boundary marker
    (800000/800001). Only the **Classic Arrow** cues (code 1 = left, 2 = right,
    12 trials per side; 1 s fixation, 1 s cue, 4 s imagery) are labelled and
    epoched. The **Cue-Execution Dual-Stage** markers (3/4 and phase markers
    13/23/33, 14/24/34) stay in ``raw.annotations`` unlabelled, because the
    released sample's marker timing does not match the documented 10 s trial
    and their side mapping cannot be verified. Curry files are read with
    :func:`mne.io.read_raw_curry`, or with a float32 sidecar reader when the
    optional ``curryreader`` dependency is missing.

    References
    ----------

    .. [1] Li, J., Wang, C., and Chen, C. (2026). A Human Motor Imagery EEG
       Dataset Covering Diverse Cognitive States and Neural Response Patterns.
       Zenodo. DOI: https://doi.org/10.5281/zenodo.20421767

    Notes
    -----

    The numeric event code to left/right assignment follows the authors'
    documented condition order (the "Left ... MI" condition is listed before the
    "Right ... MI" condition in every ``task-Task*_events.json``); the archive
    ships no explicit trigger code book. Users are advised to verify laterality
    against the EMG/EOG channels before publication.

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 64, "eog": 2, "ecg": 1, "emg": 1, "misc": 1},
            montage="standard_1020",
            hardware="Neuroscan SynAmps (64-channel Quik-Cap)",
            sensor_type="Ag/AgCl",
            electrode_type="passive",
            reference="Cz",
            ground="forehead",
            sensors=list(_EEG_CHANNELS),
            line_freq=50.0,
            impedance_threshold_kohm=10.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["HEO", "VEO"],
                has_emg=True,
                emg_channels=1,
                other_physiological=["ECG"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=5, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=10,
            class_labels=list(_EVENTS.keys()),
            trial_duration=6.0,
            study_design="Five motor-imagery tasks (left/right hand, foot, thumb, "
            "index finger and index-thumb pinch), each recorded under a Classic "
            "Arrow paradigm and a Cue-Execution dual-stage paradigm.",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events=dict(_EVENTS),
            instructions="Perform the cued left- or right-side motor imagery of "
            "the task's body part following the arrow direction.",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.20421767",
            description="Human motor-imagery EEG covering diverse cognitive states: "
            "five body-part tasks (hand, foot, thumb, index finger, pinch) under "
            "two paradigms, 64-channel Neuroscan at 1000 Hz. Public five-subject "
            "sample of a 244-participant dataset.",
            investigators=["Jianxiu Li", "Changming Wang", "Chao Chen"],
            institution="Inner Mongolia University",
            institution_address="Inner Mongolia, China",
            country="CN",
            data_url="https://doi.org/10.5281/zenodo.20421767",
            publication_year=2026,
            keywords=["EEG", "motor imagery", "BCI", "fine motor imagery"],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=5,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=5,
            trials_context="Single session with five runs (one motor-imagery task "
            "each). Only the Classic Arrow cues (12 trials per side and task) are "
            "labelled by default; the dual-stage paradigm markers are kept in the "
            "annotations.",
        ),
        file_format="Curry",
        data_processed=False,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 6)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="IMUMIA2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.20421767",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the five ``.cdt`` paths (one per task) of a subject.

        Subjects 1-5 address the sampled participants in ascending file order.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # The archive nests everything under MI_A_Dataset/MI_A_Dataset/Raw_data/.
        data_dir = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        raw_dir = data_dir / "MI_A_Dataset" / "MI_A_Dataset" / "Raw_data"
        if force_update or not raw_dir.exists():
            download_and_extract_subject_zip(
                IMUMIA2026_URL, self.code, data_dir, path, force_update, verbose
            )

        paths = []
        for task in _TASKS:
            task_dir = raw_dir / task
            cdt_files = sorted(task_dir.glob("*.cdt"), key=_subject_number)
            if len(cdt_files) < subject:
                raise FileNotFoundError(
                    f"Expected at least {subject} .cdt files under {task_dir}, "
                    f"found {len(cdt_files)}"
                )
            paths.append(str(cdt_files[subject - 1]))
        return paths

    def _get_single_subject_data(self, subject):
        runs = {
            _TASKS[task][0]: self._load_curry(cdt_path, task)
            for task, cdt_path in zip(_TASKS, self.data_path(subject))
        }
        return {"0": runs}

    @staticmethod
    def _load_curry(cdt_path, task):
        """Read one Curry recording and label its Classic Arrow cues."""
        try:
            raw = mne.io.read_raw_curry(cdt_path, preload=True, verbose="ERROR")
        except RuntimeError as exc:
            if "curryreader" not in str(exc):
                raise
            log.info("curryreader unavailable; using the Curry sidecar fallback")
            raw = IMUMIA2026._read_legacy_curry(cdt_path)

        # Mark the trailing non-EEG channels (EOG/ECG/EMG/Trigger) by type.
        present = {ch: t for ch, t in _MISC_TYPES.items() if ch in raw.ch_names}
        if present:
            raw.set_channel_types(present, verbose="ERROR")

        # Rename the two Classic Arrow cue codes to this task's side labels
        # (1 -> left, 2 -> right). Dual-stage codes (3, 4 and phase markers) and
        # the paradigm-boundary markers are left untouched in the annotations.
        _, left_label, right_label = _TASKS[task]
        descriptions = set(raw.annotations.description)
        rename = {
            code: label
            for code, label in (("1", left_label), ("2", right_label))
            if code in descriptions
        }
        if rename:
            raw.annotations.rename(rename)

        return raw

    @staticmethod
    def _read_legacy_curry(cdt_path):
        """Read this release's float32 Curry recording without curryreader.

        MNE 1.11 delegates Curry files to an optional dependency.  The IMUMIA2026
        release instead has a simple sample-major float32 ``.cdt`` payload and
        text ``.dpa``/``.ceo`` sidecars, which can be read losslessly here.
        """
        cdt_path = Path(cdt_path)
        dpa = cdt_path.with_suffix(cdt_path.suffix + ".dpa").read_text(
            encoding="utf-8-sig"
        )
        ceo_path = cdt_path.with_suffix(cdt_path.suffix + ".ceo")
        n_samples = int(_dpa_parameter(dpa, "NumSamples", cdt_path))
        n_channels = int(_dpa_parameter(dpa, "NumChannels", cdt_path))
        sfreq = float(_dpa_parameter(dpa, "SampleFreqHz", cdt_path))
        ch_names = _dpa_labels(dpa, "LABELS") + _dpa_labels(dpa, "LABELS_OTHERS")
        if len(ch_names) != n_channels:
            raise ValueError(
                f"Curry sidecar lists {len(ch_names)} channels, expected {n_channels}"
            )
        data = np.fromfile(cdt_path, dtype="<f4")
        if data.size != n_samples * n_channels:
            raise ValueError(
                f"Curry data has {data.size} values, expected {n_samples * n_channels}"
            )
        info = mne.create_info(ch_names, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(
            data.reshape(n_samples, n_channels).T * 1e-6, info, verbose="ERROR"
        )

        if ceo_path.is_file():
            ceo = ceo_path.read_text(encoding="utf-8-sig")
            match = re.search(
                r"NUMBER_LIST START_LIST.*?\n(.*?)NUMBER_LIST END_LIST", ceo, re.S
            )
            if match is not None:
                events = [
                    line.split() for line in match.group(1).splitlines() if line.strip()
                ]
                onset = [int(event[0]) / sfreq for event in events]
                description = [event[2] for event in events]
                raw.set_annotations(
                    mne.Annotations(onset, [0.0] * len(onset), description)
                )
        return raw
