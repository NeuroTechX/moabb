"""Li2026 multi-paradigm motor-imagery EEG dataset (IMU-MI_A)."""

import logging
import zipfile
from pathlib import Path

import mne

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


log = logging.getLogger(__name__)

# Public Zenodo release "MI_A_Dataset.zip" (~2.0 GB). This is a 5-subject sample
# (Sub_01, Sub_50, Sub_100, Sub_150, Sub_200) of a larger on-request dataset of
# 242 participants collected at Inner Mongolia University.
LI2026_URL = "https://zenodo.org/records/20421767/files/MI_A_Dataset.zip"

# Folder names inside the archive, one per body-part motor-imagery task.
_TASKS = ["task1", "task2", "task3", "task4", "task5"]

# MOABB run names must start with an integer (recording order) optionally
# followed by an alphanumeric description.
_RUN_NAMES = {
    "task1": "0hand",
    "task2": "1foot",
    "task3": "2thumb",
    "task4": "3indexfinger",
    "task5": "4pinch",
}

# The two Classic-Arrow cue codes (1 = first-listed condition = left,
# 2 = second-listed condition = right) map to the two sides of each task's body
# part. Codes follow the authors' consistently documented condition order
# ("Left ... MI" listed before "Right ... MI") in every task-Task*_events.json.
_TASK_SIDES = {
    "task1": ("left_hand", "right_hand"),
    "task2": ("left_foot", "right_foot"),
    "task3": ("left_thumb", "right_thumb"),
    "task4": ("left_index", "right_index"),
    "task5": ("left_pinch", "right_pinch"),
}

# Globally unique integer code per exposed class label.
_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
    "left_foot": 3,
    "right_foot": 4,
    "left_thumb": 5,
    "right_thumb": 6,
    "left_index": 7,
    "right_index": 8,
    "left_pinch": 9,
    "right_pinch": 10,
}

# 64 EEG channel labels in acquisition order (from the Curry .dpa SensorLabels).
_EEG_CHANNELS = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "M1",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "M2",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]

# 5 non-EEG channels trailing the montage, with their MNE channel types. The
# Trigger line is kept as ``misc`` (not ``stim``) so that MOABB reads the
# per-trial labels from the Curry event annotations rather than from the analog
# trigger channel.
_MISC_TYPES = {"HEO": "eog", "VEO": "eog", "EKG": "ecg", "EMG": "emg", "Trigger": "misc"}


class Li2026(BaseDataset):
    """Multi-paradigm motor-imagery EEG dataset (IMU-MI_A) [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Li2026           5       64          10                 12            4s           1000 Hz            1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Motor-imagery EEG covering diverse cognitive states, recorded with a
    64-channel Neuroscan system (10-20 layout, Cz reference, forehead ground)
    at 1000 Hz, with two synchronous EOG channels (HEO, VEO), one ECG channel
    (EKG), one EMG channel and a Trigger channel. The public Zenodo release is a
    five-subject sample (subjects Sub_01, Sub_50, Sub_100, Sub_150, Sub_200) of a
    larger dataset of 242 participants that the authors provide on request.

    Each subject performed five motor-imagery tasks, one per recording file,
    spanning gross and fine movements: left/right hand (``task1``), left/right
    foot (``task2``), left/right thumb (``task3``), left/right index finger
    (``task4``) and left/right index-thumb pinch (``task5``). This loader exposes
    the five tasks as five runs of a single session, and labels every trial by
    its body part and side, giving ten distinct classes overall (two per task).

    Each recording contains two paradigms run back to back and separated by a
    boundary marker (codes 800000/800001):

    - **Classic Arrow paradigm**: a 1 s fixation, a 1 s left/right arrow cue and a
      4 s imagery period, with one cue marker per trial (code 1 = left, code 2 =
      right, 12 trials per side). This loader epochs these clean, one-per-trial
      cues by default; the marker cadence is 5.0 s and the exposed interval spans
      a 4 s imagery window from the cue.
    - **Cue-Execution Dual-Stage paradigm**: a longer trial adding a countdown and
      a slight real movement after imagery (base code 3 = left, 4 = right, 17
      trials per side, plus phase sub-markers 13/23/33 and 14/24/34). These
      markers are preserved in ``raw.annotations`` but are not epoched by default,
      because the released sample's marker-to-trial timing does not match the
      documented 10 s trial structure and the numeric-to-side mapping cannot be
      verified from the shipped files.

    Data are Compumedics Neuroscan Curry files (``.cdt`` plus ``.cdt.dpa`` and
    ``.cdt.ceo`` sidecars) read with :func:`mne.io.read_raw_curry`; per-trial
    event codes live in the ``.cdt.ceo`` event file. Electrode positions are the
    real 3-D coordinates stored in the Curry files.

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

    .. versionadded:: 1.5.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
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
            "sample of a 242-participant dataset.",
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
            code="Li2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.20421767",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download/extract the shared archive and return the subject's files.

        Parameters
        ----------
        subject : int
            Subject number (1-5), addressing the five sampled participants in
            ascending file order (Sub_01, Sub_50, Sub_100, Sub_150, Sub_200).
        path : None | str
            Storage location override.
        force_update : bool
            Re-download even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of str
            The five ``.cdt`` file paths for this subject, one per task.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        zip_path = Path(dl.data_dl(LI2026_URL, self.code, path, force_update, verbose))
        extract_dir = zip_path.parent

        # The archive nests everything under MI_A_Dataset/MI_A_Dataset/Raw_data/.
        raw_dir = extract_dir / "MI_A_Dataset" / "MI_A_Dataset" / "Raw_data"
        if not raw_dir.is_dir():
            log.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        paths = []
        for task in _TASKS:
            task_dir = raw_dir / task
            cdt_files = sorted(task_dir.glob("*.cdt"))
            if len(cdt_files) < subject:
                raise FileNotFoundError(
                    f"Expected at least {subject} .cdt files under {task_dir}, "
                    f"found {len(cdt_files)}"
                )
            paths.append(str(cdt_files[subject - 1]))
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-5).

        Returns
        -------
        dict
            ``{"0": {run: Raw}}`` with one run per motor-imagery task (run names
            ``0hand``, ``1foot``, ``2thumb``, ``3indexfinger``, ``4pinch``).
        """
        cdt_paths = self.data_path(subject)

        runs = {}
        for task, cdt_path in zip(_TASKS, cdt_paths):
            runs[_RUN_NAMES[task]] = self._load_curry(cdt_path, task)
        return {"0": runs}

    @staticmethod
    def _load_curry(cdt_path, task):
        """Read one Curry recording and label its Classic Arrow cues."""
        raw = mne.io.read_raw_curry(cdt_path, preload=True, verbose="ERROR")

        # Mark the trailing non-EEG channels (EOG/ECG/EMG/Trigger) by type.
        present = {ch: t for ch, t in _MISC_TYPES.items() if ch in raw.ch_names}
        if present:
            raw.set_channel_types(present, verbose="ERROR")

        # Rename the two Classic Arrow cue codes to this task's side labels
        # (1 -> left, 2 -> right). Dual-stage codes (3, 4 and phase markers) and
        # the paradigm-boundary markers are left untouched in the annotations.
        left_label, right_label = _TASK_SIDES[task]
        descriptions = set(raw.annotations.description)
        rename = {
            code: label
            for code, label in (("1", left_label), ("2", right_label))
            if code in descriptions
        }
        if rename:
            raw.annotations.rename(rename)

        return raw
