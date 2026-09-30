"""Kueper2024 unilateral vs bilateral movement-execution dataset (Kueper et al. 2024)."""

import warnings
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
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)

from .utils import download_and_extract_subject_zip


# Single zip on Zenodo (record 10229480), ~3.1 GB, BrainVision format.
Kueper2024_URL = "https://zenodo.org/api/records/10229480/files/EEG_dataset.zip/content"

# Subject pseudo-codes (alphabetical) mapped to subjects 1..8.
SUBJECT_CODES = ["AV82", "JD68", "JV43", "QS70", "RA12", "UP28", "XP01", "ZS27"]

# 64 EEG channel names in acquisition order (from the BrainVision headers).
EEG_CHANNELS = (
    "Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 TP9 CP5 CP1 CP2 CP6 "
    "TP10 P7 P3 Pz P4 P8 PO9 O1 Oz O2 PO10 AF7 AF3 AF4 AF8 F5 F1 F2 F6 FT9 FT7 "
    "FC3 FC4 FT8 FT10 C5 C1 C2 C6 TP7 CP3 CPz CP4 TP8 P5 P1 P2 P6 PO7 PO3 POz "
    "PO4 PO8"
).split()

# 3-axis accelerometer channels (treated as misc, not EEG).
ACCEL_CHANNELS = ["x_dir", "y_dir", "z_dir"]

# Movement-onset (motion-tracking) marker per condition: right arm S100 in the
# unilateral (right-arm-only) recordings, left arm S101 in the bilateral ones.
ONSET_MARKER = {"unilateral": "S100", "bilateral": "S101"}


class Kueper2024(BaseDataset):
    """Unilateral vs bilateral movement-execution EEG dataset [1]_, [2]_.

    Eight healthy participants performed self-initiated, self-paced reaching
    movements with the right arm only (``unilateral``, button press) or with both
    arms (``bilateral``), 3 sets of 40 movements per condition (subject ``XP01``
    has an extra unilateral set), recorded with a 64-channel LiveAmp at 500 Hz plus
    a 3-axis accelerometer (``misc``). The two conditions are the two classes, so
    they share one session and every recording set is a run. One event per trial
    is placed at the motion-tracking movement onset (``S100`` right arm in
    unilateral, ``S101`` left arm in bilateral recordings); other markers are
    dropped.

    Paper audit (2026-09-30): the 500 Hz rate applies to the EEG (the paper's
    2000 Hz figure is the Cometa EMG, which is not part of the release); the
    paper states "3 sets of 40 self-initiated movements" per task, i.e. 120
    trials per class by design, and 6 sets per subject (7 for ``XP01``). The
    ground electrode (``AFz``) is not stated in the paper.

    References
    ----------

    .. [1] Kueper, N., Kim, S. K., & Kirchner, E. A. (2023). EEG Dataset of
       Unilateral and Bilateral Movement Executions [Data set]. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.10229480

    .. [2] Kueper, N., Kim, S. K., & Kirchner, E. A. (2024). Avoidance of
       specific calibration sessions in motor intention recognition for
       exoskeleton-supported rehabilitation through transfer learning on EEG
       data. Scientific Reports, 14(1), 16690.
       DOI: https://doi.org/10.1038/s41598-024-65910-8

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types={"eeg": 64, "misc": 3},
            montage="extended 10-20",
            hardware="Brain Products LiveAmp64 (wireless, active electrodes)",
            cap_manufacturer="Brain Products GmbH",
            cap_model="Acticap slim",
            sensor_type="active",
            electrode_type="active",
            reference="FCz",
            ground="AFz",
            sensors=EEG_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=False),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="healthy",
            gender={"male": 4, "female": 4},
            age_mean=25.5,
            age_std=4.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["unilateral", "bilateral"],
            trial_duration=3.0,
            study_design="Self-initiated, self-paced reaching movements in two "
            "conditions: unilateral (right arm only, button press) and bilateral "
            "(both arms synchronously). 3 sets of 40 movements per condition.",
            feedback_type="none",
            synchronicity="self-paced",
            mode="offline",
            events={"unilateral": 1, "bilateral": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41598-024-65910-8",
            description="EEG dataset of self-initiated unilateral (right arm) and "
            "bilateral movement executions from 8 healthy subjects, for movement "
            "intention recognition in exoskeleton-supported rehabilitation.",
            investigators=["Niklas Kueper", "Su Kyoung Kim", "Elsa Andrea Kirchner"],
            institution="German Research Centre for Artificial Intelligence (DFKI)",
            country="DE",
            data_url="https://doi.org/10.5281/zenodo.10229480",
            publication_year=2024,
            keywords=[
                "EEG",
                "movement intention",
                "ERP",
                "BCI",
                "stroke rehabilitation",
                "LRP",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        tags=Tags(modality=["Motor"], type=["Movement Execution"]),
        data_structure=DataStructureMetadata(
            n_trials=240,
            n_trials_per_class={"unilateral": 120, "bilateral": 120},
            n_blocks=6,
            trials_context=(
                "3 sets of 40 self-initiated movements per task (unilateral, "
                "bilateral) by design; one run per set, 6 runs per subject "
                "(subject XP01 has an extra unilateral set in the release)."
            ),
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            highpass_hz=0.1,
            lowpass_hz=131.0,
            notes="Hardware-prefiltered by the amplifier to 0.1-131 Hz; no further "
            "preprocessing applied.",
        ),
        file_format="BrainVision",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, len(SUBJECT_CODES) + 1)),
            sessions_per_subject=1,
            events={"unilateral": 1, "bilateral": 2},
            code="Kueper2024",
            interval=(-2.0, 1.0),
            paradigm="imagery",
            doi="10.1038/s41598-024-65910-8",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of BrainVision header paths for a single subject."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        data_dir = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        root = data_dir / "EEG_dataset"
        if force_update or not root.exists():
            download_and_extract_subject_zip(
                Kueper2024_URL,
                self.code,
                data_dir,
                path,
                force_update,
                verbose,
                redownload_corrupted=True,
            )
        code = SUBJECT_CODES[subject - 1]
        return [
            str(p)
            for condition in ("unilateral", "bilateral")
            for p in sorted((root / "EEG" / condition / code).glob("*.vhdr"))
        ]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        runs = {}
        for run_idx, vhdr in enumerate(self.data_path(subject)):
            condition = "bilateral" if "bilateral" in Path(vhdr).parts else "unilateral"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            raw.set_channel_types(
                {ch: "misc" for ch in ACCEL_CHANNELS if ch in raw.ch_names}
            )
            # Keep only the movement-onset markers, relabelled by condition.
            keep = [
                d.replace(" ", "").split("/")[-1] == ONSET_MARKER[condition]
                for d in raw.annotations.description
            ]
            onsets = raw.annotations.onset[keep]
            raw.set_annotations(
                mne.Annotations(onsets, [0.0] * len(onsets), [condition] * len(onsets))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage("colin27_1020", match_case=False, on_missing="ignore")
            runs[f"{run_idx}{condition}"] = raw
        return {"0": runs}
