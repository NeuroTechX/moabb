"""DFKI2023 unilateral vs bilateral movement-execution dataset (Kueper 2024)."""

import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

import mne

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)


# Single zip on Zenodo (record 10229480), ~3.1 GB, BrainVision format.
DFKI2023_URL = "https://zenodo.org/api/records/10229480/files/EEG_dataset.zip/content"

# Subject pseudo-codes (alphabetical) mapped to subjects 1..8.
SUBJECT_CODES = ["AV82", "JD68", "JV43", "QS70", "RA12", "UP28", "XP01", "ZS27"]

# 64 EEG channel names in acquisition order (from the BrainVision headers).
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT9",
    "FT7",
    "FC3",
    "FC4",
    "FT8",
    "FT10",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CPz",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
]

# 3-axis accelerometer channels (treated as misc, not EEG).
ACCEL_CHANNELS = ["x_dir", "y_dir", "z_dir"]

# Movement-onset (motion-tracking) marker per condition: right arm S100 in the
# unilateral (right-arm-only) recordings, left arm S101 in the bilateral ones.
ONSET_MARKER = {"unilateral": "S100", "bilateral": "S101"}


class DFKI2023(BaseDataset):
    """Unilateral vs bilateral movement-execution EEG dataset [1]_, [2]_.

    .. admonition:: Dataset summary

        =========  =======  =======  =====================  ===============  ===============  ===========
        Name         #Subj    #Chan  #Classes                 #Trials / class  Trials length    Freq (Hz)
        =========  =======  =======  =====================  ===============  ===============  ===========
        DFKI2023         8       64  2 (unilat., bilat.)                ~120              3s          500
        =========  =======  =======  =====================  ===============  ===============  ===========

    **Dataset description**

    EEG recordings from 8 healthy participants (4 male, 4 female; mean age
    25.5 +/- 4.0 years) performing self-initiated, self-paced reaching
    movements in two conditions:

    * ``unilateral`` -- right arm only, pressing a button;
    * ``bilateral`` -- synchronous movement of both arms.

    Each condition consists of 3 sets of 40 self-initiated movements (~120
    trials per condition; subject ``XP01`` has an extra unilateral set). Each
    trial began with a resting period of at least 5 s followed by a
    self-initiated reaching movement. This loader anchors one epoch per trial on
    the motion-tracking movement-onset marker (``S100`` for the right arm in the
    unilateral recordings, ``S101`` for the left arm in the bilateral
    recordings) and labels it by the movement condition, giving a binary
    unilateral-vs-bilateral movement-execution classification problem.

    EEG was acquired with a wireless Brain Products LiveAmp64 amplifier and an
    Acticap slim cap (active electrodes), 64 EEG channels in the extended 10-20
    system (reference FCz, ground AFz) plus a 3-axis accelerometer, at 500 Hz,
    hardware-prefiltered to 0.1-131 Hz.

    Because the two conditions are the two classes, both are pooled into a
    single session; each recording set is exposed as a separate run.

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

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=64,
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
        tags=Tags(modality=["Motor"], type=["Movement Execution"]),
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
        self.events = {"unilateral": 1, "bilateral": 2}
        super().__init__(
            subjects=list(range(1, len(SUBJECT_CODES) + 1)),
            sessions_per_subject=1,
            events=self.events,
            code="DFKI2023",
            interval=(-2.0, 1.0),
            paradigm="imagery",
            doi="10.1038/s41598-024-65910-8",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _condition_of(self, vhdr_path):
        return "bilateral" if "bilateral" in Path(vhdr_path).parts else "unilateral"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of BrainVision header paths for a single subject."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_zip = Path(dl.data_dl(DFKI2023_URL, self.code, force_update=force_update))
        path_folder = path_zip.parent

        # Extract the archive once.
        if not (path_folder / "EEG_dataset").is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(dl.data_dl(DFKI2023_URL, self.code, force_update=True))
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        code = SUBJECT_CODES[subject - 1]
        eeg_root = path_folder / "EEG_dataset" / "EEG"

        subject_paths = []
        for condition in ("unilateral", "bilateral"):
            cond_dir = eeg_root / condition / code
            if cond_dir.is_dir():
                subject_paths.extend(sorted(cond_dir.glob("*.vhdr")))

        return [str(p) for p in subject_paths]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        runs = {}
        for run_idx, vhdr in enumerate(self.data_path(subject)):
            condition = self._condition_of(vhdr)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

            # Tag the 3 accelerometer channels as misc (not EEG).
            accel = [ch for ch in ACCEL_CHANNELS if ch in raw.ch_names]
            if accel:
                raw.set_channel_types(dict.fromkeys(accel, "misc"))

            # Keep only the movement-onset markers, relabelled by condition.
            marker = ONSET_MARKER[condition]
            keep = [marker in d.replace(" ", "") for d in raw.annotations.description]
            onsets = raw.annotations.onset[keep]
            annotations = mne.Annotations(
                onset=onsets,
                duration=[0.0] * len(onsets),
                description=[condition] * len(onsets),
            )
            raw.set_annotations(annotations)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage("standard_1020", match_case=False, on_missing="ignore")

            runs[f"{run_idx}{condition}"] = raw

        return {"0": runs}
