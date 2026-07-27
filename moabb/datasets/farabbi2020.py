"""Farabbi2020 Motor-Imagery EEG dataset during robot-arm control.

Farabbi, A., Ghiringhelli, F., Mainardi, L., Sanches, J. M., Moreno, P.,
Santos-Victor, J., Figueiredo, P., and Vourvopoulos, A. (2020).
"Motor-Imagery EEG Dataset During Robot-Arm Control."
Data DOI: 10.5281/zenodo.5882500
"""

import logging
import warnings
import zipfile
from pathlib import Path

import mne
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# Per-subject ZIPs (01.zip .. 12.zip). The plain /files/<name> endpoint serves
# the bytes directly and yields a distinct local filename per subject, unlike
# the /content endpoint which would collide on the name "content".
ZENODO_BASE = "https://zenodo.org/records/5882500/files"

# 32 EEG channel names in acquisition order (from chanlocs.locs, actiCAP 10-20).
_EEG_CHANNELS = [
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "Cz",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
]

# 3 accelerometer channels trailing the EEG montage.
_ACC_CHANNELS = ["ACC_X", "ACC_Y", "ACC_Z"]


class Farabbi2020(BaseDataset):
    """Motor-Imagery EEG dataset during robot-arm control [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Farabbi2020     12       32           2                            4s              250 Hz            3
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Twelve healthy, BCI-naive subjects performed cue-based left- vs right-hand
    motor imagery to control a Baxter robot arm reaching toward objects, over
    three sessions on three consecutive days (each session lasting at most two
    hours). EEG was recorded with a 32-channel LiveAmp system (actiCAP active
    electrodes, Brain Products GmbH) at 250 Hz, plus 3 accelerometer channels.

    Each session contains a resting-state recording (ignored here) and two motor
    imagery feedback conditions -- a first-person and a third-person robot-arm
    view -- each split into a training run and an online run. This loader exposes
    the four motor-imagery runs per session (first-person training/online,
    third-person training/online) and drops the resting-state recording.

    Each trial lasted 6 seconds (2 seconds baseline followed by 4 seconds of
    motor imagery), preceded by a green cross about one second before onset, with
    inter-trial intervals randomly drawn between 1.5 and 3.5 seconds. Event code
    769 marks the left-hand cue onset (class 1) and 770 the right-hand cue onset
    (class 2). The exposed interval spans the 4-second imagery period following
    the cue.

    References
    ----------

    .. [1] Farabbi, A., Ghiringhelli, F., Mainardi, L., Sanches, J. M.,
       Moreno, P., Santos-Victor, J., Figueiredo, P., and Vourvopoulos, A.
       (2020). Motor-Imagery EEG Dataset During Robot-Arm Control. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.5882500

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=32,
            channel_types={"eeg": 32, "misc": 3},
            montage="standard_1020",
            hardware="LiveAmp 32 (Brain Products GmbH)",
            cap_manufacturer="Brain Products GmbH",
            cap_model="actiCAP",
            sensor_type="active Ag/AgCl",
            electrode_type="active",
            reference=None,
            ground=None,
            software=None,
            sensors=list(_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            bci_experience="naive",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=6.0,
            study_design="Cue-based left- vs right-hand motor imagery controlling a Baxter robot arm reaching toward objects, under first-person and third-person visual feedback.",
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="online",
            has_training_test_split=True,
            events={"left_hand": 769, "right_hand": 770},
            instructions="Imagine left- or right-hand movement following the cue to steer the robot arm toward the target object.",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.5882500",
            description="Motor-imagery EEG from 12 healthy naive subjects performing cued left/right-hand imagery to control a Baxter robot arm, across three sessions with first- and third-person visual feedback conditions.",
            investigators=[
                "Andrea Farabbi",
                "Fabiola Ghiringhelli",
                "Luca Mainardi",
                "Joao Miguel Sanches",
                "Plinio Moreno",
                "Jose Santos-Victor",
                "Patricia Figueiredo",
                "Athanasios Vourvopoulos",
            ],
            institution="Politecnico di Milano; Instituto Superior Tecnico, Universidade de Lisboa",
            country="IT",
            data_url="https://doi.org/10.5281/zenodo.5882500",
            publication_year=2020,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "robot arm",
                "neurorehabilitation",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=3,
        runs_per_session=4,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["robot arm control", "neurorehabilitation"],
            environment="lab",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=4,
            trials_context="Three sessions per subject, each with four motor-imagery runs (first-person training/online, third-person training/online) plus an ignored resting-state recording. Each trial: 2 s baseline + 4 s imagery.",
        ),
        file_format="GDF",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=3,
            events={"left_hand": 769, "right_hand": 770},
            code="Farabbi2020",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.5882500",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the path to the extracted per-subject ZIP directory.

        Downloads the subject's ZIP from Zenodo and extracts it if needed.

        Parameters
        ----------
        subject : int
            Subject number (1-12).
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
            Single-element list with the path to the extracted subject folder.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        url = f"{ZENODO_BASE}/{subject:02d}.zip"
        zip_path = Path(dl.data_dl(url, self.code, path, force_update, verbose))
        extract_dir = zip_path.parent

        # The ZIP extracts a top-level "NN/" folder (e.g. "01/").
        subject_dir = extract_dir / f"{subject:02d}"
        if not subject_dir.is_dir():
            log.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        return [str(subject_dir)]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-12).

        Returns
        -------
        dict
            ``{session_str: {run_str: Raw}}`` with the four motor-imagery runs
            per session (resting-state recordings are excluded).
        """
        subject_dir = Path(self.data_path(subject)[0])

        # Session folders look like "01_session_1", "02_session_2", ...
        session_dirs = sorted(
            (d for d in subject_dir.iterdir() if d.is_dir() and "session" in d.name),
            key=lambda d: d.name,
        )

        montage = make_standard_montage("standard_1020")
        sessions = {}
        for sess_idx, sess_dir in enumerate(session_dirs):
            # All motor-imagery GDFs in this session, excluding resting-state.
            gdf_files = sorted(
                p for p in sess_dir.rglob("*.gdf") if "rest" not in p.name.lower()
            )
            runs = {}
            for run_idx, gdf_path in enumerate(gdf_files):
                runs[str(run_idx)] = self._load_gdf(gdf_path, montage)
            if runs:
                sessions[str(sess_idx)] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No motor-imagery GDF files found for subject {subject} "
                f"under {subject_dir}"
            )

        return sessions

    def _load_gdf(self, gdf_path, montage):
        """Read one GDF run and standardize channels/events."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)

        # The file carries 32 EEG + 3 ACC channels in acquisition order. Rename
        # positionally to the standard 10-20 names from chanlocs.locs.
        n_named = len(_EEG_CHANNELS) + len(_ACC_CHANNELS)
        target = _EEG_CHANNELS + _ACC_CHANNELS
        rename_map = {}
        for i, ch in enumerate(raw.ch_names[:n_named]):
            if ch != target[i]:
                rename_map[ch] = target[i]
        if rename_map:
            raw.rename_channels(rename_map)

        # Mark accelerometer channels as misc.
        acc_present = {ch: "misc" for ch in _ACC_CHANNELS if ch in raw.ch_names}
        if acc_present:
            raw.set_channel_types(acc_present)

        if not self.return_all_modalities:
            raw.pick([ch for ch in _EEG_CHANNELS if ch in raw.ch_names])

        raw.set_montage(montage, on_missing="ignore")

        # GDF event codes: 769 -> left-hand cue, 770 -> right-hand cue.
        rename_ann = {
            desc: label
            for desc, label in (("769", "left_hand"), ("770", "right_hand"))
            if desc in set(raw.annotations.description)
        }
        if rename_ann:
            raw.annotations.rename(rename_ann)

        return raw
