"""Farabbi2020 Motor-Imagery EEG dataset during robot-arm control."""

import warnings
from pathlib import Path

import mne
from mne.channels import make_standard_montage

from moabb.datasets.base import BaseDataset
from moabb.datasets.batista2022 import _EEG_CHANNELS
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

from .utils import download_and_extract_zip, resolve_montage_name


# Per-subject ZIPs (01.zip .. 12.zip). The plain /files/<name> endpoint yields a
# distinct local filename per subject, unlike the colliding /content endpoint.
ZENODO_BASE = "https://zenodo.org/records/5882500/files"

# 3 accelerometer channels trailing the 32 EEG channels (same LiveAmp/actiCAP
# layout as Batista2022, from chanlocs.locs).
_ACC_CHANNELS = ["ACC_X", "ACC_Y", "ACC_Z"]


class Farabbi2020(BaseDataset):
    """Motor-Imagery EEG dataset during robot-arm control [1]_.

    Twelve healthy, BCI-naive subjects performed cue-based left- vs right-hand
    motor imagery to steer a Baxter robot arm, over three sessions on consecutive
    days, recorded with a 32-channel LiveAmp at 250 Hz plus 3 accelerometer
    channels. Each session has a first-person and a third-person view condition,
    each with a training and an online run; the loader exposes these four runs and
    drops the resting-state recording. GDF events 769/770 mark the left/right cue,
    followed by the 4 s imagery period. Channels are renamed positionally to the
    10-20 labels; only EEG is returned unless ``return_all_modalities=True``.

    References
    ----------

    .. [1] Farabbi, A., Ghiringhelli, F., Mainardi, L., Sanches, J. M.,
       Moreno, P., Santos-Victor, J., Figueiredo, P., and Vourvopoulos, A.
       (2020). Motor-Imagery EEG Dataset During Robot-Arm Control. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.5882500

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
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
        """Return ``[subject_dir]``, downloading and extracting the subject's ZIP."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        url = f"{ZENODO_BASE}/{subject:02d}.zip"
        folder = f"{subject:02d}"  # the ZIP holds a top-level "NN/" folder
        return [
            str(
                download_and_extract_zip(
                    url, self.code, folder, path, force_update, verbose
                )
            )
        ]

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: Raw}}`` with the four motor-imagery runs."""
        subject_dir = Path(self.data_path(subject)[0])
        # Session folders look like "01_session_1", "02_session_2", ...
        session_dirs = sorted(
            d for d in subject_dir.iterdir() if d.is_dir() and "session" in d.name
        )
        montage = make_standard_montage(resolve_montage_name("colin27_1020"))
        sessions = {}
        for sess_idx, sess_dir in enumerate(session_dirs):
            gdf_files = sorted(
                p for p in sess_dir.rglob("*.gdf") if "rest" not in p.name.lower()
            )
            if gdf_files:
                sessions[str(sess_idx)] = {
                    str(i): self._load_gdf(f, montage) for i, f in enumerate(gdf_files)
                }
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

        # 32 EEG + 3 ACC channels in acquisition order: rename positionally.
        target = _EEG_CHANNELS + _ACC_CHANNELS
        raw.rename_channels({ch: t for ch, t in zip(raw.ch_names, target) if ch != t})
        raw.set_channel_types({ch: "misc" for ch in _ACC_CHANNELS if ch in raw.ch_names})
        if not self.return_all_modalities:
            raw.pick([ch for ch in _EEG_CHANNELS if ch in raw.ch_names])
        raw.set_montage(montage, on_missing="ignore")

        # GDF event codes: 769 -> left-hand cue, 770 -> right-hand cue.
        present = set(raw.annotations.description)
        raw.annotations.rename(
            {
                d: c
                for d, c in (("769", "left_hand"), ("770", "right_hand"))
                if d in present
            }
        )
        return raw
