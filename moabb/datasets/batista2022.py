"""Batista2022 motor-imagery EEG dataset (NeuRow VR/haptics BCI training)."""

import logging
import warnings
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

from .utils import download_and_extract_subject_zip, rename_stimulus_codes


log = logging.getLogger(__name__)

# Per-subject ZIPs live on version record 7664069 (concept DOI 10.5281/zenodo.7664068).
# The plain /records/<id>/files/<name> endpoint yields a distinct local filename per
# subject ("sub-01.zip", ...), unlike the /api .../content endpoint.
ZENODO_BASE = "https://zenodo.org/records/7664069/files"

# MOABB subject id -> Zenodo file stem. There is no sub-10 / sub-11 on the record;
# the three pilot subjects (sub-p01..sub-p03) are appended as 20-22.
_SUBJECT_MAP = {i: f"sub-{i:02d}" for i in [*range(1, 10), *range(12, 20)]}
_SUBJECT_MAP.update({20: "sub-p01", 21: "sub-p02", 22: "sub-p03"})

# 32 EEG channels in acquisition order (LiveAmp 32 / actiCAP, 10-20 layout); shared
# with Farabbi2020, recorded by the same group with the same cap.
_EEG_CHANNELS = (
    "Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 TP9 CP5 CP1 Pz P3 P7 O1 "
    "Oz O2 P4 P8 TP10 CP6 CP2 Cz C4 T8 FT10 FC6 FC2 F4 F8 Fp2"
).split()

# AUX channels (BIP2AUX adapter) and the built-in accelerometer: raw label ->
# (descriptive name, channel type).
_AUX = {
    "Aux1": ("PPG", "misc"),
    "Aux2": ("RESP", "resp"),
    "Aux3": ("ECG", "ecg"),
    "x_dir": ("ACC_X", "misc"),
    "y_dir": ("ACC_Y", "misc"),
    "z_dir": ("ACC_Z", "misc"),
}

# Data-borne class markers in the BrainVision .vmrk: S 7 -> left, S 8 -> right hand.
_CLASS_CODES = {7: "left_hand", 8: "right_hand"}


class Batista2022(BaseDataset):
    """Motor-imagery EEG during NeuRow VR/haptics BCI training [1]_.

    Twenty healthy volunteers performed cue-based left- vs right-hand motor
    imagery of a bimanual rowing task in one lab session, under Graz-style
    feedback and the NeuRow virtual-reality environment (monitor or head-mounted
    display, with or without vibrotactile haptics). EEG was recorded with a
    32-channel LiveAmp at 500 Hz together with PPG, respiration, ECG and a 3-axis
    accelerometer.

    Each subject's session contains up to five imagery runs, one per condition
    (``MI``, ``MIMO``, ``MIMOHP``, ``MIMOVR``, ``MIMOVRHP``); the motor-execution
    ``ME`` control run is excluded. Trials are locked to the class marker
    (``S 7`` left / ``S 8`` right) and span the 5 s imagery/feedback window. Only
    the 32 EEG channels are returned unless ``return_all_modalities=True``.

    The Zenodo record (10.5281/zenodo.7664068) reports 20 healthy volunteers
    (mean age 24.79 years, SD 3.54); the release holds ``sub-01``..``sub-19``
    without ``sub-10``/``sub-11`` (no markers) plus the three pilot subjects
    ``sub-p01``..``sub-p03``, exposed here as subjects 20-22, i.e. 20 subjects.
    The record does not state the institution or country of acquisition.

    References
    ----------

    .. [1] Batista, D., Caetano, G., Fleury, M., Figueiredo, P., and
       Vourvopoulos, A. (2022). Physiological Signals During Motor Imagery
       Brain-Computer Interface Training Using Virtual Reality and Haptics.
       Zenodo. DOI: https://doi.org/10.5281/zenodo.7664068

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types={"eeg": 32, "ecg": 1, "resp": 1, "misc": 4},
            montage="standard_1020",
            hardware="LiveAmp 32 (Brain Products GmbH)",
            cap_manufacturer="Brain Products GmbH",
            cap_model="actiCAP",
            sensor_type="active Ag/AgCl",
            electrode_type="active",
            reference=None,
            ground=None,
            software="BrainVision Recorder",
            sensors=list(_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=20,
            health_status="healthy",
            age_mean=24.79,
            age_std=3.54,
            bci_experience=None,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=5.0,
            study_design=(
                "Within-subject, randomized-order cue-based left- vs right-hand "
                "motor imagery of a bimanual rowing task, compared across Graz "
                "and NeuRow virtual-reality conditions with optional haptic and "
                "head-mounted-display feedback, plus a motor-execution control."
            ),
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual", "tactile"],
            synchronicity="cue-based",
            mode="online",
            has_training_test_split=False,
            events={"left_hand": 7, "right_hand": 8},
            instructions=(
                "Imagine moving the left or right paddle (bimanual rowing) "
                "following the directional cue."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.7664068",
            description=(
                "Motor-imagery EEG plus PPG, respiration, ECG and accelerometry "
                "from 20 healthy volunteers performing cued left/right-hand "
                "motor imagery of a rowing task during VR/haptics BCI training."
            ),
            investigators=[
                "Diogo Batista",
                "Gustavo Caetano",
                "Mathis Fleury",
                "Patricia Figueiredo",
                "Athanasios Vourvopoulos",
            ],
            institution="Instituto Superior Tecnico, Universidade de Lisboa",
            country="PT",
            data_url="https://zenodo.org/records/7664069",
            publication_year=2022,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "virtual reality",
                "haptics",
                "neurorehabilitation",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=5,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        bci_application=BCIApplicationMetadata(
            applications=["motor rehabilitation", "BCI training"],
            environment="lab",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=5,
            trials_context=(
                "One lab session per subject with up to five imagery runs (one per "
                "condition: MI, MIMO, MIMOHP, MIMOVR, MIMOVRHP; the motor-"
                "execution ME control is excluded). "
                "Trials are cue-locked to the left/right-hand marker with a 5 s "
                "imagery/feedback window."
            ),
        ),
        file_format="BrainVision",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=sorted(_SUBJECT_MAP),
            sessions_per_subject=1,
            events={"left_hand": 7, "right_hand": 8},
            code="Batista2022",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.5281/zenodo.7664068",
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
        stem = _SUBJECT_MAP[subject]
        url = f"{ZENODO_BASE}/{stem}.zip"
        data_dir = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        if force_update or not (data_dir / stem).exists():
            download_and_extract_subject_zip(
                url, self.code, data_dir, path, force_update, verbose
            )
        return [str(data_dir / stem)]

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: Raw}}`` with one run per imagery condition."""
        subject_dir = Path(self.data_path(subject)[0])
        montage = make_standard_montage("colin27_1020")
        sessions = {}
        ses_dirs = sorted(
            d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")
        )
        for ses_idx, ses_dir in enumerate(ses_dirs):
            runs = {}
            for run_idx, vhdr in enumerate(sorted(ses_dir.rglob("*.vhdr"))):
                name = self._run_name(vhdr)
                if name != "ME":
                    runs[f"{run_idx}{name}"] = self._load_run(vhdr, montage)
            if runs:
                # MOABB keys start with an integer index, then a description.
                sessions[f"{ses_idx}{ses_dir.name.replace('ses-', '')}"] = runs
        if not sessions:
            raise FileNotFoundError(
                f"No BrainVision runs found for subject {subject} under {subject_dir}"
            )
        return sessions

    @staticmethod
    def _run_name(vhdr):
        """Condition name from the ``task-`` token of a ``.vhdr`` filename.

        Folder names are unreliable (one folder is MIMOHPVR while its file is
        MIMOVRHP), so the ``graz`` / ``neurow`` prefix is stripped from the token.
        """
        token = vhdr.stem.split("task-")[-1]
        for prefix in ("graz", "neurow"):
            if token.startswith(prefix):
                return token[len(prefix) :]
        return token

    def _load_run(self, vhdr, montage):
        """Read one BrainVision run and standardize channels/events."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Some headers still name the pre-BIDS DataFile/MarkerFile.
            raw = mne.io.read_raw_brainvision(
                vhdr,
                preload=True,
                verbose=False,
                overrides={
                    "data_fname": vhdr.with_suffix(".eeg").name,
                    "marker_fname": vhdr.with_suffix(".vmrk").name,
                },
            )

        aux = {ch: _AUX[ch] for ch in raw.ch_names if ch in _AUX}
        raw.rename_channels({ch: name for ch, (name, _) in aux.items()})
        raw.set_channel_types(dict(aux.values()))
        if not self.return_all_modalities:
            raw.pick([ch for ch in _EEG_CHANNELS if ch in raw.ch_names])
        raw.set_montage(montage, on_missing="ignore")

        # Rename the data-borne class markers to MOABB class labels.
        rename_stimulus_codes(raw, _CLASS_CODES)
        return raw
