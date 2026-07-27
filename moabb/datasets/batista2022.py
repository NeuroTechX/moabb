"""Batista2022 motor-imagery EEG dataset (NeuRow VR/haptics BCI training).

Batista, D., Caetano, G., Fleury, M., Figueiredo, P., and Vourvopoulos, A.
(2022). "Physiological Signals During Motor Imagery Brain-Computer Interface
Training Using Virtual Reality and Haptics."
Concept DOI: 10.5281/zenodo.7664068 (this loader downloads version record
10.5281/zenodo.7664069, where the per-subject files live).
"""

import logging
import re
import tempfile
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

# Per-subject ZIPs live on the version record 7664069. Using the plain
# /records/<id>/files/<name> endpoint (not the /api .../content endpoint)
# yields a distinct local filename per subject ("sub-01.zip", ...).
ZENODO_BASE = "https://zenodo.org/records/7664069/files"

# Mapping from MOABB integer subject id to the Zenodo file stem. The real
# numbered subjects keep their id (note there is no sub-10 / sub-11 on the
# record); the three pilot subjects (sub-p01..sub-p03) are appended as 20-22.
_SUBJECT_MAP = {
    1: "sub-01",
    2: "sub-02",
    3: "sub-03",
    4: "sub-04",
    5: "sub-05",
    6: "sub-06",
    7: "sub-07",
    8: "sub-08",
    9: "sub-09",
    12: "sub-12",
    13: "sub-13",
    14: "sub-14",
    15: "sub-15",
    16: "sub-16",
    17: "sub-17",
    18: "sub-18",
    19: "sub-19",
    20: "sub-p01",
    21: "sub-p02",
    22: "sub-p03",
}

# 32 EEG channel names in acquisition order (actiCAP CLA-32, 10-20 layout).
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

# AUX channels recorded synchronously through the BIP2AUX adapter, plus the
# built-in 3-axis accelerometer. Renamed from the raw Aux1/Aux2/Aux3/x_dir/...
# labels to descriptive names, then typed.
_AUX_RENAME = {
    "Aux1": "PPG",
    "Aux2": "RESP",
    "Aux3": "ECG",
    "x_dir": "ACC_X",
    "y_dir": "ACC_Y",
    "z_dir": "ACC_Z",
}
_AUX_TYPES = {
    "PPG": "misc",
    "RESP": "resp",
    "ECG": "ecg",
    "ACC_X": "misc",
    "ACC_Y": "misc",
    "ACC_Z": "misc",
}

# Data-borne class markers in the BrainVision .vmrk: Stimulus S 7 -> left hand
# (class 1), Stimulus S 8 -> right hand (class 2).
_CLASS_CODES = {7: "left_hand", 8: "right_hand"}


class Batista2022(BaseDataset):
    """Motor-imagery EEG during NeuRow VR/haptics BCI training [1]_.

    .. admonition:: Dataset summary

        =============  =======  =======  ==========  =================  ============  ===============  ===========
        Name             #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =============  =======  =======  ==========  =================  ============  ===============  ===========
        Batista2022         20       32           2                ~20            5s              500 Hz            1
        =============  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Twenty healthy volunteers performed cue-based left- vs right-hand motor
    imagery of a bimanual rowing task (moving one virtual paddle in each hand)
    across several experimental conditions in a single lab session. Four
    conditions used NeuRow, a virtual-reality environment rendering first-person
    virtual arms, and the others used abstract Graz-style feedback. EEG was
    recorded with a 32-channel LiveAmp system (actiCAP active electrodes, Brain
    Products GmbH) at 500 Hz, together with PPG, respiration and ECG (through the
    BIP2AUX adapter) and a 3-axis accelerometer.

    Each subject's session is exposed as one MOABB session containing up to six
    runs, one per condition:

    - ``MI``       : standard Graz motor imagery (fixation cross + arrows).
    - ``MIMO``     : motor imagery / motor observation with NeuRow on a monitor.
    - ``MIMOHP``   : as MIMO, with vibrotactile haptic feedback.
    - ``MIMOVR``   : as MIMO, displayed through a VR head-mounted display.
    - ``MIMOVRHP`` : VR head-mounted display with haptic feedback.
    - ``ME``       : motor *execution* control (finger tapping), recorded only
      from the ten subjects acquired after subject 07, so it is absent for the
      earlier subjects.

    Every condition shares the same trial structure and class markers, so all
    runs contribute left-/right-hand trials. Each trial is cue-locked: the class
    marker (S 7 left / S 8 right) starts the imagery/feedback period and the
    end-of-trial marker follows 5 s later; the exposed interval spans that 5 s
    window. ``MI`` runs carry roughly 20 trials per class, the ``ME`` control
    roughly 9 per class.

    By default only the 32 EEG channels are returned; pass
    ``return_all_modalities=True`` to also keep the PPG, respiration, ECG and
    accelerometer channels.

    References
    ----------

    .. [1] Batista, D., Caetano, G., Fleury, M., Figueiredo, P., and
       Vourvopoulos, A. (2022). Physiological Signals During Motor Imagery
       Brain-Computer Interface Training Using Virtual Reality and Haptics.
       Zenodo. DOI: https://doi.org/10.5281/zenodo.7664068

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
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
        runs_per_session=6,
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
            n_blocks=6,
            trials_context=(
                "One lab session per subject with up to six runs (one per "
                "condition: MI, MIMO, MIMOHP, MIMOVR, MIMOVRHP and a motor-"
                "execution ME control present only for the ten later subjects). "
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
        """Return the path to the extracted per-subject directory.

        Downloads the subject's ZIP from Zenodo and extracts it if needed.

        Parameters
        ----------
        subject : int
            Subject id (one of ``self.subject_list``).
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

        stem = _SUBJECT_MAP[subject]
        url = f"{ZENODO_BASE}/{stem}.zip"
        zip_path = Path(dl.data_dl(url, self.code, path, force_update, verbose))
        extract_dir = zip_path.parent

        # The ZIP extracts a top-level "<stem>/" folder (e.g. "sub-01/").
        subject_dir = extract_dir / stem
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
            Subject id.

        Returns
        -------
        dict
            ``{session_str: {run_str: Raw}}`` with one run per experimental
            condition found for the subject.
        """
        subject_dir = Path(self.data_path(subject)[0])

        montage = make_standard_montage("standard_1020")
        sessions = {}
        ses_dirs = sorted(
            d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")
        )
        for ses_idx, ses_dir in enumerate(ses_dirs):
            ses_key = ses_dir.name.replace("ses-", "")
            runs = {}
            for run_idx, vhdr in enumerate(sorted(ses_dir.rglob("*.vhdr"))):
                runs[f"{run_idx}{self._run_name(vhdr)}"] = self._load_run(vhdr, montage)
            if runs:
                # MOABB requires session/run keys to start with an integer index
                # (optionally followed by a letters+digits description).
                sessions[f"{ses_idx}{ses_key}"] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No BrainVision runs found for subject {subject} under {subject_dir}"
            )

        return sessions

    @staticmethod
    def _run_name(vhdr):
        """Derive a clean condition/run name from a .vhdr filename.

        Filenames look like ``sub-01_ses-lab1_task-grazMI.vhdr`` or
        ``..._task-neurowMIMOVR.vhdr``; the folder name is not reliable (one
        folder is named MIMOHPVR while its file is MIMOVRHP), so the run name is
        taken from the ``task-`` token with the ``graz`` / ``neurow`` acquisition
        prefix stripped.
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
            raw = self._read_brainvision(vhdr)

        rename = {k: v for k, v in _AUX_RENAME.items() if k in raw.ch_names}
        if rename:
            raw.rename_channels(rename)
        types = {k: v for k, v in _AUX_TYPES.items() if k in raw.ch_names}
        if types:
            raw.set_channel_types(types)

        if not self.return_all_modalities:
            raw.pick([ch for ch in _EEG_CHANNELS if ch in raw.ch_names])

        raw.set_montage(montage, on_missing="ignore")

        # Rename the data-borne class markers to MOABB class labels.
        rename_ann = {}
        for desc in set(raw.annotations.description):
            match = re.search(r"S\s*(\d+)\s*$", desc)
            if match:
                label = _CLASS_CODES.get(int(match.group(1)))
                if label is not None:
                    rename_ann[desc] = label
        if rename_ann:
            raw.annotations.rename(rename_ann)

        return raw

    @staticmethod
    def _read_brainvision(vhdr):
        """Read a BrainVision header, repairing a bad marker reference in-memory.

        Two pilot recordings on the Zenodo record name the data and marker
        files from a different session.  The corresponding ``.eeg`` and
        ``.vmrk`` files are present beside the header with the same stem, so
        use a short-lived corrected header rather than changing downloaded
        source files.
        """
        vhdr = Path(vhdr)
        header = vhdr.read_text(encoding="utf-8")
        repaired_any = False

        def repair_reference(match):
            nonlocal repaired_any
            key, filename = match.groups()
            expected = vhdr.with_suffix({"DataFile": ".eeg", "MarkerFile": ".vmrk"}[key])
            if (vhdr.parent / filename).is_file() or not expected.is_file():
                return match.group(0)
            repaired_any = True
            return f"{key}={expected.name}"

        repaired = re.sub(
            r"(?m)^(DataFile|MarkerFile)=([^\r\n]+)", repair_reference, header
        )
        if not repaired_any:
            return mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".vhdr",
            prefix=f".{vhdr.stem}-",
            dir=vhdr.parent,
            delete=False,
        ) as tmp:
            tmp.write(repaired)
            repaired_vhdr = Path(tmp.name)
        try:
            return mne.io.read_raw_brainvision(
                str(repaired_vhdr), preload=True, verbose=False
            )
        finally:
            repaired_vhdr.unlink(missing_ok=True)
