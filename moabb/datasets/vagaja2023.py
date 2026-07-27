"""Vagaja2023 motor-imagery EEG dataset (embodiment priming + MI-BCI in VR).

Vagaja, K., and Vourvopoulos, A. (2023). "Electrophysiological Signals of
Embodiment and MI-BCI Training in VR."
Concept DOI: 10.5281/zenodo.8086085 (this loader downloads version record
10.5281/zenodo.8086086, where the single GROUPS.zip archive lives).
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

# The whole study ships as one archive (all subjects, ~2.0 GiB) on the version
# record 8086086. The plain /records/<id>/files/<name> endpoint serves the bytes
# directly and yields a clean local filename ("GROUPS.zip").
ZENODO_URL = "https://zenodo.org/records/8086086/files/GROUPS.zip"

# Data-borne group assignment (between-subject design). Subject ids are globally
# unique across the two groups, so a MOABB integer subject maps directly onto the
# numeric part of the "SUB<nn>" folder name. Embodied = virtual-hand-illusion
# priming; Control = non-embodied priming.
_SUBJECT_GROUP = {
    3: "Embodied",
    5: "Embodied",
    6: "Embodied",
    7: "Embodied",
    8: "Embodied",
    9: "Embodied",
    10: "Embodied",
    12: "Embodied",
    14: "Embodied",
    15: "Embodied",
    16: "Embodied",
    29: "Embodied",
    31: "Embodied",
    17: "Control",
    18: "Control",
    19: "Control",
    20: "Control",
    21: "Control",
    22: "Control",
    23: "Control",
    24: "Control",
    25: "Control",
    26: "Control",
    27: "Control",
    28: "Control",
    30: "Control",
}

# 32 EEG channel names in acquisition order (LiveAmp 32, actiCAP 10-20 layout),
# read verbatim from the BrainVision .vhdr [Channel Infos] section.
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

# AUX channels recorded synchronously through the BIP2AUX adapter (bipolar EMG
# and skin temperature) plus the amplifier's built-in 3-axis accelerometer. The
# accelerometer axes are renamed to descriptive labels; EMGL/EMGR/Temp keep the
# names used in the .vhdr.
_AUX_RENAME = {"x_dir": "ACC_X", "y_dir": "ACC_Y", "z_dir": "ACC_Z"}
_AUX_TYPES = {
    "EMGL": "emg",
    "EMGR": "emg",
    "Temp": "misc",
    "ACC_X": "misc",
    "ACC_Y": "misc",
    "ACC_Z": "misc",
}

# Data-borne class markers in the BrainVision .vmrk: Stimulus S 7 -> left hand
# (class 1), Stimulus S 8 -> right hand (class 2).
_CLASS_CODES = {7: "left_hand", 8: "right_hand"}


class Vagaja2023(BaseDataset):
    """Motor-imagery EEG during embodiment-primed MI-BCI training in VR [1]_.

    .. admonition:: Dataset summary

        =============  =======  =======  ==========  =================  ============  ===============  ===========
        Name             #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =============  =======  =======  ==========  =================  ============  ===============  ===========
        Vagaja2023          26       32           2                 20            10s            500 Hz              1
        =============  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Twenty-six healthy volunteers performed cue-based left- vs right-hand motor
    imagery in an immersive virtual-reality environment (Oculus Rift CV1) during
    a single lab session. A between-subject design tested whether a virtual
    embodiment (virtual-hand-illusion) priming phase changes the subsequent
    motor-imagery training: participants were randomly assigned to an
    ``Embodied`` group (N=13, embodiment induced) or a ``Control`` group (N=13,
    embodiment broken). EEG was recorded with a 32-channel LiveAmp system
    (actiCAP active electrodes, Brain Products GmbH) at 500 Hz, together with
    bipolar EMG from both forearms, skin temperature (through the BIP2AUX
    adapter) and a 3-axis accelerometer.

    Each recorded subject contributes three BrainVision runs -- a resting-state
    baseline, the embodiment (virtual-hand-illusion) phase and the MI training
    phase -- but only the **MI** run carries left-/right-hand class markers, so
    this loader exposes that run alone as a single MOABB session. Each trial is
    cue-locked: the class marker (S 7 left / S 8 right) starts the imagery period
    and the end-of-trial marker follows 10 s later (continuous feedback begins
    1.25 s after the cue), so the exposed interval spans that 10 s window. Every
    subject provides 20 left-hand and 20 right-hand trials (40 in total).

    Subject ids are globally unique across groups. The Embodied group holds
    subjects 3, 5-10, 12, 14-16, 29 and 31; the Control group holds subjects
    17-28 and 30.

    By default only the 32 EEG channels are returned; pass
    ``return_all_modalities=True`` to also keep the EMG, temperature and
    accelerometer channels.

    References
    ----------

    .. [1] Vagaja, K., and Vourvopoulos, A. (2023). Electrophysiological Signals
       of Embodiment and MI-BCI Training in VR. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.8086086

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32, "emg": 2, "misc": 4},
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
            n_subjects=26,
            health_status="healthy",
            bci_experience=None,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=10.0,
            study_design=(
                "Between-subject, randomized-group cue-based left- vs right-hand "
                "motor imagery in immersive VR, comparing an embodied (virtual-"
                "hand-illusion) priming group against a non-embodied control "
                "group, each with a single MI training run of 40 trials."
            ),
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="online",
            has_training_test_split=False,
            events={"left_hand": 7, "right_hand": 8},
            instructions=(
                "Imagine moving the left or right hand following the directional "
                "cue while immersed in the virtual-reality environment."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.8086086",
            description=(
                "Motor-imagery EEG plus bipolar EMG, skin temperature and "
                "accelerometry from 26 healthy volunteers performing cued "
                "left/right-hand motor imagery during embodiment-primed MI-BCI "
                "training in virtual reality."
            ),
            investigators=["Katarina Vagaja", "Athanasios Vourvopoulos"],
            institution="Instituto Superior Tecnico, Universidade de Lisboa",
            country="PT",
            data_url="https://zenodo.org/records/8086086",
            publication_year=2023,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "virtual reality",
                "embodiment",
                "virtual hand illusion",
                "neurorehabilitation",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
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
            imagery_duration_s=10.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=1,
            trials_context=(
                "One lab session per subject. Each subject has resting-state, "
                "embodiment and MI BrainVision recordings, but only the MI run "
                "carries left/right-hand class markers and is exposed here. "
                "Trials are cue-locked to the left/right-hand marker with a 10 s "
                "imagery/feedback window (20 trials per class)."
            ),
        ),
        file_format="BrainVision",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=sorted(_SUBJECT_GROUP),
            sessions_per_subject=1,
            events={"left_hand": 7, "right_hand": 8},
            code="Vagaja2023",
            interval=[0, 10],
            paradigm="imagery",
            doi="10.5281/zenodo.8086086",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the path to the extracted per-subject directory.

        Downloads the single GROUPS.zip archive from Zenodo (once) and extracts
        it if needed, then returns the folder for this subject.

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

        zip_path = Path(dl.data_dl(ZENODO_URL, self.code, path, force_update, verbose))
        extract_dir = zip_path.parent

        group = _SUBJECT_GROUP[subject]
        subject_dir = extract_dir / group / f"SUB{subject:02d}"

        # The archive extracts "Control/" and "Embodied/" directly into the
        # parent dir; extract the whole archive on first access.
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
            ``{"0": {"0": Raw}}`` with the single motor-imagery run.
        """
        subject_dir = Path(self.data_path(subject)[0])

        vhdr = subject_dir / f"SUB{subject:02d}_MI.vhdr"
        if not vhdr.is_file():
            raise FileNotFoundError(
                f"No motor-imagery BrainVision run found for subject {subject} at {vhdr}"
            )

        montage = make_standard_montage("standard_1020")
        return {"0": {"0": self._load_run(vhdr, montage)}}

    @staticmethod
    def _read_brainvision(vhdr):
        """Read BrainVision data, repairing a stale same-stem BIDS header.

        The published subject-31 MI header retains lower-case acquisition
        names with an extra underscore, while the archive contains the
        adjacent `SUB31_MI.eeg` and `SUB31_MI.vmrk` files.  Only a missing
        reference with an existing same-stem sibling is repaired, in a
        temporary header, leaving the downloaded archive untouched.
        """
        vhdr = Path(vhdr)
        header = vhdr.read_text(encoding="utf-8")
        repaired = header
        for field, suffix in (("DataFile", ".eeg"), ("MarkerFile", ".vmrk")):
            match = re.search(rf"(?m)^{field}=(.+)$", repaired)
            if match is None:
                raise ValueError(f"Missing {field} entry in BrainVision header {vhdr}")
            referenced = vhdr.parent / match.group(1).strip()
            if referenced.exists():
                continue
            sibling = vhdr.with_suffix(suffix)
            if not sibling.exists():
                raise FileNotFoundError(
                    f"{vhdr} references missing {referenced.name}; expected BIDS "
                    f"sibling {sibling.name} is also absent."
                )
            repaired = re.sub(
                rf"(?m)^{field}=.+$", f"{field}={sibling.name}", repaired
            )

        temporary = None
        try:
            path = vhdr
            if repaired != header:
                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".vhdr",
                    dir=vhdr.parent,
                    encoding="utf-8",
                    delete=False,
                ) as fout:
                    fout.write(repaired)
                    temporary = Path(fout.name)
                path = temporary
            return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

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
