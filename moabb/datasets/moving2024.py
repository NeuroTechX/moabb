"""MOVING2024 motor imagery / motor execution dataset."""

import warnings
from pathlib import Path

import mne

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
from moabb.datasets.utils import extract_rar


# EDF archive on Zenodo (record 12804784). The companion virtual_glove.rar
# holds the Leap-Motion kinematics and is not needed for the EEG paradigm.
MOVING2024_URL = "https://zenodo.org/records/12804784/files/edf.rar"

# 32 EEG channels in the order stored in the EDF header. Trailing X, Y, Z are
# the Enobio accelerometer axes (set to ``misc``, not EEG).
MOVING2024_EEG_CHANNELS = [
    "P7",
    "P4",
    "Cz",
    "Pz",
    "P3",
    "P8",
    "O1",
    "O2",
    "T8",
    "F8",
    "C4",
    "F4",
    "Fp2",
    "Fz",
    "C3",
    "F3",
    "Fp1",
    "T7",
    "F7",
    "Oz",
    "PO4",
    "FC6",
    "FC2",
    "AF4",
    "CP6",
    "CP2",
    "CP1",
    "CP5",
    "FC1",
    "FC5",
    "AF3",
    "PO3",
]


class MOVING2024(BaseDataset):
    """Motor imagery / motor execution dataset from the MOVING study [1]_ [2]_.

    **Dataset description**

    The MOVING dataset couples 32-channel dry-electrode EEG with hand-kinematic
    tracking from a Virtual Glove (two orthogonal Leap Motion Controllers).
    Eleven healthy participants performed three right-hand movements --
    open/close, wrist rotation and finger tapping -- each preceded by a rest
    baseline and performed both as motor imagery (MI) and as motor execution
    (ME).

    Each subject completed a single continuous ~10-minute recording made of 8
    repetitions of a fixed block. Within a block, each of the three movements is
    presented as a ``rest -> MI -> ME`` triplet, and the movements always appear
    in the same order (open/close, wrist rotation, finger tapping). Every 6 s
    action period is preceded by a 2 s fixation cross, giving the trigger
    stream:

    ======== ================= ==========
    Trigger  Phase             Class
    ======== ================= ==========
    #1       rest (open/close) rest
    #3       MI open/close     open_close
    #5       ME open/close     open_close
    #7       rest (wrist)      rest
    #9       MI wrist rotation wrist_rotation
    #11      ME wrist rotation wrist_rotation
    #13      rest (finger)     rest
    #15      MI finger tapping finger_tapping
    #17      ME finger tapping finger_tapping
    ======== ================= ==========

    The even triggers (#2, #4, ... #16) mark the 2 s fixation crosses and are
    not used as classes. Each odd trigger occurs 8 times (once per block), so
    every movement contributes 8 MI and 8 ME trials, and the shared rest class
    contributes 24 trials.

    The default paradigm exposes the four **motor-imagery** classes
    (``rest``, ``open_close``, ``wrist_rotation``, ``finger_tapping``). Pass
    ``execution=True`` to expose the matching four **motor-execution** classes
    instead. EEG is recorded with a dry 32-channel Enobio system (10-20
    layout) at 500 Hz; the three trailing accelerometer axes are kept as
    ``misc`` channels.

    References
    ----------

    .. [1] Mattei, E., Lozzi, D., Di Matteo, A., Cipriani, A., Manes, C., &
       Placidi, G. (2024). MOVING: A Multi-Modal Dataset of EEG Signals and
       Virtual Glove Hand Tracking. Sensors, 24(16), 5207.
       DOI: https://doi.org/10.3390/s24165207

    .. [2] Mattei, E., Lozzi, D., Di Matteo, A., Placidi, G., Manes, C., &
       Cipriani, A. (2024). MOVING dataset [Data set]. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.12804784

    Notes
    -----
    Extraction of ``edf.rar`` requires ``unrar``, ``unar`` or ``7z`` to be
    installed on the system.

    .. versionadded:: 1.2.1

    """

    # Trigger label -> class label. Rest triggers are shared between the MI and
    # ME variants; the movement triggers differ.
    _IMAGERY_MAP = {
        "Trigger#1": "rest",
        "Trigger#7": "rest",
        "Trigger#13": "rest",
        "Trigger#3": "open_close",
        "Trigger#9": "wrist_rotation",
        "Trigger#15": "finger_tapping",
    }
    _EXECUTION_MAP = {
        "Trigger#1": "rest",
        "Trigger#7": "rest",
        "Trigger#13": "rest",
        "Trigger#5": "open_close",
        "Trigger#11": "wrist_rotation",
        "Trigger#17": "finger_tapping",
    }

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32, "misc": 3},
            montage="10-20",
            hardware="Neuroelectrics Enobio 32 (dry electrodes, wireless)",
            electrode_type="dry",
            sensor_type="dry",
            reference="CMS/DRL",
            sensors=MOVING2024_EEG_CHANNELS,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=11, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["rest", "open_close", "wrist_rotation", "finger_tapping"],
            trial_duration=6.0,
            study_design=(
                "rest -> motor imagery -> motor execution triplet for three "
                "right-hand movements (open/close, wrist rotation, finger "
                "tapping), 8 repetitions per subject."
            ),
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3390/s24165207",
            description=(
                "Multi-modal dataset pairing 32-channel dry EEG with Virtual "
                "Glove hand-kinematic tracking during motor imagery and motor "
                "execution of three right-hand movements."
            ),
            investigators=[
                "Enrico Mattei",
                "Daniele Lozzi",
                "Alessandro Di Matteo",
                "Alessia Cipriani",
                "Costanzo Manes",
                "Giuseppe Placidi",
            ],
            institution="University of L'Aquila",
            country="IT",
            data_url="https://doi.org/10.5281/zenodo.12804784",
            publication_year=2024,
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery", "Motor Execution"]),
        file_format="EDF",
    )

    def __init__(self, execution=False, subjects=None, sessions=None):
        self.execution = execution
        self._trigger_map = self._EXECUTION_MAP if execution else self._IMAGERY_MAP
        super().__init__(
            subjects=list(range(1, 11 + 1)),
            sessions_per_subject=1,
            events={"rest": 1, "open_close": 2, "wrist_rotation": 3, "finger_tapping": 4},
            code="MOVING2024",
            interval=[0, 6],
            paradigm="imagery",
            doi="10.3390/s24165207",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the path to a single subject's EDF file.

        Downloads and extracts ``edf.rar`` from Zenodo if needed.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-11).
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            A one-element list with the path to the subject's EDF file.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}. Must be in 1-11.")

        rar_path = Path(
            dl.data_dl(MOVING2024_URL, self.code, path, force_update, verbose)
        )
        # For this Zenodo URL, pooch stores the archive as a hashed file inside a
        # directory named ``edf.rar``; resolve the real archive file either way.
        if rar_path.is_dir():
            archives = sorted(rar_path.glob("*edf.rar")) or sorted(rar_path.glob("*.rar"))
            if not archives:
                raise FileNotFoundError(f"No .rar archive found under {rar_path}")
            rar_path = archives[0]
        extract_dir = rar_path.parent / "extracted"

        matches = self._find_subject_file(extract_dir, subject)
        if not matches or force_update:
            extract_rar(rar_path, extract_dir)
            matches = self._find_subject_file(extract_dir, subject)

        if not matches:
            available = [str(p.name) for p in extract_dir.rglob("*.edf")]
            raise FileNotFoundError(
                f"No EDF file found for subject {subject} after extracting "
                f"{rar_path} to {extract_dir}. Available EDF files: {available}"
            )

        return [str(matches[0])]

    @staticmethod
    def _find_subject_file(extract_dir, subject):
        """Find the EDF file for a subject under ``extract_dir``.

        Filenames follow ``<timestamp>_Subj_<NN>_bci_32_gesture.edf`` with a
        per-subject acquisition timestamp, so match on the ``Subj_<NN>`` token.
        """
        extract_dir = Path(extract_dir)
        if not extract_dir.is_dir():
            return []
        token = f"Subj_{subject:02d}_"
        return sorted(p for p in extract_dir.rglob("*.edf") if token in p.name)

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"0": {"0": raw}}`` for the single session and run.
        """
        file_path = self.data_path(subject)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Mark the three accelerometer axes as non-EEG.
            misc = {ch: "misc" for ch in ("X", "Y", "Z") if ch in raw.ch_names}
            if misc:
                raw.set_channel_types(misc)
            raw.set_montage("standard_1020", on_missing="ignore", verbose=False)

        # Relabel numbered triggers into class labels; drop everything else
        # (fixation crosses and the other modality's action triggers).
        ann = raw.annotations
        keep = [i for i, d in enumerate(ann.description) if d in self._trigger_map]
        if not keep:
            found = sorted(set(ann.description))
            raise RuntimeError(
                f"MOVING2024 subject {subject}: no annotation matched the "
                f"expected trigger labels {sorted(self._trigger_map)}. Found "
                f"annotation descriptions: {found}. The EDF trigger naming may "
                f"differ from the assumed 'Trigger#<N>' scheme; without a match "
                f"the paradigm would silently yield zero epochs."
            )
        new_desc = [self._trigger_map[ann.description[i]] for i in keep]
        new_ann = mne.Annotations(
            onset=ann.onset[keep],
            duration=ann.duration[keep],
            description=new_desc,
            orig_time=ann.orig_time,
        )
        raw.set_annotations(new_ann)

        return {"0": {"0": raw}}
