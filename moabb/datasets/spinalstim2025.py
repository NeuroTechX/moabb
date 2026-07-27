"""SpinalStim2025 longitudinal motor-imagery BCI dataset (TESS neuromodulation)."""

import warnings
import zipfile as z
from pathlib import Path

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
    Tags,
)


# Zenodo record 15454355 (concept DOI 10.5281/zenodo.15454354)
SPINALSTIM2025_BASE = "https://zenodo.org/records/15454355/files/{name}?download=1"

# Per-cohort archives on Zenodo. The 27 unique participants are spread across
# three archives (the d2 "SlowLearners" archive is a 6-month follow-up of four
# d1 subjects, i.e. repeat sessions of the same people, so it is not a separate
# participant and is intentionally excluded here).
_ARCHIVES = {
    "d1": ("d1_Main_Group_n20.zip", "d1_Main_Group_n20"),
    "d3": ("d3_SinglePulse_n5.zip", "d3_SinglePulse_n5"),
    "d4": ("d4_SCI_patients.zip", "d4_SCI_patients"),
}

# subject index (1..27) -> (archive key, zero-padded id token used in the
# on-disk folder name "Subject_<token>_..._Offline")
_SUBJECT_MAP = {}
_d1_rest = [4, 8, 9, 10, 13, 17, 18, 19, 20, 22]  # REST_n10 cohort
_d1_tess = [2, 3, 5, 6, 7, 11, 14, 15, 16, 21]  # TESS_n10 cohort
_d3 = [501, 502, 503, 504, 505]  # SinglePulse cohort
for _i, _raw in enumerate(_d1_rest + _d1_tess, start=1):
    _SUBJECT_MAP[_i] = ("d1", f"{_raw:03d}")
for _i, _raw in enumerate(_d3, start=21):
    _SUBJECT_MAP[_i] = ("d3", f"{_raw:03d}")
_SUBJECT_MAP[26] = ("d4", "0001")
_SUBJECT_MAP[27] = ("d4", "0002")

# The 32 EEG electrodes recorded (in file order). M1/M2 are mastoids.
_EEG_CHANNELS = [
    "FP1",
    "FPZ",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "M1",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "M2",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "POZ",
    "O1",
    "OZ",
    "O2",
]

# Auxiliary sensor channels present in the GDF header.
_AUX_CHANNELS = ["sens7", "sens8", "sens9"]

# GDF event type codes for the two motor-imagery classes (standard Graz/CNBI
# encoding: 0x0301 = left-hand cue, 0x0302 = right-hand cue).
_CLASS_CODES = {"769": "left_hand", "770": "right_hand"}


class SpinalStim2025(BaseDataset):
    """Motor-imagery BCI dataset with transcutaneous spinal stimulation [1]_.

    .. admonition:: Dataset summary

        ================ ======= ======= ========== ================= ============ ============ ===========
        Name             #Subj   #Chan   #Classes   #Trials/class     Trials len   Sampling     #Sessions
        ================ ======= ======= ========== ================= ============ ============ ===========
        SpinalStim2025   27      32      2          ~10               4 s          512 Hz       1
        ================ ======= ======= ========== ================= ============ ============ ===========

    **Dataset description**

    Longitudinal brain-computer interface (BCI) training data collected from 27
    human participants (25 able-bodied, 2 with spinal cord injury) during a study
    of motor-imagery BCI performance and transcutaneous electrical spinal
    stimulation (TESS) neuromodulation [1]_. Participants performed a two-class
    (left- vs. right-hand) kinesthetic motor-imagery task. High-resolution EEG
    (32 channels) and auxiliary EOG/reference sensors were recorded at 512 Hz and
    stored in GDF format following the CNBI/Graz recording convention.

    Participants are organised on Zenodo into cohort archives:

    - ``d1_Main_Group_n20`` -- 20 able-bodied subjects, split into a REST control
      group (n=10) and a TESS stimulation group (n=10);
    - ``d3_SinglePulse_n5`` -- 5 able-bodied single-pulse control subjects;
    - ``d4_SCI_patients`` -- 2 spinal-cord-injury participants.

    This loader exposes the **offline (calibration) recordings**, which contain
    cue-based left/right-hand motor-imagery trials. Each participant is mapped to
    a single session whose runs are the individual offline GDF recordings.

    Notes
    -----
    Only the offline cue-based recordings are exposed. The many longitudinal
    online (closed-loop feedback) recordings in the same archives are not loaded
    because they are continuous-control recordings without discrete class cues.
    The motor-imagery window (``interval``) is set to the 4 s following the class
    cue; adjust in the paradigm if a different window is desired.

    References
    ----------

    .. [1] Alawieh, H., Deland, L., Madera, J., Kumar, S., Racz, F. S.,
       Majewicz Fey, A., & Millán, J. del R. (2025). A Multi-Session EEG Dataset
       of Longitudinal Motor Imagery BCI Training with Transcutaneous Spinal
       Stimulation in Able-Bodied and Spinal Cord Injury Participants. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.15454354

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=32,
            channel_types={"eeg": 32, "eog": 3},
            montage="10-20",
            reference="mastoids (M1, M2)",
            sensors=_EEG_CHANNELS,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=True, eog_channels=3),
        ),
        participants=ParticipantMetadata(
            n_subjects=27,
            health_status="able-bodied and spinal cord injury",
            clinical_population="25 able-bodied, 2 spinal cord injury (SCI)",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            feedback_type="kinesthetic",
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.15454354",
            description=(
                "Longitudinal two-class (left/right hand) motor-imagery BCI "
                "training dataset with transcutaneous electrical spinal "
                "stimulation, in able-bodied and spinal cord injury participants."
            ),
            investigators=[
                "Hussein Alawieh",
                "Liu Deland",
                "Jonathan Madera",
                "Satyam Kumar",
                "Frigyes Samuel Racz",
                "Ann Majewicz Fey",
                "José del R. Millán",
            ],
            country="US",
            data_url="https://doi.org/10.5281/zenodo.15454354",
            publication_year=2025,
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="GDF",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 27 + 1)),
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="SpinalStim2025",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.5281/zenodo.15454354",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the offline GDF file paths for a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1..27).
        path : None | str
            Location where to look for / store the data. If None, the default
            MNE data directory is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            Sorted list of paths to the subject's offline GDF recordings.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        archive_key, token = _SUBJECT_MAP[subject]
        zip_name, root_name = _ARCHIVES[archive_key]
        url = SPINALSTIM2025_BASE.format(name=zip_name)

        path_zip = Path(dl.data_dl(url, self.code, path=path, force_update=force_update))
        path_folder = path_zip.parent

        # Extract the archive once.
        if not (path_folder / root_name).is_dir():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path_folder)

        # Collect this subject's offline recordings by globbing the extracted
        # tree: the on-disk layout differs between cohorts, so match on the
        # "Subject_<token>_..._Offline" folder rather than a fixed path.
        prefix = f"Subject_{token}_"
        subject_paths = [
            str(p)
            for p in sorted((path_folder / root_name).rglob("*.gdf"))
            if prefix in str(p) and "Offline" in str(p) and "Online" not in str(p)
        ]
        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the offline motor-imagery data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"0": {"0": raw0, "1": raw1, ...}}`` -- one session whose runs are
            the subject's offline GDF recordings.
        """
        file_paths = self.data_path(subject)
        runs = {}
        for run_idx, file_path in enumerate(file_paths):
            raw = self._read_run(file_path)
            runs[str(run_idx)] = raw
        return {"0": runs}

    @staticmethod
    def _read_run(file_path):
        """Read one GDF recording into a clean, annotated ``mne.io.Raw``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")

        # Drop the trigger channel so events are read from GDF annotations, and
        # drop any channel not part of the recorded EEG/aux set.
        keep = set(_EEG_CHANNELS) | set(_AUX_CHANNELS)
        drop = [ch for ch in raw.ch_names if ch not in keep]
        if drop:
            raw = raw.drop_channels(drop)

        aux_present = [ch for ch in _AUX_CHANNELS if ch in raw.ch_names]
        if aux_present:
            raw.set_channel_types(dict.fromkeys(aux_present, "eog"))

        # Keep only the two motor-imagery class cues and label them by class.
        onset, duration, desc = [], [], []
        for ann in raw.annotations:
            label = _CLASS_CODES.get(str(ann["description"]))
            if label is not None:
                onset.append(ann["onset"])
                duration.append(ann["duration"])
                desc.append(label)
        new_annotations = mne.Annotations(
            onset=onset, duration=duration, description=desc
        )
        raw = raw.set_annotations(new_annotations)

        montage = mne.channels.make_standard_montage("standard_1005")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = raw.set_montage(
                montage, match_case=False, on_missing="ignore", verbose=False
            )
        return raw
