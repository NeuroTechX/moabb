"""Sensory-guided motor-imagery BCI dataset from Wang et al. (2026)."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatreader import read_mat
from remotezip import RemoteZip

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
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
from moabb.datasets.utils import build_raw_from_epochs, safe_extract_zip


_DATASET_DOI = "10.1184/R1/32293995.v1"
_PAPER_DOI = "10.1038/s41467-026-75435-5"
_FIGSHARE_FILE = "https://ndownloader.figshare.com/files/{}"

# Each archive is one experimental group within the same study. The ranges are
# stable global MOABB IDs; source subject numbers restart inside every archive.
_GROUPS = {
    "joint_learning": ("JointLearning", 64710999, range(1, 16)),
    "bci2000_control": ("BCI2000Control", 64710750, range(16, 24)),
    "tactile_control": ("TactileControl", 64710993, range(24, 32)),
    "eegnet_control": ("EEGNetControl", 64710990, range(32, 40)),
}
_SUBJECT_MAPPING = {
    subject: (archive, f"S{subject - subjects.start + 1:03d}")
    for archive, _, subjects in _GROUPS.values()
    for subject in subjects
}
_FILE_IDS = {archive: file_id for archive, file_id, _ in _GROUPS.values()}

_EVENTS = {"left_hand": 1, "right_hand": 2, "hands": 3, "rest": 4}
_EEGNET_EVENT_MAP = {0: 1, 1: 2, 2: 3, 3: 4}
_BCI2000_EVENT_MAP = {1: 2, 2: 1, 3: 3, 4: 4}
_BCI2000_UD_EVENT_MAP = {1: 3, 2: 4}

_SFREQ = 1000.0
_TRIAL_DURATION_S = 5.0
_NOMINAL_TRIAL_SAMPLES = int(_TRIAL_DURATION_S * _SFREQ)
# The release does not document trialSignal's unit; confirm this conventional
# microvolt conversion with the data authors before merge.
_EEG_SCALE = 1e-6

# Canonical MNE capitalization of the 62 released EEG channels.
_CHANNELS = [
    "Fp1",
    "Fpz",
    "Fp2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POz",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "Oz",
    "O2",
    "CB2",
]

_ACQUISITION = AcquisitionMetadata(
    sampling_rate=_SFREQ,
    channel_types={"eeg": 62},
    sensors=list(_CHANNELS),
    reference="midway between Cz and CPz",
    hardware="Compumedics Neuroscan SynAmps 2/RT",
    software="BCI2000",
    filters={"notch": 60.0},
    line_freq=60.0,
    montage="standard_1005",
    impedance_threshold_kohm=5.0,
    cap_manufacturer="Compumedics Neuroscan",
    cap_model="Quik-Cap",
)

_DOCUMENTATION = DocumentationMetadata(
    doi=_DATASET_DOI,
    investigators=[
        "Hanwen Wang",
        "Yisha Zhang",
        "Maxim Karrenbach",
        "Yidan Ding",
        "Bin He",
    ],
    institution="Carnegie Mellon University",
    institution_department="Department of Biomedical Engineering",
    country="US",
    repository="KiltHub (Figshare)",
    data_url=(
        "https://kilthub.cmu.edu/articles/dataset/"
        "EEG_Datasets_for_Sensory-Guided_Joint_Learning_in_"
        "Motor_Imagery_Brain-Computer_Interfaces/32293995"
    ),
    license="CC-BY-NC-4.0",
    publication_year=2026,
    senior_author="Bin He",
    contact_info=["bhe1@andrew.cmu.edu"],
    associated_paper_doi=_PAPER_DOI,
    funding=["NS124564", "NS131069", "NS127849", "NS096761"],
    keywords=["motor imagery", "BCI", "joint learning", "tactile stimulation", "EEGNet"],
)

_METADATA = DatasetMetadata(
    acquisition=_ACQUISITION,
    participants=ParticipantMetadata(
        n_subjects=39,
        health_status="healthy",
        bci_experience="naive",
        species="homo sapiens",
    ),
    experiment=ExperimentMetadata(
        events=dict(_EVENTS),
        paradigm="imagery",
        task_type="online 1D and 2D cursor control",
        n_classes=4,
        class_labels=list(_EVENTS),
        trial_duration=_TRIAL_DURATION_S,
        tasks=[
            "left hand imagery",
            "right hand imagery",
            "bilateral hand imagery",
            "rest",
        ],
        study_design=(
            "One longitudinal MI-BCI study release with 39 unique participants. "
            "The primary randomized experiment included 31 participants: joint "
            "learning (n=15), BCI2000 control (n=8), and tactile control (n=8). "
            "A separately recruited EEGNet-control cohort added 8 participants."
        ),
        feedback_type=(
            "visual cursor feedback, with cohort-specific tactile guidance, "
            "EEGNet adaptation, or fixed BCI2000 decoding"
        ),
        stimulus_type="visual cursor and directional cue",
        stimulus_modalities=["visual", "tactile"],
        primary_modality="visual",
        synchronicity="synchronous",
        mode="online",
        has_training_test_split=True,
        stimulus_presentation={"SoftwareName": "BCI2000"},
    ),
    documentation=_DOCUMENTATION,
    sessions_per_subject=5,
    runs_per_session=None,
    contributing_labs=[
        "Carnegie Mellon University Biomedical Functional Imaging and Neuroengineering Lab"
    ],
    n_contributing_labs=1,
    data_processed=True,
    file_format="MATLAB v7.3",
    external_links={
        "source": _DOCUMENTATION.data_url,
        "paper": f"https://doi.org/{_PAPER_DOI}",
        "code": "https://github.com/bfinl/SensoryGuidedJointLearning",
    },
    tags=Tags(
        pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery", "Online BCI"]
    ),
    preprocessing=PreprocessingMetadata(
        data_state="trial-segmented",
        preprocessing_applied=True,
        preprocessing_steps=[
            "trial segmentation",
            "auxiliary-channel exclusion",
            "per-trial temporal centering in the MOABB reconstruction",
        ],
        notch_hz=60.0,
        notes=(
            "The release contains 62 selected EEG channels and states that no "
            "additional filtering or downsampling was applied. Runs use either "
            "fixed-length EEGNet trials or variable-length BCI2000 trials."
        ),
    ),
    signal_processing=SignalProcessingMetadata(
        classifiers=["weighted EEGNet", "EEGNet", "BCI2000 linear classifier"],
        feature_extraction=["EEGNet", "autoregressive spectral features"],
        frequency_bands={"online_EEGNet": [4.0, 40.0]},
        spatial_filters=["CSP-based initial sample weighting"],
    ),
    bci_application=BCIApplicationMetadata(
        applications=["cursor control"], environment="laboratory", online_feedback=True
    ),
    paradigm_specific=ParadigmSpecificMetadata(
        detected_paradigm="imagery",
        imagery_tasks=list(_EVENTS),
        imagery_duration_s=_TRIAL_DURATION_S,
    ),
    data_structure=DataStructureMetadata(
        n_trials="varies by run and subject",
        n_blocks=6,
        trials_context=(
            "Each regular session contains six runs; run0 is a separate initial "
            "BCI2000 baseline. Subjects completed 4, 6, or 8 regular sessions."
        ),
    ),
)


class Wang2026(BaseDataset):
    """Sensory-guided joint-learning MI-BCI study from Wang et al. (2026).

    The release contains 39 participants in four experimental groups:
    joint-learning (15), BCI2000 control (8), tactile control (8), and an
    independently recruited EEGNet control (8). ``group`` filters these
    conditions while preserving globally unique MOABB subject IDs.

    The public files are trial-segmented MATLAB files, not original BCI2000
    recordings, so MNE's BCI2000 reader does not apply. Fixed-length trials are
    cropped to five seconds. Variable-length trials shorter than five seconds
    are omitted rather than synthetically padded.

    Parameters
    ----------
    group : {"all", "joint_learning", "bci2000_control", "tactile_control", \
            "eegnet_control"}
        Experimental-group filter.
    subjects : list of int | None
        Optional global subject IDs within ``group``.
    sessions : list of int or str | None
        Optional sessions to retain.

    References
    ----------
    .. [1] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
       Sensory-guided human-machine joint learning accelerates the acquisition
       of motor imagery brain computer interface control. Nature
       Communications, 17, 6177.
       https://doi.org/10.1038/s41467-026-75435-5
    .. [2] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
       EEG Datasets for Sensory-Guided Joint Learning in Motor Imagery
       Brain-Computer Interfaces. Carnegie Mellon University.
       https://doi.org/10.1184/R1/32293995.v1
    """

    METADATA = _METADATA

    def __init__(
        self, group="all", subjects=None, sessions=None, *, return_all_modalities=False
    ):
        valid_groups = ("all", *_GROUPS)
        if group not in valid_groups:
            raise ValueError(f"group must be one of {valid_groups}, got {group!r}.")

        group_subjects = (
            list(_SUBJECT_MAPPING) if group == "all" else list(_GROUPS[group][2])
        )
        if subjects is None:
            selected_subjects = None if group == "all" else group_subjects
        else:
            selected_subjects = list(subjects)
            invalid = [
                subject for subject in selected_subjects if subject not in group_subjects
            ]
            if invalid:
                raise ValueError(
                    f"Subjects {invalid} are not in group {group!r}; "
                    f"valid global IDs are {group_subjects}."
                )

        self.group = group
        super().__init__(
            subjects=list(_SUBJECT_MAPPING),
            sessions_per_subject=5,
            events=dict(_EVENTS),
            code="Wang2026",
            interval=[0.0, (_NOMINAL_TRIAL_SAMPLES - 1) / _SFREQ],
            paradigm="imagery",
            doi=_DATASET_DOI,
            selected_subjects=selected_subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    @property
    def subject_mapping(self):
        """Map global IDs to ``(archive group, source subject)``."""
        return _SUBJECT_MAPPING

    def _get_single_subject_data(self, subject):
        sessions = defaultdict(dict)
        archive, source_subject = _SUBJECT_MAPPING[subject]

        for path in map(Path, self.data_path(subject)):
            run_data = read_mat(
                path,
                ignore_fields=(
                    "meta",
                    "trialInfo",
                    "trialCursorPosX",
                    "trialCursorPosY",
                    "trialFeedbackLabel",
                    "trialMoveRight",
                ),
            )["runData"]
            signal = run_data["trialSignal"]

            if isinstance(signal, list):
                targets = run_data["trialTargetCode"]
                mapping = (
                    _BCI2000_UD_EVENT_MAP
                    if path.stem.endswith("UD")
                    else _BCI2000_EVENT_MAP
                )
                keep = [
                    index
                    for index, trial in enumerate(signal)
                    if np.asarray(trial).shape[0] >= _NOMINAL_TRIAL_SAMPLES
                ]
                epochs = [
                    np.asarray(signal[index])[:_NOMINAL_TRIAL_SAMPLES].T for index in keep
                ]
                events = [
                    mapping[int(np.asarray(targets[index]).flat[0])] for index in keep
                ]
            else:
                targets = run_data["trialTargetClass"]
                epochs = np.asarray(signal)[:_NOMINAL_TRIAL_SAMPLES].transpose(2, 1, 0)
                events = [
                    _EEGNET_EVENT_MAP[int(target)] for target in np.asarray(targets)[0]
                ]

            if len(epochs) == 0:
                continue

            raw = build_raw_from_epochs(
                epochs, _CHANNELS, _SFREQ, events, "standard_1005", scale=_EEG_SCALE
            )
            raw.info["line_freq"] = 60.0
            raw.info["subject_info"] = {"his_id": f"{archive}/{source_subject}"}

            session = "0baseline"
            if "_sess" in path.stem:
                session = str(int(path.stem.split("_sess", 1)[1].split("_", 1)[0]))
            sessions[session][str(len(sessions[session]))] = raw

        if not sessions:
            raise FileNotFoundError(
                f"No five-second trials found for Wang2026 subject {subject}."
            )
        return dict(sessions)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Extract one subject without downloading its complete cohort archive."""
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject {subject}. Valid subjects: {self.subject_list}"
            )

        archive, source_subject = _SUBJECT_MAPPING[subject]
        root = Path(dl.get_dataset_path(self.code, path)) / "MNE-wang2026-data"
        subject_dir = root / archive / source_subject
        files = sorted(subject_dir.glob("*.mat"))
        if files and not force_update:
            return [str(file) for file in files]

        root.mkdir(parents=True, exist_ok=True)
        prefix = f"{archive}/{source_subject}/"
        url = _FIGSHARE_FILE.format(_FILE_IDS[archive])
        with RemoteZip(url, initial_buffer_size=1024 * 1024, timeout=300) as zip_file:
            members = [
                member
                for member in zip_file.infolist()
                if member.filename.startswith(prefix) and member.filename.endswith(".mat")
            ]
            if not members:
                raise FileNotFoundError(f"No runs found for {prefix}.")
            safe_extract_zip(zip_file, root, members=members)

        return [str(file) for file in sorted(subject_dir.glob("*.mat"))]
