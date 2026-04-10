"""Motor imagery EEG dataset with classical left/right hand paradigm.

Kaya et al. (2018), Scientific Data.
DOI: 10.1038/sdata.2018.211
"""

import logging
from pathlib import Path

import mne
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
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

# 19 EEG channels used (columns 0-9, 12-20 in o.data; skip A1, A2 at 10-11
# and X3 sync at column 21)
_EEG_CH_NAMES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]

# Indices into the 22-column data array for EEG channels only
_EEG_COL_IDX = list(range(10)) + list(range(12, 21))  # skip cols 10,11 (A1,A2), 21 (X3)

# Mapping from MOABB subject integer to original letter and list of
# (figshare_file_id, mat_filename) tuples for CLA recordings.
# Subject F session 1 (150916) has only 2 classes (left/right, no passive).
_SUBJECT_FILES = {
    1: [  # Subject A
        (9636466, "CLA-SubjectA-160108-3St-LRHand.mat")
    ],
    2: [  # Subject B
        (9636463, "CLA-SubjectB-151019-3St-LRHand.mat"),
        (9636469, "CLA-SubjectB-151020-3St-LRHand.mat"),
        (9636472, "CLA-SubjectB-151215-3St-LRHand.mat"),
    ],
    3: [  # Subject C
        (9636475, "CLA-SubjectC-151126-3St-LRHand.mat"),
        (9636478, "CLA-SubjectC-151216-3St-LRHand.mat"),
        (9636481, "CLA-SubjectC-151223-3St-LRHand.mat"),
    ],
    4: [  # Subject D
        (9636484, "CLA-SubjectD-151125-3St-LRHand.mat")
    ],
    5: [  # Subject E
        (9636487, "CLA-SubjectE-151225-3St-LRHand.mat"),
        (9636490, "CLA-SubjectE-160119-3St-LRHand.mat"),
        (9636496, "CLA-SubjectE-160122-3St-LRHand.mat"),
    ],
    6: [  # Subject F
        (9636493, "CLA-SubjectF-150916-3St-LRHand.mat"),
        (9636505, "CLA-SubjectF-150917-3St-LRHand.mat"),
        (9636502, "CLA-SubjectF-150928-3St-LRHand.mat"),
    ],
    7: [  # Subject J
        (12400406, "CLA-SubjectJ-170504-3St-LRHand-Inter.mat"),
        (12400412, "CLA-SubjectJ-170508-3St-LRHand-Inter.mat"),
        (12400409, "CLA-SubjectJ-170510-3St-LRHand-Inter.mat"),
    ],
}

_FIGSHARE_DL_BASE = "https://ndownloader.figshare.com/files"

# Event marker codes used in the CLA paradigm
_EVENT_CODES = {1: "left_hand", 2: "right_hand", 3: "passive"}


class Kaya2018(BaseDataset):
    """Classical motor imagery dataset with left hand, right hand, and rest.

    Dataset from [1]_.

    This dataset contains 19-channel EEG recordings from 7 subjects (labeled
    A-F and J in the original data, mapped to integers 1-7 here) performing
    a classical (CLA) motor imagery task. Three mental states are cued:

    - **left_hand** (code 1): left hand motor imagery
    - **right_hand** (code 2): right hand motor imagery
    - **passive** (code 3): passive/rest state

    EEG was recorded at 200 Hz with a Nihon Kohden EEG-1200 system using 19
    standard 10-20 electrodes plus A1/A2 reference/ground and an X3 sync
    channel (22 columns total in the data files). Only the 19 EEG channels
    are used by this adapter.

    Each trial consists of a 1-second visual cue followed by a 1.5-2.5
    second inter-trial interval. Subjects have between 1 and 3 recording
    sessions (CLA files) each.

    .. note::

       Subject 6 (F), session 0 (``CLA-SubjectF-150916``) contains only
       left_hand and right_hand events (no passive trials). This was one
       of the earliest recordings in the study.

       Subject 7 (J) data was recorded with an interactive BCI interface
       and has different signal resolution (0.133 uV vs 0.01 uV for other
       subjects) and a narrower dynamic range.

    The full Figshare collection contains 77 articles spanning multiple
    paradigms (CLA, HaLT, 5F, FREEFORM, NoMT). This adapter uses only the
    17 CLA files.

    References
    ----------
    .. [1] M. Kaya, M. K. Binli, E. Ozbay, H. Yanar, and Y. Mishchenko,
       "A large electroencephalographic motor imagery dataset for
       electroencephalographic brain computer interfaces," Scientific Data,
       vol. 5, p. 180211, 2018. DOI: 10.1038/sdata.2018.211
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=200.0,
            n_channels=19,
            channel_types={"eeg": 19},
            sensors=_EEG_CH_NAMES,
            montage="standard_1020",
            hardware="Nihon Kohden EEG-1200",
            reference="System 0V (0.55*(C3+C4))",
            ground="A1, A2 (earlobes)",
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=7,
            health_status="healthy",
            gender={"male": 5, "female": 2},
            age_min=20,
            age_max=35,
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            events={"left_hand": 1, "right_hand": 2, "passive": 3},
            n_classes=3,
            class_labels=["left_hand", "right_hand", "passive"],
            trial_duration=1.0,
            stimulus_type="visual arrow cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            study_design="Classical left/right hand motor imagery with passive rest",
            task_type="left_right_hand",
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/sdata.2018.211",
            investigators=[
                "Murat Kaya",
                "Mustafa Kemal Binli",
                "Erkan Ozbay",
                "Hilmi Yanar",
                "Yuriy Mishchenko",
            ],
            senior_author="Yuriy Mishchenko",
            institution="Mersin University",
            country="TR",
            repository="Figshare",
            data_url="https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698",
            license="CC-BY-4.0",
            publication_year=2018,
            keywords=["EEG", "motor imagery", "brain-computer interface", "BCI"],
        ),
        preprocessing=PreprocessingMetadata(data_state="raw"),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left_hand", "right_hand", "passive"],
            cue_duration_s=1.0,
        ),
        data_structure=DataStructureMetadata(
            trials_context="Variable number of trials per session; 1s cue + 1.5-2.5s ITI"
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SVM"],
            feature_extraction=["fourier_transform_amplitudes"],
            frequency_bands={"low_pass": [0.0, 5.0]},
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="repeated_random_split",
            cv_folds=5,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(environment="lab", online_feedback=False),
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
        file_format="MAT",
    )

    _events = {"left_hand": 1, "right_hand": 2, "passive": 3}

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 8)),
            sessions_per_subject=1,
            events=self._events,
            code="Kaya2018",
            interval=[0, 1],
            paradigm="imagery",
            doi="10.1038/sdata.2018.211",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject.

        Each CLA .mat file becomes a separate session. Within each session
        there is a single run.

        Returns
        -------
        dict
            ``{session_str: {"0": mne.io.RawArray}}``
        """
        file_paths = self.data_path(subject)
        sessions = {}

        for sess_idx, mat_path in enumerate(file_paths):
            mat = loadmat(mat_path, squeeze_me=True)
            o = mat["o"]

            # Extract fields from the struct
            data_all = o["data"].item()  # (nS, 22) in microvolts
            marker = o["marker"].item().ravel()  # (nS,)
            sfreq = float(o["sampFreq"].item())

            # Select EEG channels and convert uV -> V
            eeg_data = data_all[:, _EEG_COL_IDX].T * 1e-6  # (19, nS)

            # Create MNE info and RawArray
            info = mne.create_info(
                ch_names=list(_EEG_CH_NAMES), sfreq=sfreq, ch_types="eeg", verbose=False
            )
            raw = mne.io.RawArray(eeg_data, info, verbose=False)

            # Set standard 10-20 montage
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="warn", verbose=False)

            # Extract event onsets from marker channel transitions
            # Events are where marker transitions from 0 (or service code) to
            # a task code (1, 2, or 3).
            onsets = []
            for i in range(1, len(marker)):
                code = int(marker[i])
                if code in _EVENT_CODES and int(marker[i - 1]) != code:
                    onset_sec = i / sfreq
                    onsets.append((onset_sec, _EVENT_CODES[code]))

            if onsets:
                onset_times, descriptions = zip(*onsets)
                annotations = mne.Annotations(
                    onset=list(onset_times),
                    duration=[1.0] * len(onset_times),
                    description=list(descriptions),
                )
                raw.set_annotations(annotations, verbose=False)

            sessions[str(sess_idx)] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download and return paths to CLA .mat files for a subject.

        Parameters
        ----------
        subject : int
            Subject number (1-7).

        Returns
        -------
        list of str
            Paths to downloaded .mat files, one per session.
        """
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject {subject}. Valid subjects: {self.subject_list}"
            )

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        file_list = _SUBJECT_FILES[subject]
        paths = []

        for file_id, filename in file_list:
            local_path = data_dir / filename
            if local_path.exists() and not force_update:
                paths.append(str(local_path))
                continue

            url = f"{_FIGSHARE_DL_BASE}/{file_id}"
            dl_path = dl.data_dl(url, sign, path, force_update, verbose)

            # Pooch may save with a hash-based name; rename to expected filename
            dl_path = Path(dl_path)
            if dl_path != local_path:
                dl_path.rename(local_path)

            paths.append(str(local_path))

        return paths
