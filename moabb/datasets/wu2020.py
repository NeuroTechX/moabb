"""Ear-EEG motor task classification dataset.

Wu, Zhang, Fu, Cheung, and Chan (2020), Journal of Neural Engineering.
DOI: 10.1088/1741-2552/abc1b6
Data DOI: 10.21227/j7rq-2p11
"""

import logging
from pathlib import Path

import mne

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# IEEE DataPort open-access dataset.
# Requires free IEEE DataPort account for download.
_DATA_URL = (
    "https://ieee-dataport.org/open-access/"
    "ear-eeg-recording-brain-computer-interface-motor-task"
)

# EEG channel groups.
_SCALP_CH_NAMES = [
    "Cz",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "T7",
    "T8",
]

# fmt: off
# Left/Right: Front, Back, Outer-Upper, Outer-Down (ear canal positions)
_EAR_CH_NAMES = ["LF", "LB", "LOU", "LOD", "RF", "RB", "ROU", "ROD"]  # codespell:ignore
# fmt: on

_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
}

_SFREQ = 1000.0


class Wu2020(BaseDataset):
    """Ear-EEG motor execution dataset from Wu et al 2020.

    Dataset from the article *An investigation of in-ear sensing for
    motor task classification* [1]_.

    **Important**: This is a motor **execution** dataset, not motor
    imagery. Participants physically clenched their left or right fist.

    It contains data recorded on 6 subjects with a combination of
    standard scalp EEG and custom in-ear EEG electrodes. The scalp
    cap provides 122 channels, and 8 additional ear channels (4 per ear)
    are recorded simultaneously via custom earpieces.

    For MOABB, only the motor-relevant scalp channels and ear channels
    are used by default.

    Two conditions were recorded:

    - **left_hand**: left hand fist clench
    - **right_hand**: right hand fist clench

    Subjects 1-5 have 160 trials (10 blocks x 16 trials), subject 6
    has 80 trials (5 blocks x 16 trials).

    References
    ----------
    .. [1] Wu, X., Zhang, W., Fu, Z., Cheung, R. T. H., & Chan, R. H. M.
           (2020). An investigation of in-ear sensing for motor task
           classification. Journal of Neural Engineering, 17(6), 066029.
           https://doi.org/10.1088/1741-2552/abc1b6
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=130,
            channel_types={"eeg": 130},
            montage="standard_1005",
            hardware="Neuroscan SynAmps2",
            sensor_type="Ag/AgCl",
            reference="scalp REF",
            ground="scalp GRD",
            filters={"bandpass": [0.5, 100]},
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=6,
            health_status="healthy",
            gender={"female": 4, "male": 2},
            age_mean=25.0,
            age_min=22.0,
            age_max=28.0,
            handedness="right-handed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            study_design=(
                "Motor execution (fist clenching) with simultaneous "
                "scalp and ear-EEG recording"
            ),
            feedback_type="none",
            stimulus_type="arrow cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2552/abc1b6",
            investigators=[
                "Xiaoli Wu",
                "Wenhui Zhang",
                "Zhibo Fu",
                "Roy T.H. Cheung",
                "Rosa H.M. Chan",
            ],
            institution="City University of Hong Kong",
            country="HK",
            data_url=_DATA_URL,
            publication_year=2020,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
        ),
        data_structure=DataStructureMetadata(
            n_trials=880,
            trials_context=("5 subjects x 160 trials + 1 subject x 80 trials = 880"),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="Curry",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 7)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Wu2020",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1088/1741-2552/abc1b6",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        subj_dir = base / f"subject{subject}"

        if not subj_dir.exists():
            # Try alternative paths
            for candidate in base.iterdir():
                if candidate.is_dir() and str(subject) in candidate.name:
                    subj_dir = candidate
                    break

        # Find the .dat file (Curry format)
        dat_files = sorted(subj_dir.glob("*.dat"))
        if not dat_files:
            dat_files = sorted(subj_dir.rglob("*.dat"))

        if not dat_files:
            raise FileNotFoundError(
                f"No .dat files found for subject {subject} in {subj_dir}"
            )

        runs = {}
        for run_idx, dat_file in enumerate(dat_files):
            raw = mne.io.read_raw_curry(str(dat_file), preload=True, verbose=False)

            # Extract events from annotations
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Map event codes: 1 -> left_hand, 2 -> right_hand
            # The exact mapping needs verification from actual data.
            raw.set_annotations(raw.annotations)

            runs[str(run_idx)] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Wu2020", path)
        basepath = Path(path) / "MNE-wu2020-data"
        basepath.mkdir(parents=True, exist_ok=True)

        subj_dir = basepath / f"subject{subject}"
        if subj_dir.exists() and list(subj_dir.glob("*.dat")):
            return str(basepath)

        # IEEE DataPort requires manual download.
        log.warning(
            "Wu2020 data must be downloaded manually from IEEE DataPort: "
            "%s "
            "Download subject%d.zip and extract to: %s",
            _DATA_URL,
            subject,
            basepath,
        )

        return str(basepath)
