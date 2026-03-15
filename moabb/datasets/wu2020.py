"""Ear-EEG motor task classification dataset.

Wu, Zhang, Fu, Cheung, and Chan (2020), Journal of Neural Engineering.
DOI: 10.1088/1741-2552/abc1b6
Data DOI (original): 10.21227/j7rq-2p11
"""

import logging
from pathlib import Path

import mne

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
    SignalProcessingMetadata,
    Tags,
)
from .utils import download_and_extract_subject_zip


log = logging.getLogger(__name__)

# Zenodo re-hosted data (originally from IEEE DataPort DOI: 10.21227/j7rq-2p11).
_ZENODO_RECORD = "18961128"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# fmt: off
# Left/Right: Front, Back, Outer-Upper, Outer-Down (ear canal positions)
_EAR_CH_NAMES = ["LF", "LB", "LOU", "LOD", "RF", "RB", "ROU", "ROD"]  # codespell:ignore
# fmt: on

_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
}


class Wu2020(BaseDataset):
    """Ear-EEG motor execution dataset from Wu et al 2020.

    Dataset from the article *An investigation of in-ear sensing for
    motor task classification* [1]_.

    **Important**: This is a motor **execution** dataset, not motor
    imagery. Participants physically clenched their left or right fist.

    It contains data recorded on 6 subjects with a combination of
    standard scalp EEG and custom in-ear EEG electrodes. The scalp
    cap provides 122 channels, and 8 additional ear channels (4 per ear)
    are recorded simultaneously via custom earpieces. There is also one
    unknown misc channel ("10") and one Trigger channel, for 132 total.

    Two conditions were recorded:

    - **left_hand**: left hand fist clench (event code 1)
    - **right_hand**: right hand fist clench (event code 2)

    Trial counts vary per subject (80-240 left/right trials, 1114 total).
    Subjects have 1-4 recording blocks (Curry .dat files) each treated
    as a separate run.

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
            n_channels=122,
            channel_types={"eeg": 122, "misc": 10},
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
            stimulus_modalities=["visual", "auditory"],
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
            repository="Zenodo",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
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
            n_trials=1114,
            trials_context=("S1: 240, S2: 160, S3: 160, S4: 80, S5: 234, S6: 240 = 1114"),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["EEGNet"],
            feature_extraction=None,
            frequency_bands=None,
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="Curry",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
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
            return_all_modalities=return_all_modalities,
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

        # Find .dat files (case-insensitive, some subjects use Motor*.dat)
        dat_files = sorted(f for f in subj_dir.iterdir() if f.suffix.lower() == ".dat")
        if not dat_files:
            raise FileNotFoundError(
                f"No .dat files found for subject {subject} in {subj_dir}"
            )

        runs = {}
        for run_idx, dat_file in enumerate(dat_files):
            raw = mne.io.read_raw_curry(str(dat_file), preload=True, verbose=False)

            # Filter annotations to only left_hand (1) and right_hand (2).
            # Some files contain extra codes (800000, 800001, Impedance Check).
            desired = {"1": 1, "2": 2}
            events, _ = mne.events_from_annotations(raw, event_id=desired, verbose=False)

            # Replace annotations with only the left/right events.
            event_desc = {1: "left_hand", 2: "right_hand"}
            annot = mne.annotations_from_events(
                events=events,
                event_desc=event_desc,
                sfreq=raw.info["sfreq"],
                orig_time=raw.info["meas_date"],
            )
            raw.set_annotations(annot)

            runs[str(run_idx)] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"subject{subject}"

        # Return cached files if already extracted
        if subj_dir.is_dir() and not force_update:
            dat_files = [f for f in subj_dir.iterdir() if f.suffix.lower() == ".dat"]
            if dat_files:
                return str(data_dir)

        # Download per-subject ZIP from Zenodo and extract.
        url = f"{_ZENODO_BASE}/subject{subject}.zip"
        download_and_extract_subject_zip(url, sign, data_dir, path, force_update, verbose)

        if not any(f.suffix.lower() == ".dat" for f in subj_dir.iterdir()):
            raise FileNotFoundError(
                f"No .dat files found for subject {subject} in {subj_dir}"
            )
        return str(data_dir)
