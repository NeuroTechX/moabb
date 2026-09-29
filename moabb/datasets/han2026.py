"""Basketball motor-observation / motor-imagery EEG dataset (Han et al., 2026)."""

import warnings

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

from .utils import resolve_montage_name


# OpenNeuro dataset ID and public S3 mirror (no auth required).
_OPENNEURO_ID = "ds007327"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# We expose the single-condition "dribble" task, the only task whose trials are
# pure (single-state) motor observation or motor imagery. The two-person
# "ballpass" and "passing" tasks contain only compound sequential conditions
# (MOMO / MOMI / MIMO / MIMI, "first observe X then imagine Y") and therefore
# have no pure MO or MI trials, so they are not loaded.
_TASK = "dribble"

# 64 scalp EEG channels (ANT Neuro waveguard). The recording additionally
# carries 24 unnamed bipolar auxiliary channels (BIP1..BIP24) that are dropped
# by default (they are not listed in the BIDS channels.tsv). Note the last
# scalp label "Oz_1" is a duplicate-name artifact of the BIDS/EEGLAB export;
# it has no standard montage position and is left unplaced.
# fmt: off
_EEG_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz", "C4", "T8", "M2",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz",
    "O1", "Oz", "O2", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
    "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4",
    "P5", "P1", "P2", "P6", "PO5", "PO3", "PO4", "PO6",
    "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "Oz_1",
]
# fmt: on

# Raw EEGLAB event markers (also mirrored in the BIDS events.tsv "value"
# column) -> MOABB class names. Only the two pure single-state conditions are
# kept: A11 = motor observation, A22 = motor imagery. The other markers present
# in the dribble task (A88 = combined observe+imagine, A44 = rest, A66 =
# static, plus block/impedance/boundary markers) are ignored.
_MARKER_TO_CLASS = {"A11": "motor_observation", "A22": "motor_imagery"}
_EVENTS = {"motor_observation": 1, "motor_imagery": 2}


class Han2026(BaseDataset):
    """Basketball motor-observation / motor-imagery EEG dataset [1]_.

    Thirty-five healthy participants performed basketball-related motor
    observation (MO) and motor imagery (MI), recorded with a 64-channel ANT Neuro
    system at 1000 Hz. The loader exposes the single-handed ``dribble`` task, the
    only one with pure single-state trials: 1 s trials labelled by the EEGLAB
    markers ``A11`` (observation) and ``A22`` (imagery), about 120 per class;
    combined, rest and static markers are ignored, and exact duplicate markers
    (subjects 20, 25, 27) are dropped. The ``ballpass`` and ``passing`` tasks hold
    only compound MO/MI sequences and are not loaded. The 24 unnamed bipolar
    ``BIP*`` channels are dropped unless ``return_all_modalities=True`` (then
    ``misc``).

    References
    ----------

    .. [1] Han, J., Wang, J., Jia, L., and Tang, M. (2026). Basketball Motor
       Observation and Motor Imagery EEG Dataset. OpenNeuro.
       DOI: https://doi.org/10.18112/openneuro.ds007327.v1.1.0

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    # OpenNeuro ds007327 is the upstream source, not a verified NEMAR mirror.
    # Leave nemar_id unset until a genuine NEMAR ID is independently verified.

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 64, "misc": 24},
            montage="standard_1005",
            hardware="ANT Neuro (64-channel)",
            reference="CPz",
            ground="AFz",
            sensors=list(_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=35, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["motor_observation", "motor_imagery"],
            trial_duration=1.0,
            study_design=(
                "Basketball single-handed dribbling task with per-trial motor "
                "observation (watch a dribbling video) versus motor imagery "
                "(imagine dribbling) conditions, one second per trial."
            ),
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events=dict(_EVENTS),
            instructions=(
                "Observe the dribbling video, or imagine performing the "
                "dribbling movement, as cued."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds007327.v1.1.0",
            description=(
                "EEG from 35 healthy subjects performing basketball motor "
                "observation and motor imagery. Recorded with a 64-channel ANT "
                "Neuro system at 1000 Hz. This loader exposes the dribble task "
                "(pure motor observation vs motor imagery, 1 s trials)."
            ),
            investigators=["Jiahui Han", "Jiaqi Wang", "Lingrong Jia", "Ming Tang"],
            institution="Liaoning Normal University",
            country="CN",
            data_url="https://openneuro.org/datasets/ds007327",
            publication_year=2026,
            keywords=[
                "motor imagery",
                "motor observation",
                "action observation",
                "EEG",
                "basketball",
                "brain-computer interface",
            ],
            license="CC0",
            repository="OpenNeuro",
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
            applications=["motor imagery", "action observation"],
            environment="lab",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["motor_observation", "motor_imagery"],
            imagery_duration_s=1.0,
        ),
        data_structure=DataStructureMetadata(
            trials_context=(
                "One dribble recording per subject with roughly 120 motor "
                "observation and 120 motor imagery one-second trials, "
                "interleaved with ignored combined / rest / static blocks."
            )
        ),
        file_format="EEGLAB (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 36)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Han2026",
            interval=[0, 1],
            paradigm="imagery",
            doi="10.18112/openneuro.ds007327.v1.1.0",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the subject's dribble ``.set`` file and return ``[path]``."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        subj_str = f"sub-{subject:03d}"
        url = f"{_S3_BASE}/{subj_str}/{subj_str}_task-{_TASK}_eeg.set"
        return [str(dl.data_dl(url, self.code, path, force_update, verbose))]

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {"0": Raw}}`` for the subject's dribble recording."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(
                self.data_path(subject)[0], preload=True, verbose=False
            )

        # 24 unnamed bipolar auxiliary channels (BIP1..BIP24) are misc.
        bip = [ch for ch in raw.ch_names if ch.startswith("BIP")]
        raw.set_channel_types(dict.fromkeys(bip, "misc"))
        if not self.return_all_modalities:
            raw.drop_channels(bip)

        # Rename the two pure trial markers to their class labels; every other
        # marker is ignored downstream by events_from_annotations.
        desc = raw.annotations.description.astype("<U25")
        for marker, name in _MARKER_TO_CLASS.items():
            desc[desc == marker] = name
        raw.annotations.description = desc
        self._drop_duplicate_trial_annotations(raw)

        # standard_1005 places 63/64 scalp channels ("Oz_1" stays unplaced).
        raw.set_montage(
            make_standard_montage(resolve_montage_name("colin27_1005")),
            on_missing="ignore",
            match_case=False,
        )
        return {"0": {"0": raw}}

    @staticmethod
    def _drop_duplicate_trial_annotations(raw):
        """Remove exact duplicate pure-condition annotations.

        Subjects 20, 25, and 27 each contain a duplicated ``A11`` EEGLAB marker at
        one onset, which would duplicate windows downstream. Only same-onset,
        same-class duplicates are removed; distinct labels at an onset remain
        visible as a data error.
        """
        class_names = set(_MARKER_TO_CLASS.values())
        seen, duplicates = set(), []
        ann = raw.annotations
        for index, key in enumerate(zip(ann.onset, ann.duration, ann.description)):
            if key[2] in class_names and key in seen:
                duplicates.append(index)
            else:
                seen.add(key)
        raw.annotations.delete(duplicates)
