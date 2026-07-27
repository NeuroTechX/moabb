"""Basketball motor-observation / motor-imagery EEG dataset (Han et al., 2026).

Han, J., Wang, J., Jia, L., and Tang, M. (2026).
"Basketball Motor Observation and Motor Imagery EEG Dataset."
Data DOI: 10.18112/openneuro.ds007327.v1.1.0
Hosted on OpenNeuro (ds007327), BIDS / EEGLAB (.set) format.
"""

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


log = logging.getLogger(__name__)

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

    .. admonition:: Dataset summary

        =======  =======  =======  ==========  =================  ============  ===============  ===========
        Name       #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =======  =======  =======  ==========  =================  ============  ===============  ===========
        Han2026       35       64           2              ~120            1s           1000 Hz            1
        =======  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Thirty-five healthy participants performed basketball-related motor
    observation (MO) and motor imagery (MI) tasks while EEG was recorded with a
    64-channel ANT Neuro system at 1000 Hz (online reference CPz, ground AFz).
    Each participant completed three tasks: a single-handed ``dribble`` task, a
    two-handed ``ballpass`` task, and a two-person ``passing`` task.

    This loader exposes the ``dribble`` task, the only task whose trials are
    pure single-state conditions. In it, participants either observed a
    dribbling video (motor observation) or imagined dribbling (motor imagery),
    each labelled per trial by a data-borne EEGLAB event marker (``A11`` =
    observation, ``A22`` = imagery), corroborated by the BIDS ``events.tsv``
    ``trial_type`` column. Trials are one second long; the loader keeps the two
    pure classes (motor observation vs motor imagery, roughly 120 trials each
    per subject) and ignores the combined observe+imagine, rest and static
    control conditions of the same task.

    The two-person ``ballpass`` and ``passing`` tasks are not loaded because
    they contain only compound sequential conditions (MOMO / MOMI / MIMO /
    MIMI, "first observe X then imagine Y"), i.e. no pure MO or MI trials.

    Twenty-four unnamed bipolar auxiliary channels (``BIP1``..``BIP24``) present
    in the raw ``.set`` are dropped by default (kept as ``misc`` when
    ``return_all_modalities`` is set); only the 64 scalp electrodes are exposed.

    References
    ----------

    .. [1] Han, J., Wang, J., Jia, L., and Tang, M. (2026). Basketball Motor
       Observation and Motor Imagery EEG Dataset. OpenNeuro.
       DOI: https://doi.org/10.18112/openneuro.ds007327.v1.1.0

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
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
        """Download the subject's dribble ``.set`` file and return its path.

        Parameters
        ----------
        subject : int
            Subject number (1-35).
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
            Single-element list with the path to the downloaded ``.set`` file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        subj_str = f"sub-{subject:03d}"
        url = f"{_S3_BASE}/{subj_str}/{subj_str}_task-{_TASK}_eeg.set"
        set_path = dl.data_dl(url, self.code, path, force_update, verbose)
        return [str(set_path)]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-35).

        Returns
        -------
        dict
            ``{"0": {"0": Raw}}`` for the single dribble recording, with the
            motor-observation and motor-imagery trial markers renamed to their
            class labels.
        """
        set_path = Path(self.data_path(subject)[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

        # The .set carries 64 scalp EEG channels plus 24 unnamed bipolar
        # auxiliary channels (BIP1..BIP24). Mark the aux channels as misc and
        # drop them unless all modalities are requested.
        bip = [ch for ch in raw.ch_names if ch.startswith("BIP")]
        if bip:
            raw.set_channel_types(dict.fromkeys(bip, "misc"))
            if not self.return_all_modalities:
                raw.drop_channels(bip)

        # Rename the two pure single-state trial markers to their class labels.
        # events_from_annotations (used downstream by MOABB) then matches on
        # these descriptions; every other marker is ignored.
        desc = raw.annotations.description.astype("<U25")
        for marker, name in _MARKER_TO_CLASS.items():
            desc[desc == marker] = name
        raw.annotations.description = desc
        self._drop_duplicate_trial_annotations(raw)

        # standard_1005 places 63/64 scalp channels; the duplicate "Oz_1"
        # export artifact has no standard position and is left unplaced.
        raw.set_montage(
            make_standard_montage("standard_1005"), on_missing="ignore", match_case=False
        )

        return {"0": {"0": raw}}

    @staticmethod
    def _drop_duplicate_trial_annotations(raw):
        """Remove exact duplicate pure-condition annotations.

        Subjects 20, 25, and 27 each contain a duplicated ``A11`` EEGLAB
        marker at one onset.  Retaining both produces duplicate windows in
        downstream consumers.  Only same-onset, same-class duplicates are
        removed; distinct labels at an onset remain visible as a data error.
        """
        class_names = set(_MARKER_TO_CLASS.values())
        seen = set()
        duplicates = []
        for index, (onset, duration, description) in enumerate(
            zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
            )
        ):
            key = (onset, duration, description)
            if description in class_names and key in seen:
                duplicates.append(index)
            else:
                seen.add(key)
        if duplicates:
            raw.annotations.delete(duplicates)
