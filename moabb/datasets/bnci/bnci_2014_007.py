"""BNCI 2014-007: Affective Pacman dataset (Reuderink et al. 2011)."""

import logging

import numpy as np
from mne import Annotations
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.bnci.utils import make_raw
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

BNCI_URL = "https://lampx.tugraz.at/~bci/database/"

# Subjects S3 and S8 were withheld by the authors because they did not allow
# anonymous publication; the ten public recordings are renumbered 1-10.
_FILE_BY_SUBJECT = {
    1: "00",
    2: "01",
    3: "02",
    4: "04",
    5: "05",
    6: "06",
    7: "07",
    8: "09",
    9: "10",
    10: "11",
}

# The recordings were acquired at 512 Hz with a BioSemi ActiveTwo and are
# distributed at 128 Hz: the description PDF gives the 512 Hz recording rate,
# but the per-sample time row of I advances by 1/128 s.
SFREQ = 128.0

# Rows of the X matrix, in recording order: 32 EEG (Fp1...Cz), EXG1-8
# auxiliaries, then the physiological channels. GSR2, Erg1 and Erg2 were
# unused and contain noise only (dataset description, Table 1); they are
# never exposed.
_CH_NAMES_RECORDED = [
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
    "EXG1",
    "EXG2",
    "EXG3",
    "EXG4",
    "EXG5",
    "EXG6",
    "EXG7",
    "EXG8",
    "GSR1",
    "GSR2",
    "Erg1",
    "Erg2",
    "Resp",
    "Plet",
    "Temp",
]
_NOISE_CH_NAMES = ("GSR2", "Erg1", "Erg2")
_CH_NAMES = [name for name in _CH_NAMES_RECORDED if name not in _NOISE_CH_NAMES]
_CH_TYPES_EEG_ONLY = ["eeg"] * 32
_CH_TYPES_ALL = ["eeg"] * 32 + ["eog"] * 4 + ["emg"] * 4 + ["misc"] * 4

# Annotation description for each marker kept from Table 2 of the dataset
# description. Markers 1/2 and 3/4 are both described as left/right key
# presses; in the data, markers 3/4 and the screen freezes (5) occur
# exclusively during frustration blocks and match the ~15% of keystrokes that
# the game randomly ignored there (Reuderink et al. 2011), so they are exposed
# as distinct annotations instead of being merged into the movement classes.
_DESCRIPTION_BY_MARKER = {
    1: "left_hand",
    2: "right_hand",
    3: "ignored_left_hand",
    4: "ignored_right_hand",
    5: "screen_freeze",
    10: "init_level",
    11: "next_level",
    12: "avatar_died",
}


class BNCI2014_007(BaseDataset):
    """Affective Pacman dataset described in Reuderink et al. 2011.

    Dataset from [1]_, distributed as data set 007-2014 by BNCI Horizon 2020
    [2]_.

    Twelve subjects played a variant of the Pacman game in which a constantly
    moving Pacman avatar is turned 90 degrees to the left or right by key
    presses with the left and right index fingers. The game periodically
    induced a loss of control: in the frustration blocks the game randomly
    ignored about 15% of the keystrokes and the display lagged, which reliably
    altered the emotional state of the player, as repeatedly rated with
    valence/arousal/dominance (SAM) scales between blocks. Each session
    consists of alternating blocks of roughly two minutes (nine or ten normal
    and four to six frustration blocks per subject). The number of blocks
    varies per subject (14 to 16); each block is exposed as a separate run
    named ``<index>normal<k>`` or ``<index>frustration<k>`` (recording-order
    index, condition and per-condition counter) so that users can train on
    normal blocks and evaluate on frustration blocks, following the analysis
    of [1]_.

    This dataset contains *executed* finger movements rather than motor
    imagery; as with :class:`Schirrmeister2017`, it is mapped to the
    ``MotorImagery`` paradigm with ``left_hand``/``right_hand`` classes, and
    the executed nature is documented here and in the metadata. Following
    [1]_, classification epochs are one-second windows centered on the
    registered keystrokes (``interval=[-0.5, 0.5]``).

    The markers distinguishing ignored keystrokes (3/4) and screen freezes (5)
    are kept as separate annotations (``ignored_left_hand``,
    ``ignored_right_hand``, ``screen_freeze``) and are not merged into the
    movement classes; game events (10/11/12: level initialization, next level
    and avatar death) are annotated as well. Self-assessment responses
    (markers 100-129) fall between blocks and are not part of any run.

    EEG was recorded with 32 Ag/AgCl active electrodes placed according to the
    extended International 10-20 system. EOG (EXG1-4) and EMG over the left
    and right flexor digitorum profundus (EXG5-8) were recorded to monitor
    artifacts, together with GSR, plethysmograph, respiration and temperature
    channels. No additional filtering was applied to the sensor data. The
    unused noise channels GSR2, Erg1 and Erg2 described in the dataset
    documentation are dropped; by default only EEG channels are returned, and
    all remaining channels are exposed with ``return_all_modalities=True``.

    Subjects S3 and S8 were withheld by the authors and are not part of the
    public distribution; the ten public recordings are renumbered 1-10
    (public subjects 1-10 correspond to the original files S00, S01, S02,
    S04, S05, S06, S07, S09, S10 and S11). The data are released under
    CC BY-NC-ND 4.0.

    References
    ----------

    .. [1] Reuderink, B., Poel, M., & Nijholt, A. (2011). The impact of loss
           of control on movement BCIs. IEEE Transactions on Neural Systems
           and Rehabilitation Engineering, 19(6), 628-637.
           https://doi.org/10.1109/TNSRE.2011.2166562
    .. [2] Reuderink, B., Nijholt, A., & Poel, M. (2009). Affective Pacman: A
           frustrating game for brain-computer interface experiments. In
           Proceedings of the 3rd International Conference on Intelligent
           Technologies for Interactive Entertainment (INTETAIN 2009), LNICST
           9, 221-227. https://doi.org/10.1007/978-3-642-02315-6_23
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            channel_types={"eeg": 32, "eog": 4, "emg": 4, "misc": 4},
            hardware="BioSemi ActiveTwo",
            reference="CMS/DRL",
            sensors=_CH_NAMES_RECORDED,
            line_freq=50.0,
            software="BioSemi ActiveTwo with a custom Pacman game",
            montage="standard_1005",
            sensor_type="EEG",
        ),
        participants=ParticipantMetadata(
            n_subjects=10, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            study_design=(
                "Executed left/right index-finger key presses during affective "
                "Pacman gameplay, alternating normal and frustration "
                "(loss-of-control) blocks"
            ),
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            trial_duration=1.0,
            synchronicity="event-based",
            stimulus_presentation={
                "type": "custom Pacman game controlled by left/right key presses",
                "condition_mapping": (
                    "normal blocks: full control; frustration blocks: ~15% of "
                    "keystrokes randomly ignored, display lag"
                ),
            },
            instructions=(
                "Play the Pacman game by turning the avatar with left and "
                "right index-finger key presses"
            ),
            events={"left_hand": 1, "right_hand": 2},
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2011.2166562",
            related_paper_dois=["10.1007/978-3-642-02315-6_23"],
            publication_year=2011,
            investigators=["Boris Reuderink", "Mannes Poel", "Anton Nijholt"],
            senior_author="Anton Nijholt",
            institution="University of Twente",
            institution_department="Human Media Interaction",
            country="NL",
            how_to_acknowledge=(
                "Please cite: Reuderink, Poel & Nijholt (2011), The impact of "
                "loss of control on movement BCIs, IEEE TNSRE 19(6), 628-637, "
                "and Reuderink, Nijholt & Poel (2009), Affective Pacman: a "
                "frustrating game for brain-computer interface experiments, "
                "INTETAIN 2009, LNICST 9, 221-227."
            ),
            repository="BNCI",
            data_url="https://lampx.tugraz.at/~bci/database/007-2014/",
            license="CC-BY-NC-ND-4.0",
            keywords=[
                "electroencephalography",
                "brain-computer interface",
                "motor execution",
                "affect",
                "loss of control",
                "gaming",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=14,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Execution"]),
        preprocessing=PreprocessingMetadata(
            data_state=None,
            preprocessing_applied=None,
            filter_type=None,
            artifact_methods=None,
            re_reference=None,
            notes="No additional filtering was performed on the sensor data",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP + linear SVM", "whitening + LRP + linear SVM"],
            feature_extraction=["CSP", "bandpower", "whitening"],
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]},
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross_subject", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_execution",
            imagery_tasks=["left_index_finger_press", "right_index_finger_press"],
            cue_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials={
                "total": 19164,
                "per_subject_mean": 1916,
                "left_hand": 9041,
                "right_hand": 10123,
            },
            n_blocks=146,
            trials_context=(
                "98 normal and 48 frustration blocks across the ten public "
                "subjects (14-16 blocks per subject, roughly two minutes "
                "each); 19164 accepted key presses (markers 1/2) and 1016 "
                "ignored keystrokes (markers 3/4, frustration blocks only)"
            ),
        ),
        data_processed=False,
        file_format="MAT",
        abstract=(
            "Brain-computer interfaces (BCIs) are known to suffer from "
            "spontaneous changes in the brain activity. In this dataset, "
            "twelve subjects played an affective Pacman game controlled by "
            "executed left/right index-finger key presses while their EEG and "
            "peripheral physiology were recorded; a loss-of-control "
            "manipulation (randomly ignored keystrokes and display lag) was "
            "used to induce frustration between blocks. The data support the "
            "study of motor decoding under changing mental states."
        ),
        methodology=(
            "Participants played a custom Pacman variant turning the avatar "
            "with left/right index-finger presses. ERD-based CSP features on "
            "8-30 Hz band-pass filtered EEG and movement-related potential "
            "features were classified with linear SVMs from one-second "
            "windows centered on the registered keystrokes."
        ),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="BNCI2014-007",
            interval=[-0.5, 0.5],
            paradigm="imagery",
            doi="10.1109/TNSRE.2011.2166562",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(
                f"Subject must be between 1 and 10 for BNCI2014_007. Got {subject}."
            )

        url = f"{BNCI_URL}007-2014/S{_FILE_BY_SUBJECT[subject]}.mat"
        return [dl.data_dl(url, "BNCI2014_007", path, force_update, verbose)]

    def _get_single_subject_data(self, subject):
        mat = loadmat(
            self.data_path(subject)[0], squeeze_me=False, struct_as_record=False
        )
        ch_names = [str(np.asarray(c).flatten()[0]) for c in mat["chann"].flatten()]
        if ch_names != _CH_NAMES_RECORDED:
            raise ValueError(f"Unexpected channel list in the recording: {ch_names}")

        X = mat["X"].astype(np.float64) * 1e-6  # sensor data is stored in uV
        Y = mat["Y"].flatten().astype(int)
        block = mat["I"][1].astype(int)
        condition = mat["I"][2].astype(int)  # 0: normal, 1: frustration

        if self.return_all_modalities:
            keep = [i for i, name in enumerate(ch_names) if name not in _NOISE_CH_NAMES]
            ch_types = _CH_TYPES_ALL
        else:
            keep = list(range(32))
            ch_types = _CH_TYPES_EEG_ONLY

        sessions = {"0": {}}
        n_normal = 0
        n_frustration = 0
        # Run keys must start with the recording-order index (moabb run-name
        # convention), so blocks are named "<index>normal<k>" or
        # "<index>frustration<k>".
        for order, block_id in enumerate(np.unique(block[block >= 0])):
            is_frustration = condition[block == block_id][0]
            # A block must map to exactly one experimental condition; guard
            # against partial blocks that only exist as padding.
            conds = set(condition[block == block_id].tolist())
            if conds not in ({0}, {1}):
                log.warning("Skipping block %s with mixed conditions %s", block_id, conds)
                continue
            idx = np.flatnonzero(block == block_id)
            if is_frustration:
                run_name = f"{order}frustration{n_frustration}"
                n_frustration += 1
            else:
                run_name = f"{order}normal{n_normal}"
                n_normal += 1

            data = X[np.ix_(keep, idx)]
            onset_samples, onset_values = _marker_onsets(Y[idx])
            onsets = np.asarray(onset_samples, dtype=float) / SFREQ
            descriptions = [_DESCRIPTION_BY_MARKER[v] for v in onset_values]
            raw = make_raw(
                data,
                ch_names=[ch_names[i] for i in keep],
                ch_types=ch_types,
                sfreq=SFREQ,
                verbose=None,
            )
            if len(descriptions):
                raw.set_annotations(
                    Annotations(
                        onset=onsets,
                        duration=np.zeros(len(descriptions)),
                        description=descriptions,
                        orig_time=None,
                    ),
                    verbose=None,
                )
            sessions["0"][run_name] = raw
        return sessions


def _marker_onsets(markers):
    """Return onset indices and codes for the in-game markers of a block."""
    onsets = np.flatnonzero(np.diff(markers, prepend=markers[0] + 1) != 0)
    values = markers[onsets]
    keep = np.isin(values, list(_DESCRIPTION_BY_MARKER))
    return onsets[keep], values[keep]
