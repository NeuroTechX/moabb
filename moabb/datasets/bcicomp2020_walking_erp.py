"""BCI Competition 2020 Track 5 — ERP detection during walking.

2020 International BCI Competition, Track 5. Ambulatory P300 oddball
paradigm recorded simultaneously with scalp-EEG, ear-EEG, EOG, and
forehead IMU while subjects walked on a treadmill at 1.6 m/s. Goal:
solve the ERP classification problem under walking-induced signal
distortion.

DOI: 10.3389/fnhum.2022.898300
Data: OSF https://osf.io/pq7vb/
"""

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


_SIGN = "BCIComp2020WalkingERP"
_SFREQ = 100.0
# epo.t in the source .mat files spans -190 ms to +800 ms at 10 ms
# spacing, so the actual stimulus onset is at sample index 19 of
# each 100-sample trial (indices 0-18 are the pre-stim baseline).
_STIM_ONSET_SAMPLE = 19

# Channel layout per Data Description PDF:
#   1-32:  scalp EEG
#   33-36: EOG
#   37-50: ear-EEG (L1..L10 left, R1..R8 right, with gaps)
#   51-56: IMU (accelerometer XYZ + gyroscope XYZ)
# fmt: off
_SCALP_CHANNELS = [
    "Fp1", "Fp2", "AFz", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
_EOG_CHANNELS = ["HEOGL", "HEOGR", "VEOGU", "VEOGL"]
_EAR_CHANNELS = [
    "L1", "L2", "L4", "L5", "L6", "L7", "L9", "L10",
    "R1", "R2", "R4", "R5", "R7", "R8",
]
_IMU_CHANNELS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
# fmt: on

_CH_NAMES = _SCALP_CHANNELS + _EOG_CHANNELS + _EAR_CHANNELS + _IMU_CHANNELS
_CH_TYPES = (
    ["eeg"] * len(_SCALP_CHANNELS)
    + ["eog"] * len(_EOG_CHANNELS)
    + ["eeg"] * len(_EAR_CHANNELS)  # ear-EEG shares units with scalp EEG
    + ["misc"] * len(_IMU_CHANNELS)  # IMU stays in source units (not V)
)

_EVENTS = {"NonTarget": 1, "Target": 2}

# Stable OSF file guids for each (split, subject) pair.
# fmt: off
_OSF_URLS: dict[tuple[str, int], str] = {
    ("training", 1): "https://osf.io/download/pmrh6/",
    ("training", 2): "https://osf.io/download/buk9z/",
    ("training", 3): "https://osf.io/download/59a83/",
    ("training", 4): "https://osf.io/download/5a9x4/",
    ("training", 5): "https://osf.io/download/9ctqg/",
    ("training", 6): "https://osf.io/download/v27j9/",
    ("training", 7): "https://osf.io/download/xyq7a/",
    ("training", 8): "https://osf.io/download/3sf6g/",
    ("training", 9): "https://osf.io/download/y2nsk/",
    ("training", 10): "https://osf.io/download/wmbde/",
    ("training", 11): "https://osf.io/download/wumb8/",
    ("training", 12): "https://osf.io/download/e7nvw/",
    ("training", 13): "https://osf.io/download/2r6mf/",
    ("training", 14): "https://osf.io/download/apqg6/",
    ("training", 15): "https://osf.io/download/9yzks/",
    ("validation", 1): "https://osf.io/download/2ga9e/",
    ("validation", 2): "https://osf.io/download/9hcuw/",
    ("validation", 3): "https://osf.io/download/xvs7k/",
    ("validation", 4): "https://osf.io/download/tj5du/",
    ("validation", 5): "https://osf.io/download/6pbhk/",
    ("validation", 6): "https://osf.io/download/pyu28/",
    ("validation", 7): "https://osf.io/download/w5vhu/",
    ("validation", 8): "https://osf.io/download/5ar76/",
    ("validation", 9): "https://osf.io/download/jr72s/",
    ("validation", 10): "https://osf.io/download/db58q/",
    ("validation", 11): "https://osf.io/download/a7jhc/",
    ("validation", 12): "https://osf.io/download/2r96t/",
    ("validation", 13): "https://osf.io/download/39rcf/",
    ("validation", 14): "https://osf.io/download/y7nxp/",
    ("validation", 15): "https://osf.io/download/kx739/",
    ("test", 1): "https://osf.io/download/gqa5b/",
    ("test", 2): "https://osf.io/download/s4z3f/",
    ("test", 3): "https://osf.io/download/2bm6u/",
    ("test", 4): "https://osf.io/download/cdg5q/",
    ("test", 5): "https://osf.io/download/mys37/",
    ("test", 6): "https://osf.io/download/s7jbc/",
    ("test", 7): "https://osf.io/download/c38q9/",
    ("test", 8): "https://osf.io/download/gs6pu/",
    ("test", 9): "https://osf.io/download/r9mfn/",
    ("test", 10): "https://osf.io/download/f6wc9/",
    ("test", 11): "https://osf.io/download/akfbq/",
    ("test", 12): "https://osf.io/download/wtnhu/",
    ("test", 13): "https://osf.io/download/7b6qy/",
    ("test", 14): "https://osf.io/download/xw2zp/",
    ("test", 15): "https://osf.io/download/yde4g/",
}
# fmt: on

# Run layout: one MOABB session per subject (all trials from the same
# walking recording), three runs matching the organizer-defined
# temporal split: first 180 trials = training, next 60 = validation,
# final 60 = test.
_SPLITS: list[tuple[str, str, str]] = [
    ("training", "epo_tr", "0"),
    ("validation", "epo_val", "1"),
    ("test", "epo_te", "2"),
]

# Test-run labels, extracted from the OSF answer sheet
# Track5_Answer_Sheet_Test.xlsx. Values are 0-indexed (0=NonTarget,
# 1=Target), matching the raw xlsx cells; ``_load_epoch_mat`` maps
# them to the canonical event codes ``{NonTarget: 1, Target: 2}``.
#
# Several subjects share identical trial orders in the organizer
# answer sheet (e.g. S4-S8, S12-S15); this is a property of the
# source file, not a parsing error.
# fmt: off
_TEST_LABELS_RUN2: dict[int, tuple[int, ...]] = {
    1: (1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    2: (0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0),
    3: (0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0),
    4: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    6: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    8: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    9: (1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    10: (0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0),
    11: (1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0),
    12: (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0),
    13: (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0),
    14: (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0),
    15: (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0),
}
# fmt: on


class BCIComp2020WalkingERP(BaseDataset):
    """BCI Competition 2020 Track 5 — ERP during walking (scalp + ear-EEG + IMU).

    Dataset from the 2020 International BCI Competition [1]_.

    **Dataset Description**

    Fifteen subjects (S1-S15, aged 19-32, 11 male + 4 female) performed
    an auditory-style visual oddball ERP paradigm while walking on a
    treadmill at 1.6 m/s. On every trial a character stimulus was
    presented for 500 ms (target ``OOO`` vs non-target ``XXX``,
    target ratio 0.2), followed by a 500-1500 ms jittered rest. Each
    subject completed 300 trials in a single recording; the organizers
    split them temporally into 180 training / 60 validation / 60 test
    trials, exposed here as three runs inside a single MOABB session.

    Data is released on OSF at 100 Hz, epoched from -190 to +800 ms
    around stimulus onset (100 samples per trial, 10 ms spacing —
    the data description PDF rounds this to "-200 to 800 ms", but
    ``epo.t`` in the actual .mat files starts at -190 ms). The
    recording comprises 32 scalp EEG channels, 14 ear-EEG electrodes,
    4 EOG channels, and a 6-axis IMU on the forehead (accelerometer
    XYZ + gyroscope XYZ).

    **Channel handling**

    - 32 scalp channels are typed ``eeg`` with ``standard_1005`` montage.
    - 4 EOG channels (HEOGL/HEOGR/VEOGU/VEOGL) are typed ``eog``.
    - 14 ear-EEG channels (L1-L10, R1-R8 with gaps) are typed ``eeg``
      but have no standard-montage positions; ``set_montage`` with
      ``on_missing="ignore"`` leaves their coords as ``NaN``.
    - 6 IMU channels are typed ``misc`` so classification paradigms
      that select ``eeg`` channels ignore them automatically. They
      remain in the Raw for users who want to use them for artifact
      rejection during walking.

    Test-run labels are published as a separate answer-sheet XLSX on
    OSF rather than stored inside the test .mat files. They are
    embedded in this module as :data:`_TEST_LABELS_RUN2` so the
    loader can return labelled data for all three runs without a
    second download.

    References
    ----------
    .. [1] Jeong, J.-H. et al. (2022). 2020 International brain-computer
           interface competition: A review. Frontiers in Human Neuroscience,
           16, 898300. https://doi.org/10.3389/fnhum.2022.898300
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=len(_CH_NAMES),
            channel_types={
                "eeg": len(_SCALP_CHANNELS) + len(_EAR_CHANNELS),
                "eog": len(_EOG_CHANNELS),
                "misc": len(_IMU_CHANNELS),
            },
            montage="standard_1005",
            # Amplifier model is not stated in the Track 5 data
            # description PDF and not clearly documented in the
            # Frontiers review paper either; left unset rather than
            # guessed.
            sensors=list(_CH_NAMES),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"female": 4, "male": 11},
            age_min=19,
            age_max=32,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=_EVENTS,
            paradigm="p300",
            n_classes=len(_EVENTS),
            class_labels=list(_EVENTS.keys()),
            trial_duration=1.0,
            study_design=(
                "Visual oddball P300 (target 'OOO' vs non-target 'XXX', "
                "target ratio 0.2) performed while walking on a treadmill "
                "at 1.6 m/s to pose an ambulatory BCI problem."
            ),
            stimulus_type="visual character (OOO target, XXX non-target)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2022.898300",
            investigators=[
                "Ji-Hoon Jeong",
                "Jeong-Hyun Cho",
                "Young-Eun Lee",
                "Seo-Hyun Lee",
                "Gi-Hwan Shin",
                "Young-Seok Kweon",
                "Jose del R. Millan",
                "Klaus-Robert Muller",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            country="KR",
            publication_year=2022,
            license="CC-BY-4.0",
            data_url="https://osf.io/pq7vb/",
            repository="OSF",
            contact_info=["bcicompetition2020@gmail.com"],
            associated_paper_doi="10.3389/fnhum.2022.898300",
            keywords=[
                "P300",
                "ERP",
                "ambulatory BCI",
                "walking",
                "ear-EEG",
                "IMU",
                "BCI competition",
            ],
            description=(
                "BCI Competition 2020 Track 5: P300 oddball ERP "
                "detection during walking at 1.6 m/s with simultaneous "
                "scalp-EEG, ear-EEG, EOG and forehead IMU recording."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=3,
        tags=Tags(
            pathology=["Healthy"],
            modality=["P300"],
            type=["Research", "Competition", "Ambulatory"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "100 Hz sampling rate",
                "epoched from -190 ms to +800 ms around stimulus",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(detected_paradigm="p300"),
        data_structure=DataStructureMetadata(
            n_trials=300,
            trials_context=(
                "15 subjects * 300 trials/subject temporally split "
                "into 180 training / 60 validation / 60 test."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events=_EVENTS,
            code="BCIComp2020WalkingERP",
            interval=[-0.19, 0.8],
            paradigm="p300",
            doi="10.3389/fnhum.2022.898300",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    @staticmethod
    def _load_epoch_mat(fpath, epo_key, split, subject):
        """Load a Track 5 .mat file and return (data, labels, ch_names).

        Both splits produce labels encoded as
        ``{NonTarget: 1, Target: 2}`` regardless of the source
        convention. In the training/validation ``epo`` struct, row 0
        of ``epo.y`` is target and row 1 is non-target (see
        ``epo.className``); the answer sheet used for test labels
        ``(_TEST_LABELS_RUN2)`` uses 0 for non-target and 1 for target.
        Both are mapped to the dataset's canonical event codes via
        explicit string lookup, not positional arithmetic.
        """
        mat = loadmat(fpath, squeeze_me=False, variable_names=[epo_key])
        epo = mat[epo_key]

        # x is (n_times, n_channels, n_trials); transpose to
        # (n_trials, n_channels, n_samples).
        data = np.transpose(epo["x"][0, 0], (2, 1, 0))

        if split == "test":
            # Answer sheet: 0 = non-target, 1 = target
            target_mask = np.array(_TEST_LABELS_RUN2[subject], dtype=bool)
        else:
            # Find the target row by className, then read the one-hot
            # indicator for that row so the mapping is robust to any
            # future change in row order.
            class_names = [str(cn[0]).lower() for cn in epo["className"][0, 0][0]]
            target_row = next(
                i
                for i, name in enumerate(class_names)
                if "target" in name and "non" not in name
            )
            target_mask = epo["y"][0, 0][target_row].astype(bool)

        labels = np.where(target_mask, 2, 1)  # Target=2, NonTarget=1

        ch_names = [str(c[0]) for c in epo["clab"][0, 0][0]]
        return data, labels, ch_names

    def _download_all_splits(self, subject, path, force_update, verbose):
        """Download every split file for ``subject`` and return the paths.

        Kept separate from :meth:`data_path` so the per-subject hot path
        in :meth:`_get_single_subject_data` downloads each file once,
        instead of re-running the loop per split.
        """
        return {
            split_name: dl.data_dl(
                _OSF_URLS[(split_name, subject)],
                _SIGN,
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
            for split_name, _, _ in _SPLITS
        }

    def _get_single_subject_data(self, subject):
        """Return data for a single subject (1 session with 3 runs).

        ``epo.t`` in the source .mat files spans -190 ms to +800 ms at
        10 ms (100 Hz) spacing, so the actual stimulus onset sits at
        sample index 19 of each trial (not sample 0). We pass
        ``onset_sample=19`` to ``build_raw_from_epochs`` so the stim
        marker lands on the real stimulus, not on the start of the
        pre-stim baseline window. Without this offset the paradigm
        pipeline would treat the 190 ms baseline as "post-stimulus"
        data and replace the first 19 samples of the epoch with
        leading-buffer zeros.
        """
        paths = self._download_all_splits(
            subject, path=None, force_update=False, verbose=None
        )
        runs = {}
        for split, epo_key, run_key in _SPLITS:
            data, labels, ch_names = self._load_epoch_mat(
                paths[split], epo_key, split, subject
            )
            runs[run_key] = build_raw_from_epochs(
                data,
                ch_names,
                _SFREQ,
                labels,
                montage_name="standard_1005",
                ch_types=_CH_TYPES,
                onset_sample=_STIM_ONSET_SAMPLE,
            )
        return {"0": runs}

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        *,
        split=None,
    ):
        """Return local paths for a subject's split files.

        Downloads training + validation + test files for ``subject``
        via :func:`moabb.datasets.download.data_dl`. Returns the path
        for the requested ``split`` (defaults to ``"training"``).
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")

        paths = self._download_all_splits(subject, path, force_update, verbose)
        return paths[split] if split else paths["training"]
