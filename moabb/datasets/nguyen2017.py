"""Imagined Speech EEG dataset using Riemannian Manifold features.

Nguyen, Karavas, and Artemiadis (2017), Journal of Neural Engineering.
DOI: 10.1088/1741-2552/aa8235
Data: Zenodo mirror (`10.5281/zenodo.19502794`). The original
distribution is on Dropbox (HORC lab, ASU); the Zenodo record is a
re-packaging of the 4 condition zips with normalized filenames.
"""

from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
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
from .utils import build_raw_from_epochs, download_and_extract_subject_zip


_DOI = "10.1088/1741-2552/aa8235"
_IRB = "ASU IRB Protocols 1309009601, STUDY00001345"
_SFREQ = 256.0
_N_CHANNELS = 64

# EOG channel indices (0-indexed). Per the readme, channels [1, 10, 33, 64]
# (1-indexed) were used for EOG recording.
_EOG_INDICES = [0, 9, 32, 63]

_ZENODO_RECORD = "19502794"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

_CONDITIONS = {
    "Vowels": {
        "sign": "nguyen2017v",
        "zip_name": "Vowels.zip",
        "events": {"vowel_a": 1, "vowel_i": 2, "vowel_u": 3},
        "class_labels": ["vowel_a", "vowel_i", "vowel_u"],
        "n_classes": 3,
        "n_subjects": 8,
    },
    "ShortWords": {
        "sign": "nguyen2017s",
        "zip_name": "Short_words.zip",
        "events": {"out": 1, "in": 2, "up": 3},
        "class_labels": ["out", "in", "up"],
        "n_classes": 3,
        "n_subjects": 6,
    },
    "LongWords": {
        "sign": "nguyen2017l",
        "zip_name": "Long_words.zip",
        "events": {"cooperate": 1, "independent": 2},
        "class_labels": ["cooperate", "independent"],
        "n_classes": 2,
        "n_subjects": 6,
    },
    "ShortLongWords": {
        "sign": "nguyen2017sl",
        "zip_name": "Short_Long_words.zip",
        "events": {"cooperate": 1, "in": 2},
        "class_labels": ["cooperate", "in"],
        "n_classes": 2,
        "n_subjects": 6,
    },
}

_CH_POSITIONS = {
    "Fp1": [-2.93566145e-02, 9.03503690e-02, 5.81707230e-18],
    "Fz": [4.11329127e-18, 6.71751442e-02, 6.71751442e-02],
    "F3": [-0.05177571, 0.06393767, 0.0475],
    "F7": [-7.68566145e-02, 5.58395990e-02, 5.81707230e-18],
    "FT9": [-0.08316795, 0.02702291, -0.03711946],
    "FC5": [-0.08223207, 0.03322391, 0.03404496],
    "FC1": [-0.03398867, 0.0351963, 0.08143089],
    "C3": [-0.06717514, -0.0, 0.06717514],
    "T7": [-9.5000000e-02, -0.0000000e00, 5.8170723e-18],
    "TP9": [-0.08316795, -0.02702291, -0.03711946],
    "CP5": [-0.08279938, -0.0317837, 0.03404496],
    "CP1": [-0.03398867, -0.0351963, 0.08143089],
    "Pz": [4.11329127e-18, -6.71751442e-02, 6.71751442e-02],
    "P3": [-0.05177571, -0.06393767, 0.0475],
    "P7": [-7.68566145e-02, -5.58395990e-02, 5.81707230e-18],
    "O1": [-2.93566145e-02, -9.03503690e-02, 5.81707230e-18],
    "Oz": [5.8170723e-18, -9.5000000e-02, 5.8170723e-18],
    "O2": [2.93566145e-02, -9.03503690e-02, 5.81707230e-18],
    "P4": [0.05177571, -0.06393767, 0.0475],
    "P8": [7.68566145e-02, -5.58395990e-02, 5.81707230e-18],
    "TP10": [0.08316795, -0.02702291, -0.03711946],
    "CP6": [0.08279938, -0.0317837, 0.03404496],
    "CP2": [0.03398867, -0.0351963, 0.08143089],
    "Cz": [0.0, 0.0, 0.095],
    "C4": [0.06717514, 0.0, 0.06717514],
    "T8": [9.5000000e-02, 0.0000000e00, 5.8170723e-18],
    "FT10": [0.08316795, 0.02702291, -0.03711946],
    "FC6": [0.08279938, 0.0317837, 0.03404496],
    "FC2": [0.03398867, 0.0351963, 0.08143089],
    "F4": [0.05177571, 0.06393767, 0.0475],
    "F8": [7.68566145e-02, 5.58395990e-02, 5.81707230e-18],
    "Fp2": [2.93566145e-02, 9.03503690e-02, 5.81707230e-18],
    "AF7": [-5.58395990e-02, 7.68566145e-02, 5.81707230e-18],
    "AF3": [-0.03420902, 0.0846703, 0.02618555],
    "AFz": [5.35464328e-18, 8.74479611e-02, 3.71194572e-02],
    "F1": [-0.02685832, 0.06647668, 0.06232561],
    "F5": [-0.06891997, 0.05991122, 0.02618555],
    "FT7": [-9.03503690e-02, 2.93566145e-02, 5.81707230e-18],
    "FC3": [-0.06270797, 0.03475959, 0.06232561],
    "C1": [-0.03711946, -0.0, 0.08744796],
    "C5": [-0.08808247, -0.0, 0.03558763],
    "TP7": [-9.03503690e-02, -2.93566145e-02, 5.81707230e-18],
    "CP3": [-0.06270797, -0.03475959, 0.06232561],
    "P1": [-0.02685832, -0.06647668, 0.06232561],
    "P5": [-0.06891997, -0.05991122, 0.02618555],
    "PO7": [-5.58395990e-02, -7.68566145e-02, 5.81707230e-18],
    "PO3": [-0.03420902, -0.0846703, 0.02618555],
    "POz": [5.35464328e-18, -8.74479611e-02, 3.71194572e-02],
    "PO4": [0.03420902, -0.0846703, 0.02618555],
    "PO8": [5.58395990e-02, -7.68566145e-02, 5.81707230e-18],
    "P6": [0.06891997, -0.05991122, 0.02618555],
    "P2": [0.02685832, -0.06647668, 0.06232561],
    "CPz": [2.17911364e-18, -3.55876264e-02, 8.80824662e-02],
    "CP4": [0.06270797, -0.03475959, 0.06232561],
    "TP8": [9.03503690e-02, -2.93566145e-02, 5.81707230e-18],
    "C6": [0.08808247, 0.0, 0.03558763],
    "C2": [0.03711946, 0.0, 0.08744796],
    "FC4": [0.06270797, 0.03475959, 0.06232561],
    "FT8": [9.03503690e-02, 2.93566145e-02, 5.81707230e-18],
    "F6": [0.06891997, 0.05991122, 0.02618555],
    "AF8": [5.58395990e-02, 7.68566145e-02, 5.81707230e-18],
    "AF4": [0.03420902, 0.0846703, 0.02618555],
    "F2": [0.02685832, 0.06647668, 0.06232561],
    "Iz": [5.39349551e-18, -8.80824662e-02, -3.55876264e-02],
}

_MONTAGE = mne.channels.make_dig_montage(ch_pos=_CH_POSITIONS, coord_frame="head")

_CH_NAMES = list(_CH_POSITIONS.keys())


def _nguyen_hed(label, unit="Word"):
    """Build a HED tag string for a Nguyen 2017 imagined-speech event."""
    return (
        "(Sensory-event, Experimental-stimulus, "
        "Auditory-presentation, Visual-presentation), "
        f"(Agent-action, (Imagine, Speak, ({unit}, (Label/{label}))))"
    )


# Shared metadata sections identical across all 4 Nguyen conditions.
_NGUYEN_ACQUISITION = AcquisitionMetadata(
    sampling_rate=256.0,
    n_channels=64,
    channel_types={"eeg": 60, "eog": 4},
    montage="standard_1020",
    hardware="BrainProducts ActiCHamp",
    filters={"highpass": 8.0, "lowpass": 70.0, "notch_hz": 60.0},
    line_freq=60.0,
    sensor_type="EEG",
)

_NGUYEN_PREPROCESSING = PreprocessingMetadata(
    data_state="preprocessed",
    preprocessing_applied=True,
    preprocessing_steps=[
        "Bandpass 8-70 Hz (5th order Butterworth)",
        "60 Hz notch filter (to remove line noise)",
        "EOG artifact removal (adaptive filtering)",
        "Downsampled from 1000 Hz to 256 Hz",
    ],
    highpass_hz=8.0,
    lowpass_hz=70.0,
    notch_hz=60.0,
)

_NGUYEN_TAGS = Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"])


def _nguyen_participants(n_subjects):
    return ParticipantMetadata(
        n_subjects=n_subjects,
        health_status="healthy",
        age_min=22,
        age_max=32,
        species="human",
    )


def _nguyen_docs(keywords_suffix, description):
    return DocumentationMetadata(
        doi=_DOI,
        investigators=["Chuong H. Nguyen", "George K. Karavas", "Panagiotis Artemiadis"],
        institution="Arizona State University",
        institution_department="School for Engineering of Matter, Transport and Energy",
        institution_address="Tempe, AZ 85287, USA",
        country="US",
        publication_year=2018,
        license="other-open",
        data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
        repository="Zenodo",
        senior_author="Panagiotis Artemiadis",
        contact_info=["chuong.h.nguyen@asu.edu", "panagiotis.artemiadis@asu.edu"],
        associated_paper_doi=_DOI,
        keywords=[
            "imagined speech",
            "EEG",
            "Riemannian manifold",
            "covariance matrix",
            "relevance vector machines",
            "brain-computer interface",
            keywords_suffix,
        ],
        description=description,
    )


class _Nguyen2017Base(BaseDataset):
    """Base class for Nguyen et al. 2017 imagined speech conditions."""

    _condition = None  # Set by subclasses.
    _code_suffix = None  # Set by subclasses.

    def __init__(self, subjects=None, sessions=None):
        cfg = _CONDITIONS[self._condition]
        super().__init__(
            subjects=list(range(1, cfg["n_subjects"] + 1)),
            sessions_per_subject=1,
            events=cfg["events"],
            code=f"Nguyen2017-{self._code_suffix}",
            interval=[0, 1.5],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _mat_to_raw(self, fpath):
        """Load a .mat file and build a continuous Raw with stim channel.

        Uses the 'last_beep' variable (speech imagery period, 5s).
        Each 5s trial is split into three overlapping 2s trials:
        - Vowels/ShortWords (T=1.0s): total 4s used, split as 2s epochs with 1s overlap.
        - LongWords/ShortLongWords (T=1.4s): total 4.5s used, split as 2s epochs with 1.25s overlap.
        """
        mat = loadmat(str(fpath), squeeze_me=False)
        key = "eeg_data_wrt_task_rep_no_eog_256Hz_last_beep"
        cell = mat[key]  # shape: (n_classes, n_trials) object array

        n_classes, n_trials = cell.shape
        first_cell = cell[0, 0]
        n_ch_file = first_cell.shape[0]

        # Use first 64 channels for 80-channel subjects.
        n_ch_use = min(n_ch_file, _N_CHANNELS)

        # Splitting parameters based on literature ( Nguyen et al. 2017)
        # Epoch duration is always 2s = 512 samples at 256Hz
        epoch_len = int(2.0 * _SFREQ)
        if self._condition in ["Vowels", "ShortWords"]:
            # 4s total used, 1s overlap -> starts at 0s, 1s, 2s
            offsets = [int(0 * _SFREQ), int(1.0 * _SFREQ), int(2.0 * _SFREQ)]
        else:
            # LongWords or ShortLongWords
            offsets = [int(0 * _SFREQ), int(1.4 * _SFREQ), int(2.8 * _SFREQ)]

        # Collect all split epochs: (n_total_trials * 3, n_ch, n_times).
        all_data = []
        all_labels = []
        for cls_idx in range(n_classes):
            event_id = cls_idx + 1
            for trial_idx in range(n_trials):
                trial_data_full = cell[cls_idx, trial_idx][:n_ch_use, :]
                for offset in offsets:
                    epoch = trial_data_full[:, offset : offset + epoch_len]
                    all_data.append(epoch)
                    all_labels.append(event_id)

        data = np.array(all_data, dtype=np.float64)
        labels = np.array(all_labels, dtype=int)

        ch_types = ["eeg"] * n_ch_use
        for idx in _EOG_INDICES:
            if idx < n_ch_use:
                ch_types[idx] = "eog"

        raw = build_raw_from_epochs(
            data,
            _CH_NAMES,
            _SFREQ,
            labels,
            montage_name="standard_1005",
            ch_types=ch_types,
        )
        raw.set_montage(_MONTAGE, on_missing="ignore")
        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fpath = self.data_path(subject)
        raw = self._mat_to_raw(fpath)
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        cfg = _CONDITIONS[self._condition]
        cond_dir = (
            Path(dl.get_dataset_path(cfg["sign"], path))
            / f"MNE-{cfg['sign']}-data"
            / cfg["zip_name"].replace(".zip", "")
        )
        mat_file = cond_dir / f"sub-{subject:02d}.mat"

        if mat_file.exists() and not force_update:
            return str(mat_file)

        url = f"{_ZENODO_BASE}/{cfg['zip_name']}"
        download_and_extract_subject_zip(
            url,
            cfg["sign"],
            cond_dir,
            path=path,
            force_update=force_update,
            verbose=verbose,
        )

        if not mat_file.exists():
            raise FileNotFoundError(
                f"Expected {mat_file} after extracting {url}, but it is "
                f"missing. The Zenodo record may have been re-packaged."
            )
        return str(mat_file)


class Nguyen2017_V(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Vowels condition."""

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(8),
        experiment=ExperimentMetadata(
            events={"vowel_a": 1, "vowel_i": 2, "vowel_u": 3},
            paradigm="imagery",
            n_classes=3,
            class_labels=["vowel_a", "vowel_i", "vowel_u"],
            trial_duration=2.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.0s rhythm. "
                "Subject imagines speech at each beep and continues for "
                "3 more periods. Analyzed segment: after last beep. "
                "Each 5s trial is split into 3 overlapping epochs of 2s."
            ),
            stimulus_type="auditory + visual cue",
            stimulus_modalities=["auditory", "visual"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            hed_tags={
                "vowel_a": _nguyen_hed("a", "Phoneme"),
                "vowel_i": _nguyen_hed("i", "Phoneme"),
                "vowel_u": _nguyen_hed("u", "Phoneme"),
            },
        ),
        documentation=_nguyen_docs(
            "vowels",
            "Imagined speech EEG dataset. Paper reports 49.0+/-2.4% mean accuracy.",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=_NGUYEN_TAGS,
        preprocessing=_NGUYEN_PREPROCESSING,
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["vowel_a", "vowel_i", "vowel_u"],
            cue_duration_s=7.0,
            imagery_duration_s=5.0,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["mRVM"],
            feature_extraction=["Riemannian tangent space"],
            frequency_bands={
                "mu": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 70.0],
            },
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold", cv_folds=10, evaluation_type=["within_session"]
        ),
        data_structure=DataStructureMetadata(
            n_trials=7200,
            n_trials_per_class={"vowel_a": 2400, "vowel_i": 2400, "vowel_u": 2400},
            trials_context=(
                "8 subjects x 300 trials (100 per class), each split into 3 epochs. T=1.0 s."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "Vowels"
    _code_suffix = "V"


class Nguyen2017_S(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Short Words condition."""

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(6),
        experiment=ExperimentMetadata(
            events={"out": 1, "in": 2, "up": 3},
            paradigm="imagery",
            n_classes=3,
            class_labels=["out", "in", "up"],
            trial_duration=2.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.0s rhythm. "
                "Analyzed: 5s after last beep. Each 5s trial is split into 3 overlapping epochs of 2s."
            ),
            stimulus_type="auditory + visual cue",
            stimulus_modalities=["auditory", "visual"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            hed_tags={
                "out": _nguyen_hed("out"),
                "in": _nguyen_hed("in"),
                "up": _nguyen_hed("up"),
            },
        ),
        documentation=_nguyen_docs(
            "short words",
            "Imagined speech EEG dataset. Paper reports 50.1+/-3.5% mean accuracy.",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=_NGUYEN_TAGS,
        preprocessing=_NGUYEN_PREPROCESSING,
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["out", "in", "up"],
            cue_duration_s=7.0,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=5400,
            n_trials_per_class={"out": 1800, "in": 1800, "up": 1800},
            trials_context=(
                "6 subjects x 300 trials, each split into 3 epochs. T=1.0 s."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "ShortWords"
    _code_suffix = "S"


class Nguyen2017_L(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Long Words condition."""

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(6),
        experiment=ExperimentMetadata(
            events={"cooperate": 1, "independent": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["cooperate", "independent"],
            trial_duration=2.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.4s rhythm. "
                "Analyzed: 5s after last beep. Each 5s trial is split into 3 overlapping epochs of 2s."
            ),
            stimulus_type="auditory + visual cue",
            stimulus_modalities=["auditory", "visual"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            hed_tags={
                "cooperate": _nguyen_hed("cooperate"),
                "independent": _nguyen_hed("independent"),
            },
        ),
        documentation=_nguyen_docs(
            "long words",
            "Imagined speech EEG dataset. Paper reports 66.2+/-4.8% mean accuracy.",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=_NGUYEN_TAGS,
        preprocessing=_NGUYEN_PREPROCESSING,
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["cooperate", "independent"],
            cue_duration_s=9.8,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=3600,
            n_trials_per_class={"cooperate": 1800, "independent": 1800},
            trials_context=(
                "6 subjects x 200 trials, each split into 3 epochs. T=1.4 s."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "LongWords"
    _code_suffix = "L"


class Nguyen2017_SL(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Short vs Long Words condition."""

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(6),
        experiment=ExperimentMetadata(
            events={"cooperate": 1, "in": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["cooperate", "in"],
            trial_duration=2.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.4s rhythm. "
                "Analyzed: 5s after last beep. Each 5s trial is split into 3 overlapping epochs of 2s."
            ),
            stimulus_type="auditory + visual cue",
            stimulus_modalities=["auditory", "visual"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            hed_tags={"cooperate": _nguyen_hed("cooperate"), "in": _nguyen_hed("in")},
        ),
        documentation=_nguyen_docs(
            "short vs long words",
            "Imagined speech EEG dataset. Paper reports 73.3+/-8.9% mean accuracy.",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=_NGUYEN_TAGS,
        preprocessing=_NGUYEN_PREPROCESSING,
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["cooperate", "in"],
            cue_duration_s=9.8,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=3600,
            n_trials_per_class={"cooperate": 1800, "in": 1800},
            trials_context=(
                "6 subjects x 200 trials, each split into 3 epochs. T=1.4 s."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "ShortLongWords"
    _code_suffix = "SL"
