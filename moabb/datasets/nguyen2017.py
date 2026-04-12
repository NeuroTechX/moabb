"""Imagined Speech EEG dataset using Riemannian Manifold features.

Nguyen, Karavas, and Artemiadis (2017), Journal of Neural Engineering.
DOI: 10.1088/1741-2552/aa8235
Data: Zenodo mirror (`10.5281/zenodo.19502794`). The original
distribution is on Dropbox (HORC lab, ASU); the Zenodo record is a
re-packaging of the 4 condition zips with normalized filenames.
"""

from pathlib import Path

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

# Condition-specific configuration. The Zenodo mirror stores each
# subject as ``sub-NN.mat`` inside the per-condition zip, so only the
# subject count is needed here — the provenance mapping from the
# authors' opaque filenames to the sub-NN convention lives in the
# README.md and Read_me.txt bundled inside each zip on Zenodo.
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
        associated_paper_doi="10.1088/1741-2552/aa8235",
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
            interval=[0, 5],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _mat_to_raw(self, fpath):
        """Load a .mat file and build a continuous Raw with stim channel.

        Uses the 'last_beep' variable (speech imagery period, 5s).
        Trials are returned in class-blocked order as stored in the
        upstream .mat file; downstream code is responsible for any
        shuffling its evaluation protocol requires.
        """
        mat = loadmat(str(fpath), squeeze_me=False)
        key = "eeg_data_wrt_task_rep_no_eog_256Hz_last_beep"
        cell = mat[key]  # shape: (n_classes, n_trials) object array

        n_classes, n_trials = cell.shape
        first_cell = cell[0, 0]
        n_ch_file = first_cell.shape[0]
        n_times = first_cell.shape[1]  # 1280 = 5s at 256Hz

        # Use first 64 channels for 80-channel subjects.
        n_ch_use = min(n_ch_file, _N_CHANNELS)

        # Collect all epochs: (n_total_trials, n_ch, n_times).
        all_data = []
        all_labels = []
        for cls_idx in range(n_classes):
            event_id = cls_idx + 1
            for trial_idx in range(n_trials):
                trial_data = cell[cls_idx, trial_idx][:n_ch_use, :n_times]
                all_data.append(trial_data)
                all_labels.append(event_id)

        data = np.array(all_data, dtype=np.float64)
        labels = np.array(all_labels, dtype=int)

        # Build channel names, marking EOG channels.
        ch_names = [f"EEG{i + 1:03d}" for i in range(n_ch_use)]
        ch_types = ["eeg"] * n_ch_use
        for idx in _EOG_INDICES:
            if idx < n_ch_use:
                ch_types[idx] = "eog"

        raw = build_raw_from_epochs(
            data,
            ch_names,
            _SFREQ,
            labels,
            montage_name="standard_1005",
            ch_types=ch_types,
        )
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
        # The Zenodo mirror stores files as sub-01.mat … sub-NN.mat inside
        # each per-condition zip; the authors' opaque basenames are
        # preserved as provenance in the bundled Read_me.txt / README.md.
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
    """Nguyen 2017 Imagined Speech - Vowels condition.

    Imagined speech of three vowels: /a/, /i/, /u/.

    Dataset from Nguyen, Karavas, and Artemiadis [1]_.

    **Dataset Description**

    Eight of the 15 subjects (S4, S5, S8, S9, S11, S12, S13, S15)
    performed imagined speech of three vowels with 100 trials per
    class. EEG was recorded at 1000 Hz with 64 channels using the
    BrainProducts ActiCHamp amplifier (10/20 placement), then
    preprocessed (8-70 Hz Butterworth bandpass, 60 Hz notch, adaptive
    EOG artifact removal) and downsampled to 256 Hz. Each trial uses
    period T=1.0 s. Analyzed segment is 5 s after the last beep
    (visual cue only, no auditory evoked potentials).

    .. note::
        Channels [1, 10, 33, 64] (1-indexed) were used for EOG
        recording and should be excluded from classification.

    .. figure:: /_static/paper_figures/Nguyen2017.png
       :alt: Nguyen2017 trial structure (Fig. 3 of the JNE paper) —
             periodic beeps mark each rhythm period T, the visual cue
             stays on screen, the subject performs speech imagery; the
             extracted signal is the segment after the last beep,
             followed by 2 s rest.
       :width: 100%

       Figure 3 of [1]_ — trial structure (period T = 1.0 s for the
       Vowels condition). Class labels: ``vowel_a``, ``vowel_i``,
       ``vowel_u``. Reproduced from the author postprint at the
       University of Delaware self-archive.

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(8),
        experiment=ExperimentMetadata(
            events={"vowel_a": 1, "vowel_i": 2, "vowel_u": 3},
            paradigm="imagery",
            n_classes=3,
            class_labels=["vowel_a", "vowel_i", "vowel_u"],
            trial_duration=5.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.0s rhythm, "
                "subject imagines speech at each beep and continues for "
                "3 more periods. Analyzed segment: after last beep (no "
                "auditory evoked potentials). ~2s rest between trials."
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
            "Imagined speech EEG dataset. Paper reports 49.0+/-2.4% "
            "mean accuracy for 3-class vowels (chance 33.3%) using "
            "Riemannian manifold features + mRVM classifier. "
            "Ethics: ASU IRB Protocols 1309009601, STUDY00001345.",
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
            classifiers=["mRVM (multi-class Relevance Vector Machines)"],
            feature_extraction=["Riemannian tangent space", "covariance matrices"],
            frequency_bands={
                "mu": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 70.0],
            },
            spatial_filters=["CSP (for channel selection)"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold", cv_folds=10, evaluation_type=["within_session"]
        ),
        data_structure=DataStructureMetadata(
            n_trials=2400,
            n_trials_per_class={"vowel_a": 800, "vowel_i": 800, "vowel_u": 800},
            trials_context=(
                "8 subjects (S4, S5, S8, S9, S11, S12, S13, S15) x 300 "
                "trials (100 per class). T=1.0 s rhythm."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "Vowels"
    _code_suffix = "V"


class Nguyen2017_S(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Short Words condition.

    Imagined speech of three short words: "in", "out", "up".

    Dataset from Nguyen, Karavas, and Artemiadis [1]_.

    **Dataset Description**

    Six of the 15 subjects (S1, S3, S5, S6, S8, S12) performed imagined
    speech of three short words with 100 trials per class. Same
    recording setup as the Vowels condition, period T=1.0 s. Paper
    reports 50.1±3.5% mean accuracy with the Riemannian approach.

    .. figure:: /_static/paper_figures/Nguyen2017.png
       :alt: Nguyen2017 trial structure (Fig. 3 of the JNE paper) —
             periodic beeps mark each rhythm period T, the visual cue
             stays on screen, the subject performs speech imagery; the
             extracted signal is the segment after the last beep,
             followed by 2 s rest.
       :width: 100%

       Figure 3 of [1]_ — trial structure (period T = 1.0 s for the
       ShortWords condition). Class labels: ``out``, ``in``, ``up``.
       Reproduced from the author postprint at the University of
       Delaware self-archive.

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(6),
        experiment=ExperimentMetadata(
            events={"out": 1, "in": 2, "up": 3},
            paradigm="imagery",
            n_classes=3,
            class_labels=["out", "in", "up"],
            trial_duration=5.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.0s rhythm. "
                "Subject imagines speech at each beep and continues for "
                "3 more periods. Analyzed: 5s after last beep. Paper "
                "reports 50.1±3.5% mean accuracy."
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
            "Imagined speech EEG dataset. Paper reports 50.1+/-3.5% "
            "mean accuracy for 3-class short words (chance 33.3%) using "
            "Riemannian manifold features + mRVM classifier. "
            "Ethics: ASU IRB Protocols 1309009601, STUDY00001345.",
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
            n_trials=1800,
            n_trials_per_class={"out": 600, "in": 600, "up": 600},
            trials_context=(
                "6 subjects (S1, S3, S5, S6, S8, S12) x 300 trials "
                "(100 per class). T=1.0 s rhythm."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "ShortWords"
    _code_suffix = "S"


class Nguyen2017_L(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Long Words condition.

    Imagined speech of two long words: "cooperate", "independent".

    Dataset from Nguyen, Karavas, and Artemiadis [1]_.

    **Dataset Description**

    Six of the 15 subjects (S2, S3, S6, S7, S9, S11) performed
    imagined speech of two long words with 100 trials per class.
    EEG recorded with BrainProducts ActiCHamp, 64 channels, 10/20
    system. Period T=1.4 s (longer than the short words / vowels
    condition to accommodate the longer pronunciation). Paper
    reports 66.2±4.8% mean accuracy.

    .. figure:: /_static/paper_figures/Nguyen2017.png
       :alt: Nguyen2017 trial structure (Fig. 3 of the JNE paper) —
             periodic beeps mark each rhythm period T, the visual cue
             stays on screen, the subject performs speech imagery; the
             extracted signal is the segment after the last beep,
             followed by 2 s rest.
       :width: 100%

       Figure 3 of [1]_ — trial structure (period T = 1.4 s for the
       LongWords condition, longer than Vowels/ShortWords to
       accommodate the longer pronunciation). Class labels:
       ``cooperate``, ``independent``. Reproduced from the author
       postprint at the University of Delaware self-archive.

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(6),
        experiment=ExperimentMetadata(
            events={"cooperate": 1, "independent": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["cooperate", "independent"],
            trial_duration=5.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.4s rhythm. "
                "Subject imagines speech at each beep and continues for "
                "3 more periods. Analyzed: 5s after last beep. Paper "
                "reports 66.2±4.8% mean accuracy."
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
            "Imagined speech EEG dataset. Paper reports 66.2+/-4.8% "
            "mean accuracy for 2-class long words (chance 50.0%) using "
            "Riemannian manifold features + mRVM classifier. "
            "Ethics: ASU IRB Protocols 1309009601, STUDY00001345.",
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
            n_trials=1200,
            n_trials_per_class={"cooperate": 600, "independent": 600},
            trials_context=(
                "6 subjects (S2, S3, S6, S7, S9, S11) x 200 trials "
                "(100 per class). T=1.4 s rhythm."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "LongWords"
    _code_suffix = "L"


class Nguyen2017_SL(_Nguyen2017Base):
    """Nguyen 2017 Imagined Speech - Short vs Long Words condition.

    Imagined speech discriminating a short word ("in") from a long
    word ("cooperate").

    Dataset from Nguyen, Karavas, and Artemiadis [1]_.

    **Dataset Description**

    Six of the 15 subjects (S1, S5, S8, S9, S10, S14) performed
    imagined speech of one short ("in") and one long ("cooperate")
    word with 100 trials per class (80 for some subjects). EEG
    recorded with BrainProducts ActiCHamp, 64 channels, 10/20
    system. Period T=1.4 s. Paper reports 73.3±8.9% (Method 1,
    spatial features only) and 80.1±8.0% (Method 2, spatial +
    wavelet features) mean accuracy.

    .. figure:: /_static/paper_figures/Nguyen2017.png
       :alt: Nguyen2017 trial structure (Fig. 3 of the JNE paper) —
             periodic beeps mark each rhythm period T, the visual cue
             stays on screen, the subject performs speech imagery; the
             extracted signal is the segment after the last beep,
             followed by 2 s rest.
       :width: 100%

       Figure 3 of [1]_ — trial structure (period T = 1.4 s for the
       ShortLongWords condition). Class labels: ``cooperate``, ``in``.
       Reproduced from the author postprint at the University of
       Delaware self-archive.

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=_NGUYEN_ACQUISITION,
        participants=_nguyen_participants(6),
        experiment=ExperimentMetadata(
            events={"cooperate": 1, "in": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["cooperate", "in"],
            trial_duration=5.0,
            study_design=(
                "Auditory beep + visual cue. 5 beeps at T=1.4s rhythm. "
                "Analyzed: 5s after last beep. Paper reports 73.3±8.9% "
                "(Method 1: spatial) to 80.1±8.0% (Method 2: spatial + "
                "wavelet) mean accuracy."
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
            "Imagined speech EEG dataset. Paper reports 73.3+/-8.9% "
            "(Method 1: spatial) to 80.1+/-8.0% (Method 2: spatial + "
            "wavelet) mean accuracy for 2-class short-vs-long words "
            "(chance 50.0%) using Riemannian manifold features + mRVM "
            "classifier. "
            "Ethics: ASU IRB Protocols 1309009601, STUDY00001345.",
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
            n_trials=1200,
            n_trials_per_class={"cooperate": 600, "in": 600},
            trials_context=(
                "6 subjects (S1, S5, S8, S9, S10, S14) x 200 trials "
                "(100 per class, 80 per class for some subjects). "
                "T=1.4 s rhythm."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    _condition = "ShortLongWords"
    _code_suffix = "SL"
