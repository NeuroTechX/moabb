"""Imagined Speech EEG dataset using Riemannian Manifold features.

Nguyen, Karavas, and Artemiadis (2017), Journal of Neural Engineering.
DOI: 10.1088/1741-2552/aa8235
Data: https://www.dropbox.com/scl/fi/20j120qae7c2rlmr5lfwr/dataset.zip
"""

import logging
import zipfile
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
from .utils import build_raw_from_epochs


log = logging.getLogger(__name__)

_DOI = "10.1088/1741-2552/aa8235"
_IRB = "ASU IRB Protocols 1309009601, STUDY00001345"
_SFREQ = 256.0
_N_CHANNELS = 64

# EOG channel indices (0-indexed). Per the readme, channels [1, 10, 33, 64]
# (1-indexed) were used for EOG recording.
_EOG_INDICES = [0, 9, 32, 63]

_DROPBOX_URL = (
    "https://www.dropbox.com/scl/fi/20j120qae7c2rlmr5lfwr/dataset.zip"
    "?rlkey=0xjdairhprrakmw27d2fnesj7&dl=1"
)

# Condition-specific configuration.
_CONDITIONS = {
    "Vowels": {
        "sign": "nguyen2017v",
        "zip_name": "Vowels.zip",
        "events": {"vowel_a": 1, "vowel_i": 2, "vowel_u": 3},
        "class_labels": ["vowel_a", "vowel_i", "vowel_u"],
        "n_classes": 3,
        # Subject file basenames (sorted), excluding time_correlation file.
        "subjects": [
            "sub_4b_ch80_v_eog_removed_256Hz",
            "sub_5b_ch80_v_eog_removed_256Hz",
            "sub_8e_ch64_v_eog_removed_256Hz",
            "sub_9_ch64_v_eog_removed_256Hz",
            "sub_11_ch64_v_eog_removed_256Hz",
            "sub_12_ch64_v_eog_removed_256Hz",
            "sub_13_ch64_v_eog_removed_256Hz",
            "sub_15_ch64_v_eog_removed_256Hz",
        ],
    },
    "ShortWords": {
        "sign": "nguyen2017s",
        "zip_name": "Short_words.zip",
        "events": {"out": 1, "in": 2, "up": 3},
        "class_labels": ["out", "in", "up"],
        "n_classes": 3,
        "subjects": [
            "sub_1_ch64_s_eog_removed_256Hz",
            "sub_3_ch64_s_eog_removed_256Hz",
            "sub_5_ch64_s_eog_removed_256Hz",
            "sub_6b_ch80_s_eog_removed_256Hz",
            "sub_8g_ch64_s_eog_removed_256Hz",
            "sub_12b_ch64_s_eog_removed_256Hz",
        ],
    },
    "LongWords": {
        "sign": "nguyen2017l",
        "zip_name": "Long_words.zip",
        "events": {"cooperate": 1, "independent": 2},
        "class_labels": ["cooperate", "independent"],
        "n_classes": 2,
        "subjects": [
            "sub_2b_ch64_l_eog_removed_256Hz",
            "sub_3b_ch80_l_eog_removed_256Hz",
            "sub_6_ch64_l_eog_removed_256Hz",
            "sub_7_ch64_l_eog_removed_256Hz",
            "sub_9c_ch64_l_eog_removed_256Hz",
            "sub_11b_ch64_l_eog_removed_256Hz",
        ],
    },
    "ShortLongWords": {
        "sign": "nguyen2017sl",
        "zip_name": "Short_Long_words.zip",
        "events": {"cooperate": 1, "in": 2},
        "class_labels": ["cooperate", "in"],
        "n_classes": 2,
        # Exclude the _8s variant file for sub_14.
        "subjects": [
            "sub_1c_ch64_sl_eog_removed_256Hz_bw20",
            "sub_5c_ch64_sl_eog_removed_256Hz_bw20",
            "sub_8d_ch64_sl_eog_removed_256Hz",
            "sub_9b_ch64_sl_eog_removed_256Hz_bw20",
            "sub_10_ch64_sl_eog_removed_256Hz_bw20",
            "sub_14_ch64_sl_eog_removed_256Hz_bw20",
        ],
    },
}


class _Nguyen2017Base(BaseDataset):
    """Base class for Nguyen et al. 2017 imagined speech conditions."""

    _condition = None  # Set by subclasses.
    _code_suffix = None  # Set by subclasses.

    def __init__(self, subjects=None, sessions=None):
        cfg = _CONDITIONS[self._condition]
        n_subj = len(cfg["subjects"])
        super().__init__(
            subjects=list(range(1, n_subj + 1)),
            sessions_per_subject=1,
            events=cfg["events"],
            code=f"Nguyen2017-{self._code_suffix}",
            interval=[0, 5],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    @property
    def _cfg(self):
        return _CONDITIONS[self._condition]

    def _mat_to_raw(self, fpath):
        """Load a .mat file and build a continuous Raw with stim channel.

        Uses the 'last_beep' variable (speech imagery period, 5s).
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

        # Shuffle trials to avoid class-blocked ordering, which can
        # leak temporal autocorrelation into cross-validation folds.
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(labels))
        data = data[perm]
        labels = labels[perm]

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

    def _subject_dir(self):
        path = dl.get_dataset_path(self._cfg["sign"], None)
        return Path(path) / f"MNE-{self._cfg['sign']}-data"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        cfg = self._cfg
        base = self._subject_dir()
        base.mkdir(parents=True, exist_ok=True)

        # Map integer subject to filename.
        sub_basename = cfg["subjects"][subject - 1]
        mat_file = base / f"{sub_basename}.mat"

        if mat_file.exists() and not force_update:
            return str(mat_file)

        # Check in the manually downloaded location.
        alt_dir = Path.home() / "mne_data" / "bci_speech_imagery"
        zip_name = cfg["zip_name"]
        folder_name = zip_name.replace(".zip", "")

        # Try extracting from existing zip.
        alt_zip = alt_dir / zip_name
        if alt_zip.exists():
            with zipfile.ZipFile(str(alt_zip)) as zf:
                target = f"{folder_name}/{sub_basename}.mat"
                try:
                    info = zf.getinfo(target)
                    with zf.open(info) as src:
                        mat_file.write_bytes(src.read())
                    return str(mat_file)
                except KeyError:
                    # Try listing all mat files to find the right one.
                    for name in zf.namelist():
                        if sub_basename in name and name.endswith(".mat"):
                            with zf.open(name) as src:
                                mat_file.write_bytes(src.read())
                            return str(mat_file)

        # Attempt to download the condition zip from Dropbox.
        log.info("Downloading %s from Dropbox...", zip_name)
        try:
            downloaded = dl.data_dl(
                _DROPBOX_URL,
                "nguyen2017",
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
            downloaded = Path(downloaded)
            # The outer zip contains condition zips.
            if downloaded.suffix == ".zip" and downloaded.exists():
                from .utils import safe_extract_zip

                with zipfile.ZipFile(str(downloaded)) as outer_zf:
                    safe_extract_zip(outer_zf, alt_dir)
                # Retry extraction from the condition zip.
                alt_zip = alt_dir / zip_name
                if alt_zip.exists():
                    with zipfile.ZipFile(str(alt_zip)) as zf:
                        for name in zf.namelist():
                            if sub_basename in name and name.endswith(".mat"):
                                with zf.open(name) as src:
                                    mat_file.write_bytes(src.read())
                                return str(mat_file)
        except Exception:
            log.warning("Dropbox download failed.")

        if not mat_file.exists():
            raise FileNotFoundError(
                f"Could not find {mat_file}. Download the dataset zip from "
                f"Dropbox and place condition zips in {alt_dir}/"
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

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
            montage="standard_1020",
            hardware="BrainProducts ActiCHamp",
            filters={"highpass": 8.0, "lowpass": 70.0, "notch_hz": 60.0},
            line_freq=60.0,
            sensor_type="EEG",
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="healthy",
            gender={"male": 11, "female": 4},
            age_min=22,
            age_max=32,
            handedness="right (except S13)",
            species="human",
        ),
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
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Chuong H. Nguyen",
                "George K. Karavas",
                "Panagiotis Artemiadis",
            ],
            institution="Arizona State University",
            institution_department=(
                "School for Engineering of Matter, Transport and Energy"
            ),
            institution_address="Tempe, AZ 85287, USA",
            country="US",
            publication_year=2018,
            license="Unspecified (research use; cite paper)",
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
                "vowels",
            ],
            description=(
                "Imagined speech EEG dataset. Paper reports 49.0±2.4% "
                "mean accuracy for 3-class vowels (chance 33.3%) using "
                "Riemannian manifold features + mRVM classifier. "
                "Ethics: ASU IRB Protocols 1309009601, STUDY00001345."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
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
        ),
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

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
            montage="standard_1020",
            hardware="BrainProducts ActiCHamp",
            filters={"highpass": 8.0, "lowpass": 70.0, "notch_hz": 60.0},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=6,
            health_status="healthy",
            gender={"male": 11, "female": 4},
            age_min=22,
            age_max=32,
            handedness="right (except S13)",
            species="human",
        ),
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
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Chuong H. Nguyen",
                "George K. Karavas",
                "Panagiotis Artemiadis",
            ],
            institution="Arizona State University",
            institution_department=(
                "School for Engineering of Matter, Transport and Energy"
            ),
            country="US",
            publication_year=2018,
            license="Unspecified (research use; cite paper)",
            contact_info=["chuong.h.nguyen@asu.edu"],
            associated_paper_doi="10.1088/1741-2552/aa8235",
            keywords=["imagined speech", "EEG", "Riemannian manifold", "short words"],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "Bandpass 8-70 Hz (5th order Butterworth)",
                "60 Hz notch filter (3 Hz bandwidth)",
                "EOG artifact removal (adaptive filtering)",
                "Downsampled from 1000 Hz to 256 Hz",
            ],
            highpass_hz=8.0,
            lowpass_hz=70.0,
            notch_hz=60.0,
        ),
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

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
            montage="standard_1020",
            hardware="BrainProducts ActiCHamp",
            filters={"highpass": 8.0, "lowpass": 70.0, "notch_hz": 60.0},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=6,
            health_status="healthy",
            gender={"male": 11, "female": 4},
            age_min=22,
            age_max=32,
            handedness="right (except S13)",
            species="human",
        ),
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
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Chuong H. Nguyen",
                "George K. Karavas",
                "Panagiotis Artemiadis",
            ],
            institution="Arizona State University",
            institution_department=(
                "School for Engineering of Matter, Transport and Energy"
            ),
            country="US",
            publication_year=2018,
            license="Unspecified (research use; cite paper)",
            contact_info=["chuong.h.nguyen@asu.edu"],
            associated_paper_doi="10.1088/1741-2552/aa8235",
            keywords=["imagined speech", "EEG", "Riemannian manifold", "long words"],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "Bandpass 8-70 Hz (5th order Butterworth)",
                "60 Hz notch filter (3 Hz bandwidth)",
                "EOG artifact removal (adaptive filtering)",
                "Downsampled from 1000 Hz to 256 Hz",
            ],
            highpass_hz=8.0,
            lowpass_hz=70.0,
            notch_hz=60.0,
        ),
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

    References
    ----------
    .. [1] Nguyen, C. H., Karavas, G. K., & Artemiadis, P. (2017).
           Inferring imagined speech using EEG signals: a new approach
           using Riemannian Manifold features. Journal of Neural
           Engineering, 15(1), 016002.
           https://doi.org/10.1088/1741-2552/aa8235
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
            montage="standard_1020",
            hardware="BrainProducts ActiCHamp",
            filters={"highpass": 8.0, "lowpass": 70.0, "notch_hz": 60.0},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=6,
            health_status="healthy",
            gender={"male": 11, "female": 4},
            age_min=22,
            age_max=32,
            handedness="right (except S13)",
            species="human",
        ),
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
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Chuong H. Nguyen",
                "George K. Karavas",
                "Panagiotis Artemiadis",
            ],
            institution="Arizona State University",
            institution_department=(
                "School for Engineering of Matter, Transport and Energy"
            ),
            country="US",
            publication_year=2018,
            license="Unspecified (research use; cite paper)",
            contact_info=["chuong.h.nguyen@asu.edu"],
            associated_paper_doi="10.1088/1741-2552/aa8235",
            keywords=[
                "imagined speech",
                "EEG",
                "Riemannian manifold",
                "short vs long words",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "Bandpass 8-70 Hz (5th order Butterworth)",
                "60 Hz notch filter (3 Hz bandwidth)",
                "EOG artifact removal (adaptive filtering)",
                "Downsampled from 1000 Hz to 256 Hz",
            ],
            highpass_hz=8.0,
            lowpass_hz=70.0,
            notch_hz=60.0,
        ),
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
