"""MI-BMPI motor-imagery brain-mobile phone interface dataset (Yilmaz 2024)."""

import h5py
import mne
import numpy as np
from scipy.io import loadmat

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


_DOI = "10.1007/s00521-024-10917-5"

# Zenodo record hosting the epoched EEGLAB .set files and label .mat files.
_ZENODO_RECORD = "13626922"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# 13 sensorimotor EEG channels in acquisition order (read from chanlocs of the
# EEGLAB .set files; identical across subjects/sessions). Canonical 10-20 case
# so the standard_1020 montage attaches to all of them.
_CH_NAMES = "C5 C3 FC3 CP3 C1 Cz FCz CPz C2 C4 FC4 CP4 C6".split()

_SFREQ = 128.0

# Each epoch spans -1.0 s to +2.5 s about motor-imagery onset (448 samples at
# 128 Hz). The first 1 s is a pre-task baseline; the motor-imagery execution
# occupies 0 to 2.5 s. The event marker is placed at imagery onset (t = 0).
_BASELINE_S = 1.0

# Integer label -> gesture class name. The label .mat files store 1 and 2.
_LABEL_TO_EVENT = {1: "tap", 2: "swipe"}


class Yilmaz2024(BaseDataset):
    """MI-BMPI motor-imagery brain-mobile phone interface dataset [1]_.

    **Dataset description**

    Eight healthy subjects imagined two mobile-phone gestures, tapping the
    screen (``tap``) and swiping down with the thumb (``swipe``), in two sessions
    recorded with an Emotiv EPOC Flex at 128 Hz (13 sensorimotor channels,
    average reference). The data are epoched (1 s pre-task baseline + 2.5 s
    imagery, 96-120 trials per session); the EEGLAB ``.set`` files carry no
    events, so labels come from the separate ``*_labels.mat`` files.

    The loader rebuilds one continuous ``Raw`` per session by concatenating the
    epochs with a 0.5 s zero gap and a stim event at each imagery onset
    (1 = ``tap``, 2 = ``swipe``); the interval ends at 2.5 - 1/128 s so the gap
    is excluded.

    References
    ----------

    .. [1] Yilmaz, C. M., Yilmaz, B. H., and Kose, C. (2024). MI-BMPI motor
           imagery brain-mobile phone dataset and performance evaluation of
           voting ensembles utilizing QPDM. Neural Computing and Applications.
           DOI: https://doi.org/10.1007/s00521-024-10917-5

    Notes
    -----

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": 13},
            montage="standard_1020",
            hardware="Emotiv EPOC Flex (Model 1.0)",
            cap_manufacturer="Emotiv",
            cap_model="EPOC Flex",
            sensor_type="saline",
            electrode_type="passive",
            reference="average",
            ground=None,
            software="Emotiv Pro 2.5.1.227",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=8, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["tap", "swipe"],
            trial_duration=2.5,
            study_design=(
                "Cue-based motor imagery of two mobile-phone gestures: tapping "
                "on the screen and swiping down with a thumb; 8 subjects across "
                "2 sessions."
            ),
            feedback_type="none",
            stimulus_type="cue",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"tap": 1, "swipe": 2},
            instructions=(
                "Imagine tapping on the mobile screen, or imagine swiping down "
                "with the thumb, following the cue."
            ),
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            description=(
                "Motor-imagery EEG from 8 healthy subjects imagining two "
                "mobile-phone gestures (screen tap and thumb swipe-down) over 2 "
                "sessions, recorded with an Emotiv EPOC Flex headset at 128 Hz."
            ),
            investigators=["Cagatay Murat Yilmaz", "Beyda H. Yilmaz", "Cemal Kose"],
            institution="Karadeniz Technical University",
            country="TR",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
            publication_year=2024,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "brain-mobile phone interface",
                "EEG",
                "Emotiv EPOC Flex",
            ],
            license="CC-BY-NC-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed", preprocessing_applied=True
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["brain-mobile phone interface"],
            environment="lab",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["tap", "swipe"],
            imagery_duration_s=2.5,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=1,
            trials_context=(
                "Two sessions per subject, one run each; 96-120 trials per "
                "session, roughly balanced between the two gesture classes. "
                "Each epoch: 1 s baseline + 2.5 s imagery."
            ),
        ),
        file_format="EEGLAB (epoched)",
        data_processed=True,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=2,
            events={"tap": 1, "swipe": 2},
            code="Yilmaz2024",
            interval=[0, 2.5 - 1 / _SFREQ],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return ``[s1.set, s1_labels.mat, s2.set, s2_labels.mat]`` local paths."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for sess in (1, 2):
            stem = f"D{subject:02d}_s{sess}"
            set_url = f"{_ZENODO_BASE}/{stem}.set"
            lab_url = f"{_ZENODO_BASE}/{stem}_labels.mat"
            set_path = dl.data_dl(set_url, self.code, path, force_update, verbose)
            lab_path = dl.data_dl(lab_url, self.code, path, force_update, verbose)
            paths.extend([set_path, lab_path])

        return paths

    def _get_single_subject_data(self, subject):
        """Return ``{session: {"0": Raw}}``, one reconstructed Raw per session."""
        files = self.data_path(subject)
        return {
            str(i): {"0": self._reconstruct_raw(files[2 * i], files[2 * i + 1])}
            for i in range(2)
        }

    @staticmethod
    def _reconstruct_raw(set_path, lab_path):
        """Reconstruct a continuous ``Raw`` from one epoched session.

        The EEGLAB ``.set`` file is MATLAB v7.3 (HDF5); its ``data`` array is
        stored transposed relative to MATLAB, i.e. shape
        ``(n_trials, n_samples, n_channels)``. The label ``.mat`` file stores a
        per-trial array of integers (1 = tap, 2 = swipe).
        """
        with h5py.File(set_path, "r") as f:
            arr = np.asarray(f["data"], dtype=np.float64)

        if arr.ndim != 3:
            raise ValueError(
                f"Expected a 3-D epoched array in {set_path}, got shape {arr.shape}"
            )

        # h5py order: (n_trials, n_samples, n_channels).
        n_trials, n_samples, n_ch = arr.shape
        if n_ch != len(_CH_NAMES):
            raise ValueError(
                f"Expected {len(_CH_NAMES)} channels in {set_path}, got {n_ch}"
            )

        labels = loadmat(str(lab_path))["labels"].ravel()
        if n_trials == 0 or n_samples != 448:
            raise ValueError("Expected nonempty 448-sample epochs")
        if len(labels) != n_trials or not np.isin(labels, [1, 2]).all():
            raise ValueError("Expected one valid class label per trial")

        # Scale from microvolts to volts (MNE convention).
        arr = arr * 1e-6

        onset_offset = int(round(_BASELINE_S * _SFREQ))  # imagery onset sample
        buffer_samples = int(round(0.5 * _SFREQ))  # zero-padded gap between epochs
        stride = n_samples + buffer_samples
        total_len = n_trials * stride

        # EEG channels + one stim channel.
        all_data = np.zeros((n_ch + 1, total_len), dtype=np.float64)
        for i in range(n_trials):
            start = i * stride
            # arr[i] is (n_samples, n_channels); transpose to (n_channels, n_samples).
            all_data[:n_ch, start : start + n_samples] = arr[i].T
            all_data[n_ch, start + onset_offset] = int(labels[i])

        ch_names = list(_CH_NAMES) + ["STI 014"]
        ch_types = ["eeg"] * n_ch + ["stim"]
        info = mne.create_info(ch_names, _SFREQ, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)
        raw.set_montage("standard_1020", on_missing="warn")

        boundaries = (
            np.concatenate(
                [
                    np.arange(1, n_trials) * stride,
                    np.arange(n_trials) * stride + n_samples,
                ]
            )
            / _SFREQ
        )
        raw.set_annotations(
            mne.Annotations(onset=boundaries, duration=0.0, description="EDGE boundary")
        )
        return raw
