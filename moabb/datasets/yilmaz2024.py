"""MI-BMPI motor-imagery brain-mobile phone interface dataset (Yilmaz 2024).

Yilmaz, C. M., Yilmaz, B. H., and Kose, C. (2024). "MI-BMPI motor imagery
brain-mobile phone dataset and performance evaluation of voting ensembles
utilizing QPDM." Neural Computing and Applications.
Paper DOI: 10.1007/s00521-024-10917-5
Preprint (Research Square): 10.21203/rs.3.rs-4268007
Data: https://zenodo.org/records/13626922 and
https://doi.org/10.6084/m9.figshare.26893396
"""

import logging

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


log = logging.getLogger(__name__)

_DOI = "10.1007/s00521-024-10917-5"
_SIGN = "Yilmaz2024"

# Zenodo record hosting the epoched EEGLAB .set files and label .mat files.
_ZENODO_RECORD = "13626922"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# 13 sensorimotor EEG channels in acquisition order (read from chanlocs of the
# EEGLAB .set files; identical across subjects/sessions). Canonical 10-20 case
# so the standard_1020 montage attaches to all of them.
_CH_NAMES = [
    "C5",
    "C3",
    "FC3",
    "CP3",
    "C1",
    "Cz",
    "FCz",
    "CPz",
    "C2",
    "C4",
    "FC4",
    "CP4",
    "C6",
]

_SFREQ = 128.0

# Each epoch spans -1.0 s to +2.5 s about motor-imagery onset (448 samples at
# 128 Hz). The first 1 s is a pre-task baseline; the motor-imagery execution
# occupies 0 to 2.5 s. The event marker is placed at imagery onset (t = 0).
_BASELINE_S = 1.0

# Integer label -> gesture class name. The label .mat files store 1 and 2.
_LABEL_TO_EVENT = {1: "tap", 2: "swipe"}


class Yilmaz2024(BaseDataset):
    """MI-BMPI motor-imagery brain-mobile phone interface dataset [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Yilmaz2024       8       13           2              ~50-60           2.5s           128 Hz            2
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Eight healthy subjects performed two motor-imagery gestures designed for a
    brain-mobile phone interface (BMPI): (i) imagining tapping on the screen of
    a mobile device (class ``tap``) and (ii) imagining swiping down with a thumb
    on the screen (class ``swipe``). Each subject completed two recording
    sessions.

    EEG was recorded with a consumer-market Emotiv EPOC Flex (Model 1.0) headset
    with saline-based sensors and Emotiv Pro (2.5.1.227) software, at 128 Hz. The
    published data are restricted to 13 sensorimotor channels (C5, C3, FC3, CP3,
    C1, Cz, FCz, CPz, C2, C4, FC4, CP4, C6) and are average-referenced.

    The recordings are provided epoched. Each epoch is 3.5 s long: 1 s recorded
    before the motor-imagery task starts (a pre-task baseline) followed by 2.5 s
    recorded during motor-imagery execution. Per-session trial counts range from
    96 to 120, roughly balanced between the two classes. Trial labels are stored
    in separate ``*_labels.mat`` files (a per-trial array of 1s and 2s); the
    EEGLAB ``.set`` files carry no event structure, so the labels are taken from
    the label files.

    This loader reconstructs a continuous ``Raw`` per session by concatenating
    the epochs with a short zero-padded gap, inserting a stim-channel event at
    each imagery onset (value 1 = ``tap``, 2 = ``swipe``). The exposed interval
    spans the 2.5 s imagery period following onset.

    References
    ----------

    .. [1] Yilmaz, C. M., Yilmaz, B. H., and Kose, C. (2024). MI-BMPI motor
           imagery brain-mobile phone dataset and performance evaluation of
           voting ensembles utilizing QPDM. Neural Computing and Applications.
           DOI: https://doi.org/10.1007/s00521-024-10917-5

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=13,
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
            interval=[0, 2.5],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the subject's two sessions and return the local file paths.

        Parameters
        ----------
        subject : int
            Subject number (1-8).
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
            Paths ``[s1.set, s1_labels.mat, s2.set, s2_labels.mat]``.
        """
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
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-8).

        Returns
        -------
        dict
            ``{session_str: {"0": Raw}}`` with one reconstructed continuous
            ``Raw`` per session.
        """
        files = self.data_path(subject)

        sessions = {}
        for sess_idx in range(2):
            set_path = files[2 * sess_idx]
            lab_path = files[2 * sess_idx + 1]
            raw = self._reconstruct_raw(set_path, lab_path)
            sessions[str(sess_idx)] = {"0": raw}

        return sessions

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

        labels = loadmat(str(lab_path))["labels"].ravel().astype(int)
        if len(labels) != n_trials:
            n_trials = min(n_trials, len(labels))
            arr = arr[:n_trials]
            labels = labels[:n_trials]

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
            event_sample = start + onset_offset
            if event_sample < total_len:
                all_data[n_ch, event_sample] = int(labels[i])

        ch_names = list(_CH_NAMES) + ["STI 014"]
        ch_types = ["eeg"] * n_ch + ["stim"]
        info = mne.create_info(ch_names, _SFREQ, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)
        raw.set_montage("standard_1020", on_missing="warn")

        return raw
