"""In-ear EEG binary motor imagery dataset (OSF tq4u8)."""

import h5py
import numpy as np
import scipy.io as sio

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


_SFREQ = 500.0

# Class label -> integer code, exactly as stored in ``eegdata.labels``
# (1 = right-hand motor imagery, 2 = left-hand motor imagery).
_EVENTS = {"right_hand": 1, "left_hand": 2}

# Channel names are stored per subject in ``eegdata.data_info.channel_names``
# (a MATLAB ``string`` array). That opaque object is not readable by the common
# .mat readers, but the dataset uses a fixed acquisition montage whose ordering
# is fully determined by the number of recorded channels. The three
# configurations below were decoded from the raw MCOS string data of subjects 11
# (2 ch), 23 (4 ch) and 5 (7 ch) and cross-checked against the official
# ``subject_level_completeness.csv`` for all 26 subjects.
#   EARR / EARL : passive dry in-ear EEG (present for every subject)
#   C4 / C3 / CZ: conventional scalp EEG (16 subjects only)
#   HANDR / HANDL: auxiliary wrist EMG (25 subjects; absent for subject 11)
_CHANNELS_BY_COUNT = {
    2: ["EARR", "EARL"],
    4: ["EARR", "EARL", "HANDR", "HANDL"],
    7: ["EARR", "EARL", "C4", "C3", "CZ", "HANDR", "HANDL"],
}

# Wrist EMG channels, typed ``emg`` so EEG paradigms exclude them by default.
_EMG_CHANNELS = {"HANDR", "HANDL"}

# The in-ear pair is the only EEG montage common to every participant.  The
# optional C3/C4/CZ scalp channels cannot be included in a cross-subject cache
# without padding or fabricating measurements for the 10 two-channel subjects.
_COMMON_EEG_CHANNELS = ["EARR", "EARL"]

# Stable OSF file GUIDs for each subject's sub_NN.mat (node tq4u8, osfstorage).
# Download URL is ``https://osf.io/download/<guid>/``.
# fmt: off
_OSF_FILE_IDS = {
    1: "69959ec08a5c4258b65062d6",
    2: "69959ec146edfe4abd7b0334",
    3: "6983d15f222ce1ce94be53f1",
    4: "6983bae704cec0d97e36c6d3",
    5: "6983d21aa3a05f5bbd89a4a0",
    6: "6983d916c568a1241bb751ab",
    7: "6983c9e75bd1ed3a0bbe4ee8",
    8: "6983d917ff6bfd57c089a13b",
    9: "6983c114593a4fc2d889a898",
    10: "6983d8649e413aa783be4e94",
    11: "6983db3789d1b1de2dbe4eca",
    12: "6983dc402cca0b243d89a28a",
    13: "6983de03444d6454cdbe4eec",
    14: "6983e77ef7a8dd2be936bc89",
    15: "6983de064d9a79c6cebe4e9f",
    16: "6983dc492cca0b243d89a28c",
    17: "6983e7812cca0b243d89a60c",
    18: "6983e7687dbe8dd29989a416",
    19: "6983e70026eeed62dd36be9c",
    20: "6983e7747ff83bb7b889a394",
    21: "6983c6c2370b78de45333062",
    22: "6983e463ff6bfd57c089a44c",
    23: "6983e45c22e83db653be5018",
    24: "6983e7807dbe8dd29989a427",
    25: "6983e7817dbe8dd29989a429",
    26: "6983e781c42d5257ff36c17e",
}
# fmt: on

_OSF_DOWNLOAD = "https://osf.io/download/{guid}/"


class InEarMI2026(BaseDataset):
    """In-ear EEG binary motor imagery dataset (OSF tq4u8).

    .. admonition:: Dataset summary

        ============  =======  =======  ==========  =================  ============  ===============  ===========
        Name            #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        ============  =======  =======  ==========  =================  ============  ===============  ===========
        InEarMI2026        26        2           2              ~150            5s           500 Hz            1
        ============  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings from 26 healthy subjects performing a
    cued binary (left- vs right-hand) motor imagery task. Signals were recorded
    from two passive dry in-ear EEG electrodes (``EARR``, ``EARL``), present for
    every subject, and, for a subset of 16 subjects, three conventional scalp EEG
    electrodes (``C4``, ``C3``, ``CZ``). Two auxiliary wrist EMG channels
    (``HANDR``, ``HANDL``) are provided for 25 subjects (absent for subject 11).
    This is the first publicly available in-ear EEG dataset for motor imagery.

    Each subject completed roughly 300 trials (about 150 per class). Each trial
    comprised a 2 s fixation cross, a 2 s directional cue, a 5 s motor imagery
    period (imagined fist clench of the cued hand, no physical movement) and a
    2.5-3.5 s rest. This loader exposes the 5 s motor imagery segment of each
    retained trial. The number of retained trials varies from 260 to 300 across
    subjects because of amplitude-based artefact rejection.

    Signals were band-pass filtered 4-45 Hz (4th-order Butterworth) and
    notch filtered at 50 and 100 Hz, then stored in microvolts at 500 Hz.

    The data are distributed as 26 MATLAB v7.3 ``.mat`` files (one per subject)
    on the Open Science Framework. Each file holds a single ``eegdata`` struct
    whose ``motor_imagery`` field is an array of shape
    ``(n_trials, n_channels, n_samples)`` and whose ``labels`` field is an
    integer vector (1 = right hand, 2 = left hand). The number and ordering of
    channels are given by ``data_info.channel_names`` and are recovered here from
    the channel-count of the array (see module notes).

    Because the number of source EEG channels differs across subjects, the loader
    returns the two in-ear channels common to every subject (``EARR``, ``EARL``).
    This keeps one real, consistent EEG feature space for cross-subject analyses;
    optional scalp EEG and wrist EMG channels are not returned.

    References
    ----------

    .. [1] Introducing an In-Ear EEG Dataset for Binary Motor Imagery
       Classification. Open Science Framework.
       DOI: https://doi.org/10.17605/OSF.IO/TQ4U8

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=2,
            channel_types={"eeg": 2},
            sensors=list(_COMMON_EEG_CHANNELS),
            sensor_type="dry",
            montage="standard_1005",
            line_freq=50.0,
            filters="4-45 Hz band-pass (4th-order Butterworth); 50/100 Hz notch",
        ),
        participants=ParticipantMetadata(
            n_subjects=26,
            health_status="healthy",
            gender={"male": 16, "female": 10},
            age_min=22.0,
            age_max=33.0,
            handedness={"right": 23, "left": 3},
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="left_right_hand",
            events=_EVENTS,
            n_classes=2,
            class_labels=["right_hand", "left_hand"],
            trials_per_class={"right_hand": 150, "left_hand": 150},
            trial_duration=5.0,
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.17605/OSF.IO/TQ4U8",
            description="First publicly available in-ear EEG dataset for binary "
            "(left- vs right-hand) motor imagery, from 26 healthy subjects. Two "
            "passive dry in-ear electrodes for all subjects, plus scalp C3/C4/CZ "
            "for 16 subjects and wrist EMG for 25 subjects.",
            institution="Sharif University of Technology",
            institution_department="Institute for Convergence Science and Technology",
            country="IR",
            data_url="https://osf.io/tq4u8/",
            license="CC-BY-4.0",
            publication_year=2026,
            repository="OSF",
            keywords=["motor imagery", "in-ear EEG", "BCI", "EEG"],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
    )

    def __init__(self, selected_subjects=None, selected_sessions=None):
        super().__init__(
            subjects=list(range(1, 26 + 1)),
            sessions_per_subject=1,
            events=_EVENTS,
            code="InEarMI2026",
            interval=(0, 5),
            paradigm="imagery",
            doi="10.17605/OSF.IO/TQ4U8",
            selected_subjects=selected_subjects,
            selected_sessions=selected_sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path of a single subject's ``.mat`` file."""
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")
        url = _OSF_DOWNLOAD.format(guid=_OSF_FILE_IDS[subject])
        local = dl.data_dl(
            url, self.code, path=path, force_update=force_update, verbose=verbose
        )
        return [local]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as ``{session: {run: Raw}}``."""
        file_path = self.data_path(subject)[0]

        # The 26 subject files use two different MATLAB container formats:
        # subjects 1 and 2 are legacy MATLAB v5 (non-HDF5), the rest are v7.3
        # (HDF5). Detect the format from the fixed-length text header ("MATLAB
        # 5.0 ..." vs "MATLAB 7.3 ...") and read only the motor imagery segment,
        # the per-trial labels and the sampling rate; the large fixation, cue,
        # rest, corr and baseline arrays are skipped.
        with open(file_path, "rb") as fh:
            header = fh.read(128)
        is_hdf5 = b"MATLAB 5.0" not in header

        if is_hdf5:
            # v7.3 (HDF5). MATLAB stores arrays column-major, so h5py exposes
            # them with the dimension order reversed relative to the documented
            # (n_trials, n_channels, n_samples) convention; transpose back.
            with h5py.File(file_path, "r") as fid:
                eegdata = fid["eegdata"]
                mi = np.asarray(eegdata["motor_imagery"][()], dtype=float)
                mi = np.transpose(mi, (2, 1, 0))  # -> (trials, channels, samples)
                labels = np.asarray(eegdata["labels"][()]).ravel().astype(int)
                sfreq = float(np.asarray(eegdata["data_info/Fs"][()]).ravel()[0])
        else:
            # Legacy v5. scipy.io.loadmat already returns arrays in the
            # documented (n_trials, n_channels, n_samples) order, so NO
            # transpose is applied here.
            mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            eegdata = mat["eegdata"]
            mi = np.asarray(eegdata.motor_imagery, dtype=float)
            labels = np.asarray(eegdata.labels).ravel().astype(int)
            sfreq = float(np.asarray(eegdata.data_info.Fs).ravel()[0])

        n_channels = mi.shape[1]
        if n_channels not in _CHANNELS_BY_COUNT:
            raise ValueError(
                f"Unexpected channel count {n_channels} for subject {subject}; "
                f"expected one of {sorted(_CHANNELS_BY_COUNT)}."
            )
        ch_names = _CHANNELS_BY_COUNT[n_channels]
        ch_types = ["emg" if name in _EMG_CHANNELS else "eeg" for name in ch_names]

        raw = build_raw_from_epochs(
            mi,
            ch_names,
            sfreq,
            labels,
            montage_name="standard_1005",
            ch_types=ch_types,
            scale=1e-6,
            onset_sample=0,
        )
        raw = self._select_common_in_ear_eeg(raw, subject)
        return {"0": {"0": raw}}

    @staticmethod
    def _select_common_in_ear_eeg(raw, subject):
        """Keep shared in-ear EEG and any generated stimulus channel."""
        missing = [name for name in _COMMON_EEG_CHANNELS if name not in raw.ch_names]
        if missing:
            raise ValueError(
                f"InEarMI2026 subject {subject} is missing required in-ear "
                f"channel(s) {missing}; observed {raw.ch_names}."
            )
        # build_raw_from_epochs adds the event-carrying ``STI`` channel.  Keep
        # all real stim channels while excluding optional scalp EEG and EMG.
        stim_channels = [
            name
            for name, ch_type in zip(raw.ch_names, raw.get_channel_types())
            if ch_type == "stim"
        ]
        return raw.pick(_COMMON_EEG_CHANNELS + stim_channels)
