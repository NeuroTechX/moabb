"""ZJU-MI-EEG (MI4) four-class motor imagery dataset from Zhejiang University.

Wang, J., Yao, L., and Wang, Y. (2025). "Enhanced Online Continuous
Brain-Control by Deep Learning-Based EEG Decoding." IEEE Transactions on
Neural Systems and Rehabilitation Engineering, vol. 33, pp. 2834-2846.
DOI: 10.1109/TNSRE.2025.3591254

Data host: Hugging Face ``Jiaheng-Wang/ZJU-MI-EEG`` (MI4 subset).
https://huggingface.co/datasets/Jiaheng-Wang/ZJU-MI-EEG
"""

import numpy as np
import scipy.io as sio
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


# Base URL of the MI4 subset on the Hugging Face dataset repository. The
# "resolve/main" endpoint serves the raw file bytes. The per-subject folder
# ("sub-NN") is preserved in the local cache path, so the shared file names
# ("s1_calibration.mat", ...) do not collide between subjects.
ZJU_MI2025_BASE_URL = (
    "https://huggingface.co/datasets/Jiaheng-Wang/ZJU-MI-EEG/resolve/main/MI4"
)

# Sampling rate (Hz) reported in the MI4 dataset card.
SFREQ = 256.0

# Each stored trial spans -1 s to 4 s relative to cue onset (1280 samples).
EPOCH_START = -1.0

# 62 EEG channels in acquisition order, taken from the provided
# "62channels_gNautilus.ced" montage file and normalised to MNE casing
# (midline "Z" -> "z"). All 62 names resolve in the standard_1005 montage.
CHANNELS = [
    "Fp1",
    "Fpz",
    "Fp2",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
    "F9",
    "F10",
]

# Integer label code (data-borne, stored in the ``labels`` field) -> class name.
EVENT_ID = {"left_hand": 1, "right_hand": 2, "tongue": 3, "feet": 4}

# Two runs per session: cued calibration (240 trials) then online feedback
# (160 trials). Keys are the file basenames within each "sub-NN" folder.
RUN_FILES = {"0calibration": "calibration", "1feedback": "feedback"}


class ZjuMI2025(BaseDataset):
    """Four-class motor imagery dataset (ZJU-MI-EEG / MI4) [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        ZjuMI2025       15       62           4                200            4s            256 Hz            2
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings from 15 healthy subjects performing
    a cued four-class motor imagery task, collected at Zhejiang University and
    distributed as the "MI4" subset of the ``ZJU-MI-EEG`` Hugging Face dataset.

    Each subject took part on two separate days (mapped here to two sessions).
    Every day comprises a calibration run of 240 trials followed by an online
    feedback run of 160 trials (400 trials/day, 800 trials/subject). The four
    imagined movements are left hand, right hand, tongue and both feet, and the
    trials are balanced across classes (200 trials/class/subject).

    Signals were recorded from 62 electrodes with a gtec gNautilus system at a
    sampling rate of 256 Hz. Raw EEG was high-pass filtered above 0.1 Hz; each
    stored trial spans -1 s to 4 s relative to cue onset (1280 samples), and the
    motor imagery window used here is 0 to 4 s.

    The data are distributed as MATLAB ``.mat`` files (one per subject, day and
    run) on Hugging Face. Each file holds an ``EEG_data`` array of shape
    ``(62, 1280, n_trials)`` in microvolts and an integer ``labels`` vector
    (1 = left hand, 2 = right hand, 3 = tongue, 4 = both feet).

    References
    ----------

    .. [1] Wang, J., Yao, L., and Wang, Y. (2025). Enhanced Online Continuous
       Brain-Control by Deep Learning-Based EEG Decoding. IEEE Transactions on
       Neural Systems and Rehabilitation Engineering, 33, 2834-2846.
       DOI: 10.1109/TNSRE.2025.3591254

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="10-10",
            reference=None,
            ground=None,
            sensors=list(CHANNELS),
        ),
        participants=ParticipantMetadata(n_subjects=15, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "tongue", "feet"],
            trials_per_class={
                "left_hand": 200,
                "right_hand": 200,
                "tongue": 200,
                "feet": 200,
            },
            trial_duration=4.0,
            study_design="Cued four-class motor imagery (left hand, right hand, "
            "tongue, both feet) over two days, each with a 240-trial calibration "
            "run and a 160-trial online feedback run.",
            feedback_type="visual",
            stimulus_type="visual",
            synchronicity="cue-based",
            mode="online",
            has_training_test_split=True,
            events=dict(EVENT_ID),
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2025.3591254",
            description="Four-class motor imagery EEG from 15 healthy subjects "
            "(left hand, right hand, tongue, both feet) recorded with 62 channels "
            "at 256 Hz over two days of calibration and online feedback sessions.",
            investigators=["Jiaheng Wang", "Lin Yao", "Yueming Wang"],
            institution="Zhejiang University",
            country="CN",
            data_url="https://huggingface.co/datasets/Jiaheng-Wang/ZJU-MI-EEG",
            publication_year=2025,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "four-class",
            ],
            license="ODC-BY",
            repository="Hugging Face",
        ),
        sessions_per_subject=2,
        runs_per_session=2,
        tags=Tags(pathology=["healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "tongue", "feet"],
            imagery_duration_s=4.0,
        ),
        file_format="MAT",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15 + 1)),
            sessions_per_subject=2,
            events=dict(EVENT_ID),
            code="ZjuMI2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1109/TNSRE.2025.3591254",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of a single subject's four ``.mat`` files.

        The four files are the calibration and feedback runs of day 1 (``s1_*``)
        and day 2 (``s2_*``), in that order.

        Parameters
        ----------
        subject : int
            Subject number (1-15).
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
            The four local file paths, ordered
            ``[s1_calibration, s1_feedback, s2_calibration, s2_feedback]``.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for day in (1, 2):
            for run_kind in ("calibration", "feedback"):
                url = f"{ZJU_MI2025_BASE_URL}/sub-{subject:02d}/s{day}_{run_kind}.mat"
                paths.append(dl.data_dl(url, self.code, path, force_update, verbose))
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-15).

        Returns
        -------
        dict
            ``{session_str: {run_str: Raw}}`` with two sessions (days), each
            holding a "0calibration" and a "1feedback" run.
        """
        files = self.data_path(subject)
        # data_path yields [s1_cal, s1_fb, s2_cal, s2_fb]; group into two days.
        sessions = {}
        for sess_idx in range(2):
            runs = {}
            for run_idx, run_key in enumerate(RUN_FILES):
                file_path = files[sess_idx * 2 + run_idx]
                runs[run_key] = self._load_raw(file_path)
            sessions[str(sess_idx)] = runs
        return sessions

    @staticmethod
    def _load_raw(file_path):
        """Load one ``.mat`` run file into a continuous :class:`mne.io.RawArray`.

        Epoched trials ``(62, 1280, n_trials)`` are concatenated along time and a
        stim channel marks each trial's cue onset (t = 0) with its class code.
        """
        mat = sio.loadmat(file_path)
        data = np.asarray(
            mat["EEG_data"], dtype=float
        )  # (n_channels, n_samples, n_trials)
        labels = np.atleast_1d(np.asarray(mat["labels"]).ravel()).astype(int)

        n_channels, n_samples, n_trials = data.shape
        # Concatenate trials along time: (n_channels, n_trials * n_samples).
        cont = np.transpose(data, (0, 2, 1)).reshape(n_channels, n_trials * n_samples)
        # Convert from microvolts to volts.
        cont = cont * 1e-6

        # Sample offset of the cue (t = 0) within each epoch.
        cue_offset = int(round((0.0 - EPOCH_START) * SFREQ))

        stim = np.zeros((1, cont.shape[1]))
        for trial_idx, label in enumerate(labels[:n_trials]):
            onset = trial_idx * n_samples + cue_offset
            stim[0, onset] = label

        full = np.vstack([cont, stim])
        mne_info = create_info(
            ch_names=list(CHANNELS) + ["STI 014"],
            sfreq=SFREQ,
            ch_types=["eeg"] * n_channels + ["stim"],
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
