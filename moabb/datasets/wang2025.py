"""Wang2025: four-class MI dataset (ZJU-MI-EEG ``MI4`` release, Zhejiang University)."""

import numpy as np
import scipy.io as sio
from mne import Annotations, create_info
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
WANG2025_BASE_URL = (
    "https://huggingface.co/datasets/Jiaheng-Wang/ZJU-MI-EEG/resolve/main/MI4"
)

# Sampling rate (Hz) reported in the MI4 dataset card.
SFREQ = 256.0

# Each stored trial spans -1 s to 4 s relative to cue onset (1280 samples).
CUE_OFFSET = 256  # sample of the cue (t = 0) within each trial

# 62 EEG channels in acquisition order, taken from the provided
# "62channels_gNautilus.ced" montage file and normalised to MNE casing
# (midline "Z" -> "z"). All 62 names resolve in the standard_1005 montage.
CHANNELS = (
    "Fp1 Fpz Fp2 AF7 AF3 AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FC5 FC3 FC1 "
    "FCz FC2 FC4 FC6 FT8 T7 C5 C3 C1 Cz C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPz CP2 "
    "CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 F9 F10"
).split()

# Integer label code (data-borne, stored in the ``labels`` field) -> class name.
EVENT_ID = {"left_hand": 1, "right_hand": 2, "tongue": 3, "feet": 4}

# Two runs per session: cued calibration (240 trials) then online feedback
# (160 trials), stored as s<day>_calibration.mat / s<day>_feedback.mat.
RUN_KEYS = ("0calibration", "1feedback")


class Wang2025(BaseDataset):
    """Four-class motor imagery dataset (ZJU-MI-EEG / MI4) [1]_.

    **Dataset description**

    15 healthy subjects performed cued four-class motor imagery (left hand, right
    hand, tongue, both feet) on two days, mapped to two sessions. Each day has a
    240-trial calibration run and a 160-trial online feedback run (``MI4`` subset
    of the ``ZJU-MI-EEG`` Hugging Face dataset; 62 channels, 256 Hz). In the
    paper [1]_ each session consists of six calibration runs and four online
    feedback runs of 40 trials (10 per class); the release merges them into one
    calibration and one feedback file per day, which are the two runs exposed
    here. The paper reports a 62-channel g.USBamp amplifier (g.tec) sampled at
    256 Hz and high-pass filtered above 0.1 Hz, whereas the release ships a
    ``62channels_gNautilus.ced`` montage file; the amplifier model is therefore
    recorded as reported by the paper. Feedback trials lasted up to 10 s online,
    but the release stores every trial as the -1 to 4 s window around the cue.

    Each ``.mat`` run stores ``EEG_data`` ``(62, 1280, n_trials)`` in microvolts
    (-1 to 4 s around the cue) and integer ``labels``. The loader concatenates the
    trials, marks each cue with a stim event and converts to volts; the interval
    ends at 4 - 1/256 s.

    References
    ----------

    .. [1] Wang, J., Yao, L., and Wang, Y. (2025). Enhanced Online Continuous
       Brain-Control by Deep Learning-Based EEG Decoding. IEEE Transactions on
       Neural Systems and Rehabilitation Engineering, 33, 2834-2846.
       DOI: 10.1109/TNSRE.2025.3591254

    Notes
    -----

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"eeg": 62},
            montage="10-10",
            hardware="g.USBamp (g.tec medical engineering, Austria), 62 channels",
            filters="0.1 Hz high-pass",
            reference=None,
            ground=None,
            sensors=list(CHANNELS),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            age_mean=24.0,
            age_std=5.1,
            gender={"male": 12, "female": 3},
            bci_experience="10 BCI-naive; 5 without online BCI experience",
            species="homo sapiens",
        ),
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

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 15 + 1)),
            sessions_per_subject=2,
            events=dict(EVENT_ID),
            code="Wang2025",
            interval=[0, 4 - 1 / SFREQ],
            paradigm="imagery",
            selected_subjects=subjects,
            selected_sessions=sessions,
            doi="10.1109/TNSRE.2025.3591254",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return ``[s1_calibration, s1_feedback, s2_calibration, s2_feedback]``."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for day in (1, 2):
            for run_kind in ("calibration", "feedback"):
                url = f"{WANG2025_BASE_URL}/sub-{subject:02d}/s{day}_{run_kind}.mat"
                paths.append(dl.data_dl(url, self.code, path, force_update, verbose))
        return paths

    def _get_single_subject_data(self, subject):
        """Return ``{day: {"0calibration": Raw, "1feedback": Raw}}``."""
        files = iter(self.data_path(subject))  # [s1_cal, s1_fb, s2_cal, s2_fb]
        return {
            str(day): {key: self._load_raw(next(files)) for key in RUN_KEYS}
            for day in range(2)
        }

    @staticmethod
    def _load_raw(file_path):
        """Load one ``.mat`` run file into a continuous :class:`mne.io.RawArray`.

        Epoched trials ``(62, 1280, n_trials)`` are concatenated along time and a
        stim channel marks each trial's cue onset (t = 0) with its class code.
        """
        mat = sio.loadmat(file_path)
        data = np.asarray(mat["EEG_data"], dtype=float)  # (chan, samples, trials)
        labels = np.asarray(mat["labels"]).ravel()

        if data.ndim != 3 or data.shape[:2] != (len(CHANNELS), 1280):
            raise ValueError("Expected EEG_data shape (62, 1280, n_trials)")
        n_channels, n_samples, n_trials = data.shape
        if (
            n_trials == 0
            or len(labels) != n_trials
            or not np.isin(labels, [1, 2, 3, 4]).all()
        ):
            raise ValueError("Expected one valid class label per trial")
        # Concatenate trials along time and convert from microvolts to volts.
        cont = np.transpose(data, (0, 2, 1)).reshape(n_channels, n_trials * n_samples)
        stim = np.zeros((1, cont.shape[1]))
        stim[0, np.arange(n_trials) * n_samples + CUE_OFFSET] = labels
        full = np.vstack([cont * 1e-6, stim])
        mne_info = create_info(
            ch_names=list(CHANNELS) + ["STI 014"],
            sfreq=SFREQ,
            ch_types=["eeg"] * n_channels + ["stim"],
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        montage = make_standard_montage("colin27_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        raw.set_annotations(
            Annotations(np.arange(1, n_trials) * n_samples / SFREQ, 0.0, "EDGE boundary")
        )
        return raw
