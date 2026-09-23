"""Cross-session motor imagery dataset (SHU) from Ma et al. 2022.

Ma et al. (2022), Scientific Data.
DOI: 10.1038/s41597-022-01647-1
Data DOI: 10.6084/m9.figshare.19228725
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
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

# The SHU dataset is distributed as figshare record 19228725. Only version 1
# of the record exposes the data archives without password protection;
# versions 2 and 3 re-uploaded the same recordings inside AES-encrypted zips
# (the record description asks users to email the author for the password).
# The data paper itself cites version 1 as "open access for free download",
# so this loader targets the open version-1 ``mat.zip`` archive
# (figshare file id 36324114).
_MAT_ZIP_URL = "https://ndownloader.figshare.com/files/36324114"
_MAT_ZIP_MD5 = "f577fbd4ddcad5e358f941d27bb7c393"

# Sampling rate (Hz), from the BIDS ``task-motorimagery_eeg.json`` sidecar.
_SFREQ = 250.0

# 32 channel names in acquisition order, from the BIDS
# ``task-motorimagery_channels.tsv`` sidecar. The dataset uses the older
# 10-20 nomenclature (e.g. T3/T4/T5/T6 instead of T7/T8/P7/P8) and records
# the two mastoid channels A1/A2 among the 32 EEG channels. Name case is
# normalized here (the sidecar lists them uppercase).
# fmt: off
MA2022_CH_NAMES = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T3", "T4", "A1", "A2",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "T5", "T6",
    "PO3", "PO4", "Oz", "O1", "O2",
]
# fmt: on

# Trial labels stored in each .mat file (1 = left hand, 2 = right hand),
# consistent with the ``task-motorimagery_events.json`` sidecar.
MA2022_EVENTS = {"left_hand": 1, "right_hand": 2}

# Per-subject demographics from the BIDS ``participants.tsv`` sidecar
# (sex, age), indexed by 1-based subject number.
_DEMOGRAPHICS = {
    1: ("male", 24),
    2: ("male", 24),
    3: ("female", 23),
    4: ("female", 23),
    5: ("female", 22),
    6: ("female", 22),
    7: ("female", 24),
    8: ("male", 23),
    9: ("female", 23),
    10: ("female", 22),
    11: ("female", 22),
    12: ("male", 24),
    13: ("male", 23),
    14: ("female", 23),
    15: ("male", 22),
    16: ("female", 21),
    17: ("male", 21),
    18: ("male", 21),
    19: ("male", 23),
    20: ("female", 23),
    21: ("male", 20),
    22: ("male", 21),
    23: ("male", 23),
    24: ("female", 22),
    25: ("male", 24),
}


class Ma2022(BaseDataset):
    """Cross-session motor imagery dataset (SHU) from Ma et al. 2022.

    Dataset from [1]_.

    The SHU dataset contains EEG recordings from 25 healthy, BCI-naive
    subjects (13 males, 12 females, aged 20-24 years) performing cued
    left- vs right-hand grasping motor imagery. Each subject completed five
    independent sessions recorded on five different days, 2 to 3 days
    apart, which makes the dataset specifically suited for studying
    cross-session variability in motor imagery BCIs.

    Each session was designed with 100 trials (50 left-hand, 50 right-hand,
    randomized order). The released files retain 74 to 100 trials per
    session after the source-side bad-segment rejection described in the
    data paper, for 11,988 trials in total. Signals were recorded from 32
    EEG channels (10-20 system, unipolar reference on M1, ground on AFz)
    at 250 Hz. Every trial lasted 8 s (0-2 s rest, 2-4 s cue, 4-8 s motor
    imagery) but only the 4 s motor imagery window is stored (1000 samples
    per trial), so the analysis interval spans the full stored window.

    .. important::

       The released data is **preprocessed**: bad segments were removed,
       the baseline was corrected, and a 0.5-40 Hz FIR band-pass filter was
       applied by the authors before disclosure.

    The data are distributed as MATLAB ``.mat`` files (one per subject and
    session, named ``sub-XXX_ses-YY_task_motorimagery_eeg.mat``) bundled in
    a single figshare archive. Each file contains a ``data`` array of shape
    ``(n_trials, n_channels, n_samples)`` in microvolts and a ``labels``
    vector (1 = left hand, 2 = right hand).

    .. note::

       Only version 1 of the figshare record exposes the archives without
       password protection; versions 2 and 3 re-uploaded the same
       recordings inside AES-encrypted zips (the record description asks
       users to email the author for the password). The data paper cites
       version 1 as open access, so this loader downloads the open
       version-1 archive.

       This dataset is from the same laboratory as :class:`Yang2025`
       (WBCIC-SHU, a distinct 2025 multi-day recording) and is unrelated
       to :class:`Ma2020` (different team, different recording).

    References
    ----------
    .. [1] J. Ma, B. Yang, W. Qiu, Y. Li, S. Gao, and X. Xia, "A large EEG
       dataset for studying cross-session variability in motor imagery
       brain-computer interface," Scientific Data, vol. 9, p. 531, 2022.
       DOI: 10.1038/s41597-022-01647-1

    Notes
    -----
    .. versionadded:: 1.8.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": 32},
            montage="10-20",
            hardware="Wuhan Greentech 32-channel Ag/AgCl cap, Brickcom wireless amplifier",
            sensor_type="Ag/AgCl",
            reference="M1 (unipolar)",
            ground="AFz",
            impedance_threshold_kohm=20,
            sensors=list(MA2022_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"male": 13, "female": 12},
            age_mean=22.52,
            age_min=20,
            age_max=24,
            bci_experience="naive",
            ages=[_DEMOGRAPHICS[i][1] for i in range(1, 26)],
            sexes=[_DEMOGRAPHICS[i][0] for i in range(1, 26)],
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            events=dict(MA2022_EVENTS),
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            stimulus_type="visual cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            task_type="motor_imagery_grasping",
            feedback_type="none",
            has_training_test_split=False,
            instructions=(
                "Subjects were asked to repeatedly imagine left-hand or "
                "right-hand grasping (kinesthetic motor imagery) following "
                "the arrow cue presented on the monitor."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-022-01647-1",
            investigators=[
                "Jun Ma",
                "Banghua Yang",
                "Wenzheng Qiu",
                "Yunzhe Li",
                "Shouwei Gao",
                "Xinxing Xia",
            ],
            senior_author="Banghua Yang",
            institution="Shanghai University",
            institution_department=(
                "School of Mechatronic Engineering and Automation, "
                "Research Center of Brain-Computer Engineering"
            ),
            country="CN",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.19228725",
            license="CC-BY-4.0",
            publication_year=2022,
            ethics_approval=[
                "Shanghai Second Rehabilitation Hospital Ethics Committee "
                "(approval number: ECSHSRH 2018-0101)"
            ],
            funding=[
                "National Natural Science Foundation of China (No. 61976133)",
                "Shanghai Science and Technology Major Project (No. 2021SHZDZX)",
            ],
            keywords=["motor imagery", "EEG", "BCI", "cross-session", "cross-subject"],
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        data_processed=True,
        file_format="MAT",
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bad segment rejection (EEGLAB amplitude > 100 uV, manually confirmed)",
                "baseline removal",
                "0.5-40 Hz FIR band-pass filter",
                "epoching to the 4 s motor imagery window",
            ],
            highpass_hz=0.5,
            lowpass_hz=40.0,
            filter_type="FIR",
            artifact_methods=["amplitude threshold", "visual inspection"],
            epoch_window=[0.0, 4.0],
            notes=(
                "Preprocessing was applied by the data authors before "
                "disclosure; the released trials are the 4 s motor imagery "
                "segments only."
            ),
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=11988,
            trials_context=(
                "25 subjects x 5 sessions x up to 100 trials "
                "(74-100 retained per session after bad-segment rejection) "
                "= 11988 trials in total"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "FBCSP+SVM", "EEGNet", "DeepConvNet", "FBCNet"],
            feature_extraction=["CSP", "FBCSP"],
            frequency_bands={"mu_beta": [8.0, 30.0], "csp": [3.0, 35.0]},
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_session", "cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            environment="laboratory",
            online_feedback=False,
            applications=["motor_rehabilitation"],
        ),
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        # Each stored trial is exactly the 4 s motor-imagery window (1000
        # samples at 250 Hz, t = 0 to 3.996 s). Because mne.Epochs includes
        # the tmax sample, an interval of [0, 4] would request 1001 samples
        # per trial and thus borrow one sample from the next concatenated
        # trial (and drop the final trial of each session). Using
        # tmax = 4 - 1/sfreq selects exactly the 1000 stored samples.
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=5,
            events=dict(MA2022_EVENTS),
            code="Ma2022",
            interval=[0.0, 4.0 - 1.0 / _SFREQ],
            paradigm="imagery",
            doi="10.1038/s41597-022-01647-1",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    @staticmethod
    def _session_filenames(subject):
        """Build the five BIDS-style session filenames for one subject."""
        return [
            f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat"
            for session in range(1, 6)
        ]

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path(self.code, path)
        basepath = Path(path) / "MNE-ma2022-data"
        basepath.mkdir(parents=True, exist_ok=True)
        mat_dir = basepath / "mat"

        subject_files = [
            mat_dir / filename for filename in self._session_filenames(subject)
        ]

        if force_update or not all(f.exists() for f in subject_files):
            zip_path = basepath / "mat_files.zip"
            if force_update or not zip_path.exists():
                log.info("Downloading Ma2022 (SHU) archive (~1.4 GB) from figshare...")
                downloaded = dl.data_dl(
                    _MAT_ZIP_URL,
                    self.code,
                    path=str(basepath),
                    force_update=force_update,
                    verbose=verbose,
                )
                downloaded = Path(downloaded)
                if downloaded != zip_path:
                    downloaded.replace(zip_path)
            log.info("Extracting Ma2022 (SHU) archive...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(basepath))

        missing = [str(f) for f in subject_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing session files for subject {subject}: {missing}"
            )
        return str(mat_dir)

    # Map the stored integer label to the MOABB event name.
    _CODE_TO_NAME = {code: name for name, code in MA2022_EVENTS.items()}

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        mat_dir = Path(self.data_path(subject))

        sex, age = _DEMOGRAPHICS.get(subject, (None, None))
        _sex_map = {"male": 1, "female": 2}

        sessions = {}
        for session_idx, filename in enumerate(self._session_filenames(subject)):
            file_path = mat_dir / filename
            if not file_path.exists():
                log.warning("Missing %s", file_path)
                continue

            raw = self._mat_to_raw(file_path)

            # Attach demographics
            if sex is not None:
                raw.info["subject_info"] = {
                    "sex": _sex_map.get(sex, 0),
                    "his_id": str(subject),
                }

            sessions[str(session_idx)] = {"0": raw}

        if not sessions:
            raise FileNotFoundError(
                f"No .mat files found for subject {subject} in {mat_dir}"
            )
        return sessions

    @classmethod
    def _mat_to_raw(cls, file_path):
        """Load one ``.mat`` session file into a continuous ``mne.io.RawArray``.

        Epoched trials ``(n_trials, n_channels, n_samples)`` are concatenated
        along time. Each trial's imagery onset (t = 0) is marked with an MNE
        annotation whose description is the class name (``left_hand`` /
        ``right_hand``). Annotations are used rather than a stim channel so
        that the very first trial, which starts at sample 0, is not dropped
        by ``mne.find_events`` (which cannot detect a trigger on the first
        sample).
        """
        mat = sio.loadmat(file_path)
        data = np.asarray(mat["data"], dtype=float)  # (trials, channels, samples)
        labels = np.asarray(mat["labels"]).ravel().astype(int)

        n_trials, n_channels, n_samples = data.shape
        if n_channels != len(MA2022_CH_NAMES):
            raise ValueError(
                f"Expected {len(MA2022_CH_NAMES)} channels, got {n_channels} "
                f"in {file_path}"
            )
        if n_samples != int(round(4.0 * _SFREQ)):
            raise ValueError(
                f"Expected {int(round(4.0 * _SFREQ))} samples per trial, "
                f"got {n_samples} in {file_path}"
            )

        # Concatenate trials along time: (n_channels, n_trials * n_samples).
        cont = np.transpose(data, (1, 0, 2)).reshape(n_channels, n_trials * n_samples)
        # Convert from microvolts to volts.
        cont = cont * 1e-6

        mne_info = create_info(
            ch_names=list(MA2022_CH_NAMES), sfreq=_SFREQ, ch_types=["eeg"] * n_channels
        )
        raw = RawArray(data=cont, info=mne_info, verbose=False)

        trial_len_s = n_samples / _SFREQ
        onsets = np.arange(n_trials) * trial_len_s
        durations = np.full(n_trials, trial_len_s)
        descriptions = [cls._CODE_TO_NAME[int(label)] for label in labels[:n_trials]]
        raw.set_annotations(
            Annotations(onset=onsets, duration=durations, description=descriptions)
        )

        # T3/T4/T5/T6 and A1/A2 use the older 10-20 nomenclature and are not
        # part of the standard 10-05 montage.
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
