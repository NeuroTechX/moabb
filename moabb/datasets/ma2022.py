"""Ma2022 (SHU) cross-session motor imagery dataset."""

import logging
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import RawArray, read_raw_edf

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# The full SHU dataset (25 subjects x 5 sessions) is distributed as a single
# figshare archive. Only version 1 of the figshare record exposes the EEG
# archives without password protection; versions 2 and 3 re-uploaded the same
# recordings inside AES-encrypted zips (the README asks users to email the
# author for a password). The loader therefore targets the open version-1
# ``mat.zip`` (figshare file id 36324114).
MA2022_MAT_ZIP_URL = "https://ndownloader.figshare.com/files/36324114"
MA2022_MAT_ZIP_MD5 = "f577fbd4ddcad5e358f941d27bb7c393"
MA2022_EDF_DIRNAME = "SHU_edf"

# Sampling rate (Hz), from the BIDS ``task-motorimagery_eeg.json`` sidecar.
SFREQ = 250.0

# 32 channel names in acquisition order, from the BIDS
# ``task-motorimagery_channels.tsv`` sidecar. The dataset uses the older
# 10-20 nomenclature (e.g. T3/T4/T5/T6 instead of T7/T8/P7/P8) and includes
# the two earlobe/mastoid channels A1 and A2 among the 32 recorded channels.
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

# Trial labels are stored in each .mat file (1 = left hand, 2 = right hand).
MA2022_EVENTS = {"left_hand": 1, "right_hand": 2}


class Ma2022(BaseDataset):
    """Cross-session motor imagery dataset (SHU) from Ma et al. 2022.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Ma2022          25       32           2       up to 250            4s           250 Hz            5
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    The SHU dataset [1]_ contains electroencephalography (EEG) recordings from
    25 healthy, BCI-naive subjects performing a cued left- vs right-hand motor
    imagery task. Each subject completed five independent sessions recorded on
    five different days (2 to 3 days apart), which makes the dataset suitable
    for studying cross-session variability in motor imagery brain-computer
    interfaces.

    Each session was designed with 100 trials (50 left-hand, 50 right-hand).
    The released files contain 74 to 100 retained trials per session after the
    source-side bad-segment rejection described in the data paper, for 11,988
    trials in total. Signals were recorded from 32 EEG channels (10-20 system,
    unipolar reference on M1, ground on Afz) at a sampling rate of 250 Hz.
    Every trial lasts 8 s: a 0 to 2 s rest period, a 2 to 4 s visual cue, and a
    4 to 8 s motor imagery period. Only the 4 s motor imagery window is stored
    (1000 samples per trial), so the analysis interval spans the full stored
    window (0 to 3.996 s, i.e. 4 s minus one sample, relative to imagery
    onset).

    The data are distributed as MATLAB ``.mat`` files (one per subject and
    session, named ``sub-XXX_ses-YY_task_motorimagery_eeg.mat``) bundled in a
    single figshare archive. Each file contains a variable-length ``data``
    array of shape ``(n_trials, n_channels, n_samples)`` and an integer
    ``labels`` vector (1 = left hand, 2 = right hand).

    .. note::

        The seed reference for this dataset is the Harvard Dataverse deposit
        ``doi:10.7910/DVN/7CKTSW`` (title "Dataset for Studying Cross-Session
        Variability in Motor Imagery Brain-Computer Interface", author Jun Ma),
        which openly hosts a single verification file (subject 1, session 1) of
        the same recording. The complete 25-subject dataset is obtained from
        the open version-1 figshare archive instead. This dataset is from the
        same laboratory as :class:`Yang2025` (WBCIC-SHU, a distinct 2025
        multi-day recording) and is unrelated to :class:`Pan2023`,
        :class:`Pan2025` and :class:`Ma2020`.

    References
    ----------

    .. [1] Ma, J., Yang, B., Qiu, W., Li, Y., Gao, S., & Xia, X. (2022).
       A large EEG dataset for studying cross-session variability in motor
       imagery brain-computer interface. Scientific Data, 9, 531.
       DOI: https://doi.org/10.1038/s41597-022-01647-1
       Data: https://doi.org/10.6084/m9.figshare.19228725

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="10-20",
            reference="M1 (unipolar)",
            ground="Afz",
            sensors=list(MA2022_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"male": 13, "female": 12},
            age_min=20.0,
            age_max=24.0,
            bci_experience="naive",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 250, "right_hand": 250},
            trial_duration=4.0,
            synchronicity="cue-based",
            stimulus_type="visual",
            primary_modality="visual",
            mode="offline",
            feedback_type="none",
            events=dict(MA2022_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-022-01647-1",
            description="SHU cross-session motor imagery EEG dataset: 25 BCI-naive "
            "subjects performing cued left- vs right-hand motor imagery across five "
            "sessions recorded on five different days.",
            investigators=[
                "Jun Ma",
                "Banghua Yang",
                "Wenzheng Qiu",
                "Yunzhe Li",
                "Shouwei Gao",
                "Xinxing Xia",
            ],
            institution="Shanghai University",
            country="CN",
            data_url="https://doi.org/10.6084/m9.figshare.19228725",
            publication_year=2022,
            license="CC-BY-4.0",
            repository="figshare",
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        # Each stored trial is exactly the 4 s motor-imagery window (1000 samples
        # at 250 Hz, t = 0 to 3.996 s). Because :class:`mne.Epochs` includes the
        # tmax sample, an interval of [0, 4] would request 1001 samples per trial
        # and thus borrow one sample from the next concatenated trial (and drop
        # the final trial of each session). Using tmax = 4 - 1/sfreq selects
        # exactly the 1000 stored samples, keeping all trials and avoiding any
        # cross-trial contamination.
        super().__init__(
            subjects=list(range(1, 25 + 1)),
            sessions_per_subject=5,
            events=dict(MA2022_EVENTS),
            code="Ma2022",
            interval=[0.0, 4.0 - 1.0 / SFREQ],
            paradigm="imagery",
            doi="10.1038/s41597-022-01647-1",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return local EDF sessions, or download the open-v1 MAT fallback."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path(self.code, path)
        edf_files = self._local_edf_files(subject, path)
        if edf_files and not force_update:
            return [str(f) for f in edf_files]

        return self._mat_session_files(
            subject, path, force_update=force_update, verbose=verbose
        )

    @staticmethod
    def _session_filenames(subject, suffix):
        """Build the five BIDS-style session filenames for one subject."""
        return [
            f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.{suffix}"
            for session in range(1, 5 + 1)
        ]

    def _local_edf_files(self, subject, path):
        """Find a complete locally extracted five-session EDF set."""
        root = Path(path)
        filenames = self._session_filenames(subject, "edf")
        candidate_dirs = (
            root / MA2022_EDF_DIRNAME / "edf",
            root / "MNE-ma2022-data" / "edf",
            root / "MNE-ma2022-data" / MA2022_EDF_DIRNAME / "edf",
        )
        for edf_dir in candidate_dirs:
            files = [edf_dir / filename for filename in filenames]
            if all(f.exists() for f in files):
                return files
        return []

    def _mat_session_files(self, subject, path, force_update=False, verbose=None):
        """Download/extract and return the open-v1 MATLAB session files."""
        basepath = Path(path) / "MNE-ma2022-data"
        basepath.mkdir(parents=True, exist_ok=True)
        mat_dir = basepath / "mat"

        subject_files = [
            mat_dir / filename for filename in self._session_filenames(subject, "mat")
        ]

        if force_update or not all(f.exists() for f in subject_files):
            zip_path = basepath / "mat_files.zip"
            if force_update or not zip_path.exists():
                log.info("Downloading Ma2022 (SHU) archive (~1.4 GB) from figshare...")
                downloaded = dl.data_dl(
                    MA2022_MAT_ZIP_URL,
                    self.code,
                    path=str(basepath),
                    force_update=force_update,
                    verbose=verbose,
                )
                downloaded = Path(downloaded)
                if downloaded != zip_path:
                    downloaded.replace(zip_path)
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(basepath))

        missing = [str(f) for f in subject_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing session files for subject {subject}: {missing}"
            )
        return [str(f) for f in subject_files]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        file_paths = self.data_path(subject)

        sessions = {}
        if Path(file_paths[0]).suffix.lower() == ".edf":
            # The EDF conversion contains signals only: it has no annotations
            # or stim channel. Reuse the randomized per-trial labels from the
            # openly downloadable v1 MAT release while reading signals from
            # the locally available five-session EDF release.
            root = dl.get_dataset_path(self.code, None)
            label_paths = self._mat_session_files(subject, root)
            for session_idx, (edf_path, label_path) in enumerate(
                zip(file_paths, label_paths)
            ):
                try:
                    raw = self._edf_to_raw(edf_path, label_path)
                except ValueError as error:
                    # Nine files in the password-protected EDF release have
                    # blank numeric physical-min header fields. Preserve the
                    # complete five-session dataset by using the corresponding
                    # open-v1 MAT recording only for those malformed sessions.
                    if "could not convert string to float" not in str(error):
                        raise
                    log.warning(
                        "Unreadable EDF numeric header in %s; using %s",
                        edf_path,
                        label_path,
                    )
                    raw = self._mat_to_raw(label_path)
                sessions[str(session_idx)] = {"0": raw}
        else:
            for session_idx, file_path in enumerate(file_paths):
                sessions[str(session_idx)] = {"0": self._mat_to_raw(file_path)}
        return sessions

    # Map the stored integer label to the MOABB event name.
    _CODE_TO_NAME = {code: name for name, code in MA2022_EVENTS.items()}

    @classmethod
    def _edf_to_raw(cls, edf_path, label_path):
        """Read one EDF session and attach labels from the open-v1 MAT release."""
        raw = read_raw_edf(edf_path, preload=True, verbose=False)
        labels = (
            np.asarray(sio.loadmat(label_path, variable_names=["labels"])["labels"])
            .ravel()
            .astype(int)
        )

        if raw.info["sfreq"] != SFREQ:
            raise ValueError(
                f"Expected sampling rate {SFREQ:g} Hz, got "
                f"{raw.info['sfreq']:g} Hz in {edf_path}"
            )
        if len(raw.ch_names) != len(MA2022_CH_NAMES):
            raise ValueError(
                f"Expected {len(MA2022_CH_NAMES)} channels, got "
                f"{len(raw.ch_names)} in {edf_path}"
            )
        if [name.upper() for name in raw.ch_names] != [
            name.upper() for name in MA2022_CH_NAMES
        ]:
            raise ValueError(
                f"Unexpected channel order in {edf_path}: {raw.ch_names}"
            )
        raw.rename_channels(
            {
                observed: expected
                for observed, expected in zip(raw.ch_names, MA2022_CH_NAMES)
                if observed != expected
            }
        )

        n_samples = int(round(4.0 * SFREQ))
        expected_samples = len(labels) * n_samples
        if raw.n_times != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples for {len(labels)} trials, "
                f"got {raw.n_times} in {edf_path}"
            )

        descriptions = [cls._CODE_TO_NAME[int(label)] for label in labels]
        raw.set_annotations(
            Annotations(
                onset=np.arange(len(labels)) * (n_samples / SFREQ),
                duration=np.full(len(labels), n_samples / SFREQ),
                description=descriptions,
            )
        )
        raw.set_montage(
            make_standard_montage("standard_1005"), on_missing="ignore", verbose=False
        )
        return raw

    @classmethod
    def _mat_to_raw(cls, file_path):
        """Load one ``.mat`` session file into a continuous :class:`mne.io.RawArray`.

        Epoched trials ``(n_trials, n_channels, n_samples)`` are concatenated
        along time. Each trial's imagery onset (t = 0) is marked with an MNE
        annotation whose description is the class name (``left_hand`` /
        ``right_hand``). Annotations are used rather than a stim channel so
        that the very first trial, which starts at sample 0, is not dropped by
        :func:`mne.find_events` (which cannot detect a trigger on the first
        sample).
        """
        mat = sio.loadmat(file_path)
        data = np.asarray(mat["data"], dtype=float)  # (n_trials, n_channels, n_samples)
        labels = np.asarray(mat["labels"]).ravel().astype(int)

        n_trials, n_channels, n_samples = data.shape
        if n_channels != len(MA2022_CH_NAMES):
            raise ValueError(
                f"Expected {len(MA2022_CH_NAMES)} channels, got {n_channels} "
                f"in {file_path}"
            )

        # Concatenate trials along time: (n_channels, n_trials * n_samples).
        cont = np.transpose(data, (1, 0, 2)).reshape(n_channels, n_trials * n_samples)
        # Convert from microvolts to volts.
        cont = cont * 1e-6

        mne_info = create_info(
            ch_names=list(MA2022_CH_NAMES), sfreq=SFREQ, ch_types=["eeg"] * n_channels
        )
        raw = RawArray(data=cont, info=mne_info, verbose=False)

        trial_len_s = n_samples / SFREQ
        onsets = np.arange(n_trials) * trial_len_s
        durations = np.full(n_trials, trial_len_s)
        descriptions = [cls._CODE_TO_NAME[int(label)] for label in labels[:n_trials]]
        raw.set_annotations(
            Annotations(onset=onsets, duration=durations, description=descriptions)
        )

        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
