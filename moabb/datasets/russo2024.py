"""Russo2024 imagined-movement EEG dataset in multiple sclerosis.

Russo, J., Shiels, T., Grayden, D., Lin, C.-H. S., and John, S. (2024).
"Decoding Imagined Movement in People with Multiple Sclerosis for
Brain-Computer Interface Translation (EEG Data)."
The University of Melbourne. Dataset.
Data DOI: 10.26188/26230904
"""

import hashlib
import logging
import zipfile
from pathlib import Path

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

# Single ZIP archive on the University of Melbourne Figshare instance
# (article 26230904, DOI 10.26188/26230904, CC-BY-4.0, ~5.77 GB).
_ZIP_URL = "https://ndownloader.figshare.com/files/47539802"
_ZIP_MD5 = "a5ac85b76e3d85486bba01690f6bdd5b"

# Top-level folder inside the archive.
_ARCHIVE_ROOT = "DATASET_1748801_26070"

# Expected EEG channel count (24-channel amplifier). Used as a robust
# orientation check for the EEG_data matrix in each .mat block.
_N_CHANNELS = 24

# Published cohort (README + article overview): ten neurotypical controls
# (S1-S10) and eight participants with multiple sclerosis (P1-P8) = 18
# subjects, each with a single recording session. Integer subject ids map
# to the on-disk string ids below. Four additional non-cohort recordings in
# the archive (JL, KS4, MS8, PS) are pilot/duplicate files and are excluded.
_SUBJECT_MAP = {i: f"S{i}" for i in range(1, 11)}
_SUBJECT_MAP.update({10 + j: f"P{j}" for j in range(1, 9)})

# MS patients are subjects 11-18 (folder ids P1-P8).
_MS_SUBJECTS = set(range(11, 19))

# Imagined-block trial-label codes (prompt_times column 1). The code->meaning
# order is defined verbatim in the archive README.txt. This is a go/no-go
# protocol: five imagined-movement types crossed with a go (perform) or a
# stop (withhold) instruction.
_EVENTS = {
    "walking_go": 1,
    "left_hand_go": 2,
    "right_hand_go": 3,
    "left_foot_go": 4,
    "right_foot_go": 5,
    "walking_stop": 6,
    "left_hand_stop": 7,
    "right_hand_stop": 8,
    "left_foot_stop": 9,
    "right_foot_stop": 10,
}


class Russo2024(BaseDataset):
    """Imagined-movement EEG dataset in multiple sclerosis [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  ============  ===============  ===========
        Russo2024       18       24          10            3s             2048 Hz            1
        =========  =======  =======  ==========  ============  ===============  ===========

    **Dataset description**

    Electroencephalography from eighteen participants performing imagined
    movements under a go/no-go protocol: ten neurotypical control participants
    (folder ids ``S1``-``S10``, mapped to subjects 1-10) and eight participants
    diagnosed with multiple sclerosis (folder ids ``P1``-``P8``, mapped to
    subjects 11-18). Seven controls and one MS participant were collected by
    Shiels et al. (2018); the remainder were newly collected. Each participant
    contributed a single session of imagined-movement blocks (typically six
    blocks of about thirty trials each).

    On every trial a visual cue first indicates one of five movements (walking,
    left hand, right hand, left foot, right foot); after about five seconds a
    go/stop signal instructs the participant either to imagine the movement (go)
    or to withhold it (stop) for roughly three seconds. Each trial therefore
    belongs to one of ten labelled conditions (five movements x {go, stop}),
    stored as an integer code in the ``prompt_times`` label column of the source
    ``.mat`` file. This loader anchors each event at the go/stop signal onset
    (``prompt_times`` column 3) and exposes an interval covering the roughly
    three-second imagery window.

    EEG was recorded with a 24-channel BioSemi-style amplifier at 2048 Hz,
    referenced to AFz (50 Hz line frequency). The shared ``.mat`` files carry
    generic amplifier channel labels (``ExG1``-``ExG24``); the physical 10-20
    positions are not documented in the archive, so no montage is applied.

    .. note::

       Only the imagined-movement blocks are loaded. Some participants also have
       motor-execution ("active") blocks in the archive; those use a different
       label-code scheme and are excluded here for label consistency.

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load (1-18). Defaults to all.
    sessions : list of str | None
        Sessions to load. There is a single session per subject.

    References
    ----------
    .. [1] Russo, J., Shiels, T., Grayden, D., Lin, C.-H. S., and John, S.
           (2024). Decoding Imagined Movement in People with Multiple Sclerosis
           for Brain-Computer Interface Translation (EEG Data). The University
           of Melbourne. Dataset.
           DOI: https://doi.org/10.26188/26230904

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=2048.0,
            n_channels=24,
            channel_types={"eeg": 24},
            sensors=[f"ExG{i}" for i in range(1, 25)],
            reference="AFz",
            hardware="24-channel BioSemi-style active-electrode amplifier",
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            health_status="mixed",
            clinical_population="multiple sclerosis (8) and neurotypical controls (10)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=10,
            class_labels=list(_EVENTS.keys()),
            trial_duration=3.0,
            study_design=(
                "Go/no-go imagined-movement protocol. Five imagined movements "
                "(walking, left hand, right hand, left foot, right foot) are each "
                "cued, then a go/stop signal instructs the participant to imagine "
                "the movement (go) or withhold it (stop), giving ten labelled "
                "conditions per imagined block."
            ),
            feedback_type="none",
            stimulus_type="visual cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            events=dict(_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.26188/26230904",
            description=(
                "Imagined-movement EEG under a go/no-go protocol from eight "
                "participants with multiple sclerosis and ten neurotypical "
                "controls, recorded with a 24-channel amplifier at 2048 Hz."
            ),
            investigators=[
                "John Russo",
                "Thomas Shiels",
                "David Grayden",
                "Chin-Hsuan Sophie Lin",
                "Sam John",
            ],
            institution="The University of Melbourne",
            country="AU",
            repository="Figshare",
            data_url="https://doi.org/10.26188/26230904",
            publication_year=2024,
            license="CC-BY-4.0",
            keywords=[
                "motor imagery",
                "imagined movement",
                "multiple sclerosis",
                "brain-computer interface",
                "EEG",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        tags=Tags(
            pathology=["multiple sclerosis", "healthy"],
            modality=["motor"],
            type=["Motor Imagery"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            re_reference="AFz reference; common average of all 24 channels",
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        bci_application=BCIApplicationMetadata(
            applications=["neurorehabilitation", "assistive technology"],
            environment="clinical",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "walking",
                "left_hand",
                "right_hand",
                "left_foot",
                "right_foot",
            ],
            imagery_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=6,
            trials_context=(
                "One session per subject; typically six imagined blocks of about "
                "thirty go/no-go trials each."
            ),
        ),
        file_format="MAT",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Russo2024",
            interval=[0, 3],  # go/stop window relative to the go signal onset
            paradigm="imagery",
            doi="10.26188/26230904",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download (once) and extract the single archive, return its root path."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Russo2024", path)
        basepath = Path(path) / "MNE-russo2024-data"
        basepath.mkdir(parents=True, exist_ok=True)

        extracted_dir = basepath / _ARCHIVE_ROOT
        already_extracted = extracted_dir.is_dir() and any(extracted_dir.rglob("*.mat"))

        zip_path = basepath / "DATASET_1748801_26070.zip"
        if not zip_path.exists() and not already_extracted:
            log.info("Downloading Russo2024 dataset (~5.77 GB) from Figshare...")
            dl_path = Path(
                dl.data_dl(
                    _ZIP_URL,
                    "Russo2024",
                    path=str(basepath),
                    force_update=force_update,
                    verbose=verbose,
                )
            )
            if dl_path != zip_path:
                dl_path.rename(zip_path)
            _verify_zip_md5(zip_path)

        if zip_path.exists() and not already_extracted:
            log.info("Extracting Russo2024 dataset...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(basepath))

        return str(extracted_dir)

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: Raw}}`` for one subject's imagined blocks."""
        root = Path(self.data_path(subject))
        sid = _SUBJECT_MAP[subject]
        subject_dir = root / sid
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Missing subject folder {subject_dir}")

        block_files = sorted(
            subject_dir.glob(f"EEGsigsimagined_subject{sid}_*_block*.mat"),
            key=_block_number,
        )
        if not block_files:
            raise FileNotFoundError(
                f"No imagined-movement .mat files for subject {subject} ({sid})"
            )

        runs = {}
        for run_idx, mat_path in enumerate(block_files):
            raw = _load_block(mat_path)
            if raw is not None:
                runs[str(run_idx)] = raw

        if not runs:
            raise FileNotFoundError(
                f"No usable imagined blocks for subject {subject} ({sid})"
            )
        return {"0": runs}


def _verify_zip_md5(zip_path):
    """Check the downloaded archive against the expected MD5 (warn on mismatch).

    A mismatch is logged as a warning rather than raised: the recorded hash may
    predate a re-upload, and a hard failure on a ~5.77 GB download would be
    costly. This ties the ``_ZIP_MD5`` constant to an actual integrity check.
    """
    if not _ZIP_MD5:
        return
    md5 = hashlib.md5()
    with open(zip_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            md5.update(chunk)
    digest = md5.hexdigest()
    if digest != _ZIP_MD5:
        log.warning(
            "Russo2024 archive MD5 mismatch for %s: expected %s, got %s.",
            zip_path.name,
            _ZIP_MD5,
            digest,
        )


def _block_number(mat_path):
    """Sort key: trailing block index in the filename."""
    stem = mat_path.stem
    token = stem.rsplit("block", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return 0


def _load_block(mat_path):
    """Build a continuous :class:`mne.io.RawArray` (24 EEG + stim) from one block."""
    mat = loadmat(str(mat_path))
    eeg = np.asarray(mat["EEG_data"], dtype=np.float64)  # (n_samples, n_channels)
    if eeg.ndim != 2:
        log.warning(
            "Skipping %s: EEG_data is not 2-D (shape %s).", mat_path.name, eeg.shape
        )
        return None
    # Robust orientation check against the known channel count. The archive
    # stores samples along the long axis (n_samples, n_channels); if instead
    # the channel axis (length _N_CHANNELS) comes first, transpose it. This
    # avoids silently dropping genuinely short but valid blocks, which the
    # previous shape[0] < shape[1] heuristic did.
    if eeg.shape[1] != _N_CHANNELS and eeg.shape[0] == _N_CHANNELS:
        eeg = eeg.T
    n_samples, n_channels = eeg.shape
    if n_channels != _N_CHANNELS:
        log.warning(
            "Skipping %s: expected %d channels but EEG_data has shape %s.",
            mat_path.name,
            _N_CHANNELS,
            eeg.shape,
        )
        return None

    sfreq = float(np.ravel(mat["Fs"])[0])

    ch_names = _channel_names(mat, n_channels)
    ch_types = ["eeg"] * n_channels + ["stim"]

    # EEG assumed in microvolts -> volts for MNE; stim channel appended last.
    data = np.vstack([eeg.T * 1e-6, np.zeros((1, n_samples))])

    prompt = np.asarray(mat["prompt_times"], dtype=np.float64)
    # Column 5 (index 4) row 1 holds the EEG amplifier start time, the time
    # reference for sample 0 of EEG_data; fall back to the prompter marker.
    amp_start = prompt[0, 4] if prompt.shape[0] and prompt[0, 4] > 0 else None
    if amp_start is None:
        amp_start = float(np.ravel(mat["prompt_start_time_marker"])[0])

    valid_codes = set(_EVENTS.values())
    stim = data[-1]
    for row in prompt:
        code = int(round(row[0]))
        if code not in valid_codes:
            continue
        go_time = row[2]  # column 3: go/stop signal onset (system time)
        sample = int(round((go_time - amp_start) * sfreq))
        if 0 <= sample < n_samples:
            stim[sample] = code

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _channel_names(mat, n_channels):
    """Read channel labels from the .mat, falling back to ExG names + stim."""
    default = [f"ExG{i}" for i in range(1, n_channels + 1)]
    try:
        cells = np.ravel(mat["channel_names"])
        names = [str(np.ravel(c)[0]) for c in cells]
        if len(names) != n_channels:
            names = default
    except Exception:
        names = default
    return names + ["STI 014"]
