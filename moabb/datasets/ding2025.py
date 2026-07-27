"""EEG-BCI dataset for real-time robotic hand control at individual finger level.

Ding, Udompanyawit, Zhang and He (2025), Nature Communications.
Article DOI: 10.1038/s41467-025-61064-x
Data DOI: 10.1184/R1/29104040 (KiltHub / CMU figshare)
"""

import logging
from pathlib import Path

import mne
import numpy as np
from mne.channels import make_standard_montage
from pymatreader import read_mat

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
    SignalProcessingMetadata,
    Tags,
)
from .utils import download_and_extract_subject_zip


log = logging.getLogger(__name__)

# KiltHub (CMU figshare) per-subject ZIP file ids (article 29104040, v2).
# Verified against https://api.figshare.com/v2/articles/29104040 (2026-07).
_FIGSHARE_BASE = "https://ndownloader.figshare.com/files/"
_FILE_IDS = {
    1: 54946349,
    2: 55002860,
    3: 55002866,
    4: 55002872,
    5: 55002101,
    6: 55002311,
    7: 55002350,
    8: 55002842,
    9: 55002848,
    10: 55002854,
    11: 55002869,
    # S12 was re-uploaded in article v2 (v1 id 55002098), so its file id
    # falls outside the contiguous ~5500xxxx range of the other subjects.
    12: 57299270,
    13: 55002680,
    14: 55002857,
    15: 55002863,
    16: 55002875,
    17: 55002845,
    18: 55002851,
    19: 55002839,
    20: 55002095,
    21: 55002104,
}

# Finger class codes carried in the ``value`` field of the "Target"
# (trial-start) markers of each run's ``event`` struct. The dataset README
# legends 1 = Thumb, 2 = Index, 4 = Pinky for the online 2-/3-class control;
# code 3 is the middle finger, used only in the 4-class offline runs (verified
# as the fourth, balanced offline class in S20/OfflineImagery/R01).
_EVENTS = {"thumb": 1, "index": 2, "middle": 3, "pinky": 4}

_SFREQ = 1024.0
_N_EEG = 128

# Task arm selector -> substring that identifies the matching run folders.
_TASK_TOKEN = {"imagery": "imagery", "movement": "movement"}

# Folder-name tokens that distinguish runs *within* an acquisition session and
# are therefore stripped when deriving the MOABB session key.
_RUN_TOKENS = frozenset({"2class", "3class", "base", "finetune", "smooth"})


class _NonFiniteEEGDataError(ValueError):
    """A raw MATLAB run contains values that cannot be safely preprocessed."""


def _session_key(folder_name):
    """Collapse a run-folder name to its acquisition-session key.

    ``OnlineImagery_Sess01_2class_Base`` -> ``OnlineImagery_Sess01`` so that the
    four online sub-conditions (2-/3-class x Base/Finetune decoder) recorded on
    the same day are grouped as runs of a single session. ``OfflineImagery`` and
    ``OnlineSmoothImagery`` are left unchanged.
    """
    tokens = folder_name.split("_")
    keep = [t for t in tokens if t.lower() not in _RUN_TOKENS]
    return "_".join(keep) if keep else folder_name


class Ding2025(BaseDataset):
    """Real-time robotic-hand finger BCI dataset from Ding et al. 2025 [1]_.

    Twenty-one right-handed subjects controlled an EEG-based BCI using motor
    execution (ME) or motor imagery (MI) of individual fingers of their dominant
    (right) hand to drive the corresponding finger of a robotic hand in real
    time. EEG was recorded with a 128-channel BioSemi ActiveTwo cap (ActiView
    9.02) at 1024 Hz.

    The finger targets are **discrete** classes carried in the ``value`` field
    of each run's ``event`` struct on the ``"Target"`` (trial-start) markers:

    - **Offline** runs use all four fingers, a 4-class task
      ``thumb / index / middle / pinky`` (codes 1/2/3/4), 5 s per trial,
      5 trials per finger in randomised order.
    - **Online** runs use two control paradigms: binary ``thumb vs. pinky``
      (codes 1/4) and ternary ``thumb / index / pinky`` (codes 1/2/4), 3 s per
      trial, with a ``Base`` decoder and a ``Finetune`` decoder.
    - Sixteen of the twenty-one subjects completed three additional MI online
      sessions and two online sessions with smoothed robotic control (one ME,
      one MI).

    Each subject folder holds one folder per task condition, named
    ``TaskType(_SessYY_Zclass_Model)`` with per-run ``.mat`` files. This loader
    groups the four online sub-conditions recorded on the same day into a single
    MOABB session (e.g. ``OnlineImagery_Sess01``) and keeps the offline session
    (``OfflineImagery``) and any smoothed-control session separate. Each
    ``.mat`` stores an ``eeg`` struct (``data`` 128 x samples in microvolts,
    ``label``, ``fsample``, ...) and an ``event`` struct array
    (``type`` / ``sample`` / ``value``); online runs additionally store the
    real-time ``prediction`` and class probabilities, which are not exposed
    here.

    Parameters
    ----------
    task : str
        Which task arm to load: ``"imagery"`` (default, motor imagery) or
        ``"movement"`` (motor execution). Both arms share the identical finger
        classes and marker scheme; the dataset ``paradigm`` is reported as
        ``"imagery"``.
    subjects : list of int | None
        Subjects to load (default all 21).
    sessions : list | None
        Optional subset of sessions to load.

    Notes
    -----
    Middle-finger targets (code 3) appear only in the offline 4-class runs; the
    online runs contain 2 or 3 of the four classes. When pooled by the standard
    motor-imagery paradigm the class balance therefore varies across sessions.
    The eight runs in
    ``S07/OnlineImagery_Sess05_3class_Base`` contain non-finite EEG samples.
    Each affected raw run is skipped before scaling or later preprocessing, so
    invalid values cannot contaminate other samples through filtering.

    References
    ----------
    .. [1] Ding, Y., Udompanyawit, C., Zhang, Y., & He, B. (2025). EEG-based
           brain-computer interface enables real-time robotic hand control at
           individual finger level. Nature Communications, 16(1), 5401.
           https://doi.org/10.1038/s41467-025-61064-x
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1024.0,
            n_channels=128,
            channel_types={"eeg": 128},
            montage="biosemi128",
            hardware="BioSemi ActiveTwo, 128-channel headcap (ActiView 9.02)",
            sensor_type="Ag/AgCl active",
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=21,
            health_status="healthy",
            handedness="right-handed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=["thumb", "index", "middle", "pinky"],
            trial_duration=5.0,
            study_design=(
                "Real-time robotic-hand finger BCI. Subjects performed motor "
                "execution (ME) or motor imagery (MI) of individual fingers of "
                "the dominant (right) hand. Offline runs: 4-class "
                "(thumb/index/middle/pinky), 5 s trials, 5 trials per finger. "
                "Online runs: binary (thumb vs. pinky) and ternary "
                "(thumb/index/pinky), 3 s trials, with Base and Finetune "
                "decoders. Sixteen subjects also completed three extra MI online "
                "sessions and two online sessions with smoothed robotic control."
            ),
            feedback_type="robotic hand",
            stimulus_type="finger cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41467-025-61064-x",
            investigators=["Yidan Ding", "C. Udompanyawit", "Y. Zhang", "Bin He"],
            institution_department="Department of Biomedical Engineering",
            institution="Carnegie Mellon University",
            country="US",
            data_url="https://kilthub.cmu.edu/articles/dataset/29104040",
            publication_year=2025,
            license="CC-BY-4.0",
            funding=["NIH NS124564", "NIH NS131069", "NIH NS127849", "NIH NS096761"],
        ),
        sessions_per_subject=7,
        runs_per_session=32,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["thumb", "index", "middle", "pinky"],
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="~640 offline trials/session (20/run x 32 runs); online "
            "runs add binary/ternary finger trials. Counts vary by subject "
            "(5 subjects completed the base protocol, 16 the full protocol).",
            trials_context=(
                "Offline: 32 runs x 20 trials (5 per finger). Online: sessions "
                "of 32 runs (4 sub-conditions x 8) with 2- or 3-class finger "
                "trials."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["EEGNet"],
            feature_extraction=["deep_learning"],
            frequency_bands={"alpha_mu": [8.0, 13.0], "beta": [13.0, 30.0]},
            spatial_filters=["CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["robotic_hand"], environment="laboratory", online_feedback=True
        ),
        data_processed=False,
        file_format="MAT",
    )

    def __init__(
        self, task="imagery", subjects=None, sessions=None, *, return_all_modalities=False
    ):
        if task not in _TASK_TOKEN:
            raise ValueError(f"task must be one of {sorted(_TASK_TOKEN)}, got {task!r}")
        self.task = task
        super().__init__(
            subjects=list(range(1, 22)),
            sessions_per_subject=7,
            events=dict(_EVENTS),
            code="Ding2025",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1038/s41467-025-61064-x",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}} for one subject, filtered by task arm."""
        base = Path(self.data_path(subject)[0])
        token = _TASK_TOKEN[self.task]
        mat_files = [
            mf for mf in sorted(base.rglob("*.mat")) if token in mf.parent.name.lower()
        ]
        if not mat_files:
            raise FileNotFoundError(
                f"No {self.task} .mat files for subject {subject} in {base}"
            )

        # Group runs by acquisition session (folder with the 2-/3-class and
        # decoder tokens stripped), keyed by the descriptive session name.
        grouped = {}
        for mf in mat_files:
            grouped.setdefault(_session_key(mf.parent.name), []).append(mf)

        sessions = {}
        for session_idx, (sess_key, files) in enumerate(sorted(grouped.items())):
            runs = {}
            for run_idx, mf in enumerate(sorted(files)):
                try:
                    raw = self._load_run(mf)
                except KeyError as e:
                    # A missing expected MATLAB field leaves no trustworthy run
                    # to expose. Any other exception type is re-raised.
                    log.warning("Skipping %s: missing field %s", mf.name, e)
                    continue
                except _NonFiniteEEGDataError as e:
                    # Do not impute raw corruption: filtering would spread it
                    # across time and manufacture signal in subsequent windows.
                    log.warning("Skipping %s: %s", mf.name, e)
                    continue

                description = "".join(
                    char
                    for char in self._run_key(mf, sess_key, subject)
                    if char.isalnum()
                )
                runs[f"{run_idx}Run{description}"] = raw
            if runs:
                description = "".join(char for char in sess_key if char.isalnum())
                sessions[f"{session_idx}Session{description}"] = runs

        if not sessions:
            raise FileNotFoundError(f"No loadable {self.task} runs for subject {subject}")
        return sessions

    @staticmethod
    def _run_key(mat_path, sess_key, subject):
        """Short, unique run key: the file stem minus the SXX and session prefix."""
        stem = mat_path.stem
        prefix = f"S{subject:02d}_"
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
        if stem.startswith(sess_key + "_"):
            stem = stem[len(sess_key) + 1 :]
        return stem or mat_path.stem

    def _load_run(self, mat_path):
        """Load one run ``.mat`` into an MNE Raw with a finger-class stim channel."""
        mat = read_mat(str(mat_path))
        eeg = mat.get("eeg", mat)

        data = np.asarray(eeg["data"], dtype=float)  # (128 x n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] != _N_EEG and data.shape[1] == _N_EEG:
            data = data.T

        # Reject corrupt source data before scaling or any later filtering and
        # windowing. Replacing these values would fabricate EEG; moreover, a
        # temporal filter would spread each invalid sample into neighbouring
        # otherwise-valid data.
        if not np.isfinite(data).all():
            raise _NonFiniteEEGDataError(
                "non-finite eeg.data "
                f"(nan={np.isnan(data).sum()}, "
                f"posinf={np.isposinf(data).sum()}, "
                f"neginf={np.isneginf(data).sum()})"
            )

        labels = eeg.get("label", None)
        if labels is not None:
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            ch_names = [str(c).strip() for c in labels]
        else:
            ch_names = [f"A{i + 1}" for i in range(data.shape[0])]
        # Align channel count and label count defensively.
        if data.shape[0] > len(ch_names):
            data = data[: len(ch_names), :]
        elif data.shape[0] < len(ch_names):
            ch_names = ch_names[: data.shape[0]]

        fs = float(eeg.get("fsample", _SFREQ))

        # BioSemi records in microvolts; scale to volts for MNE.
        if np.abs(data).max() > 1e-3:
            data = data * 1e-6

        # Build a stim channel: each "Target" marker carries its finger class.
        stim = np.zeros((1, data.shape[1]))
        for sample, value in self._target_events(mat.get("event", None)):
            idx = int(sample) - 1  # FieldTrip-style event.sample is 1-based
            if 0 <= idx < data.shape[1]:
                stim[0, idx] = value

        info = mne.create_info(
            ch_names=ch_names + ["STI"],
            ch_types=["eeg"] * len(ch_names) + ["stim"],
            sfreq=fs,
        )
        raw = mne.io.RawArray(
            data=np.concatenate([data, stim], axis=0), info=info, verbose=False
        )
        raw.set_montage(make_standard_montage("biosemi128"), on_missing="ignore")
        return raw

    @staticmethod
    def _target_events(event):
        """Yield (sample, finger_code) for each "Target" marker in ``event``."""
        if not isinstance(event, dict):
            return
        types = event.get("type")
        samples = event.get("sample")
        values = event.get("value")
        if types is None or samples is None or values is None:
            return

        def _listify(x):
            if isinstance(x, np.ndarray):
                return list(x.ravel())
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        types, samples, values = _listify(types), _listify(samples), _listify(values)
        for t, s, v in zip(types, samples, values):
            if str(t).strip().lower() != "target":
                continue
            try:
                sample = float(np.asarray(s).ravel()[0])
                code = int(np.asarray(v).ravel()[0])
            except (ValueError, TypeError, IndexError):
                continue
            yield sample, code

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"S{subject:02d}"

        if subj_dir.exists() and list(subj_dir.rglob("*.mat")):
            return [str(subj_dir)]

        file_id = _FILE_IDS.get(subject)
        if file_id is None:
            raise ValueError(f"No download URL for subject {subject}")

        url = f"{_FIGSHARE_BASE}{file_id}"
        download_and_extract_subject_zip(url, sign, data_dir, path, force_update, verbose)
        return [str(subj_dir)] if subj_dir.exists() else [str(data_dir)]
