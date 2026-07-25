"""Sensory-guided motor-imagery BCI datasets from Wang et al. (2026).

Data DOI: 10.1184/R1/32293995.v1
Paper DOI: 10.1038/s41467-026-75435-5
"""

import io
import re
import shutil
import tempfile
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import mne
import numpy as np
import requests

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
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
from moabb.datasets.utils import safe_extract_zip


_DATASET_DOI = "10.1184/R1/32293995.v1"
_PAPER_DOI = "10.1038/s41467-026-75435-5"
_FIGSHARE_FILE = "https://ndownloader.figshare.com/files/{file_id}"

# Figshare article 32293995, version 1. The release publishes one monolithic
# archive per experimental arm (8.7-25.3 GB each, 63.7 GB in total) with no
# per-subject files, so a single subject is extracted from the *remote* archive
# over HTTP range requests -- see :class:`_RangeReader`. Downloading a whole arm
# to obtain one of its subjects would cost tens of gigabytes of transfer and
# permanent local storage.
#
# ``md5`` records the published version-1 whole-archive hash for provenance, so
# an upstream replacement can be detected by hand. It is deliberately not
# verified here: selective range extraction never reads the whole archive.
# Integrity is instead enforced per member, since :meth:`zipfile.ZipFile.read`
# checks every member against the CRC-32 stored in the central directory.
_ARCHIVES = {
    "BCI2000Control": {
        "file_id": 64710750,
        "filename": "BCI2000Control.zip",
        "md5": "b939407c0073fd16aee46e115245a668",
    },
    "EEGNetControl": {
        "file_id": 64710990,
        "filename": "EEGNetControl.zip",
        "md5": "c67239b074f8a7ed9b299e20cf6cc139",
    },
    "JointLearning": {
        "file_id": 64710999,
        "filename": "JointLearning.zip",
        "md5": "cde1b105f44a85d1b56685de00b4097b",
    },
    "TactileControl": {
        "file_id": 64710993,
        "filename": "TactileControl.zip",
        "md5": "17178b52771ec5beeeec609dddbe2d77",
    },
}

# Counts obtained from the version-1 ZIP central directories. They provide an
# inexpensive completeness check for subject-wise selective extraction.
_SUBJECT_FILE_COUNTS = {
    "BCI2000Control": {1: 55, 2: 31, 3: 43, 4: 55, 5: 55, 6: 43, 7: 55, 8: 55},
    "EEGNetControl": dict.fromkeys(range(1, 9), 37),
    "JointLearning": {
        1: 25,
        2: 25,
        3: 49,
        4: 25,
        5: 25,
        6: 25,
        7: 49,
        8: 49,
        9: 25,
        10: 25,
        11: 37,
        12: 49,
        13: 49,
        14: 37,
        15: 49,
    },
    "TactileControl": {1: 37, 2: 49, 3: 49, 4: 49, 5: 49, 6: 25, 7: 37, 8: 37},
}

_EVENTS = {"left_hand": 1, "right_hand": 2, "hands": 3, "rest": 4}
_EEGNET_EVENT_MAP = {0: 1, 1: 2, 2: 3, 3: 4}
_BCI2000_LR_EVENT_MAP = {1: 2, 2: 1}
_BCI2000_UD_EVENT_MAP = {1: 3, 2: 4}
_BCI2000_2D_EVENT_MAP = {1: 2, 2: 1, 3: 3, 4: 4}

_SFREQ = 1000.0
_TRIAL_DURATION_S = 5.0

# The public README does not state trialSignal's physical unit. BCI2000's
# documented brain-signal interface supplies calibrated microvolts, so the
# values are converted from microvolts to the volts required by MNE. This is an
# evidence-based inference rather than an explicit statement in the release.
_EEG_SCALE = 1e-6

# Canonical MNE capitalization of the 62 released/model-input EEG channels.
# The authors' onlineFeedback.py lists the same channels with mixed-case names.
_CHANNELS = [
    "Fp1",
    "Fpz",
    "Fp2",
    "AF3",
    "AF4",
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
    "PO5",
    "PO3",
    "POz",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "Oz",
    "O2",
    "CB2",
]

_RUN_RE = re.compile(r"_sess(?P<session>\d+)_run(?P<run>\d+)(?P<ud>UD)?\.mat$")

_ACQUISITION = AcquisitionMetadata(
    sampling_rate=_SFREQ,
    n_channels=62,
    channel_types={"eeg": 62},
    sensors=list(_CHANNELS),
    reference="midway between Cz and CPz",
    hardware="Compumedics Neuroscan SynAmps 2/RT",
    software="BCI2000",
    filters={"notch": 60.0},
    line_freq=60.0,
    montage="standard_1005",
    impedance_threshold_kohm=5.0,
    cap_manufacturer="Compumedics Neuroscan",
    cap_model="Quik-Cap",
)

_DOCUMENTATION = DocumentationMetadata(
    doi=_DATASET_DOI,
    investigators=[
        "Hanwen Wang",
        "Yisha Zhang",
        "Maxim Karrenbach",
        "Yidan Ding",
        "Bin He",
    ],
    institution="Carnegie Mellon University",
    institution_department="Department of Biomedical Engineering",
    country="US",
    repository="KiltHub (Figshare)",
    data_url=(
        "https://kilthub.cmu.edu/articles/dataset/"
        "EEG_Datasets_for_Sensory-Guided_Joint_Learning_in_"
        "Motor_Imagery_Brain-Computer_Interfaces/32293995"
    ),
    license="CC-BY-NC-4.0",
    publication_year=2026,
    senior_author="Bin He",
    contact_info=["bhe1@andrew.cmu.edu"],
    associated_paper_doi=_PAPER_DOI,
    funding=["NS124564", "NS131069", "NS127849", "NS096761"],
    keywords=["motor imagery", "BCI", "joint learning", "tactile stimulation", "EEGNet"],
)


def _make_metadata(
    group,
    n_subjects,
    sessions_per_subject,
    feedback_type,
    modalities,
    classifiers,
    feature_extraction,
    spatial_filters=None,
):
    """Create group-specific metadata without duplicating acquisition facts."""
    return DatasetMetadata(
        acquisition=_ACQUISITION,
        participants=ParticipantMetadata(
            n_subjects=n_subjects,
            health_status="healthy",
            bci_experience="naive",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            task_type="online 1D and 2D cursor control",
            n_classes=4,
            class_labels=list(_EVENTS),
            trial_duration=_TRIAL_DURATION_S,
            tasks=[
                "left hand imagery",
                "right hand imagery",
                "bilateral hand imagery",
                "rest",
            ],
            study_design=(
                f"{group} arm of a longitudinal MI-BCI study. Sessions 1-2 used "
                "left/right control, session 3 mixed left/right and up/down, and "
                "sessions 4-6 used 2D control. Sessions 7-8, when present, were "
                "long-term left/right and 2D evaluations."
            ),
            feedback_type=feedback_type,
            stimulus_type="visual cursor and directional cue",
            stimulus_modalities=modalities,
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            stimulus_presentation={"SoftwareName": "BCI2000"},
        ),
        documentation=_DOCUMENTATION,
        # Minimum across subjects, including the separately named run0 baseline.
        sessions_per_subject=sessions_per_subject,
        # The baseline has one file and session 3 can contain separate LR and
        # UD files, so the number of MOABB runs is not constant by session.
        runs_per_session=None,
        contributing_labs=[
            "Carnegie Mellon University Biomedical Functional Imaging and Neuroengineering Lab"
        ],
        n_contributing_labs=1,
        data_processed=True,
        file_format="MATLAB v7.3",
        external_links={
            "source": _DOCUMENTATION.data_url,
            "paper": f"https://doi.org/{_PAPER_DOI}",
            "code": "https://github.com/bfinl/SensoryGuidedJointLearning",
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor Imagery", "Online BCI"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="trial-segmented",
            preprocessing_applied=True,
            preprocessing_steps=["trial segmentation", "auxiliary-channel exclusion"],
            notch_hz=60.0,
            notes=(
                "The release contains 62 selected EEG channels. The README states "
                "that no additional filtering or downsampling was applied to the "
                "released EEG signals."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=classifiers,
            feature_extraction=feature_extraction,
            frequency_bands={"online_EEGNet": [4.0, 40.0]},
            spatial_filters=spatial_filters,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["cursor control"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS),
            imagery_duration_s=_TRIAL_DURATION_S,
        ),
        data_structure=DataStructureMetadata(
            n_trials="varies by run and subject",
            n_blocks=6,
            trials_context=(
                "Each regular session contains six runs; run0 is a separate "
                "initial BCI2000 baseline. Subjects completed 4, 6, or 8 regular sessions."
            ),
        ),
    )


def _mode_label(values):
    """Return the integer mode of a released sample-wise target vector."""
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Target vector contains no finite labels.")
    rounded = np.rint(values).astype(int)
    if not np.allclose(values, rounded):
        raise ValueError("Target vector contains non-integer labels.")
    labels, counts = np.unique(rounded, return_counts=True)
    return int(labels[np.argmax(counts)])


def _read_hdf5_trials(mat_path):
    """Read only signal, target, and sampling-rate fields from a v7.3 MAT file."""
    trials = []
    source_labels = []

    with h5py.File(mat_path, "r") as handle:
        if "runData" not in handle or "trialSignal" not in handle["runData"]:
            raise ValueError(f"{mat_path} does not contain runData.trialSignal.")

        run_data = handle["runData"]
        sfreq = float(np.asarray(run_data["meta/sampling_rate_hz"]).squeeze())
        if not np.isclose(sfreq, _SFREQ):
            raise ValueError(f"Unexpected sampling rate {sfreq:g} Hz in {mat_path}.")

        signal = run_data["trialSignal"]
        if signal.dtype == h5py.ref_dtype or signal.dtype.kind == "O":
            target_name = "trialTargetCode"
            if target_name not in run_data:
                raise ValueError(f"Missing runData.{target_name} in {mat_path}.")
            targets = run_data[target_name]
            signal_refs = signal[()].ravel()
            target_refs = targets[()].ravel()
            if len(signal_refs) != len(target_refs):
                raise ValueError(f"Signal/target trial count mismatch in {mat_path}.")
            for signal_ref, target_ref in zip(signal_refs, target_refs):
                trial = np.asarray(handle[signal_ref], dtype=np.float64)
                if trial.ndim != 2 or trial.shape[0] != len(_CHANNELS):
                    raise ValueError(
                        f"Unexpected variable-trial shape {trial.shape} in {mat_path}."
                    )
                trials.append(trial)
                source_labels.append(_mode_label(handle[target_ref]))
        else:
            target_name = "trialTargetClass"
            if target_name not in run_data:
                raise ValueError(f"Missing runData.{target_name} in {mat_path}.")
            # MATLAB dimensions are reversed in the underlying HDF5 datasets:
            # trials x channels x samples and trials x samples, respectively.
            signal_array = np.asarray(signal, dtype=np.float64)
            target_array = np.asarray(run_data[target_name], dtype=np.float64)
            if signal_array.ndim != 3 or signal_array.shape[1] != len(_CHANNELS):
                raise ValueError(
                    f"Unexpected fixed-trial shape {signal_array.shape} in {mat_path}."
                )
            if target_array.shape[0] != signal_array.shape[0]:
                raise ValueError(f"Signal/target trial count mismatch in {mat_path}.")
            trials.extend(signal_array)
            source_labels.extend(_mode_label(row) for row in target_array)

    return trials, source_labels, target_name


def _event_mapping(mat_path, target_name):
    """Select the documented source-label mapping for a run."""
    if target_name == "trialTargetClass":
        return _EEGNET_EVENT_MAP

    name = Path(mat_path).name
    if name.endswith("UD.mat"):
        return _BCI2000_UD_EVENT_MAP

    match = _RUN_RE.search(name)
    if match and int(match.group("session")) in {4, 5, 6, 8}:
        return _BCI2000_2D_EVENT_MAP
    return _BCI2000_LR_EVENT_MAP


def _make_raw(mat_path):
    """Convert one released run into an MNE RawArray with a stim channel."""
    trials, source_labels, target_name = _read_hdf5_trials(mat_path)
    mapping = _event_mapping(mat_path, target_name)
    try:
        event_codes = [mapping[label] for label in source_labels]
    except KeyError as exc:
        raise ValueError(
            f"Undocumented target label {exc.args[0]} in {mat_path}; expected {sorted(mapping)}."
        ) from exc

    # MOABB requires one fixed epoch interval. BCI2000 cursor-control trials may
    # end early when a target boundary is reached, so each trial is isolated and
    # only its missing tail is zero-padded to the nominal five seconds. One extra
    # sample prevents the inclusive 5 s epoch endpoint from touching the next trial.
    nominal_samples = int(round(_TRIAL_DURATION_S * _SFREQ)) + 1
    segment_lengths = [max(trial.shape[1], nominal_samples) + 1 for trial in trials]
    # Keep sample zero clear because MNE's default stim-event detection ignores
    # a non-zero initial value (there is no preceding transition to detect).
    total_samples = 1 + sum(segment_lengths)
    data = np.zeros((len(_CHANNELS) + 1, total_samples), dtype=np.float64)

    offset = 1
    nonfinite_counts = []
    for trial_index, (trial, event_code, segment_length) in enumerate(
        zip(trials, event_codes, segment_lengths)
    ):
        scaled_trial = trial * _EEG_SCALE
        nonfinite = ~np.isfinite(scaled_trial)
        if nonfinite.any():
            nonfinite_counts.append((trial_index, int(nonfinite.sum())))
            scaled_trial[nonfinite] = 0.0

        data[:-1, offset : offset + trial.shape[1]] = scaled_trial
        data[-1, offset] = event_code
        offset += segment_length

    if nonfinite_counts:
        total_nonfinite = sum(count for _, count in nonfinite_counts)
        affected_trials = ", ".join(
            f"{trial_index} ({count})" for trial_index, count in nonfinite_counts
        )
        warnings.warn(
            f"Replaced {total_nonfinite} non-finite EEG sample(s) with zero in "
            f"{mat_path}; affected zero-based trial(s): {affected_trials}.",
            RuntimeWarning,
            stacklevel=2,
        )

    info = mne.create_info(
        ch_names=[*_CHANNELS, "stim"],
        sfreq=_SFREQ,
        ch_types=["eeg"] * len(_CHANNELS) + ["stim"],
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def _path_sort_key(path):
    """Sort baseline first, then upstream session/run and LR before UD."""
    path = Path(path)
    if path.name.endswith("_run0.mat"):
        return (0, 0, 0)
    match = _RUN_RE.search(path.name)
    if match is None:
        raise ValueError(f"Unexpected Wang2026 filename: {path.name}")
    return (
        int(match.group("session")),
        int(match.group("run")),
        int(match.group("ud") is not None),
    )


class _RangeReader(io.RawIOBase):
    """Read-only file object backed by HTTP range requests.

    The release is published as four monolithic archives (8.7-25.3 GB, 63.7 GB
    total) with no per-subject files, so fetching one subject would otherwise
    mean downloading its whole group. figshare's storage honours range requests,
    which lets us pull just that subject's members out of the remote zip.
    Each request is re-resolved because figshare presigns the redirect target
    with a very short expiry.
    """

    def __init__(self, url):
        self.url = url
        self.pos = 0
        r = requests.get(
            url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60
        )
        r.raise_for_status()
        if "Content-Range" not in r.headers:
            raise OSError(f"server does not support range requests for {url}")
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n - 1, self.size - 1)
        r = requests.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            allow_redirects=True,
            timeout=300,
        )
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data


class _Wang2026Base(BaseDataset):
    """Shared implementation for one experimental arm."""

    _group = None

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        if self._group is None:
            raise TypeError("_Wang2026Base cannot be instantiated directly.")
        subject_list = sorted(_SUBJECT_FILE_COUNTS[self._group])
        super().__init__(
            subjects=subject_list,
            sessions_per_subject=self.METADATA.sessions_per_subject,
            events=dict(_EVENTS),
            code=self.__class__.__name__,
            interval=[0, _TRIAL_DURATION_S],
            paradigm="imagery",
            doi=_DATASET_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return all available sessions and runs for one subject."""
        paths = sorted(map(Path, self.data_path(subject)), key=_path_sort_key)
        sessions = defaultdict(dict)

        for mat_path in paths:
            if mat_path.name.endswith("_run0.mat"):
                session_key = "0baseline"
            else:
                match = _RUN_RE.search(mat_path.name)
                if match is None:
                    raise ValueError(f"Unexpected Wang2026 filename: {mat_path.name}")
                session_key = str(int(match.group("session")))

            run_key = str(len(sessions[session_key]))
            sessions[session_key][run_key] = _make_raw(mat_path)

        if not sessions:
            raise FileNotFoundError(
                f"No data found for subject {subject} in {self._group}."
            )
        return dict(sessions)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Extract one subject's runs out of the remote group archive.

        Only the archive members belonging to ``subject`` are transferred, using
        HTTP range requests against the published ZIP (see :class:`_RangeReader`
        and the note on :data:`_ARCHIVES`). The whole-archive download is
        deliberately avoided: one subject would otherwise cost 8.7-25.3 GB.
        """
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject {subject}. Valid subjects: {self.subject_list}"
            )

        root = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        root.mkdir(parents=True, exist_ok=True)
        subject_name = f"S{subject:03d}"
        subject_dir = root / self._group / subject_name
        expected_count = _SUBJECT_FILE_COUNTS[self._group][subject]

        existing = sorted(subject_dir.glob("*.mat")) if subject_dir.is_dir() else []
        if len(existing) == expected_count and not force_update:
            return [str(file_path) for file_path in existing]

        archive = _ARCHIVES[self._group]
        url = _FIGSHARE_FILE.format(file_id=archive["file_id"])
        prefix = f"{self._group}/{subject_name}/"

        # Extract into a temporary directory and move into place only once every
        # expected run is present, so an interrupted transfer cannot leave behind
        # a partial subject that the cache check above would accept as complete.
        with tempfile.TemporaryDirectory(prefix="wang2026-", dir=root) as tmp:
            tmp_root = Path(tmp)
            with zipfile.ZipFile(_RangeReader(url)) as zip_file:
                members = [
                    member
                    for member in zip_file.infolist()
                    if member.filename.startswith(prefix)
                    and member.filename.endswith(".mat")
                ]
                if len(members) != expected_count:
                    raise RuntimeError(
                        f"{archive['filename']} contains {len(members)} MAT files for "
                        f"{subject_name}, expected {expected_count}."
                    )
                safe_extract_zip(zip_file, tmp_root, members=members)

            extracted_dir = tmp_root / self._group / subject_name
            extracted = sorted(extracted_dir.glob("*.mat"))
            if len(extracted) != expected_count:
                raise RuntimeError(
                    f"Incomplete extraction for {self._group}/{subject_name}: "
                    f"found {len(extracted)} files, expected {expected_count}."
                )
            subject_dir.parent.mkdir(parents=True, exist_ok=True)
            if subject_dir.exists():
                shutil.rmtree(subject_dir)
            shutil.move(str(extracted_dir), str(subject_dir))

        return [str(file_path) for file_path in sorted(subject_dir.glob("*.mat"))]


class Wang2026JointLearning(_Wang2026Base):
    """Sensory-guided joint-learning arm from Wang et al. (2026).

    Fifteen BCI-naive participants received performance-contingent tactile
    guidance in paired Copy/New trials. A weighted EEGNet decoder was updated
    between training runs. Subjects completed four, six, or eight regular
    sessions, in addition to the initial BCI2000 baseline run.

    Notes
    -----
    The public release contains trial-segmented data. Variable-length BCI2000
    baseline trials are zero-padded only after their recorded endpoint so that
    MOABB can expose the nominal five-second imagery interval.

    Each experimental arm is published as one monolithic archive (8.7-25.3 GB),
    so requesting a single subject transfers only that subject's members out of
    the remote ZIP using HTTP range requests rather than downloading the whole
    arm.

    References
    ----------
    .. [1] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
           Sensory-guided human-machine joint learning accelerates the
           acquisition of motor imagery brain computer interface control.
           Nature Communications, 17, 6177.
           https://doi.org/10.1038/s41467-026-75435-5
    .. [2] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
           EEG Datasets for Sensory-Guided Joint Learning in Motor Imagery
           Brain-Computer Interfaces. Carnegie Mellon University.
           https://doi.org/10.1184/R1/32293995.v1
    """

    _group = "JointLearning"
    METADATA = _make_metadata(
        group=_group,
        n_subjects=15,
        sessions_per_subject=5,
        feedback_type="visual cursor, selective tactile guidance, and Copy/New instructions",
        modalities=["visual", "tactile"],
        classifiers=["weighted EEGNet"],
        feature_extraction=["EEGNet"],
        spatial_filters=["CSP-based initial sample weighting"],
    )


class Wang2026TactileControl(_Wang2026Base):
    """Tactile-control arm from Wang et al. (2026).

    Eight BCI-naive participants received tactile stimulation on every training
    trial, independent of performance, with conventional adaptive EEGNet
    feedback and no Copy/New instruction.

    See :class:`Wang2026JointLearning` for the common acquisition and data
    release description.
    """

    _group = "TactileControl"
    METADATA = _make_metadata(
        group=_group,
        n_subjects=8,
        sessions_per_subject=5,
        feedback_type="visual cursor and non-contingent tactile stimulation",
        modalities=["visual", "tactile"],
        classifiers=["EEGNet"],
        feature_extraction=["EEGNet"],
    )


class Wang2026EEGNetControl(_Wang2026Base):
    """Independent EEGNet-control cohort from Wang et al. (2026).

    Eight BCI-naive participants completed six regular sessions using a
    conventional adaptive EEGNet decoder, randomized non-pairwise trials, and
    no tactile stimulation or Copy/New instruction.

    See :class:`Wang2026JointLearning` for the common acquisition and data
    release description.
    """

    _group = "EEGNetControl"
    METADATA = _make_metadata(
        group=_group,
        n_subjects=8,
        sessions_per_subject=7,
        feedback_type="visual cursor with adaptive EEGNet",
        modalities=["visual"],
        classifiers=["EEGNet"],
        feature_extraction=["EEGNet"],
    )


class Wang2026BCI2000Control(_Wang2026Base):
    """BCI2000-control arm from Wang et al. (2026).

    Eight BCI-naive participants received no tactile stimulation or strategic
    instruction. The first six sessions used a fixed BCI2000 autoregressive
    decoder; available long-term sessions used EEGNet. Main-session trials are
    stored as variable-length cell arrays, while long-term runs use the fixed
    EEGNet-style arrays.

    See :class:`Wang2026JointLearning` for the common acquisition and data
    release description.
    """

    _group = "BCI2000Control"
    METADATA = _make_metadata(
        group=_group,
        n_subjects=8,
        sessions_per_subject=5,
        feedback_type="visual cursor with fixed BCI2000 autoregressive decoder",
        modalities=["visual"],
        classifiers=["BCI2000 linear classifier", "EEGNet (long-term sessions)"],
        feature_extraction=["autoregressive spectral features", "EEGNet"],
    )
