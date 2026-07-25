"""Sensory-guided motor-imagery BCI datasets from Wang et al. (2026).

Data DOI: 10.1184/R1/32293995.v1
Paper DOI: 10.1038/s41467-026-75435-5
"""

import io
import json
import re
import shutil
import tempfile
import warnings
import zipfile
import zlib
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import MappingProxyType

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

_GROUP_NAMES = MappingProxyType(
    {
        "joint_learning": "JointLearning",
        "bci2000_control": "BCI2000Control",
        "tactile_control": "TactileControl",
        "eegnet_control": "EEGNetControl",
    }
)

# The paper's primary randomized cohort occupies global IDs 1--31; the
# separately recruited EEGNet-control cohort occupies IDs 32--39. Source IDs
# restart inside every cohort archive, so the group is part of provenance.
_subject_mapping = {}
for _global_subject, _source_subject in zip(range(1, 16), range(1, 16)):
    _subject_mapping[_global_subject] = ("JointLearning", f"S{_source_subject:03d}")
for _global_subject, _source_subject in zip(range(16, 24), range(1, 9)):
    _subject_mapping[_global_subject] = ("BCI2000Control", f"S{_source_subject:03d}")
for _global_subject, _source_subject in zip(range(24, 32), range(1, 9)):
    _subject_mapping[_global_subject] = ("TactileControl", f"S{_source_subject:03d}")
for _global_subject, _source_subject in zip(range(32, 40), range(1, 9)):
    _subject_mapping[_global_subject] = ("EEGNetControl", f"S{_source_subject:03d}")
_SUBJECT_MAPPING = MappingProxyType(_subject_mapping)
del _subject_mapping, _global_subject, _source_subject

_EVENTS = {"left_hand": 1, "right_hand": 2, "hands": 3, "rest": 4}
_EEGNET_EVENT_MAP = {0: 1, 1: 2, 2: 3, 3: 4}
_BCI2000_LR_EVENT_MAP = {1: 2, 2: 1}
_BCI2000_UD_EVENT_MAP = {1: 3, 2: 4}
_BCI2000_2D_EVENT_MAP = {1: 2, 2: 1, 3: 3, 4: 4}

_SFREQ = 1000.0
_TRIAL_DURATION_S = 5.0
_NOMINAL_TRIAL_SAMPLES = int(_TRIAL_DURATION_S * _SFREQ)
_TRIAL_GUARD_SAMPLES = int(2.0 * _SFREQ)
_RANGE_BLOCK_SIZE = 8 * 1024 * 1024
_RANGE_CACHE_BLOCKS = 2
_MANIFEST_FILENAME = ".wang2026-manifest.json"

_CALIBRATION_ERROR = (
    "Wang2026 Raw construction is disabled because the public release does not "
    "document trialSignal's physical calibration. MNE Raw objects require volts, "
    "but the release metadata for DOI 10.1184/R1/32293995.v1 provides neither a "
    "signal unit nor the BCI2000 SourceChGain/SourceChOffset (or Neuroscan "
    "fResolution) needed to convert stored values. mne.io.read_raw_bci2k cannot "
    "recover this information: the release contains segmented MATLAB files rather "
    "than original BCI2000 .dat files, and MNE's BCI2000 reader does not apply "
    "gain/scaling. Obtain authoritative calibration metadata from the data authors "
    "before enabling public Raw construction."
)

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


def _make_metadata():
    """Create metadata for the complete release, including all four cohorts."""
    return DatasetMetadata(
        acquisition=_ACQUISITION,
        participants=ParticipantMetadata(
            n_subjects=39,
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
                "One longitudinal MI-BCI study release with 39 unique participants. "
                "The primary randomized experiment included 31 participants: joint "
                "learning (n=15), BCI2000 control (n=8), and tactile control (n=8). "
                "A separately recruited EEGNet-control cohort added 8 participants. "
                "Sessions 1-2 used left/right control, session 3 mixed left/right "
                "and up/down, and sessions 4-6 used 2D control. Sessions 7-8, when "
                "present, were long-term left/right and 2D evaluations."
            ),
            feedback_type=(
                "visual cursor feedback, with cohort-specific tactile guidance, "
                "EEGNet adaptation, or fixed BCI2000 decoding"
            ),
            stimulus_type="visual cursor and directional cue",
            stimulus_modalities=["visual", "tactile"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            stimulus_presentation={"SoftwareName": "BCI2000"},
        ),
        documentation=_DOCUMENTATION,
        # Minimum across the release, including the separately named baseline.
        sessions_per_subject=5,
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
            preprocessing_steps=[
                "trial segmentation",
                "auxiliary-channel exclusion",
                "per-trial temporal centering in the MOABB reconstruction",
            ],
            notch_hz=60.0,
            notes=(
                "The release contains 62 selected EEG channels. The README states "
                "that no additional filtering or downsampling was applied to the "
                "released EEG signals. Variable-length BCI2000 trials are represented "
                "with explicitly annotated zero padding up to the nominal five-second "
                "window. Public Raw construction remains disabled until authoritative "
                "physical calibration metadata is available."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["weighted EEGNet", "EEGNet", "BCI2000 linear classifier"],
            feature_extraction=["EEGNet", "autoregressive spectral features"],
            frequency_bands={"online_EEGNet": [4.0, 40.0]},
            spatial_filters=["CSP-based initial sample weighting"],
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
                "initial BCI2000 baseline. Subjects completed 4, 6, or 8 regular "
                "sessions. The 31 randomized participants and 8 independently "
                "recruited EEGNet-control participants share this release."
            ),
        ),
    )


def _strict_trial_label(values, allowed_labels):
    """Return one unambiguous integer label from a trial target track."""
    values = np.asarray(values).ravel()
    if values.size == 0:
        raise ValueError("Target vector is empty.")
    if not np.isfinite(values).all():
        raise ValueError("Target vector contains non-finite labels.")
    rounded = np.rint(values).astype(int)
    if not np.allclose(values, rounded):
        raise ValueError("Target vector contains non-integer labels.")
    labels = np.unique(rounded)
    if len(labels) != 1:
        raise ValueError(f"Target vector contains mixed labels: {labels.tolist()}.")
    label = int(labels[0])
    if label not in allowed_labels:
        raise ValueError(
            f"Target label {label} is not valid; expected one of "
            f"{sorted(allowed_labels)}."
        )
    return label


def _read_matlab_strings(handle, node):
    """Decode simple MATLAB v7.3 strings, including cell-array references."""
    values = np.asarray(node[()])
    reference_dtype = h5py.check_dtype(ref=values.dtype)
    if reference_dtype is not None:
        strings = []
        for reference in values.ravel():
            if reference:
                strings.extend(_read_matlab_strings(handle, handle[reference]))
        return strings
    if values.dtype.kind in {"S", "U"}:
        decoded = []
        for value in values.ravel():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            decoded.append(str(value))
        return [value for value in decoded if value]
    if values.dtype.kind in {"u", "i"}:
        text = "".join(chr(int(value)) for value in values.ravel() if int(value))
        return [text] if text else []
    if values.dtype.kind == "O":
        decoded = []
        for value in values.ravel():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            decoded.append(str(value))
        return [value for value in decoded if value]
    raise ValueError(f"Cannot decode MATLAB text with dtype {values.dtype}.")


def _fallback_event_mapping(mat_path, target_name):
    """Infer a mapping from storage field and filename for validation/fallback."""
    if target_name == "trialTargetClass":
        return _EEGNET_EVENT_MAP
    name = Path(mat_path).name
    if name.endswith("UD.mat"):
        return _BCI2000_UD_EVENT_MAP
    match = _RUN_RE.search(name)
    if match and int(match.group("session")) in {4, 5, 6, 8}:
        return _BCI2000_2D_EVENT_MAP
    return _BCI2000_LR_EVENT_MAP


def _mapping_from_task_type(task_type):
    """Map the release's authoritative ``meta.task_type`` to event semantics."""
    normalized = re.sub(r"[^a-z0-9]+", " ", task_type.casefold()).strip()
    tokens = set(normalized.split())
    if "eegnet" in normalized or {"eeg", "net"} <= tokens:
        return _EEGNET_EVENT_MAP
    if "bci2000" not in normalized:
        raise ValueError(f"Undocumented Wang2026 task_type {task_type!r}.")
    if "2d" in tokens or {"2", "d"} <= tokens:
        return _BCI2000_2D_EVENT_MAP
    if "up down" in normalized or "ud" in tokens:
        return _BCI2000_UD_EVENT_MAP
    if "left right" in normalized or "lr" in tokens:
        return _BCI2000_LR_EVENT_MAP
    raise ValueError(f"Undocumented Wang2026 task_type {task_type!r}.")


def _event_mapping(mat_path, target_name, task_type=None):
    """Select label semantics from metadata and validate the filename fallback."""
    fallback = _fallback_event_mapping(mat_path, target_name)
    if task_type is None:
        raise ValueError(
            f"{mat_path} has no runData.meta.task_type; label semantics cannot be "
            "validated from the release's authoritative task discriminator."
        )

    mapping = _mapping_from_task_type(task_type)
    expected_target = (
        "trialTargetClass" if mapping is _EEGNET_EVENT_MAP else "trialTargetCode"
    )
    if target_name != expected_target:
        raise ValueError(
            f"runData.meta.task_type {task_type!r} requires {expected_target}, "
            f"but {mat_path} contains {target_name}."
        )
    if mapping is not fallback:
        raise ValueError(
            f"runData.meta.task_type {task_type!r} disagrees with the validated "
            f"filename/field fallback for {Path(mat_path).name}."
        )
    return mapping


def _target_tracks(target_array, n_trials, mat_path):
    """Align a fixed target array into one sample-wise track per trial."""
    target_array = np.asarray(target_array, dtype=np.float64)
    if target_array.ndim == 1:
        if target_array.size != n_trials:
            raise ValueError(f"Signal/target trial count mismatch in {mat_path}.")
        return [target_array[index : index + 1] for index in range(n_trials)]
    if target_array.shape[0] == n_trials:
        return list(target_array)
    if target_array.ndim == 2 and target_array.shape[1] == n_trials:
        return list(target_array.T)
    raise ValueError(f"Signal/target trial count mismatch in {mat_path}.")


def _trial_info_labels(run_data, n_trials, allowed_labels, mat_path):
    """Read direct ``trialInfo.target_label`` values when HDF5 exposes them."""
    if "trialInfo" not in run_data:
        return None
    trial_info = run_data["trialInfo"]
    if not isinstance(trial_info, h5py.Group) or "target_label" not in trial_info:
        return None
    values = np.asarray(trial_info["target_label"], dtype=np.float64).ravel()
    if len(values) != n_trials:
        raise ValueError(f"runData.trialInfo.target_label count mismatch in {mat_path}.")
    return [_strict_trial_label([value], allowed_labels) for value in values]


def _read_hdf5_trials(mat_path):
    """Read and validate signal and target fields from a v7.3 MAT run."""
    trials = []
    source_labels = []

    with h5py.File(mat_path, "r") as handle:
        if "runData" not in handle or "trialSignal" not in handle["runData"]:
            raise ValueError(f"{mat_path} does not contain runData.trialSignal.")
        run_data = handle["runData"]
        if "meta" not in run_data:
            raise ValueError(f"{mat_path} does not contain runData.meta.")
        meta = run_data["meta"]
        if "sampling_rate_hz" not in meta:
            raise ValueError(f"{mat_path} has no runData.meta.sampling_rate_hz.")
        sfreq = float(np.asarray(meta["sampling_rate_hz"]).squeeze())
        if not np.isclose(sfreq, _SFREQ):
            raise ValueError(f"Unexpected sampling rate {sfreq:g} Hz in {mat_path}.")

        if "selected_channels" not in meta:
            raise ValueError(f"{mat_path} has no runData.meta.selected_channels.")
        selected_channels = _read_matlab_strings(handle, meta["selected_channels"])
        if [channel.casefold() for channel in selected_channels] != [
            channel.casefold() for channel in _CHANNELS
        ]:
            raise ValueError(
                f"Unexpected channel order in {mat_path}; expected the 62-channel "
                "published model-input order."
            )

        task_type = None
        if "task_type" in meta:
            task_types = _read_matlab_strings(handle, meta["task_type"])
            if len(task_types) != 1:
                raise ValueError(f"Invalid runData.meta.task_type in {mat_path}.")
            task_type = task_types[0]

        signal = run_data["trialSignal"]
        is_reference_array = h5py.check_dtype(ref=signal.dtype) is not None
        target_name = "trialTargetCode" if is_reference_array else "trialTargetClass"
        if target_name not in run_data:
            raise ValueError(f"Missing runData.{target_name} in {mat_path}.")
        mapping = _event_mapping(mat_path, target_name, task_type)
        allowed_labels = set(mapping)

        if is_reference_array:
            targets = run_data[target_name]
            signal_refs = signal[()].ravel()
            target_refs = targets[()].ravel()
            if len(signal_refs) != len(target_refs):
                raise ValueError(f"Signal/target trial count mismatch in {mat_path}.")
            for signal_ref, target_ref in zip(signal_refs, target_refs):
                trial = np.asarray(handle[signal_ref], dtype=np.float64)
                if (
                    trial.ndim != 2
                    or trial.shape[0] != len(_CHANNELS)
                    or trial.shape[1] == 0
                ):
                    raise ValueError(
                        f"Unexpected variable-trial shape {trial.shape} in {mat_path}."
                    )
                trials.append(trial)
                source_labels.append(
                    _strict_trial_label(handle[target_ref], allowed_labels)
                )
        else:
            signal_array = np.asarray(signal, dtype=np.float64)
            if (
                signal_array.ndim != 3
                or signal_array.shape[1] != len(_CHANNELS)
                or signal_array.shape[2] == 0
            ):
                raise ValueError(
                    f"Unexpected fixed-trial shape {signal_array.shape} in {mat_path}."
                )
            target_tracks = _target_tracks(
                run_data[target_name], signal_array.shape[0], mat_path
            )
            trials.extend(signal_array)
            source_labels.extend(
                _strict_trial_label(track, allowed_labels) for track in target_tracks
            )

        info_labels = _trial_info_labels(run_data, len(trials), allowed_labels, mat_path)
        if info_labels is not None and info_labels != source_labels:
            raise ValueError(
                f"runData.trialInfo.target_label disagrees with the sample-wise "
                f"target track in {mat_path}."
            )

    return trials, [mapping[label] for label in source_labels]


def _validated_eeg_scale(eeg_scale):
    if eeg_scale is None:
        raise RuntimeError(_CALIBRATION_ERROR)
    eeg_scale = float(eeg_scale)
    if not np.isfinite(eeg_scale) or eeg_scale <= 0:
        raise ValueError("_eeg_scale must be a finite positive volts-per-stored-unit.")
    return eeg_scale


def _make_raw(mat_path, *, _eeg_scale=None):
    """Reconstruct one run; ``_eeg_scale`` is private and used only by tests."""
    eeg_scale = _validated_eeg_scale(_eeg_scale)
    trials, event_codes = _read_hdf5_trials(mat_path)

    segment_length = _NOMINAL_TRIAL_SAMPLES + _TRIAL_GUARD_SAMPLES
    # Keep sample zero clear because MNE event detection needs a transition.
    total_samples = 1 + len(trials) * segment_length
    data = np.zeros((len(_CHANNELS) + 1, total_samples), dtype=np.float64)
    padding_onsets = []
    padding_durations = []
    guard_onsets = []
    nonfinite_counts = []

    offset = 1
    for trial_index, (trial, event_code) in enumerate(zip(trials, event_codes)):
        # BCI exports can include the endpoint at exactly 5.000 s. MOABB's
        # [0, 4.999] s convention is exactly 5000 samples, so center and retain
        # only that analysis window. This prevents a discarded endpoint from
        # influencing the samples exposed to a paradigm.
        analysis_trial = trial[:, :_NOMINAL_TRIAL_SAMPLES]
        finite = np.isfinite(analysis_trial)
        finite_per_channel = finite.sum(axis=1)
        if np.any(finite_per_channel == 0):
            raise ValueError(
                f"Trial {trial_index} in {mat_path} has a channel with no finite samples."
            )
        if not finite.all():
            nonfinite_counts.append((trial_index, int((~finite).sum())))
        channel_means = (
            np.where(finite, analysis_trial, 0.0).sum(axis=1) / finite_per_channel
        )
        centered = np.where(finite, analysis_trial - channel_means[:, None], 0.0)
        scaled_trial = centered * eeg_scale

        observed_samples = analysis_trial.shape[1]
        data[:-1, offset : offset + observed_samples] = scaled_trial
        data[-1, offset] = event_code
        if observed_samples < _NOMINAL_TRIAL_SAMPLES:
            padding_onsets.append((offset + observed_samples) / _SFREQ)
            padding_durations.append((_NOMINAL_TRIAL_SAMPLES - observed_samples) / _SFREQ)
        guard_onsets.append((offset + _NOMINAL_TRIAL_SAMPLES) / _SFREQ)
        offset += segment_length

    if nonfinite_counts:
        total_nonfinite = sum(count for _, count in nonfinite_counts)
        affected_trials = ", ".join(
            f"{trial_index} ({count})" for trial_index, count in nonfinite_counts
        )
        warnings.warn(
            f"Replaced {total_nonfinite} non-finite EEG sample(s) with zero after "
            f"per-trial centering in {mat_path}; affected zero-based trial(s): "
            f"{affected_trials}.",
            RuntimeWarning,
            stacklevel=2,
        )

    info = mne.create_info(
        ch_names=[*_CHANNELS, "stim"],
        sfreq=_SFREQ,
        ch_types=["eeg"] * len(_CHANNELS) + ["stim"],
    )
    info["line_freq"] = 60.0
    info["description"] = (
        "Wang2026 trial reconstruction: each exact 5000-sample analysis window is "
        "temporally centered; short trials have annotated zero padding; inclusive "
        "5.000-second endpoints are discarded; trials are separated by annotated "
        "2000-sample acquisition gaps."
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[*padding_onsets, *guard_onsets],
            duration=[
                *padding_durations,
                *([_TRIAL_GUARD_SAMPLES / _SFREQ] * len(guard_onsets)),
            ],
            description=[
                # ``BAD_`` makes MNE reject epochs that would otherwise treat
                # the synthetic tail as observed EEG.
                *(["BAD_WANG2026_ZERO_PADDED"] * len(padding_onsets)),
                # MNE's default filtering policy treats BAD_ACQ_SKIP as a hard
                # boundary, rather than filtering across the zero-filled guard.
                *(["BAD_ACQ_SKIP"] * len(guard_onsets)),
            ],
        )
    )
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
    """Buffered random access over strict HTTP byte-range responses."""

    def __init__(self, url, *, block_size=None, cache_blocks=_RANGE_CACHE_BLOCKS):
        self.url = url
        self.pos = 0
        self.block_size = _RANGE_BLOCK_SIZE if block_size is None else int(block_size)
        self.cache_blocks = int(cache_blocks)
        if self.block_size <= 0 or self.cache_blocks <= 0:
            raise ValueError("block_size and cache_blocks must be positive.")
        self._cache = OrderedDict()
        self.request_count = 0
        self.bytes_transferred = 0
        _, self.size = self._request_range(0, 0, expected_total=None, timeout=60)

    def _request_range(self, start, end, *, expected_total, timeout):
        """Fetch exactly one range without ever accepting an unbounded 200 body."""
        self.request_count += 1
        response = requests.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            allow_redirects=True,
            timeout=timeout,
            stream=True,
        )
        try:
            status = getattr(response, "status_code", None)
            if status == 416:
                raise OSError(
                    f"HTTP 416 for requested byte range {start}-{end} from {self.url}."
                )
            if status != 206:
                raise OSError(
                    f"Expected HTTP 206 for byte range {start}-{end} from {self.url}; "
                    f"got {status!r}."
                )

            content_range = response.headers.get("Content-Range")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
            if match is None:
                raise OSError(
                    f"Server returned malformed Content-Range {content_range!r} "
                    f"for {self.url}."
                )
            actual_start, actual_end, total = map(int, match.groups())
            if (actual_start, actual_end) != (start, end):
                raise OSError(
                    f"Content-Range {content_range!r} does not match requested "
                    f"bytes {start}-{end}."
                )
            if total <= actual_end:
                raise OSError(
                    f"Content-Range {content_range!r} has an invalid total size."
                )
            if expected_total is not None and total != expected_total:
                raise OSError(
                    f"Remote total size changed from {expected_total} to {total} "
                    f"while reading {self.url}."
                )

            expected_length = end - start + 1
            content = bytearray()
            for chunk in response.iter_content(
                chunk_size=min(1024 * 1024, expected_length + 1)
            ):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > expected_length:
                    raise OSError(
                        f"Range payload length exceeds expected {expected_length} "
                        f"bytes for {start}-{end}."
                    )
            if len(content) != expected_length:
                raise OSError(
                    f"Range payload length {len(content)} does not match expected "
                    f"{expected_length} bytes for {start}-{end}."
                )
            self.bytes_transferred += len(content)
            return bytes(content), total
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    def _get_block(self, block_start):
        try:
            block = self._cache.pop(block_start)
        except KeyError:
            block_end = min(block_start + self.block_size, self.size) - 1
            block, _ = self._request_range(
                block_start, block_end, expected_total=self.size, timeout=300
            )
        self._cache[block_start] = block
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return block

    def seek(self, off, whence=0):
        if whence not in {0, 1, 2}:
            raise ValueError(f"Invalid whence {whence}; expected 0, 1, or 2.")
        new_pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        if new_pos < 0:
            raise ValueError("Cannot seek before the start of a remote file.")
        if new_pos > self.size:
            raise ValueError("Cannot seek past end of a remote file.")
        self.pos = new_pos
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n == 0:
            return b""
        if self.pos == self.size:
            return b""
        remaining = (
            self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        )
        chunks = []
        while remaining:
            block_start = (self.pos // self.block_size) * self.block_size
            block = self._get_block(block_start)
            within_block = self.pos - block_start
            take = min(remaining, len(block) - within_block)
            if take <= 0:
                raise OSError(f"Empty read-ahead block at byte {self.pos}.")
            chunks.append(block[within_block : within_block + take])
            self.pos += take
            remaining -= take
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _crc32(path):
    checksum = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def _manifest_for_members(group, source_subject, archive, members):
    return {
        "schema_version": 1,
        "dataset_doi": _DATASET_DOI,
        "group": group,
        "source_subject": source_subject,
        "archive": {
            "file_id": archive["file_id"],
            "filename": archive["filename"],
            "md5": archive["md5"],
        },
        "members": [
            {
                "name": Path(member.filename).name,
                "size": member.file_size,
                "crc32": f"{member.CRC:08x}",
            }
            for member in sorted(members, key=lambda member: member.filename)
        ],
    }


def _cache_matches_manifest(subject_dir, expected_count, expected_identity):
    """Validate archive provenance, local member sizes, and CRC-32 values."""
    if not subject_dir.is_dir():
        return False
    files = sorted(subject_dir.glob("*.mat"))
    if len(files) != expected_count:
        return False
    try:
        manifest = json.loads(
            (subject_dir / _MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        if {
            key: manifest[key]
            for key in (
                "schema_version",
                "dataset_doi",
                "group",
                "source_subject",
                "archive",
            )
        } != expected_identity:
            return False
        members = manifest["members"]
        if len(members) != expected_count:
            return False
        if {member["name"] for member in members} != {path.name for path in files}:
            return False
        for member in members:
            path = subject_dir / member["name"]
            if path.stat().st_size != member["size"]:
                return False
            if _crc32(path) != member["crc32"]:
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


class Wang2026(BaseDataset):
    """Sensory-guided joint-learning MI-BCI study from Wang et al. (2026).

    The release contains one study dataset with 39 globally unique participants.
    Thirty-one participants were randomized to joint-learning (15), BCI2000
    control (8), or tactile-control (8) conditions. Eight independently
    recruited participants form the EEGNet-control cohort. ``group`` may filter
    these cohorts without changing their global MOABB subject IDs.

    Raw construction is intentionally blocked because the release does not
    document the physical unit or gain/offset required to convert ``trialSignal``
    into volts. :meth:`data_path` remains available for explicit archive parsing
    and provenance work, but :meth:`get_data` fails before network access until
    authoritative calibration metadata is published. MNE's BCI2000 reader is not
    applicable here because the release contains segmented MATLAB files instead
    of the original BCI2000 ``.dat`` recordings, and that reader does not apply
    BCI2000 gain/scaling.

    Parameters
    ----------
    group : {"all", "joint_learning", "bci2000_control", "tactile_control", \
            "eegnet_control"}
        Cohort filter. IDs always retain the paper-first global mapping.
    subjects : list of int | None
        Optional global subject IDs within ``group``.
    sessions : list of int or str | None
        Optional sessions to retain.

    Notes
    -----
    Released runs are trial-segmented. The private reconstruction helper
    temporally centers each trial, uses an exact 5000-sample interval
    (0--4.999 s), annotates the zero-padded tail of short BCI2000 trials, and
    inserts a 2000-sample guard between trials to prevent cross-trial filtering.

    References
    ----------
    .. [1] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
       Sensory-guided human-machine joint learning accelerates the acquisition
       of motor imagery brain computer interface control. Nature
       Communications, 17, 6177.
       https://doi.org/10.1038/s41467-026-75435-5
    .. [2] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
       EEG Datasets for Sensory-Guided Joint Learning in Motor Imagery
       Brain-Computer Interfaces. Carnegie Mellon University.
       https://doi.org/10.1184/R1/32293995.v1
    """

    METADATA = _make_metadata()

    def __init__(
        self, group="all", subjects=None, sessions=None, *, return_all_modalities=False
    ):
        valid_groups = ("all", *_GROUP_NAMES)
        if group not in valid_groups:
            raise ValueError(f"group must be one of {valid_groups}, got {group!r}.")
        self.group = group
        if group == "all":
            group_subjects = list(_SUBJECT_MAPPING)
        else:
            archive_group = _GROUP_NAMES[group]
            group_subjects = [
                subject
                for subject, (candidate_group, _) in _SUBJECT_MAPPING.items()
                if candidate_group == archive_group
            ]
        if subjects is None:
            selected_subjects = None if group == "all" else group_subjects
        else:
            selected_subjects = list(subjects)
            invalid = [
                subject for subject in selected_subjects if subject not in group_subjects
            ]
            if invalid:
                raise ValueError(
                    f"Subjects {invalid} are not in group {group!r}; valid global "
                    f"IDs are {group_subjects}."
                )
        super().__init__(
            subjects=list(_SUBJECT_MAPPING),
            sessions_per_subject=5,
            events=dict(_EVENTS),
            code="Wang2026",
            interval=[0.0, (_NOMINAL_TRIAL_SAMPLES - 1) / _SFREQ],
            paradigm="imagery",
            doi=_DATASET_DOI,
            selected_subjects=selected_subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    @property
    def subject_mapping(self):
        """Read-only global-ID mapping to ``(cohort, source subject)``."""
        return _SUBJECT_MAPPING

    def _get_single_subject_data(self, subject):
        """Fail before transfer while physical calibration is unresolved."""
        return self._get_single_subject_data_calibrated(subject)

    def _get_single_subject_data_calibrated(self, subject, *, _eeg_scale=None):
        """Private parsing path used only after an explicit calibration is supplied."""
        eeg_scale = _validated_eeg_scale(_eeg_scale)
        paths = sorted(map(Path, self.data_path(subject)), key=_path_sort_key)
        sessions = defaultdict(dict)
        group, source_subject = _SUBJECT_MAPPING[subject]

        for mat_path in paths:
            if mat_path.name.endswith("_run0.mat"):
                session_key = "0baseline"
            else:
                match = _RUN_RE.search(mat_path.name)
                if match is None:
                    raise ValueError(f"Unexpected Wang2026 filename: {mat_path.name}")
                session_key = str(int(match.group("session")))

            run_key = str(len(sessions[session_key]))
            raw = _make_raw(mat_path, _eeg_scale=eeg_scale)
            raw.info["subject_info"] = {"his_id": f"{group}/{source_subject}"}
            sessions[session_key][run_key] = raw

        if not sessions:
            raise FileNotFoundError(
                f"No data found for Wang2026 subject {subject} "
                f"({group}/{source_subject})."
            )
        return dict(sessions)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Extract one global subject's runs from its remote cohort archive.

        Only the archive members belonging to ``subject`` are transferred, using
        HTTP range requests against the published ZIP (see :class:`_RangeReader`
        and the note on :data:`_ARCHIVES`). The whole-archive download is
        deliberately avoided: one subject would otherwise cost 8.7-25.3 GB.
        """
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject {subject}. Valid subjects: {self.subject_list}"
            )

        group, subject_name = _SUBJECT_MAPPING[subject]
        source_subject = int(subject_name.removeprefix("S"))
        root = Path(dl.get_dataset_path(self.code, path)) / "MNE-wang2026-data"
        root.mkdir(parents=True, exist_ok=True)
        subject_dir = root / group / subject_name
        expected_count = _SUBJECT_FILE_COUNTS[group][source_subject]
        archive = _ARCHIVES[group]
        expected_identity = {
            "schema_version": 1,
            "dataset_doi": _DATASET_DOI,
            "group": group,
            "source_subject": subject_name,
            "archive": {
                "file_id": archive["file_id"],
                "filename": archive["filename"],
                "md5": archive["md5"],
            },
        }
        if not force_update and _cache_matches_manifest(
            subject_dir, expected_count, expected_identity
        ):
            return [str(file_path) for file_path in sorted(subject_dir.glob("*.mat"))]

        url = _FIGSHARE_FILE.format(file_id=archive["file_id"])
        prefix = f"{group}/{subject_name}/"

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

            extracted_dir = tmp_root / group / subject_name
            extracted = sorted(extracted_dir.glob("*.mat"))
            if len(extracted) != expected_count:
                raise RuntimeError(
                    f"Incomplete extraction for {group}/{subject_name}: "
                    f"found {len(extracted)} files, expected {expected_count}."
                )
            manifest = _manifest_for_members(group, subject_name, archive, members)
            (extracted_dir / _MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            subject_dir.parent.mkdir(parents=True, exist_ok=True)
            if subject_dir.exists():
                shutil.rmtree(subject_dir)
            shutil.move(str(extracted_dir), str(subject_dir))

        return [str(file_path) for file_path in sorted(subject_dir.glob("*.mat"))]
