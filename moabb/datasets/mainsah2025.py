"""BigP3BCI dataset — the largest public P300 BCI dataset.

Mainsah, Fleeting, Balmat, Sellers, & Collins (2025).
PhysioNet DOI: 10.13026/0byy-ry86

Contains ~267 subjects across 20 studies (A through S2), each with
16 or 32 EEG channels sampled at 256 Hz. Studies use either a 6x6 or
9x8 character grid. Some studies include ALS patients.

Each study is exposed as a separate MOABB dataset class (e.g.,
``Mainsah2025_A``, ``Mainsah2025_B``, ..., ``Mainsah2025_S2``).
"""

import logging
import re
import warnings
from functools import partialmethod
from pathlib import Path

import mne
import numpy as np

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_BASE_URL = "https://physionet.org/files/bigp3bci/1.0.0/"
_MANIFEST_URL = _BASE_URL + "SHA256SUMS.txt"
_DOI = "10.13026/0byy-ry86"
_SIGN = "Mainsah2025"

# Module-level manifest cache (parsed once, shared across instances)
_manifest_cache = None

# fmt: off
_STUDY_CONFIGS = {
    "A": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16, 17, 19],
        "n_sessions": 1,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "B": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        "n_sessions": 8,
        "n_eeg": 16,
        "grid": "6x6",
        "has_als": True,
    },
    "C": {
        "subjects": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22],
        "n_sessions": 1,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "D": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "n_sessions": 1,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "E": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": False,
    },
    "F": {
        "subjects": [3, 5, 6, 7, 8, 20, 21, 23, 24, 25],
        "n_sessions": 3,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": True,
    },
    "G": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": False,
    },
    "H": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": False,
    },
    "I": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": False,
    },
    "J": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "6x6",
        "has_als": False,
    },
    "K": {
        "subjects": [1, 2, 3, 4, 9],
        "n_sessions": 2,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": False,
    },
    "L": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "6x6",
        "has_als": True,
    },
    "M": {
        "subjects": [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        "n_sessions": 1,
        "n_eeg": 16,
        "grid": "9x8",
        "has_als": False,
    },
    "N": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8],
        "n_sessions": 2,
        "n_eeg": 16,
        "grid": "6x6",
        "has_als": True,
    },
    "O": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "n_sessions": 2,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "P": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "n_sessions": 2,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "Q": {
        "subjects": list(range(1, 37)),
        "n_sessions": 3,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "R": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "n_sessions": 2,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "S1": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "n_sessions": 1,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
    "S2": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28],
        "n_sessions": 1,
        "n_eeg": 32,
        "grid": "9x8",
        "has_als": False,
    },
}
# fmt: on

# Shared documentation metadata
_DOCUMENTATION = DocumentationMetadata(
    doi=_DOI,
    description=(
        "BigP3BCI: the largest public P300 BCI dataset, containing EEG "
        "recordings from ~267 subjects across 20 studies using 6x6 or 9x8 "
        "character grids with various stimulus paradigms."
    ),
    investigators=[
        "Boyla Mainsah",
        "Chance Fleeting",
        "Thomas Balmat",
        "Eric Sellers",
        "Leslie Collins",
    ],
    institution="Duke University; East Tennessee State University",
    country="US",
    repository="PhysioNet",
    data_url="https://physionet.org/content/bigp3bci/1.0.0/",
    publication_year=2025,
    license="CC-BY-4.0",
)


def _parse_manifest(manifest_path):
    """Parse SHA256SUMS.txt into a nested dict of EDF file paths.

    Returns
    -------
    dict
        ``{study: {subj_id: {session_int: [relative_paths]}}}``
    """
    result = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line.endswith(".edf"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            path = parts[1]
            m = re.match(
                r"bigP3BCI-data/Study(\w+)/(\w+)/SE(\d+)/"
                r"(Test|Train)/(\w+)/(.+\.edf)",
                path,
            )
            if not m:
                continue
            study, subj_str, session_str = m.group(1), m.group(2), m.group(3)
            session = int(session_str)

            result.setdefault(study, {})
            result[study].setdefault(subj_str, {})
            result[study][subj_str].setdefault(session, [])
            result[study][subj_str][session].append(path)

    # Sort file lists for deterministic ordering
    for study in result.values():
        for subj in study.values():
            for sess in subj:
                subj[sess].sort()
    return result


def _get_manifest_path():
    """Return local path for the cached manifest file."""
    path = Path(dl.get_dataset_path(_SIGN, None))
    root = path / f"MNE-{_SIGN}-data"
    root.mkdir(parents=True, exist_ok=True)
    return root / "SHA256SUMS.txt"


class Mainsah2025(BaseDataset):
    """Base class for the BigP3BCI dataset by Mainsah et al. 2025 [1]_.

    BigP3BCI is the largest publicly available P300-based BCI dataset,
    containing EEG data from approximately 267 participants across 20
    studies (A through S2). Studies were conducted at Duke University
    and East Tennessee State University using g.USBamp amplifiers with
    either 16 or 32 active/passive EEG electrodes sampled at 256 Hz.

    Each study explores a different P300 speller paradigm variant
    (checkerboard, row-column, dynamic, adaptive, etc.) with either
    a 6x6 or 9x8 character grid. Studies L through R include
    participants with ALS.

    Each EDF+ file corresponds to one spelling block and contains:

    - EEG channels (``EEG_<electrode>`` labels)
    - Stimulus marker channels: ``StimulusBegin`` (0/1 onset flag),
      ``StimulusType`` (0=NonTarget, 1=Target)
    - Additional channels: ``StimulusCode``, ``SelectedTarget``,
      ``SelectedRow``, ``SelectedColumn``, ``PhaseInSequence``,
      ``CurrentTarget``

    Events are extracted from the ``StimulusBegin`` and ``StimulusType``
    signal channels.

    This base class is not intended to be instantiated directly.
    Use the study-specific subclasses (e.g., ``Mainsah2025_A``).

    References
    ----------
    .. [1] Mainsah BO, Fleeting CE, Balmat TJ, Sellers EW, Collins LM.
       BigP3BCI: A large P300-based brain-computer interface dataset.
       PhysioNet, 2025. DOI: https://doi.org/10.13026/0byy-ry86
    """

    def __init__(self, study, subjects=None, sessions=None):
        config = _STUDY_CONFIGS[study]
        self._study = study

        super().__init__(
            subjects=config["subjects"],
            sessions_per_subject=config["n_sessions"],
            events={"Target": 2, "NonTarget": 1},
            code=f"Mainsah2025-{study}",
            interval=[0, 1.0],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    @staticmethod
    def _get_manifest():
        """Download and parse the SHA256SUMS manifest (cached in memory)."""
        global _manifest_cache
        if _manifest_cache is None:
            manifest_path = _get_manifest_path()
            dl.download_if_missing(str(manifest_path), _MANIFEST_URL, warn_missing=False)
            _manifest_cache = _parse_manifest(manifest_path)
        return _manifest_cache

    def _subject_str(self, subject):
        """Convert integer subject ID to the dataset's string format."""
        return f"{self._study}_{subject:02d}"

    def _get_subject_manifest(self, subject):
        """Return ``{session_int: [rel_paths]}`` for *subject*."""
        manifest = self._get_manifest()
        subj_str = self._subject_str(subject)
        subj_manifest = manifest.get(self._study, {}).get(subj_str, {})
        if not subj_manifest:
            raise FileNotFoundError(
                f"No EDF files found in manifest for study={self._study}, "
                f"subject={subj_str}"
            )
        return subj_manifest

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download and return local paths for all EDF files of a subject.

        Returns
        -------
        list of str
            Local file paths to the downloaded EDF files, sorted by
            session then filename.
        """
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject {subject} for study {self._study}. "
                f"Valid: {self.subject_list}"
            )

        subj_manifest = self._get_subject_manifest(subject)
        base_path = Path(dl.get_dataset_path(_SIGN, path))
        root = base_path / f"MNE-{_SIGN}-data"

        local_paths = []
        for session in sorted(subj_manifest.keys()):
            for rel_path in subj_manifest[session]:
                local_file = root / rel_path
                dl.download_if_missing(
                    str(local_file), _BASE_URL + rel_path, warn_missing=False
                )
                local_paths.append(str(local_file))

        return local_paths

    def _get_single_subject_data(self, subject):
        """Load all EDF files for one subject and return session/run dict.

        Returns
        -------
        dict
            ``{session_str: {run_str: mne.io.Raw}}``
        """
        # Ensure all files are downloaded first
        self.data_path(subject)

        subj_manifest = self._get_subject_manifest(subject)
        base_path = Path(dl.get_dataset_path(_SIGN, None))
        root = base_path / f"MNE-{_SIGN}-data"

        # Map actual session numbers to 0-indexed MOABB sessions
        all_sessions = sorted(subj_manifest.keys())

        sessions = {}
        for sess_idx, sess_num in enumerate(all_sessions):
            sess_key = str(sess_idx)
            run_idx = 0
            for rel_path in subj_manifest[sess_num]:
                local_file = root / rel_path
                try:
                    raw = self._load_edf(str(local_file))
                except Exception:
                    log.warning("Failed to load %s, skipping.", local_file)
                    continue

                sessions.setdefault(sess_key, {})
                sessions[sess_key][str(run_idx)] = raw
                run_idx += 1

        return sessions

    @staticmethod
    def _load_edf(edf_path):
        """Load a single EDF file, extract P300 events, return Raw.

        Reads the EDF, extracts Target/NonTarget events from the
        StimulusBegin and StimulusType signal channels, builds a
        synthetic stim channel, strips the ``EEG_`` prefix from
        channel names, and sets a standard 10-20 montage.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Extract stimulus signals before dropping non-EEG channels
        required = {"StimulusBegin", "StimulusType"}
        if not required.issubset(raw.ch_names):
            raise ValueError(f"Missing StimulusBegin/StimulusType in {edf_path}")
        stim_begin, stim_type = raw.get_data(picks=["StimulusBegin", "StimulusType"])

        # Find rising edges of StimulusBegin (transition from 0 to 1)
        onsets = np.where(np.diff(stim_begin) > 0.5)[0] + 1

        # Build stim channel: Target=2, NonTarget=1 (vectorized)
        stim_data = np.zeros(raw.n_times)
        valid = onsets[onsets < len(stim_type)]
        stim_data[valid] = np.where(stim_type[valid] > 0.5, 2, 1)

        # Keep only EEG channels
        eeg_picks = [ch for ch in raw.ch_names if ch.startswith("EEG_")]
        if not eeg_picks:
            raise ValueError(f"No EEG channels found in {edf_path}")
        raw.pick(eeg_picks)

        # Strip EEG_ prefix and fix non-standard names in a single pass
        _ch_fixes = {"EEG_FP1": "Fp1", "EEG_FP2": "Fp2"}
        rename = {ch: _ch_fixes.get(ch, ch.replace("EEG_", "")) for ch in raw.ch_names}
        raw.rename_channels(rename)

        # Set montage (warn on missing channels rather than error)
        raw.set_montage("standard_1020", on_missing="warn")

        # Add synthetic stim channel
        stim_info = mne.create_info(["STI"], raw.info["sfreq"], ["stim"])
        stim_raw = mne.io.RawArray(stim_data[np.newaxis], stim_info, verbose=False)
        raw.add_channels([stim_raw], force_update_info=True)

        return raw


def _make_study_metadata(study):
    """Build DatasetMetadata for a given study."""
    config = _STUDY_CONFIGS[study]
    return DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=config["n_eeg"],
            channel_types={"eeg": config["n_eeg"]},
            hardware="g.USBamp (g.tec)",
            montage="standard_1020",
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=len(config["subjects"]),
            health_status="patients" if config["has_als"] else "healthy",
            clinical_population="ALS" if config["has_als"] else None,
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            events={"Target": 2, "NonTarget": 1},
            n_classes=2,
            class_labels=["Target", "NonTarget"],
        ),
        documentation=_DOCUMENTATION,
        sessions_per_subject=config["n_sessions"],
        paradigm_specific=ParadigmSpecificMetadata(detected_paradigm="p300"),
        signal_processing=SignalProcessingMetadata(
            classifiers=None,
            feature_extraction=["P300_ERP_detection"],
            frequency_bands=None,
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="calibration-then-test", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller"], environment="laboratory", online_feedback=True
        ),
        tags=Tags(modality=["visual"], type=["perception"]),
    )


# ---------------------------------------------------------------------------
# Study-specific subclasses (one per study, 20 total)
# ---------------------------------------------------------------------------


class Mainsah2025_A(Mainsah2025):
    """BigP3BCI Study A — 6x6 checkerboard/row-column/random (13 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "A")
    METADATA = _make_study_metadata("A")


class Mainsah2025_B(Mainsah2025):
    """BigP3BCI Study B — 6x6 checkerboard, multi-session (19 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "B")
    METADATA = _make_study_metadata("B")


class Mainsah2025_C(Mainsah2025):
    """BigP3BCI Study C — 6x6 checkerboard with ERN (19 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "C")
    METADATA = _make_study_metadata("C")


class Mainsah2025_D(Mainsah2025):
    """BigP3BCI Study D — 6x6 dynamic/row-column (17 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "D")
    METADATA = _make_study_metadata("D")


class Mainsah2025_E(Mainsah2025):
    """BigP3BCI Study E — 6x6 checkerboard (8 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "E")
    METADATA = _make_study_metadata("E")


class Mainsah2025_F(Mainsah2025):
    """BigP3BCI Study F — 6x6 multi-paradigm, 3 sessions (10 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "F")
    METADATA = _make_study_metadata("F")


class Mainsah2025_G(Mainsah2025):
    """BigP3BCI Study G — 9x8 checkerboard/dynamic (20 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "G")
    METADATA = _make_study_metadata("G")


class Mainsah2025_H(Mainsah2025):
    """BigP3BCI Study H — 9x8 checkerboard with gaze conditions (16 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "H")
    METADATA = _make_study_metadata("H")


class Mainsah2025_I(Mainsah2025):
    """BigP3BCI Study I — 9x8 checkerboard/performance-based (13 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "I")
    METADATA = _make_study_metadata("I")


class Mainsah2025_J(Mainsah2025):
    """BigP3BCI Study J — 9x8 performance-based/row-column (20 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "J")
    METADATA = _make_study_metadata("J")


class Mainsah2025_K(Mainsah2025):
    """BigP3BCI Study K — 9x8 adaptive/checkerboard, 2 sessions (5 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "K")
    METADATA = _make_study_metadata("K")


class Mainsah2025_L(Mainsah2025):
    """BigP3BCI Study L — 6x6 multi-paradigm (11 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "L")
    METADATA = _make_study_metadata("L")


class Mainsah2025_M(Mainsah2025):
    """BigP3BCI Study M — 9x8 adaptive/checkerboard (21 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "M")
    METADATA = _make_study_metadata("M")


class Mainsah2025_N(Mainsah2025):
    """BigP3BCI Study N — 9x8 dry/wet electrode comparison (8 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "N")
    METADATA = _make_study_metadata("N")


class Mainsah2025_O(Mainsah2025):
    """BigP3BCI Study O — 9x8 supervised/checkerboard (18 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "O")
    METADATA = _make_study_metadata("O")


class Mainsah2025_P(Mainsah2025):
    """BigP3BCI Study P — 9x8 predictive/non-predictive spelling (19 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "P")
    METADATA = _make_study_metadata("P")


class Mainsah2025_Q(Mainsah2025):
    """BigP3BCI Study Q — 6x6 color intensification (36 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "Q")
    METADATA = _make_study_metadata("Q")


class Mainsah2025_R(Mainsah2025):
    """BigP3BCI Study R — 9x8 multi-face paradigms (20 ALS subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "R")
    METADATA = _make_study_metadata("R")


class Mainsah2025_S1(Mainsah2025):
    """BigP3BCI Study S1 — 9x8 face/house paradigm (10 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "S1")
    METADATA = _make_study_metadata("S1")


class Mainsah2025_S2(Mainsah2025):
    """BigP3BCI Study S2 — 9x8 house/tool paradigm (24 healthy subjects)."""

    __init__ = partialmethod(Mainsah2025.__init__, "S2")
    METADATA = _make_study_metadata("S2")
