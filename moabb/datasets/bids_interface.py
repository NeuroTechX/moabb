"""BIDS Interface for MOABB.

========================
This module contains the BIDS interface for MOABB, which allows to convert
any MOABB dataset to BIDS with Cache.
We can convert at the Raw, Epochs or Array level.
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import abc
import csv
import datetime
import json
import logging
import re
import shutil
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

import mne
import mne_bids
import pandas as pd
from numpy import load as np_load
from numpy import save as np_save

import moabb
from moabb.analysis.results import get_digest
from moabb.datasets import download as dl
from moabb.datasets._channel_pick import pick_channels_for_modalities


if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkey-patch mne_bids to produce BIDS-compliant output by default.
#
# BIDS validator v2.4.0 requires a SpatialReference key in the JSON sidecar
# for *_electrodes.tsv (when a ``space`` entity is present, e.g.
# space-CapTrak).  mne_bids does not create this sidecar, so we wrap
# ``_write_dig_bids`` to produce the missing ``*_electrodes.json`` file.
#
# For standard scalp EEG, electrode positions are in CapTrak coordinates
# (defined by nasion, LPA, RPA landmarks on the individual's head).  There
# is no external template image, so SpatialReference is ``"n/a"``.
# ---------------------------------------------------------------------------

import mne_bids.dig as _mne_bids_dig  # noqa: E402


_orig_write_dig_bids = _mne_bids_dig._write_dig_bids


def _write_dig_bids_with_electrodes_json(*args, **kwargs):
    """Wrap mne_bids _write_dig_bids to also create electrodes.json sidecar."""
    _orig_write_dig_bids(*args, **kwargs)
    # Extract what we need for the electrodes.json sidecar
    bids_path = args[0]
    overwrite = kwargs.get("overwrite", False)
    # Create electrodes.json sidecar next to every electrodes.tsv that was
    # written.  The filenames include the space entity (e.g. space-CapTrak),
    # so we glob for them.
    sub_dir = Path(bids_path.root) / f"sub-{bids_path.subject}"
    if bids_path.session is not None:
        sub_dir = sub_dir / f"ses-{bids_path.session}"
    eeg_dir = sub_dir / bids_path.datatype
    if eeg_dir.is_dir():
        for elec_tsv in eeg_dir.glob("*_electrodes.tsv"):
            elec_json = elec_tsv.with_suffix(".json")
            if not elec_json.exists() or overwrite:
                with open(elec_json, "w") as f:
                    json.dump({"SpatialReference": "n/a"}, f, indent="\t")


# Apply patch so write_raw_bids() creates electrodes.json automatically.
_mne_bids_dig._write_dig_bids = _write_dig_bids_with_electrodes_json
import mne_bids.write as _mne_bids_write  # noqa: E402


_mne_bids_write._write_dig_bids = _write_dig_bids_with_electrodes_json

# ---------------------------------------------------------------------------

# Known amplifier manufacturer lookup for splitting hardware into
# Manufacturer + ManufacturersModelName
_MANUFACTURER_LOOKUP = {
    "BrainAmp": "Brain Products",
    "BrainAmp DC": "Brain Products",
    "BrainAmp MR": "Brain Products",
    "BrainAmp MR plus": "Brain Products",
    "actiCHamp": "Brain Products",
    "actiCHamp Plus": "Brain Products",
    "LiveAmp": "Brain Products",
    "g.USBamp": "g.tec",
    "g.HIamp": "g.tec",
    "g.Nautilus": "g.tec",
    "Biosemi ActiveTwo": "BioSemi",
    "ActiveTwo": "BioSemi",
    "NeurOne": "Mega Electronics",
    "NeuScan": "Compumedics",
    "Neuroscan SynAmps2": "Compumedics Neuroscan",
    "SynAmps2": "Compumedics Neuroscan",
    "SynAmps": "Compumedics Neuroscan",
    "ANT Neuro eego": "ANT Neuro",
    "eego sports": "ANT Neuro",
    "EGI": "Electrical Geodesics",
}


def _split_manufacturer(hardware):
    """Split a hardware string into (Manufacturer, ModelName).

    Uses a lookup table for known amplifiers. Falls back to using the
    full string for both fields.
    """
    if not hardware:
        return None, None
    for model, manufacturer in _MANUFACTURER_LOOKUP.items():
        if model.lower() in hardware.lower():
            return manufacturer, hardware
    return hardware, hardware


_SEX_LABEL_TO_CODE = {"male": 1, "female": 2}
_SEX_CODE_TO_LABEL = {v: k for k, v in _SEX_LABEL_TO_CODE.items()}
_HAND_LABEL_TO_CODE = {"right": 1, "left": 2, "ambidextrous": 3}
_HAND_CODE_TO_LABEL = {v: k for k, v in _HAND_LABEL_TO_CODE.items()}


def _is_missing_participant_value(value):
    """Return True when a participant value should be treated as unknown."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "n/a", "na", "none", "unknown", "null"}
    return False


def _get_subject_list_value(values, subject_idx):
    """Return per-subject value from list-like metadata fields."""
    if values is None or subject_idx >= len(values):
        return None
    value = values[subject_idx]
    if _is_missing_participant_value(value):
        return None
    return value


def _normalize_sex_value(value):
    """Normalize sex values to BIDS labels ('male'/'female')."""
    if _is_missing_participant_value(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) in _SEX_CODE_TO_LABEL:
        return _SEX_CODE_TO_LABEL[int(value)]
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "male": "male",
            "m": "male",
            "man": "male",
            "1": "male",
            "female": "female",
            "f": "female",
            "woman": "female",
            "2": "female",
        }
        return mapping.get(normalized)
    return None


def _normalize_hand_value(value):
    """Normalize handedness values to BIDS labels."""
    if _is_missing_participant_value(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) in _HAND_CODE_TO_LABEL:
        return _HAND_CODE_TO_LABEL[int(value)]
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", " ").replace("-", " ")
        mapping = {
            "right": "right",
            "r": "right",
            "right handed": "right",
            "rh": "right",
            "left": "left",
            "l": "left",
            "left handed": "left",
            "lh": "left",
            "ambidextrous": "ambidextrous",
            "ambi": "ambidextrous",
            "both": "ambidextrous",
            "mixed": "ambidextrous",
            "3": "ambidextrous",
            "1": "right",
            "2": "left",
        }
        return mapping.get(normalized)
    return None


def _resolve_subject_age(participants, subject_idx, raw=None):
    """Resolve age with priority: per-subject list -> raw -> aggregate mean -> n/a."""
    age = _get_subject_list_value(participants.ages, subject_idx)
    if age is not None:
        return age

    if raw is not None and hasattr(raw, "_moabb_subject_age"):
        raw_age = getattr(raw, "_moabb_subject_age")
        if not _is_missing_participant_value(raw_age):
            return raw_age

    if participants.age_mean is not None:
        return participants.age_mean
    return "n/a"


def _resolve_subject_sex(participants, subject_idx, raw=None):
    """Resolve sex with priority: per-subject list -> raw subject_info -> n/a."""
    sex = _normalize_sex_value(_get_subject_list_value(participants.sexes, subject_idx))
    if sex is not None:
        return sex

    if raw is not None:
        subject_info = raw.info.get("subject_info") or {}
        sex = _normalize_sex_value(subject_info.get("sex"))
        if sex is not None:
            return sex
    return "n/a"


def _resolve_subject_hand(participants, subject_idx, raw=None):
    """Resolve hand with robust fallback: list -> raw -> aggregate -> n/a."""
    hand = _normalize_hand_value(
        _get_subject_list_value(participants.handedness_list, subject_idx)
    )
    if hand is not None:
        return hand

    if raw is not None:
        subject_info = raw.info.get("subject_info") or {}
        hand = _normalize_hand_value(subject_info.get("hand"))
        if hand is not None:
            return hand

    if isinstance(participants.handedness, dict):
        counts = {"right": 0, "left": 0, "ambidextrous": 0}
        for label, count in participants.handedness.items():
            normalized = _normalize_hand_value(label)
            if normalized in counts and isinstance(count, (int, float)):
                counts[normalized] += count
        total = sum(counts.values())
        if total > 0:
            for label, count in counts.items():
                if count == total:
                    return label
    elif isinstance(participants.handedness, str):
        hand = _normalize_hand_value(participants.handedness)
        if hand is not None:
            return hand

    return "n/a"


def _enrich_raw_info_from_metadata(raw, metadata, subject):
    """Set ``raw.info`` fields from dataset metadata before ``write_raw_bids``.

    Enriches ``subject_info`` (sex, hand) and ``line_freq`` so that
    ``mne_bids.write_raw_bids`` auto-generates richer sidecars and
    ``participants.tsv`` entries.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw object whose info will be modified in-place.
    metadata : DatasetMetadata
        The dataset metadata.
    subject : int
        The MOABB subject number (1-based).
    """
    if metadata is None:
        return

    # Use metadata line_freq if raw doesn't have one
    if raw.info.get("line_freq", None) is None and metadata.acquisition:
        raw.info["line_freq"] = metadata.acquisition.line_freq

    participants = metadata.participants
    if participants is None:
        return

    subject_info = raw.info.get("subject_info") or {}
    subject_idx = subject - 1  # MOABB subjects are 1-based

    # Sex: BIDS uses FHIR codes (0=unknown, 1=male, 2=female).
    # Per-subject list takes priority over aggregate gender dict.
    sex = _normalize_sex_value(_get_subject_list_value(participants.sexes, subject_idx))
    if sex is not None:
        subject_info["sex"] = _SEX_LABEL_TO_CODE[sex]
    elif participants.gender:
        # Fallback: only set if the population is homogeneous (all one gender)
        total = sum(participants.gender.values())
        if participants.gender.get("male", 0) == total:
            subject_info["sex"] = 1
        elif participants.gender.get("female", 0) == total:
            subject_info["sex"] = 2

    if (participants.gender or participants.sexes) and "his_id" not in subject_info:
        subject_info.setdefault("his_id", str(subject))

    # Handedness: BIDS uses 1=right, 2=left, 3=ambidextrous.
    # Per-subject list takes priority over aggregate handedness.
    hand = _normalize_hand_value(
        _get_subject_list_value(participants.handedness_list, subject_idx)
    )
    if hand is not None:
        subject_info["hand"] = _HAND_LABEL_TO_CODE[hand]
    elif isinstance(participants.handedness, dict):
        total = sum(participants.handedness.values())
        if participants.handedness.get("right", 0) == total:
            subject_info["hand"] = 1
        elif participants.handedness.get("left", 0) == total:
            subject_info["hand"] = 2
    elif isinstance(participants.handedness, str):
        h = participants.handedness.lower()
        if "right" in h and "left" not in h:
            subject_info["hand"] = 1
        elif "left" in h and "right" not in h:
            subject_info["hand"] = 2

    if subject_info:
        raw.info["subject_info"] = subject_info


_PARADIGM_COG_ATLAS = {
    "imagery": "https://www.cognitiveatlas.org/task/id/trm_4c8a834779883/",
    "p300": "https://www.cognitiveatlas.org/task/id/tsk_GxjZBNiJorj1K/",
}

# Paradigm-level HED tag mappings for events.json sidecar enrichment.
# Maps (paradigm, event_name) → HED tag string using HED schema 8.4.0.

# Shared sensory prefix for MI events where the specific visual stimulus is
# unknown — datasets that use arrows or other cues override via hed_tags
# in their ExperimentMetadata.
_MI_SENSORY = "(Sensory-event, Experimental-stimulus, Visual-presentation)"

_PARADIGM_HED_TAGS = {
    # ── Motor Imagery ──
    # Each MI stimulus event decomposes into two top-level groups per HED
    # annotation semantics (Rules 2b/2e/2f):
    #   1. Sensory-event — what the participant sees (generic here; datasets
    #      with known cue types override via ExperimentMetadata.hed_tags).
    #   2. (Agent-action, (Imagine, Move, ...)) — what the participant does.
    "imagery": {
        # Common MI events — generic sensory prefix (no arrow assumption)
        "left_hand": f"{_MI_SENSORY}, (Agent-action, (Imagine, Move, (Left, Hand)))",
        "right_hand": f"{_MI_SENSORY}, (Agent-action, (Imagine, Move, (Right, Hand)))",
        "feet": f"{_MI_SENSORY}, (Agent-action, (Imagine, Move, Foot))",
        "tongue": f"{_MI_SENSORY}, (Agent-action, (Imagine, Move, Tongue))",
        "both_hand": f"{_MI_SENSORY}, (Agent-action, (Imagine, Move, Hand))",
        "hands": f"{_MI_SENSORY}, (Agent-action, (Imagine, Move, Hand))",
        "rest": "Sensory-event, Experimental-stimulus, Visual-presentation, Rest",
        "right_hand_right_foot": (
            f"{_MI_SENSORY}, "
            "(Agent-action, (Imagine, Move, (Right, Hand)), "
            "(Imagine, Move, (Right, Foot)))"
        ),
        "palmar_grasp": f"{_MI_SENSORY}, (Agent-action, (Imagine, Grasp, Hand))",
        "lateral_grasp": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Grasp, Hand, (Label/lateral)))"
        ),
        # Weibo2014 — compound limb motor imagery
        "left_hand_right_foot": (
            f"{_MI_SENSORY}, "
            "(Agent-action, (Imagine, Move, (Left, Hand)), "
            "(Imagine, Move, (Right, Foot)))"
        ),
        "right_hand_left_foot": (
            f"{_MI_SENSORY}, "
            "(Agent-action, (Imagine, Move, (Right, Hand)), "
            "(Imagine, Move, (Left, Foot)))"
        ),
        # Ofner2017 — upper limb motor imagery
        "right_elbow_flexion": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Flex, (Right, Elbow)))"
        ),
        "right_elbow_extension": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Stretch, (Right, Elbow)))"
        ),
        "right_supination": (
            f"{_MI_SENSORY}, "
            "(Agent-action, (Imagine, Turn, (Right, Forearm), (Label/supination)))"
        ),
        "right_pronation": (
            f"{_MI_SENSORY}, "
            "(Agent-action, (Imagine, Turn, (Right, Forearm), (Label/pronation)))"
        ),
        "right_hand_close": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Close, (Right, Hand)))"
        ),
        "right_hand_open": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Open, (Right, Hand)))"
        ),
        # BNCI2019_001 — motor imagery without laterality prefix
        "hand_open": f"{_MI_SENSORY}, (Agent-action, (Imagine, Open, Hand))",
        "pronation": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Turn, Forearm, (Label/pronation)))"
        ),
        "supination": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Turn, Forearm, (Label/supination)))"
        ),
        # BNCI2015_004 — mental/cognitive tasks
        "math": f"{_MI_SENSORY}, (Agent-action, (Imagine, Think, (Label/math)))",
        "letter": f"{_MI_SENSORY}, (Agent-action, (Imagine, Think, (Label/letter)))",
        "rotation": f"{_MI_SENSORY}, (Agent-action, (Imagine, Think, (Label/rotation)))",
        "count": f"{_MI_SENSORY}, (Agent-action, (Imagine, Count))",
        "baseline": "Sensory-event, Experimental-stimulus, Visual-presentation, Rest",
        # Shin2017B — mental arithmetic
        "subtraction": (
            f"{_MI_SENSORY}, (Agent-action, (Imagine, Think, (Label/subtraction)))"
        ),
        # BNCI2024_001 — handwritten character writing
        "letter_a": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/a)))",
        "letter_d": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/d)))",
        "letter_e": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/e)))",
        "letter_f": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/f)))",
        "letter_j": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/j)))",
        "letter_n": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/n)))",
        "letter_o": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/o)))",
        "letter_s": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/s)))",
        "letter_t": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/t)))",
        "letter_v": f"{_MI_SENSORY}, (Agent-action, (Write, Hand, (Label/v)))",
        # BNCI2022_001 — drone piloting / waypoint events
        "trajectory_start": "Experiment-structure, (Label/trajectory_start)",
        "waypoint_hit": "Experiment-structure, (Label/waypoint_hit)",
        "waypoint_miss": "Experiment-structure, (Label/waypoint_miss)",
        "trajectory_end": "Experiment-structure, (Label/trajectory_end)",
        # BNCI2025_001 — discrete reaching (direction × speed × distance)
        "up_slow_near": f"{_MI_SENSORY}, (Agent-action, (Reach, Upward, (Label/slow), (Label/near)))",
        "up_slow_far": f"{_MI_SENSORY}, (Agent-action, (Reach, Upward, (Label/slow), (Label/far)))",
        "up_fast_near": f"{_MI_SENSORY}, (Agent-action, (Reach, Upward, (Label/fast), (Label/near)))",
        "up_fast_far": f"{_MI_SENSORY}, (Agent-action, (Reach, Upward, (Label/fast), (Label/far)))",
        "down_slow_near": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Downward, (Label/slow), (Label/near)))"
        ),
        "down_slow_far": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Downward, (Label/slow), (Label/far)))"
        ),
        "down_fast_near": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Downward, (Label/fast), (Label/near)))"
        ),
        "down_fast_far": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Downward, (Label/fast), (Label/far)))"
        ),
        "left_slow_near": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Left, (Label/slow), (Label/near)))"
        ),
        "left_slow_far": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Left, (Label/slow), (Label/far)))"
        ),
        "left_fast_near": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Left, (Label/fast), (Label/near)))"
        ),
        "left_fast_far": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Left, (Label/fast), (Label/far)))"
        ),
        "right_slow_near": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Right, (Label/slow), (Label/near)))"
        ),
        "right_slow_far": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Right, (Label/slow), (Label/far)))"
        ),
        "right_fast_near": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Right, (Label/fast), (Label/near)))"
        ),
        "right_fast_far": (
            f"{_MI_SENSORY}, (Agent-action, (Reach, Right, (Label/fast), (Label/far)))"
        ),
        # BNCI2025_002 — continuous 2D trajectory tracking
        "snakerun": "Experiment-structure, (Label/snakerun)",
        "freerun": "Experiment-structure, (Label/freerun)",
        "eyerun": "Experiment-structure, (Label/eyerun)",
        # Cue event for datasets that expose it separately
        "cue": "Sensory-event, Cue, (Auditory-presentation, Tone), (Visual-presentation, Cross)",
    },
    # ── P300 / Oddball ──
    "p300": {
        "Target": "Sensory-event, Experimental-stimulus, Visual-presentation, Target",
        "NonTarget": (
            "Sensory-event, Experimental-stimulus, Visual-presentation, Non-target"
        ),
    },
    # ── Resting State ──
    "rstate": {
        "closed": "Experiment-structure, (Rest, (Close, Eye))",
        "open": "Experiment-structure, (Rest, (Open, Eye))",
        "on": "Experiment-structure, Rest",
        "off": "Experiment-structure, Rest",
        # Hinss2021 — resting state + cognitive workload levels
        "rs": "Experiment-structure, Rest",
        "easy": "Experiment-structure, (Label/easy)",
        "medium": "Experiment-structure, (Label/medium)",
        "diff": "Experiment-structure, (Label/difficult)",
    },
    # ── c-VEP ──
    "cvep": {
        "0.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_0_0)",
        "1.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_1_0)",
        "0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_0)",
        "1": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_1)",
        # MartinezCagigal2023Pary — p-ary intensity levels
        "2.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_2_0)",
        "3.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_3_0)",
        "4.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_4_0)",
        "5.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_5_0)",
        "6.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_6_0)",
        "7.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_7_0)",
        "8.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_8_0)",
        "9.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_9_0)",
        "10.0": "Sensory-event, Experimental-stimulus, Visual-presentation, (Label/intensity_10_0)",
    },
}


def _build_sidecar_enrichment(metadata):
    """Build a dict of BIDS EEG sidecar fields from dataset metadata.

    The returned dict can be passed to ``mne_bids.update_sidecar_json``
    to enrich ``*_eeg.json`` after ``write_raw_bids()``.

    Parameters
    ----------
    metadata : DatasetMetadata
        The dataset metadata.

    Returns
    -------
    dict
        BIDS EEG sidecar key-value pairs. Empty if metadata is None.
    """
    if metadata is None:
        return {}

    entries = {}
    acq = metadata.acquisition
    doc = metadata.documentation
    exp = metadata.experiment
    prep = metadata.preprocessing

    if acq:
        # EEGReference (REQUIRED)
        if acq.reference:
            entries["EEGReference"] = acq.reference

        # EEGGround (RECOMMENDED)
        if acq.ground:
            entries["EEGGround"] = acq.ground

        # Manufacturer and ManufacturersModelName (RECOMMENDED)
        if acq.hardware:
            manufacturer, model = _split_manufacturer(acq.hardware)
            if manufacturer:
                entries["Manufacturer"] = manufacturer
            if model:
                entries["ManufacturersModelName"] = model

        # SoftwareVersions (RECOMMENDED)
        if acq.software:
            entries["SoftwareVersions"] = acq.software

        # EEGPlacementScheme (RECOMMENDED)
        if acq.montage:
            montage_descriptions = {
                "standard_1005": "10-05 system",
                "standard_1020": "10-20 system",
                "10-20": "10-20 system",
                "10-05": "10-05 system",
                "10-10": "10-10 system",
                "biosemi128": "BioSemi 128-channel cap",
                "biosemi256": "BioSemi 256-channel cap",
                "biosemi64": "BioSemi 64-channel cap",
                "biosemi32": "BioSemi 32-channel cap",
                "biosemi16": "BioSemi 16-channel cap",
                "GSN-HydroCel-128": "Geodesic Sensor Net 128",
                "GSN-HydroCel-256": "Geodesic Sensor Net 256",
                "GSN-HydroCel-64_1.0": "Geodesic Sensor Net 64",
            }
            entries["EEGPlacementScheme"] = montage_descriptions.get(
                acq.montage, acq.montage
            )

        # CapManufacturer (RECOMMENDED)
        if acq.cap_manufacturer:
            entries["CapManufacturer"] = acq.cap_manufacturer

        # CapManufacturersModelName (RECOMMENDED)
        if acq.cap_model:
            entries["CapManufacturersModelName"] = acq.cap_model

    # EEGReference fallback: use prep.re_reference when acq.reference is absent
    if "EEGReference" not in entries and prep and prep.re_reference:
        entries["EEGReference"] = prep.re_reference

    # EEGReference is REQUIRED by BIDS — ensure always present
    entries.setdefault("EEGReference", "n/a")

    # HardwareFilters and SoftwareFilters
    if prep and any(
        v is not None
        for v in [prep.bandpass, prep.highpass_hz, prep.lowpass_hz, prep.notch_hz]
    ):
        hw_filters = {}
        if prep.bandpass:
            if isinstance(prep.bandpass, dict):
                hw_filters["Bandpass"] = {
                    k: v for k, v in prep.bandpass.items() if v is not None
                }
            elif isinstance(prep.bandpass, list) and len(prep.bandpass) == 2:
                hw_filters["Bandpass"] = {
                    "LowCutoffFrequency": prep.bandpass[0],
                    "HighCutoffFrequency": prep.bandpass[1],
                }
        else:
            # Build from individual highpass/lowpass
            bp = {}
            if prep.highpass_hz is not None:
                bp["LowCutoffFrequency"] = prep.highpass_hz
            if prep.lowpass_hz is not None:
                bp["HighCutoffFrequency"] = prep.lowpass_hz
            if bp:
                hw_filters["Bandpass"] = bp

        if prep.notch_hz is not None:
            notch = prep.notch_hz
            if isinstance(notch, (int, float)):
                notch = [notch]
            hw_filters["Notch"] = {"CutoffFrequency": notch}

        # Enrich Bandpass with filter type and order if available
        if "Bandpass" in hw_filters:
            if prep.filter_type is not None:
                hw_filters["Bandpass"]["Type"] = prep.filter_type
            if prep.filter_order is not None:
                hw_filters["Bandpass"]["Order"] = prep.filter_order

        if hw_filters:
            entries["HardwareFilters"] = hw_filters

    # HardwareFilters fallback: use acq.filters when prep filters are absent
    if "HardwareFilters" not in entries and acq and acq.filters:
        if isinstance(acq.filters, dict):
            # BIDS requires nested structure: {"FilterName": {"key": "value"}}
            # If the dict is flat (no nested dicts), wrap it under a filter name.
            if acq.filters and not any(isinstance(v, dict) for v in acq.filters.values()):
                entries["HardwareFilters"] = {"HardwareFilter": acq.filters}
            else:
                entries["HardwareFilters"] = acq.filters
        else:
            entries["HardwareFilters"] = {"HardwareFilter": str(acq.filters)}

    # HardwareFilters (RECOMMENDED) — "n/a" when not described
    entries.setdefault("HardwareFilters", "n/a")

    # SoftwareFilters — build from preprocessing_steps if available
    if prep and prep.preprocessing_steps:
        entries.setdefault(
            "SoftwareFilters",
            {"preprocessing_steps": {"Description": ", ".join(prep.preprocessing_steps)}},
        )

    # SoftwareFilters is REQUIRED — set to "n/a" when not described
    entries.setdefault("SoftwareFilters", "n/a")

    # TaskDescription (RECOMMENDED)
    if exp and exp.study_design:
        task_desc = exp.study_design

        # Prepend task_type if available
        if exp.task_type:
            task_desc = f"[{exp.task_type}] {task_desc}"

        # Append timing details from experiment and paradigm_specific
        timing_parts = []
        if exp.trial_duration is not None:
            timing_parts.append(f"Trial duration: {exp.trial_duration}s")
        ps = metadata.paradigm_specific
        if ps:
            if ps.cue_duration_s is not None:
                timing_parts.append(f"cue: {ps.cue_duration_s}s")
            if ps.imagery_duration_s is not None:
                timing_parts.append(f"imagery: {ps.imagery_duration_s}s")
            if ps.isi_ms is not None:
                timing_parts.append(f"ISI: {ps.isi_ms}ms")
            if ps.soa_ms is not None:
                timing_parts.append(f"SOA: {ps.soa_ms}ms")
            if ps.stimulus_frequencies_hz is not None:
                freqs = ", ".join(str(f) for f in ps.stimulus_frequencies_hz)
                timing_parts.append(f"stimulus frequencies: [{freqs}] Hz")
        if timing_parts:
            task_desc += " " + ", ".join(timing_parts) + "."

        entries["TaskDescription"] = task_desc

    # Instructions (RECOMMENDED)
    if exp and exp.instructions:
        entries["Instructions"] = exp.instructions
    else:
        entries.setdefault("Instructions", "n/a")

    # CogAtlasID (RECOMMENDED) — explicit value or paradigm-based fallback
    if exp:
        if exp.cog_atlas_id:
            entries["CogAtlasID"] = exp.cog_atlas_id
        elif exp.paradigm in _PARADIGM_COG_ATLAS:
            entries["CogAtlasID"] = _PARADIGM_COG_ATLAS[exp.paradigm]

        # CogPOID (RECOMMENDED)
        if exp.cog_po_id:
            entries["CogPOID"] = exp.cog_po_id

    # InstitutionName (RECOMMENDED)
    entries["InstitutionName"] = doc.institution if doc and doc.institution else "n/a"

    # InstitutionAddress (RECOMMENDED) — fall back to country
    if doc and doc.institution_address:
        entries["InstitutionAddress"] = doc.institution_address
    elif doc and doc.country:
        entries["InstitutionAddress"] = doc.country
    else:
        entries["InstitutionAddress"] = "n/a"

    # InstitutionalDepartmentName (RECOMMENDED)
    entries["InstitutionalDepartmentName"] = (
        doc.institution_department if doc and doc.institution_department else "n/a"
    )

    # CapManufacturer (RECOMMENDED) — ensure always present
    entries.setdefault("CapManufacturer", "n/a")

    # CapManufacturersModelName (RECOMMENDED)
    entries.setdefault("CapManufacturersModelName", "n/a")

    # ManufacturersModelName (RECOMMENDED)
    entries.setdefault("ManufacturersModelName", "n/a")

    # SubjectArtefactDescription (RECOMMENDED) — from impedance threshold
    if acq and acq.impedance_threshold_kohm is not None:
        imp = acq.impedance_threshold_kohm
        if isinstance(imp, dict):
            parts = [f"{ch}: {v} kOhm" for ch, v in imp.items()]
            entries["SubjectArtefactDescription"] = "Impedance kept below " + ", ".join(
                parts
            )
        else:
            entries["SubjectArtefactDescription"] = f"Impedance kept below {imp} kOhm"
    entries.setdefault("SubjectArtefactDescription", "n/a")

    # DeviceSerialNumber (RECOMMENDED)
    entries.setdefault("DeviceSerialNumber", "n/a")

    # RecordingType (RECOMMENDED) — MOABB stores raw continuous data
    entries["RecordingType"] = "continuous"

    # Description (RECOMMENDED by BIDS for all sidecars)
    if doc and doc.description:
        entries.setdefault("Description", doc.description)
    else:
        entries.setdefault("Description", "EEG recording.")

    return entries


def _build_dataset_description_kwargs(dataset):
    """Build enriched kwargs for ``mne_bids.make_dataset_description``.

    Parameters
    ----------
    dataset : BaseDataset
        The MOABB dataset instance.

    Returns
    -------
    dict
        Keyword arguments for ``make_dataset_description()``.
    """
    kwargs = dict(
        name=dataset.code,
        hed_version="8.4.0",
        dataset_type="derivative",
        generated_by=[
            dict(
                CodeURL="https://github.com/NeuroTechX/moabb",
                Name="moabb",
                Description="Mother of All BCI Benchmarks",
                Version=moabb.__version__,
            )
        ],
        source_datasets=[
            dict(
                DOI=dataset.doi or "n/a",
            )
        ],
    )

    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return kwargs

    doc = metadata.documentation

    if doc:
        # Enrich source_datasets with URL.
        sd = dict(DOI=dataset.doi or "n/a")
        if doc.data_url:
            sd["URL"] = doc.data_url
        if len(sd) > 1:  # more than just DOI
            kwargs["source_datasets"] = [sd]

        # data_license
        if doc.license:
            kwargs["data_license"] = doc.license

        # authors
        if doc.investigators:
            kwargs["authors"] = doc.investigators

        # funding
        if doc.funding:
            kwargs["funding"] = doc.funding

        # references_and_links
        refs = []
        if doc.data_url:
            refs.append(doc.data_url)
        if doc.associated_paper_doi:
            refs.append(doc.associated_paper_doi)
        if refs:
            kwargs["references_and_links"] = refs

        # doi — BIDS requires the format "doi:<value>"
        if doc.doi:
            doi_val = doc.doi
            if not doi_val.startswith("doi:"):
                doi_val = f"doi:{doi_val}"
            kwargs["doi"] = doi_val

        # acknowledgements
        if doc.acknowledgements:
            kwargs["acknowledgements"] = doc.acknowledgements

        # how_to_acknowledge
        if doc.how_to_acknowledge:
            kwargs["how_to_acknowledge"] = doc.how_to_acknowledge

        # ethics_approvals
        if doc.ethics_approval:
            kwargs["ethics_approvals"] = doc.ethics_approval

    return kwargs


def _update_participants_tsv(root, subject, metadata, raw=None):
    """Patch ``participants.tsv`` with demographic data from metadata.

    Adds ``age`` and ``group`` columns. Updates the ``participants.json``
    sidecar with column descriptions for newly added columns.

    Parameters
    ----------
    root : Path
        Root of the BIDS dataset.
    subject : int
        MOABB subject number (1-based).
    metadata : DatasetMetadata
        The dataset metadata.
    """
    if metadata is None:
        return

    participants = metadata.participants
    if participants is None:
        return

    tsv_path = Path(root) / "participants.tsv"
    if not tsv_path.exists():
        return

    subject_idx = subject - 1  # MOABB subjects are 1-based
    age = _resolve_subject_age(participants, subject_idx, raw=raw)

    # Determine group (clinical_population takes priority over health_status)
    group = "n/a"
    if participants.clinical_population:
        group = participants.clinical_population
    elif participants.health_status:
        group = participants.health_status

    sex = _resolve_subject_sex(participants, subject_idx, raw=raw)
    hand = _resolve_subject_hand(participants, subject_idx, raw=raw)

    # BCI experience
    bci_experience = "n/a"
    if participants.bci_experience:
        bci_experience = participants.bci_experience

    # Species (BIDS RECOMMENDED, default "homo sapiens")
    species = participants.species if participants.species else "n/a"

    # Read existing TSV (utf-8-sig strips BOM written by mne_bids)
    rows = []
    with open(tsv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)

    # Add columns if missing
    for col in ("age", "group", "sex", "hand", "species", "bci_experience"):
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = "n/a"

    # Update the row for this subject
    sub_id = f"sub-{subject}"
    for row in rows:
        if row.get("participant_id") == sub_id:
            if age != "n/a" and row.get("age", "n/a") == "n/a":
                row["age"] = age
            if group != "n/a" and row.get("group", "n/a") == "n/a":
                row["group"] = group
            if sex != "n/a" and row.get("sex", "n/a") == "n/a":
                row["sex"] = sex
            if hand != "n/a" and row.get("hand", "n/a") == "n/a":
                row["hand"] = hand
            if species != "n/a" and row.get("species", "n/a") == "n/a":
                row["species"] = species
            if bci_experience != "n/a" and row.get("bci_experience", "n/a") == "n/a":
                row["bci_experience"] = bci_experience

    # Write back
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Update participants.json sidecar with column descriptions
    json_path = Path(root) / "participants.json"
    sidecar = {}
    if json_path.exists():
        with open(json_path) as f:
            sidecar = json.load(f)

    updated = False
    # Top-level Description (RECOMMENDED by BIDS for all sidecars)
    if "Description" not in sidecar:
        if participants.health_status:
            sidecar["Description"] = (
                f"Participant demographics ({participants.health_status})."
            )
        else:
            sidecar["Description"] = "Participant demographic and metadata information."
        updated = True
    if "age" not in sidecar:
        age_desc = "Age of the participant"
        if participants.age_min is not None and participants.age_max is not None:
            age_desc = (
                f"Age of the participant "
                f"(range: {participants.age_min}-{participants.age_max} years)"
            )
        sidecar["age"] = {
            "Description": age_desc,
            "Units": "years",
        }
        updated = True
    if "group" not in sidecar:
        sidecar["group"] = {
            "Description": "Group the participant belonged to "
            "(e.g., healthy, patients)",
        }
        updated = True
    if "sex" not in sidecar:
        sidecar["sex"] = {
            "Description": "Sex of the participant",
            "Levels": {"male": "Male", "female": "Female", "other": "Other"},
        }
        updated = True
    if "hand" not in sidecar:
        sidecar["hand"] = {
            "Description": "Handedness of the participant",
            "Levels": {
                "right": "Right-handed",
                "left": "Left-handed",
                "ambidextrous": "Ambidextrous",
            },
        }
        updated = True
    if "species" not in sidecar:
        sidecar["species"] = {
            "Description": "Species of the participant (binomial name)",
        }
        updated = True
    if "bci_experience" not in sidecar:
        sidecar["bci_experience"] = {
            "Description": "BCI experience level of the participant",
        }
        updated = True
    # Ensure mne_bids-generated columns have Description (RECOMMENDED)
    if "participant_id" not in sidecar:
        sidecar["participant_id"] = {
            "Description": "Unique participant identifier",
        }
        updated = True
    if "weight" not in sidecar:
        sidecar["weight"] = {
            "Description": "Body weight of the participant",
            "Units": "kg",
        }
        updated = True
    if "height" not in sidecar:
        sidecar["height"] = {
            "Description": "Body height of the participant",
            "Units": "m",
        }
        updated = True

    if updated:
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent="\t")


def _update_electrodes_tsv(bids_path, metadata):
    """Patch ``*_electrodes.tsv`` with material and type columns from metadata.

    Adds BIDS RECOMMENDED ``type`` and ``material`` columns to the electrodes
    TSV file created by ``mne_bids.write_raw_bids()``.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        The BIDS path for the current recording.
    metadata : DatasetMetadata
        The dataset metadata.
    """
    if metadata is None:
        return

    acq = metadata.acquisition
    if acq is None:
        return

    electrode_type = acq.electrode_type
    # Prefer electrode_material; fall back to sensor_type for backward compat
    electrode_material = acq.electrode_material or acq.sensor_type
    if not electrode_type and not electrode_material:
        return

    root = Path(bids_path.root)
    subject = bids_path.subject
    # Find electrodes TSV files for this subject
    import glob as glob_mod

    pattern = str(root / f"sub-{subject}" / "**" / "*_electrodes.tsv")
    tsv_files = glob_mod.glob(pattern, recursive=True)
    if not tsv_files:
        return

    for tsv_file in tsv_files:
        tsv_path = Path(tsv_file)
        rows = []
        with open(tsv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = list(reader.fieldnames) if reader.fieldnames else []
            rows = list(reader)

        changed = False
        if electrode_type and "type" not in fieldnames:
            fieldnames.append("type")
            for row in rows:
                row["type"] = electrode_type
            changed = True
        if electrode_material and "material" not in fieldnames:
            fieldnames.append("material")
            for row in rows:
                row["material"] = electrode_material
            changed = True

        if changed:
            with open(tsv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)


def _build_hed_sidecar_annotations(dataset):
    """Build HED tag mapping for events.json sidecar trial_type column.

    Resolution order:

    1. Per-dataset override via ``metadata.experiment.hed_tags``
    2. Static paradigm-level lookup from ``_PARADIGM_HED_TAGS``
    3. Dynamic SSVEP frequency-based tags
    4. Generic ``Label/<event_name>`` fallback for any remaining events

    Parameters
    ----------
    dataset : BaseDataset
        The MOABB dataset instance.

    Returns
    -------
    dict
        Mapping of event names to HED tag strings.
    """
    metadata = getattr(dataset, "metadata", None)
    paradigm = dataset.paradigm
    event_names = list(dataset.event_id.keys())
    hed = {}

    # 1. Per-dataset override from metadata.experiment.hed_tags
    # Applied as highest-priority layer, remaining events still get defaults
    if metadata and metadata.experiment and metadata.experiment.hed_tags:
        for name in event_names:
            if name in metadata.experiment.hed_tags:
                hed[name] = metadata.experiment.hed_tags[name]

    # 2. Paradigm-level static defaults (don't overwrite custom overrides)
    if paradigm in _PARADIGM_HED_TAGS:
        mapping = _PARADIGM_HED_TAGS[paradigm]
        for name in event_names:
            if name not in hed and name in mapping:
                hed[name] = mapping[name]

    # 3. SSVEP: dynamic frequency-based tags
    if paradigm == "ssvep":
        for name in event_names:
            if name not in hed:
                if name == "rest":
                    hed[name] = "Experiment-structure, Rest"
                else:
                    safe_freq = name.replace(".", "_")
                    hed[name] = (
                        f"Sensory-event, Experimental-stimulus, Visual-presentation, (Label/{safe_freq})"
                    )

    # 4. Label fallback — every event gets a valid HED tag
    for name in event_names:
        if name not in hed:
            # Sanitize name for HED nameClass (alphanumeric, hyphens, underscores)
            safe_name = name.replace(".", "_").replace(" ", "_")
            hed[name] = f"Sensory-event, (Label/{safe_name})"

    return hed


def _update_events_json_sidecar(bids_path, hed_tags, metadata):
    """Enrich the events.json sidecar with HED and stimulus presentation.

    Adds ``"HED"`` tags to the ``"trial_type"`` column and
    ``"StimulusPresentation"`` when available from metadata.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        BIDS path for the current recording.
    hed_tags : dict
        Mapping of event names to HED tag strings.
    metadata : DatasetMetadata or None
        The dataset metadata.
    """
    events_json_path = bids_path.copy().update(suffix="events", extension=".json").fpath
    if not events_json_path.exists():
        return

    with open(events_json_path) as f:
        sidecar = json.load(f)

    changed = False

    # Top-level Description (RECOMMENDED by BIDS for all sidecars)
    if "Description" not in sidecar:
        # Build a descriptive string from task and event classes
        task = bids_path.task
        event_classes = sorted(hed_tags.keys()) if hed_tags else []
        if task and event_classes:
            classes_str = ", ".join(event_classes)
            sidecar["Description"] = f"Event annotations for {task} task ({classes_str})."
        elif task:
            sidecar["Description"] = f"Event annotations for {task} task."
        else:
            sidecar["Description"] = "Event annotations for the recording."
        changed = True

    # HED annotations
    if hed_tags:
        if "trial_type" not in sidecar:
            sidecar["trial_type"] = {
                "Description": "The type, category, or name of the event."
            }
        existing_hed = sidecar["trial_type"].get("HED", {})
        merged = {**hed_tags, **existing_hed}
        sidecar["trial_type"]["HED"] = merged
        changed = True

    # StimulusPresentation (RECOMMENDED)
    if (
        metadata
        and metadata.experiment
        and metadata.experiment.stimulus_presentation
        and "StimulusPresentation" not in sidecar
    ):
        sidecar["StimulusPresentation"] = metadata.experiment.stimulus_presentation
        changed = True

    # StimulusPresentation fallback: build from individual metadata fields.
    # SoftwareName is standard BIDS; StimulusType, StimulusModalities,
    # PrimaryModality, and Environment are MOABB extension fields.
    elif metadata and metadata.experiment and "StimulusPresentation" not in sidecar:
        exp = metadata.experiment
        acq = metadata.acquisition if metadata.acquisition else None
        bci = metadata.bci_application if metadata.bci_application else None
        sp = {}
        if acq and acq.software:
            sp["SoftwareName"] = acq.software
        if exp.stimulus_type:
            sp["StimulusType"] = exp.stimulus_type
        if exp.stimulus_modalities:
            sp["StimulusModalities"] = exp.stimulus_modalities
        if exp.primary_modality:
            sp["PrimaryModality"] = exp.primary_modality
        if bci and bci.environment:
            sp["Environment"] = bci.environment
        if sp:
            sidecar["StimulusPresentation"] = sp
            changed = True

    if changed:
        with open(events_json_path, "w") as f:
            json.dump(sidecar, f, indent="\t")


def _update_dataset_description_extra(root, metadata):
    """Merge extra fields into ``dataset_description.json``.

    Adds BIDS fields (like ``Keywords``) that ``mne_bids.make_dataset_description()``
    does not support as kwargs.

    Parameters
    ----------
    root : Path
        Root of the BIDS dataset.
    metadata : DatasetMetadata
        The dataset metadata.
    """
    if metadata is None:
        return

    desc_path = Path(root) / "dataset_description.json"
    if not desc_path.exists():
        return

    # Determine keywords: explicit > derived from tags
    keywords = None
    doc = metadata.documentation
    if doc and doc.keywords:
        keywords = doc.keywords
    elif metadata.tags:
        kw = []
        if metadata.tags.pathology:
            kw.extend(metadata.tags.pathology)
        if metadata.tags.modality:
            kw.extend(metadata.tags.modality)
        if metadata.tags.type:
            kw.extend(metadata.tags.type)
        if metadata.experiment:
            kw.append(metadata.experiment.paradigm)
        bci = metadata.bci_application
        if bci and bci.applications:
            kw.extend(bci.applications)
        if kw:
            keywords = list(dict.fromkeys(kw))  # deduplicate, preserve order

    with open(desc_path) as f:
        desc = json.load(f)

    changed = False
    if keywords and "Keywords" not in desc:
        desc["Keywords"] = keywords
        changed = True

    # PublicationYear is a MOABB extension (not standard BIDS); we write it
    # directly because mne_bids.make_dataset_description() rejects unknown keys.
    if doc and doc.publication_year and "PublicationYear" not in desc:
        desc["PublicationYear"] = doc.publication_year
        changed = True

    if changed:
        with open(desc_path, "w") as f:
            json.dump(desc, f, indent="\t")


def _write_metadata_yaml(root, dataset):
    """Write a ``<ClassName>.metadata.yaml`` sidecar into the BIDS root.

    Serialises the full ``DatasetMetadata`` dataclass to YAML so that every
    metadata field is available alongside the converted BIDS data.

    Parameters
    ----------
    root : Path
        Root of the BIDS dataset.
    dataset : BaseDataset
        The MOABB dataset instance.
    """
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return

    try:
        from dataclasses import asdict

        import yaml
    except ImportError:
        log.debug("PyYAML not installed — skipping metadata YAML export.")
        return

    def _clean(d):
        """Remove None, empty dict/list, and False values recursively."""
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            if v is None or v is False:
                continue
            if isinstance(v, dict):
                v = _clean(v)
                if not v:
                    continue
            if isinstance(v, list) and len(v) == 0:
                continue
            out[k] = v
        return out

    data = _clean(asdict(metadata))
    data["_dataset"] = {
        "class": type(dataset).__name__,
        "code": dataset.code,
        "doi": dataset.doi,
        "paradigm": dataset.paradigm,
        "n_subjects": len(dataset.subject_list),
        "interval": dataset.interval,
    }

    code_dir = Path(root) / "code"
    code_dir.mkdir(exist_ok=True)
    yaml_path = code_dir / f"{type(dataset).__name__}.metadata.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
            width=120,
        )


def _build_readme(dataset):
    """Build a comprehensive BIDS README from dataset docstring and metadata.

    Generates a plain-text README for the BIDS dataset root, incorporating
    the class docstring, core dataset attributes, and all available structured
    metadata sections.

    Parameters
    ----------
    dataset : BaseDataset
        The MOABB dataset instance.

    Returns
    -------
    str
        The README content.
    """
    lines = []
    metadata = getattr(dataset, "metadata", None)
    doc = metadata.documentation if metadata else None
    acq = metadata.acquisition if metadata else None
    part = metadata.participants if metadata else None
    exp = metadata.experiment if metadata else None
    preproc = metadata.preprocessing if metadata else None
    sig = metadata.signal_processing if metadata else None
    cv = metadata.cross_validation if metadata else None
    perf = metadata.performance if metadata else None
    bci = metadata.bci_application if metadata else None
    ps = metadata.paradigm_specific if metadata else None
    ds = metadata.data_structure if metadata else None
    tags = metadata.tags if metadata else None

    # ── Title ──
    lines.append(dataset.code)
    lines.append("=" * len(dataset.code))
    lines.append("")

    # ── Description from docstring ──
    raw_doc = getattr(type(dataset), "__doc__", "") or ""
    if raw_doc:
        # Extract the original (hand-written) docstring portion.
        # MetaclassDataset prepends auto-generated blocks *after* the first
        # paragraph.  The original docstring starts at the first paragraph and
        # ends before the ".. admonition::" blocks or the auto-generated table.
        paragraphs = raw_doc.split("\n\n")
        desc_paragraphs = []
        for para in paragraphs:
            stripped = para.strip()
            # Stop at auto-generated admonition blocks or rst table markers
            if stripped.startswith(".. admonition::") or stripped.startswith("+--"):
                break
            desc_paragraphs.append(stripped)
        if desc_paragraphs:
            # Clean indentation from each line
            for para in desc_paragraphs:
                clean = " ".join(line.strip() for line in para.splitlines())
                if clean:
                    lines.append(clean)
                    lines.append("")

    # ── Dataset Overview ──
    lines.append("Dataset Overview")
    lines.append("-" * 16)
    _kv(lines, "Code", dataset.code)
    _kv(lines, "Paradigm", dataset.paradigm)
    _kv(lines, "DOI", dataset.doi)
    _kv(lines, "Subjects", len(dataset.subject_list))
    _kv(lines, "Sessions per subject", dataset.n_sessions)
    _kv(lines, "Events", _format_dict(dataset.event_id))
    _kv(lines, "Trial interval", f"{dataset.interval} s" if dataset.interval else None)
    if metadata:
        _kv(
            lines,
            "Runs per session",
            metadata.runs_per_session if metadata.runs_per_session > 1 else None,
        )
        if metadata.sessions:
            _kv(lines, "Session IDs", ", ".join(str(s) for s in metadata.sessions))
        _kv(lines, "File format", metadata.file_format)
        _kv(
            lines,
            "Data preprocessed",
            metadata.data_processed if metadata.data_processed else None,
        )
        if metadata.contributing_labs:
            _kv(lines, "Contributing labs", ", ".join(metadata.contributing_labs))
        elif metadata.n_contributing_labs:
            _kv(lines, "Number of contributing labs", metadata.n_contributing_labs)
    lines.append("")

    # ── Acquisition ──
    if acq:
        lines.append("Acquisition")
        lines.append("-" * 11)
        _kv(lines, "Sampling rate", f"{acq.sampling_rate} Hz")
        _kv(lines, "Number of channels", acq.n_channels)
        _kv(lines, "Channel types", _format_dict(acq.channel_types))
        _kv(lines, "Channel names", ", ".join(acq.sensors) if acq.sensors else None)
        _kv(lines, "Montage", acq.montage)
        _kv(lines, "Hardware", acq.hardware)
        _kv(lines, "Software", acq.software)
        _kv(lines, "Reference", acq.reference)
        _kv(lines, "Ground", acq.ground)
        _kv(lines, "Sensor type", acq.sensor_type)
        _kv(lines, "Line frequency", f"{acq.line_freq} Hz")
        _kv(lines, "Online filters", acq.filters)
        _kv(
            lines,
            "Impedance threshold",
            (
                f"{acq.impedance_threshold_kohm} kOhm"
                if acq.impedance_threshold_kohm
                else None
            ),
        )
        _kv(lines, "Cap manufacturer", acq.cap_manufacturer)
        _kv(lines, "Cap model", acq.cap_model)
        _kv(lines, "Electrode type", acq.electrode_type)
        _kv(lines, "Electrode material", acq.electrode_material)
        if acq.auxiliary_channels:
            aux = acq.auxiliary_channels
            aux_parts = []
            if aux.has_eog and aux.eog_channels:
                eog_str = f"EOG ({aux.eog_channels} ch"
                if aux.eog_type:
                    eog_str += f", {', '.join(aux.eog_type)}"
                eog_str += ")"
                aux_parts.append(eog_str)
            if aux.has_emg and aux.emg_channels:
                aux_parts.append(f"EMG ({aux.emg_channels} ch)")
            if aux.other_physiological:
                aux_parts.extend(aux.other_physiological)
            if aux_parts:
                _kv(lines, "Auxiliary channels", ", ".join(aux_parts))
        lines.append("")

    # ── Participants ──
    if part:
        lines.append("Participants")
        lines.append("-" * 12)
        _kv(lines, "Number of subjects", part.n_subjects)
        _kv(lines, "Health status", part.health_status)
        _kv(lines, "Clinical population", part.clinical_population)
        # Age
        age_parts = []
        if part.age_mean is not None:
            age_parts.append(f"mean={part.age_mean}")
        if part.age_std is not None:
            age_parts.append(f"std={part.age_std}")
        if part.age_min is not None:
            age_parts.append(f"min={part.age_min}")
        if part.age_max is not None:
            age_parts.append(f"max={part.age_max}")
        if age_parts:
            _kv(lines, "Age", ", ".join(age_parts))
        _kv(lines, "Gender distribution", _format_dict(part.gender))
        _kv(lines, "Handedness", part.handedness)
        _kv(lines, "BCI experience", part.bci_experience)
        _kv(lines, "Species", part.species if part.species != "homo sapiens" else None)
        lines.append("")

    # ── Experiment ──
    if exp:
        lines.append("Experimental Protocol")
        lines.append("-" * 21)
        _kv(lines, "Paradigm", exp.paradigm)
        _kv(lines, "Task type", exp.task_type)
        _kv(lines, "Number of classes", exp.n_classes)
        _kv(
            lines,
            "Class labels",
            ", ".join(exp.class_labels) if exp.class_labels else None,
        )
        _kv(
            lines,
            "Trial duration",
            f"{exp.trial_duration} s" if exp.trial_duration else None,
        )
        _kv(lines, "Trials per class", _format_dict(exp.trials_per_class))
        if exp.tasks:
            _kv(lines, "Tasks", ", ".join(exp.tasks))
        _kv(lines, "Study design", exp.study_design)
        _kv(lines, "Study domain", exp.study_domain)
        _kv(lines, "Feedback type", exp.feedback_type)
        _kv(lines, "Stimulus type", exp.stimulus_type)
        _kv(
            lines,
            "Stimulus modalities",
            ", ".join(exp.stimulus_modalities) if exp.stimulus_modalities else None,
        )
        _kv(lines, "Primary modality", exp.primary_modality)
        _kv(lines, "Synchronicity", exp.synchronicity)
        _kv(lines, "Mode", exp.mode)
        _kv(lines, "Training/test split", exp.has_training_test_split)
        _kv(lines, "Instructions", exp.instructions)
        _kv(lines, "Cognitive Atlas ID", exp.cog_atlas_id)
        _kv(lines, "Cognitive Paradigm Ontology ID", exp.cog_po_id)
        if exp.stimulus_presentation:
            _kv(lines, "Stimulus presentation", _format_dict(exp.stimulus_presentation))
        lines.append("")

    # ── HED Event Annotations ──
    hed_tags = _build_hed_sidecar_annotations(dataset)
    if hed_tags:
        lines.append("HED Event Annotations")
        lines.append("-" * 21)
        lines.append(
            "  Schema: HED 8.4.0 | " "Browse: https://www.hedtags.org/hed-schema-browser"
        )
        lines.append("")
        for event_name, tag_str in hed_tags.items():
            elements = _split_hed_top_level(tag_str)
            nodes = [_hed_element_to_tree(e) for e in elements]
            lines.append(f"  {event_name}")
            lines.extend(_render_hed_tree(nodes))
            lines.append("")

    # ── Paradigm-Specific ──
    if ps:
        has_content = any(
            getattr(ps, f, None) is not None
            for f in (
                "detected_paradigm",
                "stimulus_frequencies_hz",
                "frequency_resolution_hz",
                "code_type",
                "code_length",
                "n_targets",
                "n_repetitions",
                "isi_ms",
                "soa_ms",
                "imagery_tasks",
                "cue_duration_s",
                "imagery_duration_s",
            )
        )
        if has_content:
            lines.append("Paradigm-Specific Parameters")
            lines.append("-" * 28)
            _kv(lines, "Detected paradigm", ps.detected_paradigm)
            # SSVEP
            _kv(
                lines,
                "Stimulus frequencies",
                (
                    f"{ps.stimulus_frequencies_hz} Hz"
                    if ps.stimulus_frequencies_hz
                    else None
                ),
            )
            _kv(
                lines,
                "Frequency resolution",
                (
                    f"{ps.frequency_resolution_hz} Hz"
                    if ps.frequency_resolution_hz
                    else None
                ),
            )
            # c-VEP
            _kv(lines, "Code type", ps.code_type)
            _kv(lines, "Code length", ps.code_length)
            # P300
            _kv(lines, "Number of targets", ps.n_targets)
            _kv(lines, "Number of repetitions", ps.n_repetitions)
            _kv(
                lines, "Inter-stimulus interval", f"{ps.isi_ms} ms" if ps.isi_ms else None
            )
            _kv(
                lines,
                "Stimulus onset asynchrony",
                f"{ps.soa_ms} ms" if ps.soa_ms else None,
            )
            # Motor Imagery
            _kv(
                lines,
                "Imagery tasks",
                ", ".join(ps.imagery_tasks) if ps.imagery_tasks else None,
            )
            _kv(
                lines,
                "Cue duration",
                f"{ps.cue_duration_s} s" if ps.cue_duration_s else None,
            )
            _kv(
                lines,
                "Imagery duration",
                f"{ps.imagery_duration_s} s" if ps.imagery_duration_s else None,
            )
            lines.append("")

    # ── Data Structure ──
    if ds:
        has_content = any(
            getattr(ds, f, None) is not None
            for f in (
                "n_trials",
                "n_trials_per_class",
                "n_blocks",
                "block_duration_s",
                "trials_context",
            )
        )
        if has_content:
            lines.append("Data Structure")
            lines.append("-" * 14)
            _kv(lines, "Trials", ds.n_trials)
            _kv(
                lines,
                "Trials per class",
                (
                    _format_dict(ds.n_trials_per_class)
                    if isinstance(ds.n_trials_per_class, dict)
                    else ds.n_trials_per_class
                ),
            )
            _kv(lines, "Blocks per session", ds.n_blocks)
            _kv(
                lines,
                "Block duration",
                f"{ds.block_duration_s} s" if ds.block_duration_s else None,
            )
            _kv(lines, "Trials context", ds.trials_context)
            lines.append("")

    # ── Preprocessing ──
    if preproc:
        has_content = any(
            getattr(preproc, f, None) is not None
            for f in (
                "data_state",
                "preprocessing_applied",
                "preprocessing_steps",
                "highpass_hz",
                "lowpass_hz",
                "bandpass",
                "notch_hz",
                "filter_type",
                "filter_order",
                "artifact_methods",
                "re_reference",
                "downsampled_to_hz",
                "epoch_window",
                "notes",
            )
        )
        if has_content:
            lines.append("Preprocessing")
            lines.append("-" * 13)
            _kv(lines, "Data state", preproc.data_state)
            _kv(lines, "Preprocessing applied", preproc.preprocessing_applied)
            if preproc.preprocessing_steps:
                _kv(lines, "Steps", ", ".join(preproc.preprocessing_steps))
            _kv(
                lines,
                "Highpass filter",
                f"{preproc.highpass_hz} Hz" if preproc.highpass_hz else None,
            )
            _kv(
                lines,
                "Lowpass filter",
                f"{preproc.lowpass_hz} Hz" if preproc.lowpass_hz else None,
            )
            _kv(lines, "Bandpass filter", preproc.bandpass)
            _kv(
                lines,
                "Notch filter",
                f"{preproc.notch_hz} Hz" if preproc.notch_hz else None,
            )
            _kv(lines, "Filter type", preproc.filter_type)
            _kv(lines, "Filter order", preproc.filter_order)
            if preproc.artifact_methods:
                _kv(lines, "Artifact methods", ", ".join(preproc.artifact_methods))
            _kv(lines, "Re-reference", preproc.re_reference)
            _kv(
                lines,
                "Downsampled to",
                f"{preproc.downsampled_to_hz} Hz" if preproc.downsampled_to_hz else None,
            )
            _kv(lines, "Epoch window", preproc.epoch_window)
            _kv(lines, "Notes", preproc.notes)
            lines.append("")

    # ── Signal Processing ──
    if sig:
        has_content = any(
            getattr(sig, f, None) is not None
            for f in (
                "classifiers",
                "feature_extraction",
                "frequency_bands",
                "spatial_filters",
            )
        )
        if has_content:
            lines.append("Signal Processing")
            lines.append("-" * 17)
            if sig.classifiers:
                _kv(lines, "Classifiers", ", ".join(sig.classifiers))
            if sig.feature_extraction:
                _kv(lines, "Feature extraction", ", ".join(sig.feature_extraction))
            if sig.frequency_bands:
                band_parts = []
                _freq_band_display = {"analyzed_range": "analyzed"}
                for name, val in sig.frequency_bands.items():
                    if val:
                        display_name = _freq_band_display.get(name, name)
                        band_parts.append(f"{display_name}={val} Hz")
                if band_parts:
                    _kv(lines, "Frequency bands", "; ".join(band_parts))
            if sig.spatial_filters:
                _kv(lines, "Spatial filters", ", ".join(sig.spatial_filters))
            lines.append("")

    # ── Cross-Validation ──
    if cv:
        has_content = any(
            getattr(cv, f, None) is not None
            for f in ("cv_method", "cv_folds", "evaluation_type")
        )
        if has_content:
            lines.append("Cross-Validation")
            lines.append("-" * 16)
            _kv(lines, "Method", cv.cv_method)
            _kv(lines, "Folds", cv.cv_folds)
            if cv.evaluation_type:
                _kv(lines, "Evaluation type", ", ".join(cv.evaluation_type))
            lines.append("")

    # ── Performance ──
    if perf:
        _perf_units = {
            "accuracy_percent": "%",
            "itr_bits_per_min": " bits/min",
        }
        lines.append("Performance (Original Study)")
        lines.append("-" * 28)
        for key, val in perf.items():
            if val is not None:
                if key == "other_metrics" and isinstance(val, dict):
                    formatted = ", ".join(f"{k}={v}" for k, v in val.items())
                    _kv(lines, "Other Metrics", formatted)
                else:
                    unit = _perf_units.get(key, "")
                    label = key.replace("_percent", "").replace("_bits_per_min", "")
                    label = label.replace("_", " ").title()
                    _kv(lines, label, f"{val}{unit}")
        lines.append("")

    # ── BCI Application ──
    if bci:
        has_content = any(
            getattr(bci, f, None) is not None
            for f in ("applications", "environment", "online_feedback")
        )
        if has_content:
            lines.append("BCI Application")
            lines.append("-" * 15)
            if bci.applications:
                _kv(lines, "Applications", ", ".join(bci.applications))
            _kv(lines, "Environment", bci.environment)
            _kv(lines, "Online feedback", bci.online_feedback)
            lines.append("")

    # ── Tags ──
    if tags:
        has_content = any(
            getattr(tags, f, None) is not None for f in ("pathology", "modality", "type")
        )
        if has_content:
            lines.append("Tags")
            lines.append("-" * 4)
            if tags.pathology:
                _kv(lines, "Pathology", ", ".join(tags.pathology))
            if tags.modality:
                _kv(lines, "Modality", ", ".join(tags.modality))
            if tags.type:
                _kv(lines, "Type", ", ".join(tags.type))
            lines.append("")

    # ── Documentation ──
    if doc:
        has_content = any(
            getattr(doc, f, None) is not None
            for f in (
                "doi",
                "description",
                "investigators",
                "institution",
                "country",
                "repository",
                "data_url",
                "license",
                "publication_year",
                "funding",
                "senior_author",
                "contact_info",
                "associated_paper_doi",
                "institution_address",
                "institution_department",
                "ethics_approval",
                "acknowledgements",
                "how_to_acknowledge",
                "keywords",
            )
        )
        if has_content:
            lines.append("Documentation")
            lines.append("-" * 13)
            _kv(lines, "Description", doc.description)
            _kv(lines, "DOI", doc.doi)
            _kv(lines, "Associated paper DOI", doc.associated_paper_doi)
            _kv(lines, "License", doc.license)
            if doc.investigators:
                _kv(lines, "Investigators", ", ".join(doc.investigators))
            _kv(lines, "Senior author", doc.senior_author)
            if doc.contact_info:
                _kv(lines, "Contact", "; ".join(doc.contact_info))
            _kv(lines, "Institution", doc.institution)
            _kv(lines, "Department", doc.institution_department)
            _kv(lines, "Address", doc.institution_address)
            _kv(lines, "Country", doc.country)
            _kv(lines, "Repository", doc.repository)
            _kv(lines, "Data URL", doc.data_url)
            _kv(lines, "Publication year", doc.publication_year)
            if doc.funding:
                _kv(lines, "Funding", "; ".join(doc.funding))
            if doc.ethics_approval:
                _kv(lines, "Ethics approval", "; ".join(doc.ethics_approval))
            _kv(lines, "Acknowledgements", doc.acknowledgements)
            _kv(lines, "How to acknowledge", doc.how_to_acknowledge)
            if doc.keywords:
                _kv(lines, "Keywords", ", ".join(doc.keywords))
            lines.append("")

    # ── External Links ──
    ext = metadata.external_links if metadata else None
    if ext:
        lines.append("External Links")
        lines.append("-" * 14)
        for name, url in ext.items():
            _kv(lines, name.replace("_", " ").title(), url)
        lines.append("")

    # ── Abstract / Methodology ──
    if metadata and metadata.abstract:
        lines.append("Abstract")
        lines.append("-" * 8)
        lines.append(metadata.abstract)
        lines.append("")
    if metadata and metadata.methodology:
        lines.append("Methodology")
        lines.append("-" * 11)
        lines.append(metadata.methodology)
        lines.append("")

    # ── References ──
    # Extract references from the original class docstring (not the auto-generated one)
    orig_doc = type(dataset).__doc__ or ""
    refs = _extract_references_from_docstring(orig_doc)
    lines.append("References")
    lines.append("-" * 10)
    if refs:
        lines.append(refs)
    lines.append(
        "Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., "
        "Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Hochenberger, R., "
        "Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and "
        "Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into "
        "the BIDS format and facilitating their analysis. Journal of Open Source "
        "Software 4: (1896). https://doi.org/10.21105/joss.01896"
    )
    lines.append("")
    lines.append(
        "Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., "
        "Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an "
        "extension to the brain imaging data structure for "
        "electroencephalography. Scientific Data, 6, 103. "
        "https://doi.org/10.1038/s41597-019-0104-8"
    )
    lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append(
        f"Generated by MOABB {moabb.__version__} " "(Mother of All BCI Benchmarks)"
    )
    lines.append("https://github.com/NeuroTechX/moabb")
    lines.append("")

    return "\n".join(lines)


def _kv(lines, key, value):
    """Append a key-value line if the value is not None or 'n/a'."""
    if value is None or value == "n/a":
        return
    lines.append(f"  {key}: {value}")


def _format_dict(d):
    """Format a dict as a compact key=value string."""
    if not d:
        return None
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _extract_references_from_docstring(docstring):
    """Extract reference citations from a dataset class docstring.

    Looks for the ``references`` / ``References`` section and ``.. [N]``
    citation directives, returning clean plain-text references.
    """
    if not docstring:
        return ""
    # Collect individual reference blocks: each starts with ``.. [N]``
    references = []
    current_ref = []
    in_refs = False
    for line in docstring.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        # Detect start of references section
        if low in ("references", "references:") or low.startswith(".. ["):
            in_refs = True
        if low.startswith(".. admonition::") or low.startswith("+--"):
            if in_refs:
                break
        if not in_refs:
            continue
        # Skip section headers and underlines
        if low in ("references", "references:", "") or set(stripped) <= {"-", "="}:
            if current_ref:
                references.append(" ".join(current_ref))
                current_ref = []
            continue
        # New citation directive starts a new reference
        if re.match(r"^\.\.\s*\[\d+\]", stripped):
            if current_ref:
                references.append(" ".join(current_ref))
                current_ref = []
            cleaned = re.sub(r"^\.\.\s*\[\d+\]\s*", "", stripped)
            if cleaned:
                current_ref.append(cleaned)
        else:
            # Continuation line of current reference
            current_ref.append(stripped)
    if current_ref:
        references.append(" ".join(current_ref))
    return "\n\n".join(references)


def _split_hed_top_level(hed_str):
    """Split a HED tag string by commas at the top level.

    Respects parenthetical grouping so that commas inside groups
    are not treated as separators.

    Parameters
    ----------
    hed_str : str
        A HED tag string, e.g. ``"Sensory-event, (Imagine, (Move, Hand))"``.

    Returns
    -------
    list of str
        Top-level elements (tags and groups with outer parentheses preserved).
    """
    elements = []
    depth = 0
    current = []
    for char in hed_str:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            s = "".join(current).strip()
            if s:
                elements.append(s)
            current = []
        else:
            current.append(char)
    s = "".join(current).strip()
    if s:
        elements.append(s)
    return elements


def _hed_element_to_tree(element):
    """Convert a single HED element (tag or group) to a tree node.

    Returns ``(label, children)`` where *children* is a list of similar
    tuples.  Leaf-only groups (no nested sub-groups) are rendered as a
    single comma-separated label to keep the tree compact.

    Parameters
    ----------
    element : str
        A HED tag (``"Sensory-event"``) or group
        (``"(Imagine, (Move, Hand))"``).

    Returns
    -------
    tuple
        ``(label, children)`` tree node.
    """
    element = element.strip()
    if element.startswith("(") and element.endswith(")"):
        inner = element[1:-1]
        parts = _split_hed_top_level(inner)
        if not parts:
            return (element, [])
        # Leaf-only group (no nested sub-groups) → inline label
        has_subgroups = any(p.strip().startswith("(") for p in parts)
        if not has_subgroups:
            return (", ".join(p.strip() for p in parts), [])
        # Mixed: first element is the head, rest become children
        head = parts[0].strip()
        rest = parts[1:]
        return (head, [_hed_element_to_tree(r) for r in rest])
    return (element, [])


def _render_hed_tree(nodes, prefix="    "):
    """Render tree nodes as ASCII art using box-drawing characters.

    Parameters
    ----------
    nodes : list of tuple
        Each node is ``(label, children)`` as returned by
        :func:`_hed_element_to_tree`.
    prefix : str
        Current indentation prefix.

    Returns
    -------
    list of str
        Lines of the rendered tree.
    """
    lines = []
    for i, (label, children) in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        lines.append(f"{prefix}{connector}{label}")
        if children:
            extension = "   " if is_last else "\u2502  "
            lines.extend(_render_hed_tree(children, prefix + extension))
    return lines


def get_bids_root(code, path=None):
    """Path to the root of the BIDS structure used for caching.

    See :class:`moabb.datasets.base.BaseDataset` and
    :class:`moabb.datasets.base.CacheConfig` for more information
     on the MOABB caching mechanism.

    Parameters
    ----------
    code : str
        The dataset code from the MOABB dataset.
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.

    Returns
    -------
    root : Path
        Path to the root of the BIDS structure.
    """

    mne_path = Path(dl.get_dataset_path(code, path))
    cache_dir = f"MNE-BIDS-{camel_to_kebab_case(code)}"
    root = mne_path / cache_dir
    return root


def camel_to_kebab_case(name):
    """Converts a CamelCase string to kebab-case."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", name).lower()


def subject_moabb_to_bids(subject: int):
    """Convert the subject number to string (subject)."""
    return str(subject)


def subject_bids_to_moabb(subject: str):
    """Convert the subject string to int(subject)."""
    return int(subject)


def run_moabb_to_bids(run: str):
    """Convert the run to run index plus eventually description."""
    p = r"([0-9]+)(|[a-zA-Z]+[a-zA-Z0-9]*)"
    idx, desc = re.fullmatch(p, run).groups()
    out = {"run": idx}
    if desc:
        out["recording"] = desc
    return out


def run_bids_to_moabb(path: mne_bids.BIDSPath):
    """Extracts the run index plus eventually description from a path."""
    if path.recording is None:
        return path.run
    return f"{path.run}{path.recording}"


@dataclass
class BIDSInterfaceBase(abc.ABC):
    """Base class for BIDSInterface.

    This dataclass is used to convert a MOABB dataset to MOABB BIDS.
    It is used by the ``get_data`` method of any MOABB dataset.

    Parameters
    ----------
    dataset : BaseDataset
        The dataset to convert.
    subject : int
        The subject to convert.
    path : str
        The path to the BIDS dataset.
    process_pipeline : Pipeline
        The processing pipeline used to convert the data.
    verbose : str
        The verbosity level.

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    dataset: "BaseDataset"
    subject: int
    path: str = None
    process_pipeline: "Pipeline" = None
    verbose: str = None
    _dataset_type: str = "derivative"

    @property
    def processing_params(self):
        """Return the processing parameters."""
        # TODO: add dataset kwargs
        return self.process_pipeline

    @property
    def desc(self):
        """Return the description of the processing pipeline."""
        return get_digest(self.processing_params)

    def __repr__(self):
        """Return the representation of the BIDSInterface."""
        desc = self.desc
        desc_str = f"{desc:.7}" if desc is not None else "None"
        return (
            f"{self.dataset.code!r} sub-{self.subject} "
            f"suffix-{self._suffix} desc-{desc_str}"
        )

    @property
    def root(self):
        """Return the root path of the BIDS dataset."""
        return get_bids_root(self.dataset.code, self.path)

    def _lock_file(self, session):
        """Return the lock file path for a specific session.

        This file is saved after writing all runs for a session to ensure
        the session's data was completely saved. It is stored in the
        ``code/`` folder of the BIDS dataset root, which is
        BIDS-validator exempt.
        """
        return (
            self.root
            / "code"
            / f"sub-{subject_moabb_to_bids(self.subject)}_ses-{session}_desc-{self.desc}_lockfile.json"
        )

    @property
    def _migration_lock_file(self):
        """Per-subject lock file used for backward compatibility.

        This was the lock file format used between the initial BIDS caching
        implementation migration to the ``code/`` folder and the switch to
        per-session lock files.
        """
        return (
            self.root
            / "code"
            / f"sub-{subject_moabb_to_bids(self.subject)}_desc-{self.desc}_lockfile.json"
        )

    @property
    def _legacy_lock_file(self):
        """Return the legacy lock file path for backward compatibility.

        In the original implementation, the lock file was stored inside the
        subject folder of the BIDS structure. This property allows loading
        caches that were created with the old path.
        """
        return mne_bids.BIDSPath(
            root=self.root,
            subject=subject_moabb_to_bids(self.subject),
            description=self.desc,
            extension=".json",
            suffix="lockfile",  # necessary for unofficial files
            check=False,
        )

    def erase(self):
        """Erase the cache of the subject if it exists."""
        log.info("Starting erasing cache of %s...", repr(self))

        if not self.root.exists():
            log.info("No cache directory at %s, nothing to erase.", self.root)
            return

        # Find all matching paths to determine which sessions exist
        paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=subject_moabb_to_bids(self.subject),
            descriptions=self.desc,
            check=self._check,
            suffixes=self._suffix,
            extensions=self._extension,
        )
        sessions = set(p.session for p in paths)

        # Remove lock files FIRST, before calling session_path.rm(). In some
        # versions of mne_bids, rm() globs all files under root and finds our
        # lock files (named with BIDS entity syntax). It then derives a wrong
        # "canonical" BIDS path and tries to unlink a non-existent file.
        code_dir = self.root / "code"
        if code_dir.exists():
            pattern = f"sub-{subject_moabb_to_bids(self.subject)}_ses-*_desc-{self.desc}_lockfile.json"
            for lock_file in code_dir.glob(pattern):
                lock_file.unlink()
        # Remove migration-style per-subject lock file if present
        if self._migration_lock_file.exists():
            self._migration_lock_file.unlink()
        # Remove original legacy lock file if present
        legacy = self._legacy_lock_file
        if legacy.fpath.exists():
            legacy.fpath.unlink()

        # Remove data files per session to avoid mne_bids failing when
        # looking up scans.tsv across multiple sessions.  Note: mne_bids
        # rm() automatically calls rmtree on the subject directory when
        # the last session is removed (i.e. no remaining files under
        # sub-{subject}/), so empty directories are cleaned up.
        for session in sessions:
            session_path = mne_bids.BIDSPath(
                root=self.root,
                subject=subject_moabb_to_bids(self.subject),
                session=session,
                description=self.desc,
                check=False,
            )
            try:
                session_path.rm(safe_remove=False)
            except RuntimeError:
                session_dir = (
                    Path(self.root)
                    / f"sub-{subject_moabb_to_bids(self.subject)}"
                    / f"ses-{session}"
                )
                if session_dir.is_dir():
                    shutil.rmtree(session_dir)
        log.info("Finished erasing cache of %s.", repr(self))

    def load(self, preload=False):
        """Load the cache of the subject if it exists and returns it as
        a nested dictionary with the following structure::

            sessions_data = {'session_id':
                        {'run_id': run}
                    }

        If the cache is not present, returns None.
        """
        log.info("Attempting to retrieve cache of %s...", repr(self))
        code_dir = self.root / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # Check for non-session-aware legacy lock files (backward compatibility)
        legacy_lock_exists = (
            self._migration_lock_file.exists() or self._legacy_lock_file.fpath.exists()
        )
        # Ensure the legacy BIDSPath directory exists for mne_bids compatibility
        self._legacy_lock_file.mkdir(exist_ok=True)

        paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=subject_moabb_to_bids(self.subject),
            descriptions=self.desc,
            extensions=self._extension,
            check=self._check,
            # datatypes="eeg", # commented for compatibility with cache saved in previous versions
            suffixes=self._suffix,
        )

        if not paths:
            log.info("No cache found at %s.", str(code_dir))
            return None

        # Check per-session lock files unless a legacy (non-session-aware) lock
        # file exists, which indicates the whole subject was already cached.
        if not legacy_lock_exists:
            found_sessions = {path.session for path in paths}
            missing = [s for s in found_sessions if not self._lock_file(s).exists()]
            if missing:
                log.info("No cache found at %s.", str(code_dir))
                return None

        sessions_data = {}
        for path in paths:
            session_moabb = path.session
            session = sessions_data.setdefault(session_moabb, {})
            run = self._load_file(path, preload=preload)
            session[run_bids_to_moabb(path)] = run
        log.info("Finished reading cache of %s", repr(self))
        return sessions_data

    def save(self, sessions_data):
        """Save the cache of the subject.
        The data to be saved should be a nested dictionary
        with the following structure::

            sessions_data = {'session_id':
                        {'run_id': run}
                    }

        If a ``run`` is None, it will be skipped.

        The type of the ``run`` object can vary (see the subclases).
        """
        log.info("Starting caching %s", repr(self))
        mne_bids.BIDSPath(root=self.root).mkdir(exist_ok=True)

        lock_data = dict(processing_params=str(self.processing_params))

        # Write .bidsignore so the BIDS validator skips MOABB lockfiles
        bidsignore = Path(self.root) / ".bidsignore"
        if not bidsignore.exists():
            bidsignore.write_text("*_lockfile.json\n")

        for session, runs in sessions_data.items():
            for run, obj in runs.items():
                if obj is None:
                    log.warning(
                        "Skipping caching %s session %s run %s because it is None.",
                        repr(self),
                        session,
                        run,
                    )
                    continue

                run_kwargs = run_moabb_to_bids(run)
                bids_path = mne_bids.BIDSPath(
                    root=self.root,
                    subject=subject_moabb_to_bids(self.subject),
                    session=session,
                    task=self.dataset.paradigm,
                    **run_kwargs,
                    description=self.desc,
                    extension=self._extension,
                    datatype="eeg",
                    suffix=self._suffix,
                    check=self._check,
                )

                bids_path.mkdir(exist_ok=True)
                self._write_file(bids_path, obj)

            self._write_lock_file(session, lock_data)

        # Write dataset_description.json after all files so that it
        # overwrites any version created internally by mne_bids.write_raw_bids.
        desc_kwargs = _build_dataset_description_kwargs(self.dataset)
        mne_bids.make_dataset_description(
            path=str(self.root),
            overwrite=True,
            verbose=self.verbose,
            **desc_kwargs,
        )

        # Merge extra fields (Keywords, etc.) that make_dataset_description
        # doesn't support as kwargs
        metadata = getattr(self.dataset, "metadata", None)
        _update_dataset_description_extra(self.root, metadata)

        # Write comprehensive README (after write_raw_bids to overwrite its
        # boilerplate README with our enriched version)
        readme_path = Path(self.root) / "README"
        readme_path.write_text(_build_readme(self.dataset), encoding="utf-8")

        # Write full metadata as YAML sidecar
        _write_metadata_yaml(self.root, self.dataset)

        log.info("Finished caching %s to disk.", repr(self))

    def _write_lock_file(self, session, lock_data):
        """Write the lock file for a session to signal that saving is complete."""
        lock_file = self._lock_file(session)
        log.debug("Writing %s", lock_file)
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with lock_file.open("w") as file:
            json.dump(lock_data, file)

    @abc.abstractmethod
    def _load_file(self, bids_path, preload):
        pass

    @abc.abstractmethod
    def _write_file(self, bids_path, obj):
        pass

    @property
    @abc.abstractmethod
    def _extension(self):
        pass

    @property
    @abc.abstractmethod
    def _check(self):
        pass

    @property
    @abc.abstractmethod
    def _suffix(self):
        pass


_FORMAT_EXTENSION_MAP = {
    "EDF": ".edf",
    "BrainVision": ".vhdr",
    "EEGLAB": ".set",
}


class BIDSInterfaceRawEDF(BIDSInterfaceBase):
    """BIDS Interface for Raw EEG files.

    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``mne.io.BaseRaw`` instance."""

    _format = "EDF"

    @property
    def _extension(self):
        return _FORMAT_EXTENSION_MAP[self._format]

    @property
    def _check(self):
        return True

    @property
    def _suffix(self):
        return "eeg"

    def _load_file(self, bids_path, preload):
        raw = mne_bids.read_raw_bids(
            bids_path, extra_params=dict(preload=preload), verbose=self.verbose
        )
        return raw

    def _write_file(self, bids_path, raw):
        if not raw.annotations:
            raise ValueError(
                "Raw object must have annotations to be saved in BIDS format."
                "Use the SetRawAnnotations pipeline for this."
            )
        if raw.info.get("line_freq", None) is None:
            # specify line frequency if not present as required by BIDS
            raw.info["line_freq"] = 50

        # Enrich raw.info from dataset metadata (sex, hand, line_freq)
        metadata = getattr(self.dataset, "metadata", None)
        _enrich_raw_info_from_metadata(raw, metadata, self.subject)

        if raw.info.get("subject_info", None) is None:
            # specify subject info as required by BIDS
            raw.info["subject_info"] = {
                "his_id": subject_moabb_to_bids(self.subject),
            }
        if raw.info.get("device_info", None) is None:
            # specify device info as required by BIDS
            raw.info["device_info"] = {"type": "eeg"}

        # Recover a meaningful meas_date when the original file lacks one.
        # Many EEG files ship with epoch-zero (1970-01-01) or None.  Use the
        # dataset's publication_year as a best-effort approximation so that
        # the BIDS acq_time field is not obviously wrong.
        _epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        meas_date = raw.info.get("meas_date", None)
        if meas_date is None or meas_date == _epoch:
            pub_year = None
            if metadata and metadata.documentation:
                pub_year = metadata.documentation.publication_year
            if pub_year:
                raw.set_meas_date(
                    datetime.datetime(pub_year, 1, 1, tzinfo=datetime.timezone.utc)
                )

        # Otherwise, the montage would still have the stim channel
        # which is dropped by mne_bids.write_raw_bids:
        picks = pick_channels_for_modalities(raw.info, self.dataset.return_all_modalities)
        raw.pick(picks)

        # By using the same anonymization `daysback` number we can
        # preserve the longitudinal structure of multiple sessions for a
        # single subject and the relation between subjects. Be sure to
        # change or delete this number before putting code online, you
        # wouldn't want to inadvertently de-anonymize your data.
        #
        # Note that we do not need to pass any events, as the dataset
        # is already equipped with annotations, which will be converted to
        # BIDS events automatically.

        # Suppress mne_bids informational warnings about format conversion.
        # "Converting data files to EDF format" — we explicitly request the
        # format via self._format, so this is expected.
        # "Encountered data in "double" format" — mne_bids internally handles
        # the float64->float32 downcast for EDF; we cannot pre-convert because
        # MNE Epochs.save() requires float64 data.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Converting data files to EDF", RuntimeWarning
            )
            warnings.filterwarnings(
                "ignore",
                'Encountered data in "double" format',
                RuntimeWarning,
            )
            # Save annotation extras before write_raw_bids (which may
            # strip them).  We patch events.tsv afterwards.
            ann_extras = getattr(raw.annotations, "extras", None)
            has_extras = ann_extras is not None and any(ann_extras)

            mne_bids.write_raw_bids(
                raw,
                bids_path,
                format=self._format,
                allow_preload=True,
                montage=raw.get_montage(),
                overwrite=True,
                verbose=self.verbose,
            )

            # Append per-event metadata from annotation extras (e.g.
            # triallength for Stieger2021) as extra columns in events.tsv.
            if has_extras:
                events_path = bids_path.copy().update(suffix="events", extension=".tsv")
                events_fpath = events_path.fpath
                if events_fpath.exists():
                    df = pd.read_csv(str(events_fpath), sep="\t")
                    extras_df = pd.DataFrame(ann_extras)
                    if len(extras_df) == len(df):
                        for col in extras_df.columns:
                            df[col] = extras_df[col]
                        df.to_csv(str(events_fpath), sep="\t", index=False, na_rep="n/a")
                    else:
                        log.warning(
                            "Annotation extras length (%d) does not match "
                            "events.tsv rows (%d); skipping extras.",
                            len(extras_df),
                            len(df),
                        )

        # Post-write enrichment: update EEG sidecar with metadata fields
        if metadata is not None:
            sidecar_entries = _build_sidecar_enrichment(metadata)
            if sidecar_entries:
                sidecar_path = bids_path.copy().update(extension=".json")
                mne_bids.update_sidecar_json(sidecar_path, sidecar_entries)

            # Fix mne_bids key casing: MiscChannelCount → MISCChannelCount
            sidecar_fpath = bids_path.copy().update(extension=".json").fpath
            if sidecar_fpath.exists():
                with open(sidecar_fpath) as f:
                    sc = json.load(f)
                if "MiscChannelCount" in sc and "MISCChannelCount" not in sc:
                    sc["MISCChannelCount"] = sc.pop("MiscChannelCount")
                    with open(sidecar_fpath, "w") as f:
                        json.dump(sc, f, indent="\t")

            # Patch participants.tsv with demographic data
            _update_participants_tsv(bids_path.root, self.subject, metadata, raw=raw)

            # Patch electrodes.tsv with material and type
            _update_electrodes_tsv(bids_path, metadata)

        # SpatialReference in electrodes.json sidecars is handled by the
        # monkey-patched _write_dig_bids function at module level.

        # Enrich events.json sidecar with HED annotations and stimulus info
        hed_tags = _build_hed_sidecar_annotations(self.dataset)
        _update_events_json_sidecar(bids_path, hed_tags, metadata)

        # Create scans.json sidecar at session level (next to scans.tsv)
        ses_dir = bids_path.root / f"sub-{bids_path.subject}"
        if bids_path.session is not None:
            ses_dir = ses_dir / f"ses-{bids_path.session}"
        scans_tsv_files = list(ses_dir.glob("*_scans.tsv"))
        if scans_tsv_files:
            scans_json_path = scans_tsv_files[0].with_suffix(".json")
            if not scans_json_path.exists():
                scans_sidecar = {
                    "Description": "Scans file listing data acquisitions.",
                    "filename": {
                        "Description": "Relative path to the data file.",
                    },
                    "acq_time": {
                        "Description": "Acquisition date and time.",
                    },
                }
                with open(scans_json_path, "w") as f:
                    json.dump(scans_sidecar, f, indent="\t")

        # Create channels.json sidecar (Description RECOMMENDED)
        channels_tsv = bids_path.copy().update(suffix="channels", extension=".tsv")
        if channels_tsv.fpath.exists():
            channels_json_path = channels_tsv.fpath.with_suffix(".json")
            if not channels_json_path.exists():
                # Build descriptive string from channel info and metadata
                n_ch = len(raw.ch_names)
                ch_types = set(mne.channel_type(raw.info, i) for i in range(n_ch))
                types_str = ", ".join(sorted(t.upper() for t in ch_types))
                channels_desc = f"Channel information ({n_ch} {types_str} channels)."
                # Add auxiliary channel info from metadata
                if metadata is not None:
                    acq = metadata.acquisition
                    if acq and acq.channel_types:
                        aux_parts = []
                        non_eeg = {
                            k: v for k, v in acq.channel_types.items() if k != "eeg"
                        }
                        for ch_type, count in sorted(non_eeg.items()):
                            aux_parts.append(f"{count} {ch_type.upper()}")
                        if acq.auxiliary_channels:
                            aux = acq.auxiliary_channels
                            if aux.other_physiological:
                                existing = [a.split()[-1] for a in aux_parts]
                                for p in aux.other_physiological:
                                    if p.upper() not in existing:
                                        aux_parts.append(p)
                        if aux_parts:
                            channels_desc += (
                                " Original recording also included "
                                + ", ".join(aux_parts)
                                + "."
                            )
                channels_sidecar = {
                    "Description": channels_desc,
                }
                with open(channels_json_path, "w") as f:
                    json.dump(channels_sidecar, f, indent="\t")


class BIDSInterfaceEpochs(BIDSInterfaceBase):
    """This interface is used to cache mne-epochs to disk.

    Pseudo-BIDS format is used to store the data.


    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``mne.Epochs`` instance.
    """

    @property
    def _extension(self):
        return ".fif"

    @property
    def _check(self):
        return False

    @property
    def _suffix(self):
        return "epo"

    def _load_file(self, bids_path, preload):
        epochs = mne.read_epochs(bids_path.fpath, preload=preload, verbose=self.verbose)
        return epochs

    def _write_file(self, bids_path, epochs):
        epochs.save(bids_path.fpath, overwrite=False, verbose=self.verbose)


class BIDSInterfaceNumpyArray(BIDSInterfaceBase):
    """This interface is used to cache numpy arrays to disk.

    MOABB Pseudo-BIDS format is used to store the data.

    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``OrderedDict`` with keys ``"X"`` and
    ``"events"``. Both values are expected to be ``numpy.ndarray``.
    """

    @property
    def _extension(self):
        return ".npy"

    @property
    def _check(self):
        return False

    @property
    def _suffix(self):
        return "array"

    def _load_file(self, bids_path, preload):
        if preload:
            raise ValueError("preload must be False for numpy arrays")
        events_fname = mne_bids.write._find_matching_sidecar(
            bids_path,
            suffix="events",
            extension=".eve",  # mne convention
            on_error="raise",
        )
        log.debug("Reading %s", bids_path.fpath)
        X = np_load(bids_path.fpath)
        events = mne.read_events(events_fname, verbose=self.verbose)
        return OrderedDict([("X", X), ("events", events)])

    def _write_file(self, bids_path, obj):
        events_path = bids_path.copy().update(
            suffix="events",
            extension=".eve",
        )
        log.debug("Writing %s", bids_path.fpath)
        np_save(bids_path.fpath, obj["X"])
        log.debug("Wrote %s", bids_path.fpath)
        mne.write_events(
            filename=events_path.fpath,
            events=obj["events"],
            overwrite=False,
            verbose=self.verbose,
        )


class StepType(Enum):
    """Enum corresponding to the type of data returned
    by a pipeline step."""

    RAW = "raw"
    EPOCHS = "epochs"
    ARRAY = "array"


_interface_map: Dict[StepType, Type[BIDSInterfaceBase]] = {
    StepType.RAW: BIDSInterfaceRawEDF,
    StepType.EPOCHS: BIDSInterfaceEpochs,
    StepType.ARRAY: BIDSInterfaceNumpyArray,
}


@dataclass
class _BIDSInterfaceRawEDFNoDesc(BIDSInterfaceRawEDF):
    """BIDSInterfaceRawEDF variant that saves without a description hash.

    Used internally by :meth:`~moabb.datasets.base.BaseDataset.convert_to_bids` to produce BIDS files
    whose names do not contain a ``desc-<hash>`` entity.
    """

    _dataset_type: str = "raw"
    _format: str = "EDF"

    @property
    def desc(self):
        return None

    def _write_lock_file(self, session, lock_data):
        """Do not write a lock file for public BIDS conversion."""

    def erase(self):
        """Remove the subject's BIDS directory entirely."""
        subject_dir = self.root / f"sub-{subject_moabb_to_bids(self.subject)}"
        if subject_dir.exists():
            log.info("Starting erasing BIDS data of %s...", repr(self))
            shutil.rmtree(subject_dir)
            log.info("Finished erasing BIDS data of %s.", repr(self))
