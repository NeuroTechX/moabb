"""Kodera2023 left/right-hand motor-imagery EEG dataset (University of West Bohemia)."""

import re
import warnings
from pathlib import Path

import mne
import numpy as np
from mne.channels import make_standard_montage

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

from .utils import download_and_extract_subject_zip


# Single Zenodo archive (concept DOI 10.5281/zenodo.7893846 -> version 7893847).
KODERA2023_URL = "https://zenodo.org/records/7893847/files/data.zip"

# Event codes exposed to the paradigm. The left/right class is carried by the
# recording file name, not by the BrainVision markers.
EVENTS = {"left_hand": 1, "right_hand": 2}

# The two trailing unnamed channels of the 500 Hz cohort ("17", "18") are not
# scalp EEG and are marked as misc.
_AUX_CHANNELS = {"17", "18"}

# The nine scalp electrodes every recording has (the 500 Hz layout adds seven);
# their ordered intersection is the cross-subject channel space, never padded.
_COMMON_EEG_CHANNELS = ("Fz", "Cz", "Pz", "F3", "F4", "P3", "P4", "C3", "C4")

# The sixteen scalp electrodes of the 2020-2021 recordings (500 Hz), in file
# order. T3-T6 are the legacy names of T7, T8, P7 and P8.
_FULL_EEG_CHANNELS = (
    "Fp1", "F3", "F4", "C3", "C4", "P3", "P4", "F7", "F8",
    "T3", "T4", "T5", "T6", "Cz", "Fz", "Pz",
)  # fmt: skip

# Two recording setups (checked on every header of the Zenodo archive): the
# 2020-2021 recordings hold 16 scalp + 2 auxiliary channels at 500 Hz, the
# 2021-2023 recordings (``HR_*`` names and the 03_04_2023 short names) the
# nine shared channels at 1000 Hz. MOABB paradigms stack every subject of a
# dataset into one array, so the default returns all subjects on the shared
# channels resampled to the common (lowest) rate.
_SFREQ = 500.0
_SETUPS = {
    "16ch": {"sfreq": 500.0, "channels": _FULL_EEG_CHANNELS},
    "9ch": {"sfreq": 1000.0, "channels": _COMMON_EEG_CHANNELS},
}

# Download folder shared by every ``cohort`` variant (their codes differ).
_SIGN = "Kodera2023"

# One subject == one (recording-date, person-token) group of recordings holding
# both left- and right-hand runs. Short names follow
# ``<idx><initial><ddmmyyyy><lh|rh><run>``; ``HR_<ddmmyyyy>_<NN>_*`` names carry
# Czech ``leva``/``prava`` suffixes. The unlabelled ``HR_02092021_02_...`` is excluded.
SUBJECTS = [
    (date, token)
    for date, tokens in (
        ("01_12_2020", "1z 2z"),
        ("02_09_2021", "S03"),
        ("03_04_2023", "1m 2m 3m 4m 5m"),
        ("07_10_2021", "S09 S10"),
        ("10_12_2020", "1m 2z"),
        ("14_01_2021", "1m 2m 3z"),
        ("14_10_2021", "S12 S13"),
        ("21_01_2021", "1m 2z 3z 4z"),
        ("23_09_2021", "S04 S05"),
        ("28_01_2021", "1z 2z 3z"),
        ("30_09_2021", "S06 S07 S08"),
    )
    for token in tokens.split()
]

# Recording setup of each date folder (see ``_SETUPS``).
_FOLDER_SETUP = {
    "01_12_2020": "16ch",
    "10_12_2020": "16ch",
    "14_01_2021": "16ch",
    "21_01_2021": "16ch",
    "28_01_2021": "16ch",
    "02_09_2021": "9ch",
    "23_09_2021": "9ch",
    "30_09_2021": "9ch",
    "07_10_2021": "9ch",
    "14_10_2021": "9ch",
    "03_04_2023": "9ch",
}

# Subjects (1-based, MOABB numbering) and dataset code of each ``cohort``
# option. ``is_abbrev("Kodera2023", code)`` only lets lower-case letters,
# digits and dashes follow a letter of the class name, and nothing follow its
# final digits, so the cohort tag sits before the year.
_COHORTS = {
    None: {"subjects": list(range(1, len(SUBJECTS) + 1)), "code": "Kodera2023"},
    **{
        setup: {
            "subjects": [
                i + 1
                for i, (folder, _) in enumerate(SUBJECTS)
                if _FOLDER_SETUP[folder] == setup
            ],
            "code": f"Kodera-{setup}2023",
        }
        for setup in _SETUPS
    },
}


def _class_from_stem(stem):
    """Return the class encoded in a recording file name, or None.

    ``leva``/``prava`` (Czech left/right) for the ``HR_*`` cohort, the ``lh``/``rh``
    token before the trailing run digit for the short-name cohort.
    """
    s = stem.lower()
    if "leva" in s:
        return "left_hand"
    if "prava" in s:
        return "right_hand"
    m = re.search(r"(lh|rh)\d+$", s)
    if m:
        return "left_hand" if m.group(1) == "lh" else "right_hand"
    return None


def _matches_subject(stem, date_folder, token):
    """Whether a recording stem belongs to a given (date, person-token) subject."""
    date8 = date_folder.replace("_", "")
    if token.startswith("S") and token[1:].isdigit():
        # HR_<ddmmyyyy>_<NN>_...
        return stem.startswith(f"HR_{date8}_{token[1:]}_")
    # <idx><initial><ddmmyyyy><lh|rh><run>
    return stem.startswith(token) and date8 in stem


def _make_metadata(cohort=None):
    """``DatasetMetadata`` of the variant returned for ``cohort``."""
    setup = _SETUPS.get(cohort)
    channels = list(setup["channels"] if setup else _COMMON_EEG_CHANNELS)
    sfreq = setup["sfreq"] if setup else _SFREQ
    n_subjects = len(_COHORTS[cohort]["subjects"])
    description = {
        None: "Left/right-hand motor-imagery EEG (University of West Bohemia): "
        "29 subjects from two recording setups (16 channels at 500 Hz and 9 "
        "channels at 1000 Hz), returned on the 9 shared channels at 500 Hz; "
        "2 classes, BrainVision format.",
        "16ch": "Left/right-hand motor-imagery EEG (University of West "
        "Bohemia), 2020-2021 recordings: 14 subjects, 16 channels at 500 Hz, "
        "2 classes, BrainVision format.",
        "9ch": "Left/right-hand motor-imagery EEG (University of West "
        "Bohemia), 2021-2023 recordings: 15 subjects, 9 channels at 1000 Hz, "
        "2 classes, BrainVision format.",
    }[cohort]
    return DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=sfreq,
            channel_types={"eeg": len(channels)},
            montage="standard_1020",
            hardware="BrainVision Recorder (BrainProducts)",
            reference=None,
            ground=None,
            sensors=channels,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=n_subjects, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            study_design="Cue-based single-class left- or right-hand motor "
            "imagery; each recording is one class, with the imagery cue onset "
            "marked by the S 1 BrainVision stimulus marker.",
            stimulus_modalities=["visual"],
            mode="offline",
            events=dict(EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.7893846",
            description=description,
            investigators=[
                "Jakub Kodera",
                "Roman Moucek",
                "Pavel Moutner",
                "Pavel Mochura",
                "Josef Yassin Saleh",
                "Petr Bruha",
                "Jana Solcova",
                "Lukas Vareka",
                "Pavel Snejdar",
            ],
            institution="University of West Bohemia",
            country="CZ",
            data_url="https://doi.org/10.5281/zenodo.7893846",
            publication_year=2023,
            keywords=[
                "motor imagery",
                "EEG",
                "BCI",
                "brain-computer interface",
                "left hand",
                "right hand",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        file_format="BrainVision",
    )


class Kodera2023(BaseDataset):
    """Left/right-hand motor-imagery EEG dataset [1]_.

    EEG recorded at the University of West Bohemia (Pilsen) during cue-based left-
    versus right-hand motor imagery. Each recording holds a single class, encoded
    in its file name; the ``S 1`` BrainVision marker is the per-trial imagery cue
    (the markers themselves are identical across classes).

    A subject is one (recording-date, person) group holding both left- and
    right-hand runs: 29 subjects, one session of two to four runs each. Every run
    is annotated at its ``S 1`` onsets with the run's class. The one unlabelled
    recording is excluded.

    **Recording setups.** The archive mixes two setups (every BrainVision header
    of the Zenodo record was checked):

    - the 2020-2021 recordings (date folders ``01_12_2020`` to ``28_01_2021``,
      short names such as ``1z01122020lh1``; 14 subjects): 16 scalp channels
      (Fp1, F3, F4, C3, C4, P3, P4, F7, F8, T3, T4, T5, T6, Cz, Fz, Pz; T3-T6 are
      the legacy names of T7, T8, P7, P8) plus two unnamed auxiliary channels,
      sampled at **500 Hz**, DC-coupled with large slow drifts;
    - the 2021-2023 recordings (``HR_<date>_<nn>_..._leva`` names and the
      ``03_04_2023`` short names; 15 subjects): the nine channels Fz, Cz, Pz,
      F3, F4, P3, P4, C3, C4, sampled at **1000 Hz**.

    MOABB paradigms and evaluations stack the epochs of every subject of a
    dataset into one array, so all subjects of a dataset object must return the
    same channels at the same rate. The ``cohort`` option selects which:

    - ``cohort=None`` (default, code ``Kodera2023``): all 29 subjects with the
      nine channels both setups share, in one order, at **500 Hz**. The 1000 Hz
      recordings are resampled to 500 Hz when loaded (the common, lowest rate);
      the extra scalp and auxiliary channels are dropped, never padded.
    - ``cohort="16ch"`` (code ``Kodera-16ch2023``): the 14 subjects of the
      2020-2021 setup with all 16 scalp channels at 500 Hz.
    - ``cohort="9ch"`` (code ``Kodera-9ch2023``): the 15 subjects of the
      2021-2023 setup with their nine channels at the native 1000 Hz.

    Subject numbers are the same in every option; asking a cohort for a subject
    of the other setup raises a ``ValueError``. Each option has its own ``code``
    and ``METADATA``, so cached epochs and results never mix channel sets or
    rates; downloads are shared.

    The Zenodo record carries only the title "EEG motor imagery", the creator
    list and a one-line description ("EEG dataset used for automatic motor
    imagery detection."); there is no linked paper. Subject, channel, rate and
    class information above is derived from the archive contents and could not
    be checked against a publication (paper audit, 2026-09-30).

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load (restricted to the selected ``cohort``).
    sessions : list of int | str | None
        Sessions to keep.
    cohort : None | "16ch" | "9ch"
        ``None`` returns all 29 subjects on the nine shared channels at
        500 Hz; ``"16ch"`` the 14 subjects of the 2020-2021 setup with 16
        channels at 500 Hz; ``"9ch"`` the 15 subjects of the 2021-2023 setup
        at 1000 Hz.

    References
    ----------

    .. [1] Kodera, J., Moucek, R., Moutner, P., Mochura, P., Saleh, J. Y.,
       Bruha, P., Solcova, J., Vareka, L., and Snejdar, P. (2023). EEG motor
       imagery [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7893846

    .. versionadded:: 1.8.0

    """

    nemar_id = "nm000295"

    METADATA = _make_metadata(None)

    def __init__(self, subjects=None, sessions=None, *, cohort=None):
        if cohort not in _COHORTS:
            raise ValueError(f"cohort must be None, '16ch' or '9ch', got {cohort!r}")
        self.cohort = cohort
        if cohort is not None:
            # Instance override: METADATA describes what this variant returns.
            self.METADATA = _make_metadata(cohort)
        if subjects is not None:
            self._check_cohort(subjects)
        super().__init__(
            subjects=list(_COHORTS[cohort]["subjects"]),
            sessions_per_subject=1,
            events=dict(EVENTS),
            code=_COHORTS[cohort]["code"],
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.7893846",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    @property
    def metadata(self):
        """Structured metadata of this ``cohort`` variant."""
        if self.cohort is None:
            return super().metadata
        return self.METADATA

    def _check_cohort(self, subjects):
        """Raise a ``ValueError`` naming the setup of out-of-cohort subjects."""
        if self.cohort is None:
            return
        allowed = _COHORTS[self.cohort]["subjects"]
        outside = [s for s in subjects if s not in allowed]
        if outside:
            raise ValueError(
                f"Kodera2023(cohort={self.cohort!r}) holds subjects {allowed} "
                f"only; subject(s) {outside} were recorded with the other setup. "
                f"Use cohort=None for all 29 subjects on the 9 shared channels at "
                f"500 Hz, or cohort='16ch' / cohort='9ch'."
            )

    def get_data(self, subjects=None, cache_config=None, process_pipeline=None, n_jobs=1):
        if subjects is not None:
            self._check_cohort(subjects)
        return super().get_data(
            subjects=subjects,
            cache_config=cache_config,
            process_pipeline=process_pipeline,
            n_jobs=n_jobs,
        )

    get_data.__doc__ = BaseDataset.get_data.__doc__

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the BrainVision header paths of a single subject's labelled runs."""
        self._check_cohort([subject])
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        # One download folder for every cohort variant, whatever self.code is.
        data_dir = (
            Path(dl.get_dataset_path(_SIGN, path)) / f"MNE-{_SIGN.lower()}-data" / "data"
        )
        if force_update or not data_dir.exists():
            download_and_extract_subject_zip(
                KODERA2023_URL, _SIGN, data_dir.parent, path, force_update, verbose
            )
        date_folder, token = SUBJECTS[subject - 1]
        folder = data_dir / date_folder
        vhdrs = sorted(
            p
            for p in folder.glob("*.vhdr")
            if _matches_subject(p.stem, date_folder, token)
            and _class_from_stem(p.stem) is not None
        )
        if not vhdrs:
            raise FileNotFoundError(
                f"No labelled recordings found for subject {subject} "
                f"({date_folder}, {token}) under {folder}"
            )
        return [str(p) for p in vhdrs]

    def _read_run(self, vhdr_path):
        """Read one BrainVision run and annotate it with its class at S 1 onsets.

        The run keeps the channels of the selected ``cohort`` (the nine shared
        ones by default) and, by default, is resampled to 500 Hz.
        """
        cls = _class_from_stem(Path(vhdr_path).stem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

        setup = _SETUPS.get(self.cohort)
        channels = setup["channels"] if setup else _COMMON_EEG_CHANNELS
        raw.set_channel_types({ch: "misc" for ch in raw.ch_names if ch in _AUX_CHANNELS})
        missing = [ch for ch in channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Kodera2023 recording is missing required EEG channels {missing}: "
                f"{vhdr_path}"
            )
        raw.pick(list(channels))

        # Only the two rates of the archive are expected; anything else is a
        # file this loader was not checked against.
        sfreq = raw.info["sfreq"]
        expected = setup["sfreq"] if setup else (500.0, 1000.0)
        if sfreq not in np.atleast_1d(expected):
            raise ValueError(
                f"Kodera2023 recording sampled at {sfreq:g} Hz, expected "
                f"{expected} Hz: {vhdr_path}"
            )
        if setup is None and sfreq != _SFREQ:
            # Common rate of the default dataset: the lowest of the archive.
            raw.resample(_SFREQ, verbose=False)
        raw.set_montage(
            make_standard_montage("colin27_1020"), on_missing="ignore", match_case=False
        )

        # Keep only the S 1 imagery-cue onsets, relabelled with the run's class.
        ann = raw.annotations
        onsets = [o for o, d in zip(ann.onset, ann.description) if d.endswith("S  1")]
        raw.set_annotations(
            mne.Annotations(onsets, [0.0] * len(onsets), [cls] * len(onsets))
        )
        return raw

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {run: Raw}}``: one session with the subject's runs."""
        return {
            "0": {
                str(i): self._read_run(v) for i, v in enumerate(self.data_path(subject))
            }
        }
