"""Kodera2023 left/right-hand motor-imagery EEG dataset (University of West Bohemia)."""

import re
import warnings
from pathlib import Path

import mne
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


class Kodera2023(BaseDataset):
    """Left/right-hand motor-imagery EEG dataset [1]_.

    EEG recorded at the University of West Bohemia (Pilsen) during cue-based left-
    versus right-hand motor imagery. Each recording holds a single class, encoded
    in its file name; the ``S 1`` BrainVision marker is the per-trial imagery cue
    (the markers themselves are identical across classes). The archive mixes a
    16-channel 500 Hz cohort (short names such as ``1z01122020lh1``, legacy
    T3-T6 labels, DC-coupled with large slow drifts) and a 9-channel 1000 Hz
    cohort (``HR_<date>_<nn>_..._leva``).

    A subject is one (recording-date, person) group holding both left- and
    right-hand runs: 29 subjects, one session of two to four runs each. Every run
    is annotated at its ``S 1`` onsets with the run's class. To make both cohorts
    comparable the loader keeps only their ordered nine-channel intersection (Fz,
    Cz, Pz, F3, F4, P3, P4, C3, C4); the extra scalp and two unnamed auxiliary
    channels are dropped, never padded. The one unlabelled recording is excluded.

    The Zenodo record carries only the title "EEG motor imagery", the creator
    list and a one-line description ("EEG dataset used for automatic motor
    imagery detection."); there is no linked paper. Subject, channel, rate and
    class information above is derived from the archive contents and could not
    be checked against a publication (paper audit, 2026-09-30).

    References
    ----------

    .. [1] Kodera, J., Moucek, R., Moutner, P., Mochura, P., Saleh, J. Y.,
       Bruha, P., Solcova, J., Vareka, L., and Snejdar, P. (2023). EEG motor
       imagery [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7893846

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types={"eeg": 9},
            montage="standard_1020",
            hardware="BrainVision Recorder (BrainProducts)",
            reference=None,
            ground=None,
            sensors=["Fz", "Cz", "Pz", "F3", "F4", "P3", "P4", "C3", "C4"],
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=29, health_status="healthy", species="homo sapiens"
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
            description="Left/right-hand motor-imagery EEG (University of West "
            "Bohemia): 29 subjects across two cohorts (16-channel 500 Hz and "
            "9-channel 1000 Hz), 2 classes, BrainVision format.",
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

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, len(SUBJECTS) + 1)),
            sessions_per_subject=1,
            events=dict(EVENTS),
            code="Kodera2023",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.7893846",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the BrainVision header paths of a single subject's labelled runs."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        data_dir = (
            Path(dl.get_dataset_path(self.code, path))
            / f"MNE-{self.code.lower()}-data"
            / "data"
        )
        if force_update or not data_dir.exists():
            download_and_extract_subject_zip(
                KODERA2023_URL, self.code, data_dir.parent, path, force_update, verbose
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
        """Read one BrainVision run and annotate it with its class at S 1 onsets."""
        cls = _class_from_stem(Path(vhdr_path).stem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

        raw.set_channel_types({ch: "misc" for ch in raw.ch_names if ch in _AUX_CHANNELS})
        missing = [ch for ch in _COMMON_EEG_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                "Kodera2023 recording is missing required shared EEG channels "
                f"{missing}: {vhdr_path}"
            )
        raw.pick(list(_COMMON_EEG_CHANNELS))
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
