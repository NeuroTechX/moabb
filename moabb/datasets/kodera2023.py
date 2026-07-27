"""Kodera2023 left/right-hand motor-imagery EEG dataset (University of West Bohemia)."""

import logging
import re
import warnings
import zipfile
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


log = logging.getLogger(__name__)

# Single Zenodo archive (concept DOI 10.5281/zenodo.7893846 -> version 7893847).
KODERA2023_URL = "https://zenodo.org/records/7893847/files/data.zip"

# Event codes exposed to the paradigm. The left/right class is carried by the
# recording file name (see module notes), not by the BrainVision markers.
EVENTS = {"left_hand": 1, "right_hand": 2}

# The two trailing unnamed channels of the 500 Hz cohort ("17", "18") are not
# scalp EEG and are marked as misc.
_AUX_CHANNELS = {"17", "18"}

# Every source recording contains these nine scalp electrodes.  The 500 Hz
# layout has seven additional scalp electrodes, whereas the 1000 Hz layout
# contains only this set.  Keep the ordered intersection for a genuine
# cross-subject channel space; do not pad the 9-channel recordings or invent
# signals for the absent optional electrodes.
_COMMON_EEG_CHANNELS = ("Fz", "Cz", "Pz", "F3", "F4", "P3", "P4", "C3", "C4")

# Deterministic subject map: one subject == one (recording-date, person-token)
# group of BrainVision recordings that contains both left- and right-hand runs.
# Short names follow ``<idx><initial><ddmmyyyy><lh|rh><run>``; ``HR_*`` names
# carry Czech ``leva``/``prava`` = left/right suffixes. The source layouts vary
# by recording date as well as file-name family, so channel normalization is
# derived from the actual header rather than inferred from this token. The
# single unlabelled recording (``HR_02092021_02_...``, no side suffix) is
# excluded.
SUBJECTS = [
    ("01_12_2020", "1z"),
    ("01_12_2020", "2z"),
    ("02_09_2021", "S03"),
    ("03_04_2023", "1m"),
    ("03_04_2023", "2m"),
    ("03_04_2023", "3m"),
    ("03_04_2023", "4m"),
    ("03_04_2023", "5m"),
    ("07_10_2021", "S09"),
    ("07_10_2021", "S10"),
    ("10_12_2020", "1m"),
    ("10_12_2020", "2z"),
    ("14_01_2021", "1m"),
    ("14_01_2021", "2m"),
    ("14_01_2021", "3z"),
    ("14_10_2021", "S12"),
    ("14_10_2021", "S13"),
    ("21_01_2021", "1m"),
    ("21_01_2021", "2z"),
    ("21_01_2021", "3z"),
    ("21_01_2021", "4z"),
    ("23_09_2021", "S04"),
    ("23_09_2021", "S05"),
    ("28_01_2021", "1z"),
    ("28_01_2021", "2z"),
    ("28_01_2021", "3z"),
    ("30_09_2021", "S06"),
    ("30_09_2021", "S07"),
    ("30_09_2021", "S08"),
]


def _class_from_stem(stem):
    """Return the motor-imagery class carried by a recording file name.

    The left/right label lives in the file name: ``leva``/``prava`` (Czech for
    left/right) for the ``HR_*`` cohort and the ``lh``/``rh`` token before the
    trailing run digit for the short-name cohort.

    Parameters
    ----------
    stem : str
        Recording file name without extension.

    Returns
    -------
    str or None
        ``"left_hand"``, ``"right_hand"`` or ``None`` when no side is encoded.
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

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Kodera2023      29       9           2              ~20-30            4s        500/1000 Hz            1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    EEG recorded at the University of West Bohemia (Pilsen) while participants
    performed cue-based left- versus right-hand motor imagery, published on
    Zenodo as "EEG motor imagery" for automatic motor-imagery detection. Each
    recording is a single-class session (all left-hand or all right-hand
    imagery); the class is encoded in the recording file name and the per-trial
    imagery cue onset is the ``S 1`` BrainVision stimulus marker.

    The archive mixes two acquisition cohorts recorded on eleven dates:

    * a 16-channel cohort sampled at 500 Hz (Fp1, F3, F4, C3, C4, P3, P4, F7,
      F8, T3, T4, T5, T6, Cz, Fz, Pz, plus two unnamed auxiliary channels),
      with short file names such as ``1z01122020lh1``;
    * a 9-channel cohort sampled at 1000 Hz (Fz, Cz, Pz, F3, F4, P3, P4, C3, C4;
      a subset of the 16-channel montage), with file names such as
      ``HR_07102021_09_s_vibratory_s_haptikou_leva``.

    A subject here is one (recording-date, person) group of recordings holding
    both left- and right-hand runs; 29 such subjects are exposed, each as a
    single session with two to four runs. Each run is annotated at its ``S 1``
    imagery cue onsets with the run's class label, so the ``LeftRightImagery``
    paradigm epochs both classes across a subject's runs. To make subjects from
    the two acquisition layouts comparable, the loader exposes their ordered
    nine-channel scalp intersection (Fz, Cz, Pz, F3, F4, P3, P4, C3, C4).
    The seven 500 Hz-only scalp channels and the two unnamed auxiliary channels
    are deliberately omitted rather than padded or fabricated.

    Notes
    -----
    The BrainVision markers (``S 1``, ``S 2`` and the auxiliary ``S 4``/``S 5``)
    are identical between left- and right-hand recordings and therefore do not
    encode the class; the class is data-borne through the recording file name.
    Each ``S 1`` (imagery cue) is followed by an ``S 2`` after 2.5 s (500 Hz
    cohort) or 5 s (1000 Hz cohort). The 500 Hz cohort is DC-coupled and carries
    large slow drifts that a band-pass filter removes. Electrode names use the
    legacy T3/T4/T5/T6 labels; the standard_1020 montage is attached with
    ``on_missing="ignore"``.

    References
    ----------

    .. [1] Kodera, J., Moucek, R., Moutner, P., Mochura, P., Saleh, J. Y.,
       Bruha, P., Solcova, J., Vareka, L., and Snejdar, P. (2023). EEG motor
       imagery [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7893846

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=9,
            channel_types={"eeg": 9},
            montage="standard_1020",
            hardware="BrainVision Recorder (BrainProducts)",
            reference=None,
            ground=None,
            sensors=[
                "Fz",
                "Cz",
                "Pz",
                "F3",
                "F4",
                "P3",
                "P4",
                "C3",
                "C4",
            ],
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

    def _extracted_root(self, path=None, force_update=False, verbose=None):
        """Download and extract the archive, returning the ``data`` folder."""
        zip_path = Path(
            dl.data_dl(KODERA2023_URL, self.code, path, force_update, verbose)
        )
        data_dir = zip_path.parent / "data"
        if not data_dir.is_dir():
            log.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_path.parent)
        return data_dir

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the recording (.vhdr) paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-29).
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list of str
            The BrainVision header paths for the subject's labelled runs.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        data_dir = self._extracted_root(
            path=path, force_update=force_update, verbose=verbose
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

        aux = {ch: "misc" for ch in raw.ch_names if ch in _AUX_CHANNELS}
        if aux:
            raw.set_channel_types(aux)

        missing = [ch for ch in _COMMON_EEG_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                "Kodera2023 recording is missing required shared EEG channels "
                f"{missing}: {vhdr_path}"
            )
        # The 16-channel recordings contain seven optional scalp electrodes;
        # keeping their exact ordered intersection with the 9-channel cohort
        # is the only lossless cross-subject normalization available here.
        raw.pick(list(_COMMON_EEG_CHANNELS))
        raw.set_montage(
            make_standard_montage("standard_1020"), on_missing="ignore", match_case=False
        )

        # Keep only the S 1 imagery-cue onsets and relabel them with the run's
        # (file-name-encoded) class, so the paradigm can epoch both classes.
        ann = raw.annotations
        onsets = [
            ann.onset[i] for i in range(len(ann)) if ann.description[i].endswith("S  1")
        ]
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=[0.0] * len(onsets),
                description=[cls] * len(onsets),
            )
        )
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-29).

        Returns
        -------
        dict
            ``{"0": {run_str: Raw}}`` - one session with the subject's runs.
        """
        runs = {}
        for run_idx, vhdr in enumerate(self.data_path(subject)):
            runs[str(run_idx)] = self._read_run(vhdr)
        return {"0": runs}
