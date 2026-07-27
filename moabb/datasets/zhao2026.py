"""EEG dataset for grip-force-level motor imagery (Zhao, 2026).

Zhao, J. (2026). EEG Dataset for Grip-Force-Level Motor Imagery. Zenodo.
Concept DOI: 10.5281/zenodo.21470455 (latest version: 10.5281/zenodo.21470456)
"""

import logging
import warnings

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
    PreprocessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# The published record (10.5281/zenodo.21470456) serves the raw Neuroscan
# ``.cnt`` files directly from the plain /records/<id>/files/<name> endpoint,
# which yields a distinct local filename per file (the /api .../content
# endpoint would collide on the literal name "content").
ZENODO_BASE = "https://zenodo.org/records/21470456/files"

# The single released subject and its file-name token (S<subject>-<token>-*).
_SUBJECT_TOKENS = {1: "GWY"}

# The two motor-imagery task runs. A third file, ``S1-GWY-REST.cnt``, holds a
# resting-state recording (no task markers) and is intentionally excluded.
_TASK_RUNS = ["1", "2"]

# 60 EEG channels in acquisition order (extended 10-10 layout).
_EEG_CHANNELS = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
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
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "O1",
    "Oz",
    "O2",
]

# Raw onset markers embedded in the .cnt annotations -> MOABB class names.
# Each trial carries a paired onset marker "<level>1" and offset marker
# "<level>2"; the first digit (1/2/3) is the graded grip-force level. Only the
# three onset markers are exposed as class events; the offset markers are
# dropped.
_ONSET_MAP = {"11": "level_1", "21": "level_2", "31": "level_3"}


class Zhao2026(BaseDataset):
    """Grip-force-level motor imagery EEG dataset [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Zhao2026         1       60           3                 40            8s             1000 Hz            1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    EEG recorded while a subject performed kinesthetic motor imagery of a
    hand grip at three graded force levels. Each imagery trial lasts about
    8 seconds and is delimited in the raw Neuroscan ``.cnt`` file by a pair of
    embedded event markers: an onset marker whose first digit encodes the
    force level (``"11"``, ``"21"``, ``"31"`` for levels 1, 2 and 3) followed
    ~8 s later by the matching offset marker (``"12"``, ``"22"``, ``"32"``).
    The three graded force levels are treated here as the three decoding
    classes.

    Signals were acquired with a 60-channel Neuroscan system at 1000 Hz. The
    released record contains a single subject (``S1-GWY``) with two motor-imagery
    task runs and one resting-state recording. This loader exposes the two task
    runs as a single session with two runs and drops the resting-state file.
    Across the inspected task run each force level is presented 20 times, in
    randomised order, so a subject has 40 trials per class over the two runs.

    Class labels are data-borne: they come directly from the onset event
    markers stored in the ``.cnt`` files, not from acquisition order. The
    ordinal names ``level_1``/``level_2``/``level_3`` follow the marker first
    digit; the exact grip-force magnitudes (e.g. % of maximum voluntary
    contraction) are not documented in the Zenodo record.

    References
    ----------

    .. [1] Zhao, J. (2026). EEG Dataset for Grip-Force-Level Motor Imagery.
       Zenodo. DOI: https://doi.org/10.5281/zenodo.21470455

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=60,
            channel_types={"eeg": 60},
            montage="standard_1005",
            hardware="Neuroscan",
            reference=None,
            ground=None,
            sensors=list(_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(n_subjects=1, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["level_1", "level_2", "level_3"],
            trial_duration=8.0,
            study_design=(
                "Kinesthetic motor imagery of a hand grip at three graded "
                "force levels; 20 trials per level per task run in randomised "
                "order, two task runs plus one resting-state recording for a "
                "single subject."
            ),
            synchronicity="cue-based",
            mode="offline",
            events={"level_1": 1, "level_2": 2, "level_3": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.21470456",
            description=(
                "Grip-force-level motor-imagery EEG: a single subject imagines "
                "a hand grip at three graded force levels while 60-channel "
                "Neuroscan EEG is recorded at 1000 Hz; force level per trial is "
                "encoded by onset markers embedded in the raw .cnt files."
            ),
            investigators=["Jialing Zhao"],
            data_url="https://doi.org/10.5281/zenodo.21470455",
            country="CN",
            publication_year=2026,
            keywords=[
                "motor imagery",
                "grip force",
                "graded force",
                "EEG",
                "brain-computer interface",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        file_format="CNT",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(_SUBJECT_TOKENS.keys()),
            sessions_per_subject=1,
            events={"level_1": 1, "level_2": 2, "level_3": 3},
            code="Zhao2026",
            interval=[0, 8],
            paradigm="imagery",
            doi="10.5281/zenodo.21470456",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the subject's two motor-imagery ``.cnt`` files.

        Parameters
        ----------
        subject : int
            Subject number (only ``1`` is available).
        path : None | str
            Storage location override.
        force_update : bool
            Re-download even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of str
            Local paths to the two task ``.cnt`` files, in run order.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        token = _SUBJECT_TOKENS[subject]
        paths = []
        for suffix in _TASK_RUNS:
            url = f"{ZENODO_BASE}/S{subject}-{token}-{suffix}.cnt"
            paths.append(dl.data_dl(url, self.code, path, force_update, verbose))
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (only ``1`` is available).

        Returns
        -------
        dict
            ``{"0": {run_str: Raw}}`` with the two motor-imagery task runs; the
            resting-state recording is excluded.
        """
        files = self.data_path(subject)
        montage = make_standard_montage("standard_1005")

        runs = {}
        for run_idx, fpath in enumerate(files):
            raw = mne.io.read_raw_cnt(str(fpath), preload=True, verbose=False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage(
                    montage, match_case=False, on_missing="warn", verbose=False
                )

            # Keep only the three onset markers, renamed to class names; drop
            # the paired offset markers ("12"/"22"/"32"). Labels are read
            # straight from the embedded .cnt annotations.
            ann = raw.annotations
            onset, duration, description = [], [], []
            for o, d, desc in zip(ann.onset, ann.duration, ann.description):
                if str(desc) in _ONSET_MAP:
                    onset.append(o)
                    duration.append(d)
                    description.append(_ONSET_MAP[str(desc)])
            raw.set_annotations(
                mne.Annotations(
                    onset=onset,
                    duration=duration,
                    description=description,
                    orig_time=ann.orig_time,
                )
            )

            runs[str(run_idx)] = raw

        return {"0": runs}
