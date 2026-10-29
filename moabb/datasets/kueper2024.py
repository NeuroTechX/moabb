"""Kueper2024 unilateral and bilateral movement-execution dataset (Kueper et al. 2024)."""

import warnings
from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)

from .utils import download_and_extract_subject_zip


# Version 2 of the Zenodo record (concept 10229479; record 17940098, 2025-12-15),
# ~4.3 GB, BrainVision format. It adds six unilateral-only subjects; the EEG and
# marker files of the eight subjects of version 1 (record 10229480) are
# byte-identical in both versions.
Kueper2024_URL = "https://zenodo.org/api/records/17940098/files/EEG_dataset.zip/content"
Kueper2024_ZIP = "EEG_dataset_v2.zip"

# Subject pseudo-codes mapped to subjects 1..14. Subjects 1-8 are the first
# study (unilateral and bilateral, alphabetical, numbering unchanged since
# version 1); subjects 9-14 are the second study (unilateral only, codes ending
# in "D").
SUBJECT_CODES = [
    "AV82",
    "JD68",
    "JV43",
    "QS70",
    "RA12",
    "UP28",
    "XP01",
    "ZS27",
    "AF64D",
    "AU12D",
    "AW98D",
    "BR39D",
    "BS34D",
    "BZ29D",
]

# 64 EEG channel names in acquisition order (from the BrainVision headers).
EEG_CHANNELS = (
    "Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 TP9 CP5 CP1 CP2 CP6 "
    "TP10 P7 P3 Pz P4 P8 PO9 O1 Oz O2 PO10 AF7 AF3 AF4 AF8 F5 F1 F2 F6 FT9 FT7 "
    "FC3 FC4 FT8 FT10 C5 C1 C2 C6 TP7 CP3 CPz CP4 TP8 P5 P1 P2 P6 PO7 PO3 POz "
    "PO4 PO8"
).split()

# 3-axis accelerometer channels (treated as misc, not EEG).
ACCEL_CHANNELS = ["x_dir", "y_dir", "z_dir"]

# Movement onset from motion tracking: right arm S100 in the unilateral
# recordings, left arm S101 in the bilateral ones (first study only).
ONSET_MARKER = {"unilateral": "S100", "bilateral": "S101"}
# Release of the hand switch of the same arm: the movement onset used when a
# recording has no motion-tracking marker (second study). Across the correct
# trials of the first study it follows the motion-tracking onset by a median
# of 54 ms (interquartile range 14-96 ms, 1,918 trials).
SWITCH_RELEASE = {"unilateral": "S22", "bilateral": "S24"}
# Hands put back on the switches (right S23, left S25): start of a rest.
SWITCH_PUSH = ("S23", "S25")
# End of a correct trial; error trials (S3, rest shorter than 5 s) and
# additional trials (S27) have none.
END_OF_CORRECT_TRIAL = "S128"

# The protocol requires a rest of at least 5 s, hands on the switches, before
# each movement. A rest event is placed REST_OFFSET s before the onset, so the
# default interval (-2, 1) covers -5 to -2 s before the onset.
MIN_REST = 5.0
REST_OFFSET = 3.0


def _marker(description):
    """'Stimulus/S 100' -> 'S100'."""
    return description.replace(" ", "").split("/")[-1]


def _trial_events(annotations, condition):
    """Return (movement onsets, rest onsets) of the correct trials of a run.

    A movement onset is the motion-tracking onset, or the hand-switch release
    when the run has no motion-tracking marker. A trial is correct when an
    end-of-correct-trial marker (S128) follows its onset before the next onset.
    A rest event is placed ``REST_OFFSET`` s before a correct onset when the
    hands were back on the switches at least ``MIN_REST`` s before it.
    """
    times = {}
    for onset, description in zip(annotations.onset, annotations.description):
        times.setdefault(_marker(description), []).append(onset)
    onsets = np.sort(
        times.get(ONSET_MARKER[condition]) or times.get(SWITCH_RELEASE[condition], [])
    )
    ends = np.sort(times.get(END_OF_CORRECT_TRIAL, []))
    pushes = np.sort([t for m in SWITCH_PUSH for t in times.get(m, [])])
    nxt = np.append(onsets[1:], np.inf)
    movement, rest = [], []
    for onset, next_onset in zip(onsets, nxt):
        if not np.any((ends > onset) & (ends < next_onset)):
            continue
        movement.append(onset)
        before = pushes[pushes < onset]
        if len(before) and before[-1] <= onset - MIN_REST:
            rest.append(onset - REST_OFFSET)
    return movement, rest


class Kueper2024(BaseDataset):
    """Unilateral and bilateral movement-execution EEG dataset [1]_, [2]_, [3]_.

    Fourteen healthy right-handed participants executed (not imagined)
    self-initiated, self-paced reaching movements. In the ``unilateral``
    condition they pressed a button in front of them with the right thumb; in
    the ``bilateral`` condition they reached with both arms synchronously. After
    each movement they returned their hands to rest on hand switches. Every
    trial is a rest of at least 5 s with the hands on the switches (a shorter
    rest triggered an error message), the movement onset (time 0), then the
    self-paced execution until the hands are back at rest. Each condition was
    recorded in 3 sets of 40 movements (subject ``XP01`` has its third
    unilateral set split over two recordings).

    The first study recorded 8 participants (subjects 1-8) in both conditions
    in a shielded cabin. The second study recorded 6 more participants
    (subjects 9-14) in the unilateral condition only, in a standard laboratory
    [3]_. EEG was recorded with a 64-channel LiveAmp at 500 Hz (reference FCz,
    ground AFz), plus a 3-axis accelerometer kept as ``misc`` channels.

    Events: ``unilateral`` and ``bilateral`` are placed at the movement onset
    of the correct trials, from motion tracking (``S100`` right arm,
    ``S101`` left arm) in the first study and from the release of the right
    hand switch (``S22``) in the second study, which has no motion-tracking
    marker. In the first study the hand-switch release follows the
    motion-tracking onset by a median of 54 ms. Error trials (``S3``) and
    additional trials (``S27``) are dropped: only onsets followed by an
    end-of-correct-trial marker (``S128``) are kept. ``rest`` is placed 3 s
    before a correct onset when the hands were back on the switches at least
    5 s before it, so with the default interval (-2, 1) a rest epoch covers 5
    to 2 s before the movement onset. The two conditions share one session and
    every recording set is a run.

    The second study has no ``bilateral`` runs: use the ``unilateral`` and
    ``rest`` classes to evaluate all 14 subjects. The loader keeps 5,163
    trials: 1,662 ``unilateral``, 959 ``bilateral`` and 2,542 ``rest``.
    Subjects 1-8 have 239-240 movement trials; subjects 9-14 have 116-119,
    not 120, because most second-study recordings also send the
    end-of-correct-trial marker when the hands are first placed on the
    switches, before any movement, so the last movement of the set has none.

    Paper audit (2026-09-30): the 500 Hz rate applies to the EEG (the paper's
    2000 Hz figure is the Cometa EMG, which is not part of the release). The
    ground electrode (``AFz``) is given in the release's readme, not in the
    paper.

    References
    ----------

    .. [1] Kueper, N., Kim, S. K., & Kirchner, E. A. (2023). EEG Dataset of
       Unilateral and Bilateral Movement Executions [Data set]. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.10229480

    .. [2] Kueper, N., Kim, S. K., & Kirchner, E. A. (2024). Avoidance of
       specific calibration sessions in motor intention recognition for
       exoskeleton-supported rehabilitation through transfer learning on EEG
       data. Scientific Reports, 14(1), 16690.
       DOI: https://doi.org/10.1038/s41598-024-65910-8

    .. [3] Kueper, N., Kim, S. K., Chari, K., & Kirchner, E. A. (2025). EEG
       dataset of unilateral and bilateral movement executions (version 2)
       [Data set]. Zenodo. DOI: https://doi.org/10.5281/zenodo.17940098

    Notes
    -----

    .. versionadded:: 1.8.0

    .. versionchanged:: 1.8.0
       Version 2 of the release (subjects 9-14), ``rest`` class, error and
       additional trials dropped (:gh:`1333`).

    """

    # The NEMAR deposit holds version 1 (subjects 1-8); subjects 9-14 are
    # downloaded from Zenodo.
    nemar_id = "nm000319"

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types={"eeg": 64, "misc": 3},
            montage="extended 10-20",
            hardware="Brain Products LiveAmp64 (wireless, active electrodes)",
            cap_manufacturer="Brain Products GmbH",
            cap_model="Acticap slim",
            sensor_type="active",
            electrode_type="active",
            reference="FCz",
            ground="AFz",
            sensors=EEG_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=False),
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            gender={"male": 6, "female": 8},
            age_mean=24.57,
            age_std=3.06,
            age_median=24.0,
            age_min=19,
            age_max=30,
            handedness="right",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["unilateral", "bilateral", "rest"],
            trial_duration=3.0,
            study_design="Executed (not imagined), self-initiated, self-paced "
            "reaching movements: unilateral (right thumb presses a button in front "
            "of the participant) and bilateral (both arms synchronously), each "
            "followed by a return of the hands to hand switches. Each movement "
            "follows a rest of at least 5 s with the hands on the switches; a "
            "shorter rest triggered an error message. 3 sets of 40 movements per "
            "condition. First study: 8 subjects, both conditions; second study: "
            "6 subjects, unilateral only.",
            feedback_type="none",
            synchronicity="self-paced",
            mode="offline",
            events={"unilateral": 1, "bilateral": 2, "rest": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41598-024-65910-8",
            description="EEG dataset of self-initiated, executed unilateral "
            "(right arm) and bilateral reaching movements from 14 healthy subjects "
            "(8 in both conditions, 6 in the unilateral condition only), for "
            "movement intention recognition in exoskeleton-supported "
            "rehabilitation.",
            investigators=[
                "Niklas Kueper",
                "Su Kyoung Kim",
                "Kartik Chari",
                "Elsa Andrea Kirchner",
            ],
            institution="German Research Centre for Artificial Intelligence (DFKI)",
            country="DE",
            data_url="https://doi.org/10.5281/zenodo.17940098",
            publication_year=2024,
            keywords=[
                "EEG",
                "movement intention",
                "movement execution",
                "ERP",
                "BCI",
                "stroke rehabilitation",
                "LRP",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        tags=Tags(modality=["Motor"], type=["Movement Execution"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_execution",
            imagery_tasks=["unilateral", "bilateral", "rest"],
        ),
        data_structure=DataStructureMetadata(
            n_trials=240,
            n_trials_per_class={"unilateral": 120, "bilateral": 120},
            n_blocks=6,
            trials_context=(
                "3 sets of 40 executed movements per condition by design; one run "
                "per set. Subjects 1-8: 6 runs (subject XP01: 7 recordings, its "
                "third unilateral set split in two), 240 movement trials. "
                "Subjects 9-14: 3 unilateral runs, 120 movement trials. Error and "
                "additional trials are dropped (only onsets followed by the "
                "end-of-correct-trial marker S128 are kept); one rest trial is "
                "added per correct movement whose preceding rest lasted at least "
                "5 s. Loaded: 1,662 unilateral, 959 bilateral and 2,542 rest "
                "trials (subjects 9-14: 116-119 movements, as most of their "
                "recordings also mark the first hand placement with S128)."
            ),
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            highpass_hz=0.1,
            lowpass_hz=131.0,
            notes="Hardware-prefiltered by the amplifier to 0.1-131 Hz; no further "
            "preprocessing applied.",
        ),
        file_format="BrainVision",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, len(SUBJECT_CODES) + 1)),
            sessions_per_subject=1,
            events={"unilateral": 1, "bilateral": 2, "rest": 3},
            code="Kueper2024",
            interval=(-2.0, 1.0),
            paradigm="imagery",
            doi="10.1038/s41598-024-65910-8",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of BrainVision header paths for a single subject."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        data_dir = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        root = data_dir / "EEG_dataset"
        code = SUBJECT_CODES[subject - 1]

        def headers():
            return [
                str(p)
                for condition in ("unilateral", "bilateral")
                for p in sorted((root / "EEG" / condition / code).glob("*.vhdr"))
            ]

        # Check this subject's own files, not the root folder: a cache made by
        # version 1, or seeded from the NEMAR deposit, has subjects 1-8 only.
        if force_update or not headers():
            download_and_extract_subject_zip(
                Kueper2024_URL,
                self.code,
                data_dir,
                path,
                force_update,
                verbose,
                fname=Kueper2024_ZIP,
                redownload_corrupted=True,
                skip_existing=not force_update,
            )
        return headers()

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        runs = {}
        for run_idx, vhdr in enumerate(self.data_path(subject)):
            condition = "bilateral" if "bilateral" in Path(vhdr).parts else "unilateral"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            raw.set_channel_types(
                {ch: "misc" for ch in ACCEL_CHANNELS if ch in raw.ch_names}
            )
            movement, rest = _trial_events(raw.annotations, condition)
            onsets = np.concatenate([movement, rest])
            labels = [condition] * len(movement) + ["rest"] * len(rest)
            order = np.argsort(onsets, kind="stable")
            raw.set_annotations(
                mne.Annotations(
                    onsets[order], [0.0] * len(onsets), np.asarray(labels)[order]
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage("colin27_1020", match_case=False, on_missing="ignore")
            runs[f"{run_idx}{condition}"] = raw
        return {"0": runs}
