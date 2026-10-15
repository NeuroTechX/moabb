"""Peters2026: BCI-FIT longitudinal P300 calibration dataset (OpenNeuro ds007720).

Peters et al. (2026), OpenNeuro ds007720 (CC0).
Data DOI: 10.18112/openneuro.ds007720.v1.0.0
"""

import logging
import warnings
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne.channels import get_builtin_montages, make_standard_montage

from ._openneuro_mirror import OpenNeuroMirrorMixin
from .base import BaseBIDSDataset
from .download import data_dl, get_dataset_path, nemar_dl
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
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# Download folder shared by every ``cap`` variant (their codes differ).
_SIGN = "Peters2026"
_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds007720"
_SFREQ = 300.0

# Calibration tasks and the layout tag each one gets in the MOABB run name.
_LAYOUTS = {"MatrixCalibration": "matrix", "RSVPCalibration": "rsvp"}

# ``events.tsv`` codes every flash twice at the same sample: once as
# ``target``/``nontarget`` and once with the presented character. The numeric
# ``value`` column is assigned per file and is not stable across runs.
_EVENTS = {"Target": 2, "NonTarget": 1}
_TRIAL_TYPES = {"target": "Target", "nontarget": "NonTarget"}

# Calibration runs of release 1.0.1: subject -> visit -> (task, BIDS run).
# fmt: off
_M, _R = "MatrixCalibration", "RSVPCalibration"
_CALIBRATION_RUNS = {
    1: {1: ((_M, 1), (_R, 2)), 2: ((_R, 1), (_M, 6)), 3: ((_R, 1), (_M, 7)),
        4: ((_M, 1), (_M, 6)), 5: ((_M, 1), (_M, 6)), 6: ((_M, 1), (_M, 7)),
        7: ((_M, 1), (_M, 6)), 8: ((_M, 1), (_M, 2), (_M, 7)),
        9: ((_M, 1), (_M, 6))},
    2: {1: ((_M, 1), (_R, 2)), 2: ((_M, 1), (_R, 7)), 3: ((_R, 1), (_M, 8)),
        4: ((_M, 1), (_R, 6), (_R, 7)), 5: ((_R, 1), (_M, 8)),
        6: ((_R, 1), (_M, 6)), 7: ((_M, 1), (_M, 2), (_M, 7)),
        8: ((_M, 1), (_M, 6)), 9: ((_M, 1), (_M, 6)), 10: ((_M, 1), (_M, 6))},
    3: {1: ((_M, 2), (_R, 3)), 2: ((_M, 1), (_R, 6), (_R, 7)),
        3: ((_R, 1), (_M, 6)), 4: ((_R, 1), (_M, 6)), 5: ((_M, 1), (_M, 7)),
        6: ((_M, 1), (_M, 6)), 7: ((_M, 1), (_M, 6)), 8: ((_M, 1), (_M, 6)),
        9: ((_M, 1), (_M, 6))},
    4: {1: ((_M, 1), (_R, 2)), 2: ((_R, 1), (_M, 6)), 3: ((_M, 1), (_M, 7)),
        4: ((_M, 1), (_M, 6)),
        5: ((_M, 1), (_M, 2), (_M, 4), (_M, 9), (_M, 10)),
        6: ((_M, 1), (_M, 6))},
    5: {1: ((_M, 1), (_R, 2)), 2: ((_M, 1), (_M, 6)), 3: ((_M, 1), (_M, 6)),
        4: ((_M, 1), (_M, 6)), 5: ((_M, 1), (_M, 3), (_M, 8)),
        6: ((_M, 1), (_M, 6))},
}
# fmt: on

# Visits recorded with the DSI-Flex; every other visit used the DSI-24
# (``CapManufacturersModelName`` in the ``*_eeg.json`` sidecars).
_DSI_FLEX_VISITS = {1: {1}, 2: {1, 2, 3, 4, 5, 6, 7}}
_CAPS = ("DSI-24", "DSI-Flex")

# Channel names as stored in the release (``channels.tsv``). The DSI-24 keeps
# the old 10-20 temporal names and also records the A2 ear electrode, which
# belongs to the linked-ear reference and is typed ``misc`` by this loader.
# fmt: off
_DSI24_EEG = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
# fmt: on
# The README lists FCz among the DSI-Flex channels, but FCz is that cap's
# ground; the files and BciPy's DSI-Flex device definition call the channel Cz.
_DSI_FLEX_EEG = ["Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
_EAR_CHANNELS = ["A2"]
_CHANNEL_ORDER = (
    _DSI24_EEG + [ch for ch in _DSI_FLEX_EEG if ch not in _DSI24_EEG] + _EAR_CHANNELS
)
_COMMON_EEG = [ch for ch in _DSI24_EEG if ch in _DSI_FLEX_EEG]

_RUN_SUFFIXES = (
    "eeg.vhdr",
    "eeg.vmrk",
    "eeg.eeg",
    "eeg.json",
    "channels.tsv",
    "events.tsv",
    "events.json",
)
_ROOT_FILES = ("dataset_description.json", "participants.tsv", "participants.json")


def _cap(subject, visit):
    return "DSI-Flex" if visit in _DSI_FLEX_VISITS.get(subject, ()) else "DSI-24"


def _run_names(subject, visit):
    """MOABB run names of a visit, ``{bids_run: "<k><layout>"}``.

    ``k`` numbers the visit's calibration runs in BIDS run order, so every
    name keeps a unique index (MOABB rejects duplicate run indices) and the
    layout stays readable, e.g. ``"0matrix"``, ``"1rsvp"``.
    """
    runs = sorted(_CALIBRATION_RUNS[subject][visit], key=lambda tr: tr[1])
    return {run: f"{k}{_LAYOUTS[task]}" for k, (task, run) in enumerate(runs)}


class Peters2026(OpenNeuroMirrorMixin, BaseBIDSDataset):
    """BCI-FIT longitudinal P300 calibration dataset from Peters et al. (2026).

    BCI-FIT (BCI Functional Implementation Toolkit) is a customization
    protocol for non-implantable communication BCIs [1]_. Five participants
    with speech and/or physical impairments due to amyotrophic lateral
    sclerosis (ALS; ``sub-P1`` to ``sub-P5``, demographics withheld by the
    authors) used the Matrix and RSVP typing layouts of BciPy 2.0 [2]_ over
    6 to 10 visits each (9, 10, 9, 6 and 6). Between visits the system was
    customized (layout, timing, inquiry length, cap) from typing performance,
    participant preference, clinical observations and offline simulations.

    **Task and events.** Each calibration inquiry shows the target
    character (1 s), a cue (0.5 s) and then a stream of characters from a
    28-symbol alphabet (A-Z, ``<`` backspace, ``_`` space), flashed one at
    a time; the participant watches for the target. MOABB returns one
    ``Target`` or ``NonTarget`` epoch per flash, from the ``target`` and
    ``nontarget`` rows of the BIDS ``events.tsv`` (each flash is also listed
    with its character, which :meth:`get_additional_metadata` returns as
    ``character``). Inquiry-preview and key-press rows are not epoched (the
    files have no prompt or cue rows). The README describes the default
    settings (55 inquiries of 14 flashes at 5 Hz, 10% of inquiries without
    a target, 4 s blank); the released runs follow the per-visit
    customization:

    - stimulus onset asynchrony of 83 to 337 ms (about 12 Hz to 3 Hz), most
      runs at 237 ms (Matrix) or 203 ms (RSVP);
    - 14 flashes per inquiry, or 28 in five Matrix runs of ``sub-P2``
      (visits 7 to 10), which have a target in every inquiry;
    - eight runs with an inquiry preview confirmed by a switch press
      (``sub-P2`` visits 4 and 5 RSVP, ``sub-P4`` visits 3 to 5), most of
      them with 46 to 50 inquiries and few or no target-free inquiries.

    The 88 calibration runs hold 4,472 inquiries and 66,150 flashes
    (4,109 ``Target``, 62,041 ``NonTarget``). Seven runs end early and are
    kept as released: 1 inquiry (``sub-P1`` visit 8 run 2, ``sub-P4``
    visit 5 run 10, ``sub-P5`` visit 5 run 3), 2 or 5 inquiries
    (``sub-P4`` visit 5 runs 4 and 1, ``sub-P3`` visit 2 RSVP run 6) and
    33 inquiries (``sub-P2`` visit 7 run 1).

    **Sessions and runs.** One MOABB session per visit, ``"0"`` for
    ``ses-1`` to ``"9"`` for ``ses-10``. Runs are named ``"<k>matrix"`` or
    ``"<k>rsvp"``, with ``k`` the order of the calibration run within the
    visit (BIDS run order), so analyses can select or split by layout from
    the run name. 70 runs are Matrix and 18 RSVP; RSVP calibration only
    occurs in the early visits (``sub-P1`` 1-3, ``sub-P2`` 1-6, ``sub-P3``
    1-4, ``sub-P4`` 1-2, ``sub-P5`` 1), later visits are Matrix-only.

    **Caps and channels.** Each visit used one dry-electrode cap
    (Wearable Sensing): the DSI-24 (19 scalp channels, 10-20 names with
    T3/T4/T5/T6, plus the A2 ear electrode, typed ``misc`` here) or the
    DSI-Flex (7 channels). ``sub-P3``, ``sub-P4`` and ``sub-P5`` used the
    DSI-24 throughout; ``sub-P1`` used the DSI-Flex at visit 1 and the
    DSI-24 afterwards; ``sub-P2`` used the DSI-Flex at visits 1 to 7 and
    the DSI-24 at visits 8 to 10. The two caps share only **Cz, P3, Pz and
    P4**. The DSI-Flex channels are Cz, Oz, P3, P4, PO7, PO8 and Pz, as
    named in the released files. The README lists FCz instead of Cz, but
    FCz is the DSI-Flex ground electrode (README and every DSI-Flex
    ``*_eeg.json``), which cannot also be a recording site; BciPy's own
    DSI-Flex device definition labels the cap's first sensor Cz, and the
    BciPy BIDS conversion only keeps channels named in its 10-20 list,
    which has no FCz. This loader therefore keeps Cz and the standard Cz
    position. The ERPs and spectra of P1 and P2 could not tell Cz from FCz
    (ratios to Pz varied more between visits than between the two
    positions), so the evidence for Cz is documentary.

    MOABB evaluations load all subjects and visits into one array, so each
    variant of this loader returns a single channel set. By default
    (``cap=None``) all 40 visits are returned with the four shared
    channels Cz, P3, Pz and P4. ``cap="DSI-24"`` returns the 32 DSI-24
    visits (all five subjects; ``sub-P1`` visits 2-9, ``sub-P2`` visits
    8-10) with the 19 DSI-24 channels (and A2 with
    ``return_all_modalities=True``), and ``cap="DSI-Flex"`` the 8 DSI-Flex
    visits (``sub-P1`` and ``sub-P2``) with its 7 channels. Session keys
    always number the visit, so ``cap="DSI-24"`` starts at session ``"1"``
    for ``sub-P1``. Positions come from the standard 10-20 template; the
    release's electrode files hold template positions too, not digitized
    ones.

    **Signal.** Linked-ear reference ((A1 + A2) / 2) with an active
    common-mode follower; ground at FCz (DSI-Flex) or Fpz (DSI-24);
    hardware band 0.003-150 Hz. The released files are sampled at 300 Hz
    and are not filtered: the sidecars declare no software filter, and the
    recordings carry power above 20 Hz and, in most runs, 60 Hz line
    noise. The README's downsampling to 150 Hz, 1-20 Hz band-pass and
    60 Hz notch were therefore not applied to the distributed files; MOABB
    paradigms apply their own band-pass.

    **Timing and the late RSVP P300.** Flash onsets are BciPy's
    display-flip timestamps, shifted during the BIDS conversion by the
    static offset of the device (0.1 s), the same for both layouts; the
    release has no trigger or photodiode channel. In the 16 visits with
    both layouts, the Pz Target minus NonTarget wave peaks later in RSVP
    than in Matrix runs (cross-correlation delay 50 to 327 ms, median
    92 ms; per participant the peak is at 380-450 ms in Matrix and
    456-713 ms in RSVP runs), while the flash-locked steady-state visual
    response has the same latency in both layouts (median difference 7 ms,
    95% CI -19 to 16 ms, in the 13 visits at 203 vs 237 ms SOA). A trigger offset would delay both, so the late
    RSVP P300 is a genuine latency difference and the events are not
    shifted. The [0, 0.8] s window still contains it.

    **Not included.** The CopyPhrase runs (copy-spelling) are excluded:
    they are closed loop, with the classifier choosing the next inquiries
    and stopping at a 0.8 probability threshold, so they have no fixed
    Target/NonTarget design. The NEMAR mirror ``on007720`` of the OpenNeuro
    release is used when available, with OpenNeuro as fallback; only the
    calibration files are downloaded (about 0.9 GB of the 2.3 GB release).

    Please cite the BCI-FIT poster [1]_ (the authors' requested citation)
    and the OpenNeuro release [3]_. Release 1.0.1 only changed text
    relative to 1.0.0; OpenNeuro has registered a DOI for 1.0.0 only.

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load (1-5 for ``sub-P1`` to ``sub-P5``).
    sessions : list of int | str | None
        Sessions (visits) to keep, ``"0"`` being ``ses-1``.
    cap : None | "DSI-24" | "DSI-Flex"
        ``None`` returns all visits with the channels both caps share
        (Cz, P3, Pz, P4); a cap name returns only that cap's visits with
        its full channel set. Each variant has its own ``code``
        (``Peters2026``, ``Peters-dsi24-2026``, ``Peters-dsiflex-2026``), so
        caches and results stay apart; downloads are shared.
    return_all_modalities : bool | dict
        Also return non-EEG channels (A2 with ``cap="DSI-24"``).

    References
    ----------
    .. [1] B. Peters, M. Kinsella, D. Klee, M. Lawhead, T. Memmott,
       S. Spaulding, B. Oken and M. Fried-Oken, "BCI-FIT: Effects of cBCI
       customization on performance," poster, 11th International BCI
       Meeting, Banff, AB, Canada, June 2-5, 2025.
    .. [2] T. Memmott, A. Koçanaoğulları, M. Lawhead, D. Klee, S. Dudy,
       M. Fried-Oken and B. Oken, "BciPy: brain-computer interface software
       in Python," Brain-Computer Interfaces, vol. 8, no. 4, pp. 137-153,
       2021. DOI: 10.1080/2326263X.2021.1878727
    .. [3] B. Peters, D. Klee, M. Kinsella, Y. Hendin, T. Memmott,
       M. Lawhead, S. Ananthoju, B. Celik, S. Spaulding, B. Oken and
       M. Fried-Oken, "BCI-FIT: A customization protocol for communication
       brain-computer interface systems," OpenNeuro ds007720, 2026.
       DOI: 10.18112/openneuro.ds007720.v1.0.0

    Notes
    -----
    .. versionadded:: 1.8.0
    """

    nemar_id = "on007720"
    nemar_subject_template = "P{subject}"
    # Raw EEG files of the calibration tasks only (CopyPhrase is not loaded).
    nemar_bids_filters = {"scope": "raw", "datatype": "eeg", "task": list(_LAYOUTS)}
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": len(_CHANNEL_ORDER) - 1, "misc": 1},
            montage="standard_1020",
            hardware="Wearable Sensing DSI-24 or DSI-Flex (dry electrodes)",
            software="BciPy 2.0",
            cap_manufacturer="Wearable Sensing",
            cap_model="DSI-24 (19 ch) or DSI-Flex (7 ch), chosen per visit",
            sensor_type="dry",
            electrode_type="dry",
            reference="linked ears ((A1 + A2) / 2), active common-mode follower",
            ground="FCz (DSI-Flex) or Fpz (DSI-24)",
            filters={"highpass": 0.003, "lowpass": 150},
            sensors=list(_CHANNEL_ORDER),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=5,
            health_status="patients",
            clinical_population=(
                "amyotrophic lateral sclerosis (ALS) with speech and/or "
                "physical impairments"
            ),
            bci_experience="mixed",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="calibration",
            events=dict(_EVENTS),
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=0.8,
            tasks=list(_LAYOUTS),
            study_design=(
                "Longitudinal customization protocol: calibration and "
                "copy-spelling with Matrix and RSVP layouts over 6-10 visits, "
                "with system settings adjusted between visits. Calibration "
                "inquiries: target prompt, cue, then a stream of 14 (or 28) "
                "flashed characters from a 28-symbol alphabet."
            ),
            feedback_type="none",
            stimulus_type="matrix or RSVP speller",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions=(
                "Watch for the intensification or presentation of the "
                "prompted target character."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds007720.v1.0.0",
            description=(
                "BCI-FIT: longitudinal Matrix and RSVP P300 calibration "
                "recordings of five participants with ALS using a "
                "customized non-implantable communication BCI."
            ),
            investigators=[
                "Betts Peters",
                "Daniel Klee",
                "Michelle Kinsella",
                "Yonah Hendin",
                "Tab Memmott",
                "Matthew Lawhead",
                "Srikar Ananthoju",
                "Basak Celik",
                "Scott Spaulding",
                "Barry Oken",
                "Melanie Fried-Oken",
            ],
            senior_author="Melanie Fried-Oken",
            contact_info=["Betts Peters (petersbe@ohsu.edu)"],
            institution="Oregon Health & Science University",
            institution_address="3181 SW Sam Jackson Park Rd, Portland, OR 97239, USA",
            country="US",
            repository="OpenNeuro",
            data_url="https://openneuro.org/datasets/ds007720",
            license="CC0",
            publication_year=2026,
            related_paper_dois=["10.1080/2326263X.2021.1878727"],
            funding=[
                "National Institutes of Health, National Institute on "
                "Deafness and Other Communication Disorders, grant R01DC009834"
            ],
            ethics_approval=[
                "Oregon Health & Science University Institutional Review "
                "Board (IRB#26456)"
            ],
            how_to_acknowledge=(
                "Peters, B., Kinsella, M., Klee, D., Lawhead, M., Memmott, T., "
                "Spaulding, S., Oken, B., & Fried-Oken, M. (2025, June 2-5). "
                "BCI-FIT: Effects of cBCI customization on performance "
                "[Poster presentation]. 11th International BCI Meeting, "
                "Banff, AB, Canada."
            ),
            keywords=[
                "P300",
                "ERP",
                "BCI",
                "ALS",
                "RSVP",
                "matrix speller",
                "dry electrodes",
                "longitudinal",
                "cross-session",
            ],
        ),
        sessions_per_subject=6,
        runs_per_session=2,
        data_processed=False,
        file_format="BrainVision (BIDS)",
        external_links={
            "nemar": "https://nemar.org/dataexplorer/detail?dataset_id=on007720",
            "bcipy": "https://github.com/CAMBI-tech/BciPy",
        },
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            notes=(
                "Files are distributed unfiltered at the 300 Hz acquisition "
                "rate (60 Hz line noise in most runs); the README's downsampling to "
                "150 Hz, 1-20 Hz band-pass and 60 Hz notch are not applied to "
                "the distributed files."
            ),
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", n_targets=28, soa_ms=237.0
        ),
        data_structure=DataStructureMetadata(
            n_trials=66150,
            n_trials_per_class={"Target": 4109, "NonTarget": 62041},
            trials_context=(
                "88 calibration runs over 40 visits (9, 10, 9, 6 and 6 per "
                "subject), 4472 inquiries, 66150 flashes. Per subject: "
                "P1 901 T / 12973 NT, P2 1100 / 18836, P3 904 / 13026, "
                "P4 603 / 8553, P5 601 / 8653; the summary table gives the "
                "per-subject mean."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["regularized discriminant analysis (BciPy default)"]
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_session", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            environment="home or clinic",
            online_feedback=False,
            applications=["communication", "speller"],
        ),
        tags=Tags(pathology=["ALS"], modality=["visual"], type=["erp"]),
    )

    def __init__(
        self, subjects=None, sessions=None, *, cap=None, return_all_modalities=False
    ):
        if cap not in (None, *_CAPS):
            raise ValueError(f"cap must be None, 'DSI-24' or 'DSI-Flex', got {cap!r}")
        self.cap = cap
        # Channels returned by this variant, in a fixed order.
        self._channels = {
            None: list(_COMMON_EEG),
            "DSI-24": _DSI24_EEG + _EAR_CHANNELS,
            "DSI-Flex": [ch for ch in _CHANNEL_ORDER if ch in _DSI_FLEX_EEG],
        }[cap]
        all_subjects = [
            s
            for s in sorted(_CALIBRATION_RUNS)
            if cap is None or any(_cap(s, v) == cap for v in _CALIBRATION_RUNS[s])
        ]
        # Variants return different visits and channels: separate codes keep
        # their epoch caches and results apart (downloads stay shared).
        code = {
            None: "Peters2026",
            "DSI-24": "Peters-dsi24-2026",
            "DSI-Flex": "Peters-dsiflex-2026",
        }
        super().__init__(
            subjects=all_subjects,
            sessions_per_subject=1 if cap == "DSI-Flex" else 3 if cap else 6,
            events=dict(_EVENTS),
            code=code[cap],
            interval=[0.0, 0.8],
            paradigm="p300",
            doi="10.18112/openneuro.ds007720.v1.0.0",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _visits(self, subject):
        """Visits (BIDS ``ses`` numbers) returned for ``subject``."""
        return [
            v
            for v in sorted(_CALIBRATION_RUNS[subject])
            if self.cap is None or _cap(subject, v) == self.cap
        ]

    # -- download ---------------------------------------------------------

    def _download_nemar(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        # Shared folder for all ``cap`` variants, whatever ``self.code`` is.
        return nemar_dl(
            self.nemar_id,
            _SIGN,
            path=path,
            force_update=force_update,
            subject=self._nemar_subject(subject),
            verbose=verbose,
            **self.nemar_bids_filters,
        )

    def download(
        self,
        subject_list=None,
        path=None,
        force_update=False,
        update_path=None,
        accept=False,
        verbose=None,
    ):
        """Download the calibration files of the given subjects."""
        for subject in self.subject_list if subject_list is None else subject_list:
            if subject not in self.subject_list:
                raise ValueError("Invalid subject number")
            self._download_subject(subject, path, force_update, update_path, verbose)

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        """Fetch the subject's calibration files (all visits) and return the BIDS root."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        if mirror_root is not None:
            return mirror_root

        bids_root = Path(get_dataset_path(_SIGN, path)) / f"MNE-{_SIGN.lower()}-data"
        rel_paths = list(_ROOT_FILES)
        sub = f"sub-P{subject}"
        for visit, runs in _CALIBRATION_RUNS[subject].items():
            ses = f"ses-{visit}"
            for task, run in runs:
                stem = f"{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_run-{run}"
                rel_paths += [f"{stem}_{suffix}" for suffix in _RUN_SUFFIXES]
        for rel_path in rel_paths:
            data_dl(
                f"{_S3_BASE}/{rel_path}",
                _SIGN,
                path=path,
                force_update=force_update,
                verbose=verbose,
                fname=rel_path,
            )
        return str(bids_root)

    # -- BIDS selection -----------------------------------------------------

    def _get_path_search_params(self, subject):
        out = {
            "tasks": list(_LAYOUTS),
            "datatypes": "eeg",
            "suffixes": "eeg",
            "extensions": ".vhdr",
        }
        if subject is not None:
            out["subjects"] = f"P{subject}"
        return out

    def bids_paths(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Calibration ``.vhdr`` paths of the selected visits, in visit/run order."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        paths = super().bids_paths(subject, path, force_update, update_path, verbose)
        found = {(int(p.session), p.task, int(p.run)): p for p in paths}
        expected = [
            (visit, task, run)
            for visit in self._visits(subject)
            for task, run in sorted(
                _CALIBRATION_RUNS[subject][visit], key=lambda tr: tr[1]
            )
        ]
        missing = [key for key in expected if key not in found]
        if missing:
            raise FileNotFoundError(
                f"Peters2026 subject P{subject} is missing calibration runs "
                f"(visit, task, run): {missing}"
            )
        return [found[key] for key in expected]

    # -- reading --------------------------------------------------------------

    @staticmethod
    def _events_table(bids_path):
        """Flash rows of a run's ``events.tsv`` with their presented character."""
        events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
        df = pd.read_csv(events_tsv, sep="\t", keep_default_na=False)
        flashes = df[df["trial_type"].isin(list(_TRIAL_TYPES))]
        characters = df[df["trial_type"].str.len() == 1].drop_duplicates("sample")
        flashes = flashes.merge(
            characters[["sample", "trial_type"]].rename(
                columns={"trial_type": "character"}
            ),
            on="sample",
            how="left",
        )
        return pd.DataFrame(
            {
                "onset": flashes["onset"].astype(float).to_numpy(),
                "sample": flashes["sample"].astype(int).to_numpy(),
                "trial_type": flashes["trial_type"].map(_TRIAL_TYPES).to_numpy(),
                "character": flashes["character"].to_numpy(),
            }
        )

    def _read_run(self, bids_path):
        with warnings.catch_warnings():
            # participants.tsv has release-specific columns (devices_used, ...).
            warnings.filterwarnings("ignore", "Unable to map the following column")
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
        raw.load_data(verbose=False)
        flashes = self._events_table(bids_path)
        raw.set_annotations(
            mne.Annotations(
                onset=flashes["onset"].to_numpy(),
                duration=np.zeros(len(flashes)),
                description=flashes["trial_type"].to_numpy(),
                orig_time=raw.annotations.orig_time,
            )
        )
        return raw

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: raw}}`` for the selected visits."""
        bids_paths = self.bids_paths(subject)
        data = {}
        for bids_path in bids_paths:
            visit, run = int(bids_path.session), int(bids_path.run)
            session = str(visit - 1)
            data.setdefault(session, {})[_run_names(subject, visit)[run]] = (
                self._read_run(bids_path)
            )

        montage = make_standard_montage(
            "colin27_1020"
            if "colin27_1020" in get_builtin_montages()
            else "standard_1020"
        )
        for runs in data.values():
            for raw in runs.values():
                # One channel set and order for the whole variant, so MOABB can
                # concatenate epochs across visits and subjects.
                raw.pick(self._channels).reorder_channels(self._channels)
                raw.set_montage(montage, verbose=False)
                ears = [ch for ch in _EAR_CHANNELS if ch in raw.ch_names]
                if ears:
                    raw.set_channel_types(
                        dict.fromkeys(ears, "misc"),
                        on_unit_change="ignore",
                        verbose=False,
                    )
        return data

    def get_additional_metadata(self, subject, session, run):
        """Per-flash metadata: label, presented character, layout, BIDS run, cap."""
        visit = int(session) + 1
        bids_run = next(
            r for r, name in _run_names(subject, visit).items() if name == run
        )
        bids_path = next(
            p
            for p in self.bids_paths(subject)
            if int(p.session) == visit and int(p.run) == bids_run
        )
        flashes = self._events_table(bids_path)
        return flashes.drop(columns="sample").assign(
            duration=0.0,
            layout=_LAYOUTS[bids_path.task],
            task=bids_path.task,
            bids_run=bids_run,
            cap=_cap(subject, visit),
            subject=subject,
            session=session,
            run=run,
        )
