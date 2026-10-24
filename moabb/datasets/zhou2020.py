"""7-day motor imagery BCI EEG dataset.

Zhou et al. (2021), Frontiers in Human Neuroscience.
DOI: 10.3389/fnhum.2021.701091
Data DOI: 10.21227/f1c7-7x89
"""

import logging
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
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import download_and_extract_subject_zip


log = logging.getLogger(__name__)


class _ColumnLayoutError(ValueError):
    """An NPZ file has fewer columns than the subject's channel layout."""


# Zenodo re-hosted data (originally from IEEE DataPort DOI: 10.21227/f1c7-7x89).
_ZENODO_RECORD = "18988317"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# Two subject groups with different channel counts. We map A1-A8 -> subjects
# 13-20 for consistent MOABB numbering; the ``cohort`` option of Zhou2020
# selects which channel set all subjects share (see ``_COHORTS``).
#
# Column layout of the NSsignal ``signal`` array. The NPZ files carry no
# channel names; these come from the authors' ``readme_MIdataset.docx``
# (Table 1, Fig. 2) and ``MI_dataset.py``, both shipped with the original IEEE
# DataPort deposit (doi:10.21227/f1c7-7x89) but not with the Zenodo copy:
#
# - both cohorts: columns 0-25 = ``_26CH_NAMES``, 26 = hEOG, 27 = vEOG
#   (Table 1; ``MI_dataset.py`` names ``data[:, 0:28]`` this way for both);
# - S-subjects (46 columns): 28-29 = M1/M2 mastoids, 30-44 = ``_S_EXTRA_NAMES``,
#   45 = trigger. Table 1 lists the 15 extra names as indices 28-42, i.e.
#   after the mastoid columns are removed, which ``MI_dataset.py`` does with
#   ``np.delete(data, [28, 29, 45], 1)  # remove M1 M2 event channel``. The
#   recordings agree: columns 28-29 are a tightly correlated, blink-free pair,
#   and the 30-44 placement fits the signal correlation structure (#1120).
# - A-subjects (29 columns): 28 = trigger.
# The trigger column is not used, events come from ``MarkOnSignal``.

# 26 EEG channel names recorded in both cohorts (the analysis subset in paper).
_26CH_NAMES = [
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
]

# 15 extra EEG channels of the S-subjects' 41-channel layout (Fig. 2, right).
_S_EXTRA_NAMES = [
    "F5",
    "AF3",
    "AF4",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "PO3",
    "POz",
    "PO4",
    "Oz",
    "F6",
]
_EOG_NAMES = ["hEOG", "vEOG"]
_EOG_COLS = [26, 27]
_S_MASTOID_NAMES = ["M1", "M2"]
_S_MASTOID_COLS = [28, 29]
_S_EXTRA_FIRST_COL = 30

# GDF/BioSig event codes in MarkOnSignal → MOABB event names.
_GDF_EVENT_MAP = {
    769: "left_hand",  # 0x0301 — class 1 (left arrow)
    770: "right_hand",  # 0x0302 — class 2 (right arrow)
    771: "feet",  # 0x0303 — class 3 (down arrow)
    780: "rest",  # 0x030C — class 12 (up arrow = idle)
}

# MOABB event_id mapping (integer codes for stim channel).
_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}

# Channel counts per subject group (EEG only, excluding EOG and trigger).
_S_N_EEG = 41  # S-subjects (1-12)
_A_N_EEG = 26  # A-subjects (13-20)

_SFREQ = 500.0

# Download folder shared by every ``cohort`` variant (their codes differ).
_SIGN = "Zhou2020"

# Subjects and dataset code of each ``cohort`` option. MOABB evaluations
# concatenate every subject of a dataset into one array, so each option
# returns a single channel set; distinct codes keep their caches and results
# apart. ``is_abbrev("Zhou2020", code)`` only lets lower-case letters, digits
# and dashes follow a letter of the class name, and nothing follow its final
# digits, so the cohort tag sits before the year ("Zhou2020-S" would fail).
_S_SUBJECTS = list(range(1, 13))
_A_SUBJECTS = list(range(13, 21))
_COHORTS = {
    None: {"subjects": _S_SUBJECTS + _A_SUBJECTS, "code": "Zhou2020"},
    "S": {"subjects": _S_SUBJECTS, "code": "Zhou-s2020"},
    "A": {"subjects": _A_SUBJECTS, "code": "Zhou-a2020"},
}


def _cohort_of(subject):
    return "S" if subject in _S_SUBJECTS else "A"


def _channel_layout(subject, all_modalities=False, cohort=None):
    """Return ``(columns, names, types)`` of the channels loaded for ``subject``.

    With ``cohort="S"`` the S-subjects (1-12) get the 41-channel layout.
    Otherwise every subject gets the 26 channels both cohorts share, so all
    subjects of a ``cohort=None`` dataset have the same channels. With
    ``all_modalities``, hEOG/vEOG (type ``eog``) are added, and with
    ``cohort="S"`` also the M1/M2 mastoids (type ``misc``), which only the
    S-subject files hold.
    """
    full_s = cohort == "S" and _cohort_of(subject) == "S"
    eeg_cols = list(range(_A_N_EEG))
    eeg_names = list(_26CH_NAMES)
    eog_cols, eog_names, misc_cols, misc_names = [], [], [], []
    if all_modalities:
        eog_cols, eog_names = list(_EOG_COLS), list(_EOG_NAMES)
    if full_s:
        eeg_cols += [_S_EXTRA_FIRST_COL + i for i in range(len(_S_EXTRA_NAMES))]
        eeg_names += _S_EXTRA_NAMES
        if all_modalities:
            misc_cols, misc_names = list(_S_MASTOID_COLS), list(_S_MASTOID_NAMES)
    columns = eeg_cols + eog_cols + misc_cols
    names = eeg_names + eog_names + misc_names
    types = ["eeg"] * len(eeg_cols) + ["eog"] * len(eog_cols) + ["misc"] * len(misc_cols)
    return columns, names, types


def _make_metadata(cohort=None):
    """``DatasetMetadata`` of the variant returned for ``cohort``.

    ``channel_types`` counts what ``return_all_modalities=True`` returns:
    the EEG channels of the variant, hEOG/vEOG, and with ``cohort="S"`` the
    M1/M2 mastoids.
    """
    n_subjects = len(_COHORTS[cohort]["subjects"])
    n_trials = n_subjects * 7 * 6 * 40
    if cohort == "S":
        channel_types = {"eeg": _S_N_EEG, "eog": 2, "misc": 2}
        sensors = list(_26CH_NAMES) + list(_S_EXTRA_NAMES)
    else:
        channel_types = {"eeg": _A_N_EEG, "eog": 2}
        sensors = list(_26CH_NAMES)
    if cohort is None:
        # Demographics of all 20 subjects; not split by group here.
        demographics = {
            "gender": {"female": 9, "male": 11},
            "age_mean": 23.2,
            "age_min": 21,
            "age_max": 27,
            "age_std": 1.47,
        }
    else:
        demographics = {}
    which = {None: "20 subjects", "S": "12 S-subjects", "A": "8 A-subjects"}[cohort]
    return DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types=channel_types,
            montage="standard_1005",
            hardware="Neuroscan SynAmps2",
            sensor_type="Ag/AgCl",
            reference="vertex (Cz)",
            ground="AFz",
            filters={"bandpass": [0.5, 100], "notch_hz": 50},
            sensors=sensors,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=n_subjects,
            health_status="healthy",
            handedness="right-handed",
            bci_experience="mixed",
            species="human",
            **demographics,
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "rest"],
            trial_duration=5.0,
            study_design=(
                "7-day longitudinal MI-BCI study without feedback training. "
                "4 classes: left hand, right hand, both feet, idle"
            ),
            feedback_type="none",
            stimulus_type="arrow cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2021.701091",
            investigators=[
                "Qing Zhou",
                "Jiafan Lin",
                "Lin Yao",
                "Yueming Wang",
                "Yan Han",
                "Kedi Xu",
            ],
            institution="Zhejiang University",
            country="CN",
            repository="Zenodo",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
            publication_year=2021,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=7,
        runs_per_session=6,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet", "rest"],
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=n_trials,
            trials_context=(f"{which} x 7 sessions x 6 runs x 40 trials = {n_trials}"),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SVM"],
            feature_extraction=["CSP"],
            frequency_bands={"classification": [8.0, 30.0]},
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold", cv_folds=10, evaluation_type=["within_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["research"], environment="laboratory", online_feedback=False
        ),
        data_processed=True,
        file_format="NPZ",
    )


class Zhou2020(BaseDataset):
    """7-day motor imagery BCI EEG dataset from Zhou et al 2021.

    Dataset from the article *Relative Power Correlates With the Decoding
    Performance of Motor Imagery Both Across Time and Subjects* [1]_.

    It contains data recorded on 20 subjects over 7 sessions (one session
    every ~2 days over 2 weeks) with no feedback training. Two groups of
    subjects were recorded with a 64-channel Neuroscan SynAmps2 system
    at 500 Hz:

    - **S-subjects** (subjects 1-12): 41 EEG + 2 EOG + 2 mastoid channels
    - **A-subjects** (subjects 13-20): 26 EEG + 2 EOG channels

    Both cohorts share 26 fronto-central electrodes (F3-F4, FC5-FC6, C5-C6,
    CP5-CP6); the S-subjects add F5, F6, AF3, AF4, P5-P6, PO3, POz, PO4 and
    Oz. The data files carry no channel names: they come from Table 1 and
    Fig. 2 of the authors' ``readme_MIdataset.docx`` and from their
    ``MI_dataset.py``, both distributed with the original IEEE DataPort
    deposit [2]_ but not with the Zenodo copy used for download. The S-subject
    files also hold the M1/M2 mastoid channels (columns 28-29, as
    ``MI_dataset.py`` drops them), which Table 1 omits; the column placement was
    checked against the recordings' correlation structure (:gh:`1120`). hEOG
    and vEOG (type ``eog``) and the mastoids (type ``misc``) are loaded only
    with ``return_all_modalities``.

    **Cohorts and channel sets.** MOABB paradigms and evaluations stack the
    epochs of every subject of a dataset into one array, so all subjects of a
    dataset object must return the same channels. The ``cohort`` option
    selects which channel set that is:

    - ``cohort=None`` (default): all 20 subjects with the 26 channels both
      cohorts share. This is the only set that covers every subject, and the
      one the paper analyses. With ``return_all_modalities``, hEOG and vEOG
      are added; the mastoids are not, as the A-subjects have none.
    - ``cohort="S"``: subjects 1-12 only, with all 41 EEG channels (and
      hEOG, vEOG, M1, M2 with ``return_all_modalities``).
    - ``cohort="A"``: subjects 13-20 only, with the 26 EEG channels (and
      hEOG, vEOG with ``return_all_modalities``).

    Subject numbers stay 1-20 in every option: S1-S12 are subjects 1-12 and
    A1-A8 subjects 13-20. Asking a cohort for a subject of the other cohort
    raises a ``ValueError``. Each option has its own ``code``
    (``Zhou2020``, ``Zhou-s2020``, ``Zhou-a2020``), so cached epochs and
    results never mix channel sets; downloads are shared. ``METADATA`` of a
    ``cohort`` instance describes that variant (subjects, trials, channels);
    the sex and age figures cover all 20 subjects, so only the default
    carries them.

    Four MI classes were recorded: left hand, right hand, both feet, and
    idle/rest. Each session contains ~6 runs of 40 trials each (10 per
    class), giving ~240 trials per session and ~1680 trials per subject.

    The data is stored as Neuroscan NSsignal NPZ files with continuous
    recordings (band-pass 0.5-100 Hz, 50 Hz notch). Events are encoded
    using GDF/BioSig codes: 769 (left), 770 (right), 771 (feet), 780 (rest).

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load (1-20, restricted to the selected ``cohort``).
    sessions : list of int | str | None
        Sessions to keep.
    cohort : None | "S" | "A"
        ``None`` returns all 20 subjects on the 26 shared channels; ``"S"``
        subjects 1-12 with all 41 EEG channels; ``"A"`` subjects 13-20 with
        their 26 channels.
    return_all_modalities : bool | dict
        Also return hEOG/vEOG (``eog``) and, with ``cohort="S"``, the M1/M2
        mastoids (``misc``).

    References
    ----------
    .. [1] Zhou, Q., Lin, J., Yao, L., Wang, Y., Han, Y., Xu, K. (2021).
           Relative Power Correlates With the Decoding Performance of Motor
           Imagery Both Across Time and Subjects. Frontiers in Human
           Neuroscience, 15, 701091.
           https://doi.org/10.3389/fnhum.2021.701091
    .. [2] Zhou, Q. et al. (2020). EEG dataset of 7-day Motor Imagery BCI.
           IEEE DataPort. https://doi.org/10.21227/f1c7-7x89
    """

    METADATA = _make_metadata(None)
    nemar_id = "nm000237"

    def __init__(
        self, subjects=None, sessions=None, *, cohort=None, return_all_modalities=False
    ):
        if cohort not in _COHORTS:
            raise ValueError(f"cohort must be None, 'S' or 'A', got {cohort!r}")
        self.cohort = cohort
        if cohort is not None:
            # Instance override: METADATA describes what this variant returns.
            self.METADATA = _make_metadata(cohort)
        if subjects is not None:
            self._check_cohort(subjects)
        super().__init__(
            subjects=list(_COHORTS[cohort]["subjects"]),
            sessions_per_subject=7,
            events=dict(_EVENTS),
            code=_COHORTS[cohort]["code"],
            interval=[0, 5],
            paradigm="imagery",
            doi="10.3389/fnhum.2021.701091",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    @property
    def metadata(self):
        """Structured metadata of this ``cohort`` variant."""
        if self.cohort is None:
            return super().metadata
        return self.METADATA

    def _check_cohort(self, subjects):
        """Raise a ``ValueError`` naming the cohort of out-of-cohort subjects."""
        if self.cohort is None:
            return
        allowed = _COHORTS[self.cohort]["subjects"]
        outside = [s for s in subjects if s not in allowed]
        if outside:
            raise ValueError(
                f"Zhou2020(cohort={self.cohort!r}) holds subjects "
                f"{allowed[0]}-{allowed[-1]} only; subject(s) {outside} are not "
                f"in this cohort. Use cohort=None for all 20 subjects on the 26 "
                f"shared channels, or cohort='S' (subjects 1-12, 41 EEG channels) "
                f"/ cohort='A' (subjects 13-20, 26 EEG channels)."
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

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        data_dir = Path(self.data_path(subject))
        subj_dir = data_dir / f"S{subject:02d}"

        layout = _channel_layout(subject, bool(self.return_all_modalities), self.cohort)

        sessions = {}
        session_dirs = sorted(d for d in subj_dir.iterdir() if d.is_dir())

        for sess_idx, sess_dir in enumerate(session_dirs):
            sess_key = str(sess_idx)
            runs = {}

            npz_files = sorted(sess_dir.glob("*.npz"))
            for run_idx, npz_file in enumerate(npz_files):
                try:
                    raw = self._npz_to_raw(npz_file, *layout)
                    runs[str(run_idx)] = raw
                except _ColumnLayoutError:
                    # A file that does not match the channel layout would load
                    # with the wrong channel names: fail instead of skipping it.
                    raise
                except Exception as e:
                    log.warning("Failed to load %s: %s", npz_file.name, e)

            if runs:
                sessions[sess_key] = runs

        if not sessions:
            raise FileNotFoundError(f"No data found for subject {subject} in {subj_dir}")
        return sessions

    def _npz_to_raw(self, npz_path, columns, ch_names, ch_types):
        """Convert an NSsignal NPZ file to MNE Raw.

        Parameters
        ----------
        npz_path : pathlib.Path
            Path to the .npz file.
        columns : list of int
            Columns of the ``signal`` array to load, see ``_channel_layout``.
        ch_names : list of str
            Channel names, one per column.
        ch_types : list of str
            Channel types, one per column.
        """
        npz = np.load(npz_path, allow_pickle=True)
        signal = npz["signal"]  # (n_samples, n_channels_total)
        mos = npz["MarkOnSignal"]  # (n_events, 2): [sample, code]
        sfreq = float(npz["SampleRate"][0])

        if signal.shape[1] <= max(columns):
            raise _ColumnLayoutError(
                f"{npz_path.name} has {signal.shape[1]} columns, expected more than "
                f"{max(columns)} for this subject's channel layout"
            )
        eeg_data = signal[:, columns].T  # (n_channels, n_samples)

        # Scale from microvolts to volts for MNE.
        if np.abs(eeg_data).max() > 1e-3:
            eeg_data = eeg_data * 1e-6

        # Build stim channel from MarkOnSignal events.
        stim = np.zeros((1, eeg_data.shape[1]))
        for sample_idx, gdf_code in mos:
            event_name = _GDF_EVENT_MAP.get(int(gdf_code))
            if event_name is not None and 0 <= sample_idx < stim.shape[1]:
                stim[0, int(sample_idx)] = _EVENTS[event_name]

        # Create MNE Raw.
        all_data = np.concatenate([eeg_data, stim], axis=0)
        ch_types_full = list(ch_types) + ["stim"]
        ch_names_full = list(ch_names) + ["STI"]
        info = mne.create_info(
            ch_names=ch_names_full, ch_types=ch_types_full, sfreq=sfreq
        )
        raw = mne.io.RawArray(data=all_data, info=info, verbose=False)
        montage = mne.channels.make_standard_montage("colin27_1005")
        raw.set_montage(montage, on_missing="warn")

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        self._check_cohort([subject])
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # One download folder for every cohort variant, whatever self.code is.
        sign = _SIGN
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"S{subject:02d}"

        # Check if subject data already exists (session dirs with NPZs).
        if subj_dir.is_dir():
            has_data = any(subj_dir.rglob("*.npz"))
            if has_data and not force_update:
                return str(data_dir)

        # The NEMAR deposit keeps the extracted subject; take it first.
        if not force_update and dl.fill_tree_from_store(data_dir, subj_dir.name):
            return str(data_dir)

        # Download per-subject ZIP from Zenodo and extract.
        url = f"{_ZENODO_BASE}/S{subject:02d}.zip"
        download_and_extract_subject_zip(url, sign, subj_dir, path, force_update, verbose)

        if not any(subj_dir.rglob("*.npz")):
            raise FileNotFoundError(
                f"No .npz files found for subject {subject} in {subj_dir}"
            )
        return str(data_dir)
