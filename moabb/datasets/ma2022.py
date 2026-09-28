"""Cross-session motor imagery dataset (SHU) from Ma et al. 2022.

Ma et al. (2022), Scientific Data.
DOI: 10.1038/s41597-022-01647-1
Data DOI: 10.6084/m9.figshare.19228725
"""

import pandas as pd
from mne.channels import get_builtin_montages, make_standard_montage
from mne_bids import events_file_to_annotation_kwargs

from .base import BaseBIDSDataset
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


# Sampling rate (Hz), from the BIDS ``task-motorimagery_eeg.json`` sidecar.
_SFREQ = 250.0

# 32 channel names in acquisition order, from the BIDS
# ``task-motorimagery_channels.tsv`` sidecar. The dataset uses the older
# 10-20 nomenclature (e.g. T3/T4/T5/T6 instead of T7/T8/P7/P8) and records
# the two mastoid channels A1/A2 among the 32 EEG channels. Name case is
# normalized here (the sidecar lists them uppercase).
# fmt: off
MA2022_CH_NAMES = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T3", "T4", "A1", "A2",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "T5", "T6",
    "PO3", "PO4", "Oz", "O1", "O2",
]
# fmt: on

# Trial labels in the BIDS events sidecars (1 = left hand, 2 = right hand),
# consistent with the ``task-motorimagery_events.json`` sidecar.
MA2022_EVENTS = {"left_hand": 1, "right_hand": 2}

# Per-subject demographics from the BIDS ``participants.tsv`` sidecar
# (sex, age), indexed by 1-based subject number.
_DEMOGRAPHICS = {
    1: ("male", 24),
    2: ("male", 24),
    3: ("female", 23),
    4: ("female", 23),
    5: ("female", 22),
    6: ("female", 22),
    7: ("female", 24),
    8: ("male", 23),
    9: ("female", 23),
    10: ("female", 22),
    11: ("female", 22),
    12: ("male", 24),
    13: ("male", 23),
    14: ("female", 23),
    15: ("male", 22),
    16: ("female", 21),
    17: ("male", 21),
    18: ("male", 21),
    19: ("male", 23),
    20: ("female", 23),
    21: ("male", 20),
    22: ("male", 21),
    23: ("male", 23),
    24: ("female", 22),
    25: ("male", 24),
}


class Ma2022(BaseBIDSDataset):
    """Cross-session motor imagery dataset (SHU) from Ma et al. 2022.

    Dataset from [1]_.

    The SHU dataset contains EEG recordings from 25 healthy, BCI-naive
    subjects (13 males, 12 females, aged 20-24 years) performing cued
    left- vs right-hand grasping motor imagery. Each subject completed five
    independent sessions recorded on five different days, 2 to 3 days
    apart, which makes the dataset specifically suited for studying
    cross-session variability in motor imagery BCIs.

    Each session was designed with 100 trials (50 left-hand, 50 right-hand,
    randomized order). The released files retain 74 to 100 trials per
    session after the source-side bad-segment rejection described in the
    data paper, for 11,988 trials in total. Signals were recorded from 32
    EEG channels (10-20 system, unipolar reference on M1, ground on AFz)
    at 250 Hz. Every trial lasted 8 s (0-2 s rest, 2-4 s cue, 4-8 s motor
    imagery) but only the 4 s motor imagery window is stored (1000 samples
    per trial), so the analysis interval spans the full stored window.

    .. important::

       The released data is **preprocessed**: bad segments were removed,
       the baseline was corrected, and a 0.5-40 Hz FIR band-pass filter was
       applied by the authors before disclosure.

    This loader reads the authors' EDF release, hosted in BIDS form on
    NEMAR (``nm000288``). Each EDF contains the retained 4 s windows
    concatenated in time, **not continuous amplifier recordings**. BIDS
    ``DatasetType: raw`` describes the deposit layout, not its processing
    state. Events and bad-channel flags are read from the BIDS sidecars;
    MNE converts the EDF microvolt calibration to SI volts.

    Nine sessions contain a channel zeroed by the authors. The deposit
    repairs only that channel's otherwise unreadable physical-range header
    and marks it bad; the untouched EDF is preserved under ``sourcedata/``.
    Bad channels remain flagged, without interpolation or deletion. Since
    MOABB's default EEG selection excludes bads, full-dataset analyses must
    use a common good-channel set (exclude F3, T6 and A2). Explicitly picking
    a bad channel includes the authors' zeroed signal. Channel positions
    are template estimates from ``standard_1020`` (called ``colin27_1020``
    in newer MNE versions), not measured locations.

    .. note::

       NEMAR is the only download source for this EDF/BIDS loader, including
       when the provider is set to ``upstream``. Failures propagate: there is
       no fallback to the scientifically different MATLAB representation.
       ``data_path`` returns the five EDF paths; sessions are numbered
       ``"0"`` to ``"4"`` in MOABB. The code ``Ma-edf2022`` isolates downloads,
       caches and evaluation results from the former MATLAB loader.

       This dataset is from the same laboratory as :class:`Yang2025`
       (WBCIC-SHU, a distinct 2025 multi-day recording) and is unrelated
       to :class:`Ma2020` (different team, different recording).

    References
    ----------
    .. [1] J. Ma, B. Yang, W. Qiu, Y. Li, S. Gao, and X. Xia, "A large EEG
       dataset for studying cross-session variability in motor imagery
       brain-computer interface," Scientific Data, vol. 9, p. 531, 2022.
       DOI: 10.1038/s41597-022-01647-1

    Notes
    -----
    .. versionadded:: 1.8.0
    """

    nemar_id = "nm000288"
    nemar_subject_template = "{subject:03d}"
    nemar_bids_filters = {"task": "motorimagery", "suffix": "eeg"}
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": 32},
            montage="standard_1020",
            hardware="Wuhan Greentech 32-channel Ag/AgCl cap, Brickcom wireless amplifier",
            sensor_type="Ag/AgCl",
            reference="M1 (unipolar)",
            ground="AFz",
            impedance_threshold_kohm=20,
            sensors=list(MA2022_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"male": 13, "female": 12},
            age_mean=22.52,
            age_min=20,
            age_max=24,
            bci_experience="naive",
            ages=[_DEMOGRAPHICS[i][1] for i in range(1, 26)],
            sexes=[_DEMOGRAPHICS[i][0] for i in range(1, 26)],
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            events=dict(MA2022_EVENTS),
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            stimulus_type="visual cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            task_type="motor_imagery_grasping",
            feedback_type="none",
            has_training_test_split=False,
            instructions=(
                "Subjects were asked to repeatedly imagine left-hand or "
                "right-hand grasping (kinesthetic motor imagery) following "
                "the arrow cue presented on the monitor."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-022-01647-1",
            investigators=[
                "Jun Ma",
                "Banghua Yang",
                "Wenzheng Qiu",
                "Yunzhe Li",
                "Shouwei Gao",
                "Xinxing Xia",
            ],
            senior_author="Banghua Yang",
            institution="Shanghai University",
            institution_department=(
                "School of Mechatronic Engineering and Automation, "
                "Research Center of Brain-Computer Engineering"
            ),
            country="CN",
            repository="NEMAR",
            data_url="https://nemar.org/dataexplorer/detail?dataset_id=nm000288",
            license="CC-BY-4.0",
            publication_year=2022,
            ethics_approval=[
                "Shanghai Second Rehabilitation Hospital Ethics Committee "
                "(approval number: ECSHSRH 2018-0101)"
            ],
            funding=[
                "National Natural Science Foundation of China (No. 61976133)",
                "Shanghai Science and Technology Major Project (No. 2021SHZDZX)",
            ],
            keywords=["motor imagery", "EEG", "BCI", "cross-session", "cross-subject"],
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        data_processed=True,
        file_format="EDF",
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bad segment rejection (EEGLAB amplitude > 100 uV, manually confirmed)",
                "baseline removal",
                "0.5-40 Hz FIR band-pass filter",
                "epoching to the 4 s motor imagery window",
            ],
            highpass_hz=0.5,
            lowpass_hz=40.0,
            filter_type="FIR",
            artifact_methods=["amplitude threshold", "visual inspection"],
            epoch_window=[0.0, 4.0],
            notes=(
                "Preprocessing was applied by the data authors before "
                "disclosure; the released trials are the 4 s motor imagery "
                "segments only."
            ),
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=11988,
            trials_context=(
                "25 subjects x 5 sessions x up to 100 trials "
                "(74-100 retained per session after bad-segment rejection) "
                "= 11988 trials in total; the summary table reports the mean "
                "239.76 retained trials per class per subject, not a balanced count"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "FBCSP+SVM", "EEGNet", "DeepConvNet", "FBCNet"],
            feature_extraction=["CSP", "FBCSP"],
            frequency_bands={"mu_beta": [8.0, 30.0], "csp": [3.0, 35.0]},
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_session", "cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            environment="laboratory",
            online_feedback=False,
            applications=["motor_rehabilitation"],
        ),
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        # Each stored trial is exactly the 4 s motor-imagery window (1000
        # samples at 250 Hz, t = 0 to 3.996 s). Because mne.Epochs includes
        # the tmax sample, an interval of [0, 4] would request 1001 samples
        # per trial and thus borrow one sample from the next concatenated
        # trial (and drop the final trial of each session). Using
        # tmax = 4 - 1/sfreq selects exactly the 1000 stored samples.
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=5,
            events=dict(MA2022_EVENTS),
            code="Ma-edf2022",
            interval=[0.0, 4.0 - 1.0 / _SFREQ],
            paradigm="imagery",
            doi="10.1038/s41597-022-01647-1",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _prefetch_nemar_sourcedata(self, subjects, verbose=None):
        # Unlike mirrored upstream loaders, we consume the BIDS EDFs directly.
        # BaseDataset's prefetch would download unused original EDFs instead.
        pass

    def download(
        self,
        subject_list=None,
        path=None,
        force_update=False,
        update_path=None,
        accept=False,
        verbose=None,
    ):
        """Download BIDS EDFs, not the original sourcedata distribution."""
        for subject in self.subject_list if subject_list is None else subject_list:
            self.data_path(subject, path, force_update, update_path, verbose)

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        return self._download_nemar(subject, path, force_update, update_path, verbose)

    def _get_path_search_params(self, subject):
        return {
            "subjects": self._nemar_subject(subject),
            "tasks": "motorimagery",
            "suffixes": "eeg",
            "datatypes": "eeg",
            "extensions": ".edf",
        }

    def bids_paths(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        paths = super().bids_paths(subject, path, force_update, update_path, verbose)
        if len(paths) != 5 or {p.session for p in paths} != {
            f"{i:02d}" for i in range(1, 6)
        }:
            raise FileNotFoundError(
                f"Expected five EDF sessions (01-05) for Ma2022 subject {subject}"
            )
        return sorted(paths, key=lambda p: p.session)

    def _get_single_subject_data(self, subject):
        sessions = super()._get_single_subject_data(subject)
        montage_name = (
            "colin27_1020"
            if "colin27_1020" in get_builtin_montages()
            else "standard_1020"
        )
        montage = make_standard_montage(montage_name)
        for runs in sessions.values():
            for raw in runs.values():
                # Covers the legacy temporal and mastoid names too. Never clear
                # BIDS bad-channel flags or rescale data already read in volts.
                raw.set_montage(montage)
        return {str(int(session) - 1): runs for session, runs in sessions.items()}

    def get_additional_metadata(self, subject, session, run):
        # Unlike BaseBIDSDataset's generic implementation, match zero-based
        # MOABB sessions to BIDS labels and handle the absent run entity.
        bids_path = next(
            p for p in self.bids_paths(subject) if int(p.session) == int(session) + 1
        )
        events_file = bids_path.find_matching_sidecar(suffix="events", extension=".tsv")
        annotations = events_file_to_annotation_kwargs(events_file)
        return pd.DataFrame(
            {
                "onset": annotations["onset"],
                "duration": annotations["duration"],
                "trial_type": annotations["description"],
            }
        ).assign(subject=subject, session=session, run=run)
