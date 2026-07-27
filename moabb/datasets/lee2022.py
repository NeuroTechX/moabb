"""Lee2022 orthopedic-impairment upper-limb motor imagery dataset (ds004022)."""

import warnings

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids


# OpenNeuro dataset ID and S3 base URL (CC0, no authentication required).
_OPENNEURO_ID = "ds004022"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# Number of runs per subject (BIDS ``run-1``/``run-2``/``run-3``).
_N_RUNS = 3

# 18 EEG channels (BrainVision actiCAP slim, extended 10% system).
# fmt: off
_CH_NAMES = [
    "C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6",
    "FC1", "FC2", "FC5", "FC6", "Fp2",
    "O1", "O2", "Oz", "P3", "P4", "Pz",
]
# fmt: on

# The four MI classes. In each trial the class is signalled by one of the
# BrainVision cue markers ``S  3``-``S  6`` (10 per run each); the generic
# imagery-onset marker ``S  8`` follows ~7.4 s later (4 s visual cue + 3 s
# ready). We relabel that imagery onset with the trial class and epoch the 5 s
# imagery window. The cue-code -> task mapping follows the paper's canonical
# order (reaching, grasping, lifting, twisting).
_CUE_TO_LABEL = {
    "S  3": "reaching",
    "S  4": "grasping",
    "S  5": "lifting",
    "S  6": "twisting",
}
_IMAGERY_ONSET_MARKER = "S  8"

_EVENTS = {"reaching": 1, "grasping": 2, "lifting": 3, "twisting": 4}


class Lee2022(BaseDataset):
    """Upper-limb motor imagery dataset from Lee et al. 2022 (ds004022).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class     Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Lee2022          7       18           4                 30             5s            500Hz            1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    Multimodal EEG (and fNIRS) acquired during motor imagery of four
    right-upper-limb movements in patients with orthopedic impairment [1]_.

    Seven participants performed visually cued motor imagery of four
    right-upper-limb movements -- **reaching**, **grasping**, **lifting** and
    **twisting** -- across three runs of 40 trials each (four tasks in random
    order, 10 trials per task per run). Each trial started with a 3 s fixation
    cross, then a 4 s visual cue, then a 3 s ready state (gray screen), followed
    by 5 s of imagined movement.

    EEG was recorded with a BrainVision actiCHamp amplifier and an actiCAP slim
    cap at 500 Hz over 18 channels (reference FCz, ground Fpz). The imagery
    period is epoched from the imagery-onset marker, relabelled by trial class.

    The dataset is hosted on OpenNeuro (ds004022) in BIDS/EEGLAB ``.set``
    format. An fNIRS modality is present in the archive but is not loaded here.

    Notes
    -----
    Trial labels are stored in the EEGLAB event structure (no ``events.tsv``).
    The cue-code to movement mapping follows the paper's canonical task order
    (reaching, grasping, lifting, twisting).

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Lee, S., Jung, H. R., Wang, I.-N., Jung, M.-K., Kim, H., &
           Kim, D.-J. (2022). Multimodal EEG and fNIRS Biosignal Acquisition
           during Motor Imagery Tasks in Patients with Orthopedic Impairment.
           OpenNeuro. https://doi.org/10.18112/openneuro.ds004022.v1.0.0
    """

    nemar_id = "ds004022"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=18,
            channel_types={"eeg": 18},
            montage="standard_1020",
            hardware="BrainVision actiCHamp",
            cap_manufacturer="BrainVision",
            cap_model="actiCAP slim",
            reference="FCz",
            ground="Fpz",
            sensors=list(_CH_NAMES),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=7,
            health_status="orthopedic impairment",
            clinical_population="patients with orthopedic impairment",
            gender={"male": 3, "female": 4},
            age_min=48.0,
            age_max=83.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=list(_EVENTS.keys()),
            trial_duration=15.0,
            study_design=(
                "Four right-upper-limb MI tasks (reaching, grasping, lifting, "
                "twisting) in random order, 40 trials per run x 3 runs. Each "
                "trial: 3 s fixation, 4 s visual cue, 3 s ready, 5 s imagery."
            ),
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds004022.v1.0.0",
            investigators=[
                "Seho Lee",
                "Hee Ra Jung",
                "In-Nea Wang",
                "Min-Kyung Jung",
                "Hakseung Kim",
                "Dong-Joo Kim",
            ],
            institution="Korea University",
            country="KR",
            data_url="https://openneuro.org/datasets/ds004022",
            publication_year=2022,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=_N_RUNS,
        tags=Tags(
            pathology=["Orthopedic Impairment"],
            modality=["Motor"],
            type=["Motor Imagery"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=120,
            trials_context=(
                "7 subjects x 3 runs x 40 trials (10 per class per run); "
                "30 trials per class per subject."
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="within_subject", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        file_format="SET (EEGLAB, BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 8)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Lee2022",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.18112/openneuro.ds004022.v1.0.0",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the ``.set``/``.fdt`` EEG files for a subject's three runs.

        Returns a list of the local ``.set`` file paths (one per run).
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        subj = f"sub-{subject:02d}"
        set_paths = []
        for run in range(1, _N_RUNS + 1):
            base = f"{subj}/eeg/{subj}_task-motorimagery_run-{run}_eeg"
            # The .set references the .fdt by name -> both must be co-located.
            dl.data_dl(f"{_S3_BASE}/{base}.fdt", self.code, force_update=force_update)
            set_path = dl.data_dl(
                f"{_S3_BASE}/{base}.set", self.code, force_update=force_update
            )
            set_paths.append(set_path)
        return set_paths

    def _get_single_subject_data(self, subject):
        """Return ``{'0': {'0': raw, '1': raw, '2': raw}}`` for one subject."""
        set_paths = self.data_path(subject)

        runs = {}
        for run_idx, set_path in enumerate(set_paths):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

            # EEGLAB channel names carry trailing whitespace padding.
            raw.rename_channels({name: name.strip() for name in raw.ch_names})

            raw.set_annotations(self._imagery_annotations(raw.annotations))
            raw.set_montage("standard_1020", match_case=False, on_missing="ignore")

            runs[str(run_idx)] = stim_channels_with_selected_ids(raw, self.event_id)

        return {"0": runs}

    @staticmethod
    def _imagery_annotations(annotations):
        """Relabel imagery-onset markers with the preceding trial's class.

        Each trial carries a class cue (``S  3``-``S  6``) followed by the
        generic imagery-onset marker (``S  8``). We emit one annotation per
        trial at the imagery onset, described by the class label.
        """
        order = np.argsort(annotations.onset)
        current = None
        onsets, labels = [], []
        for i in order:
            desc = annotations.description[i].strip()
            if desc in _CUE_TO_LABEL:
                current = _CUE_TO_LABEL[desc]
            elif desc == _IMAGERY_ONSET_MARKER and current is not None:
                onsets.append(annotations.onset[i])
                labels.append(current)
                current = None
        return mne.Annotations(
            onset=onsets,
            duration=[0.0] * len(onsets),
            description=labels,
            orig_time=annotations.orig_time,
        )
