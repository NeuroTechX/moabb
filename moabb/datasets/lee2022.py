"""Lee2022 orthopedic-impairment upper-limb motor imagery dataset (ds004022)."""

import warnings
from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets._openneuro_mirror import OpenNeuroMirrorMixin
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
from moabb.datasets.utils import resolve_montage_name, stim_channels_with_selected_ids


_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds004022"
_N_RUNS = 3

# 18 EEG channels (BrainVision actiCAP slim, extended 10% system).
# fmt: off
_CH_NAMES = [
    "C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6",
    "FC1", "FC2", "FC5", "FC6", "Fp2",
    "O1", "O2", "Oz", "P3", "P4", "Pz",
]
# fmt: on

# Each trial has one class cue ``S  3``-``S  6`` (paper's task order), then the
# generic imagery-onset marker ``S  8`` ~7.4 s later (4 s cue + 3 s ready).
_CUE_TO_LABEL = {
    "S  3": "reaching",
    "S  4": "grasping",
    "S  5": "lifting",
    "S  6": "twisting",
}
_IMAGERY_ONSET_MARKER = "S  8"

_EVENTS = {"reaching": 1, "grasping": 2, "lifting": 3, "twisting": 4}


class Lee2022(OpenNeuroMirrorMixin, BaseDataset):
    """Upper-limb motor imagery dataset from Lee et al. 2022 (ds004022).

    Seven patients with orthopedic impairment performed visually cued motor
    imagery of four right-upper-limb movements (**reaching**, **grasping**,
    **lifting**, **twisting**), 3 runs x 40 trials, each trial 3 s fixation,
    4 s cue, 3 s ready and 5 s imagery [1]_. The EEGLAB ``.set`` recordings
    are loaded; the fNIRS modality in the archive is not.

    Notes
    -----
    Trial labels live in the EEGLAB event structure (no ``events.tsv``). Each
    ``S  8`` imagery-onset marker is relabelled with the preceding class cue
    and the 5 s imagery window is epoched from it. EEGLAB channel names are
    stripped of their whitespace padding.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Lee, S., Jung, H. R., Wang, I.-N., Jung, M.-K., Kim, H., &
           Kim, D.-J. (2022). Multimodal EEG and fNIRS Biosignal Acquisition
           during Motor Imagery Tasks in Patients with Orthopedic Impairment.
           OpenNeuro. https://doi.org/10.18112/openneuro.ds004022.v1.0.0
    """

    nemar_id = "on004022"
    nemar_subject_template = "{subject:02d}"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
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
        """Return the local ``.set`` path of each of the subject's three runs."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        sub = f"sub-{subject:02d}"
        stems = [
            f"{sub}/eeg/{sub}_task-motorimagery_run-{run}_eeg"
            for run in range(1, _N_RUNS + 1)
        ]
        if mirror_root is not None:
            return [str(Path(mirror_root) / f"{stem}.set") for stem in stems]
        set_paths = []
        for stem in stems:
            # The .set references the .fdt by name -> both must be co-located.
            for ext in (".fdt", ".set"):
                local = dl.data_dl(
                    f"{_S3_BASE}/{stem}{ext}",
                    self.code,
                    path=path,
                    force_update=force_update,
                    verbose=verbose,
                )
            set_paths.append(local)
        return set_paths

    def _get_single_subject_data(self, subject):
        """Return ``{'0': {'0': raw, '1': raw, '2': raw}}`` for one subject."""
        runs = {}
        for run_idx, set_path in enumerate(self.data_path(subject)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            raw.rename_channels({name: name.strip() for name in raw.ch_names})
            raw.set_annotations(self._imagery_annotations(raw.annotations))
            raw.set_montage(
                resolve_montage_name("colin27_1020"),
                match_case=False,
                on_missing="ignore",
            )
            runs[str(run_idx)] = stim_channels_with_selected_ids(raw, self.event_id)
        return {"0": runs}

    @staticmethod
    def _imagery_annotations(annotations):
        """One annotation per trial at the ``S  8`` onset, labelled by its cue."""
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
