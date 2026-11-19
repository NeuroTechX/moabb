"""Nencki-Symfonia 3-stimulus visual oddball (OpenNeuro ds004621)."""

from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets._openneuro_mirror import OpenNeuroMirrorMixin
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids


_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds004621"

_EVENTS = {"Target": 2, "NonTarget": 1}
# BrainVision markers (task-oddball_events.json and Table 3 of the paper):
# S 6 target, S 5 standard, S 7 distractor. S 1 (key press), S 10 (break)
# and S 12 (run start) are not trials.
_MARKERS = {
    "Stimulus/S  6": "Target",
    "Stimulus/S  5": "NonTarget",
    "Stimulus/S  7": "NonTarget",
}
# EEGLAB writes these where it joined recordings (sub-37 and sub-38).
_JOIN_MARKERS = ("New Segment/", "Stimulus/boundary")


class Dzianok2022(OpenNeuroMirrorMixin, BaseDataset):
    """Nencki-Symfonia 3-stimulus visual oddball dataset [1]_.

    42 healthy right-handed adults (22 females, aged 20-34) did a visual
    oddball task with three letters: a rare target (``þ``, about 12%), a rare
    distractor (``Þ``, about 12%) and a frequent standard (``p``, about 76%).
    Each stimulus was shown for 200 ms with a 1200-1600 ms ISI. The
    participant pressed a key for the target only. A session has 660 stimuli
    in about 22 min, with breaks, stored as one BrainVision file.

    EEG was recorded at 1000 Hz from 127 actiCAP active electrodes with an
    actiCHamp amplifier, referenced to FCz with ground at FPz. The data has no
    high-pass filter. The channels ``TP9 LEFT EAR`` and ``TP10 RIGHT EAR`` are
    on the earlobes, so they have no montage position, and nor do ``O9`` and
    ``O10``, which are not in the MNE template.

    MOABB labels the target as ``Target`` and both the standard and the
    distractor as ``NonTarget``. The same deposit also holds MSIT+, simple
    reaction-time and resting-state recordings, which this class does not
    download.

    Notes
    -----
    The paper excludes sub-02, sub-14 and sub-17 from its behavioural summary
    because they also pressed the key for most distractors. Their EEG is kept.
    sub-37 and sub-38 are joined from several recordings; the joins are
    marked with ``EDGE boundary`` annotations.

    References
    ----------
    .. [1] Dzianok, P., Antonova, I., Wojciechowski, J., Dreszer, J., &
           Kublik, E. (2022). The Nencki-Symfonia electroencephalography/
           event-related potential dataset: Multiple cognitive tasks and
           resting-state data collected in a sample of healthy adults.
           GigaScience, 11, giac015.
           https://doi.org/10.1093/gigascience/giac015
    """

    nemar_id = "on004621"
    nemar_subject_template = "{subject:02d}"
    # The deposit also has MSIT+, SRT and rest recordings (about 2 GB each).
    nemar_bids_filters = {"scope": "raw", "datatype": "eeg", "task": "oddball"}
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 127},
            montage="standard_1005",
            hardware="Brain Products actiCHamp",
            sensor_type="active",
            cap_manufacturer="Brain Products",
            cap_model="actiCAP 128",
            reference="FCz",
            ground="FPz",
            filters={"lowpass": 280},
            line_freq=50.0,
            software="BrainVision Recorder",
        ),
        participants=ParticipantMetadata(
            n_subjects=42,
            health_status="healthy",
            species="homo sapiens",
            gender={"female": 22, "male": 20},
            handedness="right-handed",
            age_mean=24.62,
            age_std=4.12,
            age_min=20,
            age_max=34,
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="p300",
            n_classes=2,
            class_labels=list(_EVENTS),
            trial_duration=1.0,
            study_design=(
                "3-stimulus visual oddball: rare target (12%), rare distractor "
                "(12%), frequent standard (76%); 660 stimuli, 200 ms each, "
                "1200-1600 ms ISI; key press to targets only."
            ),
            feedback_type="none",
            stimulus_type="letters (target þ, distractor Þ, standard p)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            senior_author="Ewa Kublik",
            investigators=[
                "Patrycja Dzianok",
                "Ingrida Antonova",
                "Jakub Wojciechowski",
                "Joanna Dreszer",
                "Ewa Kublik",
            ],
            institution="Nencki Institute of Experimental Biology, Polish Academy of Sciences",
            country="PL",
            repository="OpenNeuro",
            doi="10.1093/gigascience/giac015",
            associated_paper_doi="10.18112/openneuro.ds004621.v1.0.4",
            data_url="https://openneuro.org/datasets/ds004621",
            funding=["Polish National Science Center (NCN) grant 2016/20/W/NZ4/00354"],
            ethics_approval=[
                "Research Ethics Committee, Faculty of Humanities, Nicolaus "
                "Copernicus University in Torun (No. 6/2018)"
            ],
            publication_year=2022,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["ERP"], type=["P300"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", isi_ms=1400.0
        ),
        data_structure=DataStructureMetadata(
            n_trials=660,
            n_trials_per_class={"Target": 81, "NonTarget": 579},
            trials_context=(
                "660 stimuli per subject: 81 targets, 81 distractors and "
                "498 standards (distractors and standards are NonTarget)."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["target_detection"],
            environment="laboratory",
            online_feedback=False,
        ),
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 43)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Dzianok2022",
            interval=[0, 1],
            paradigm="p300",
            doi="10.1093/gigascience/giac015",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the subject's oddball ``.vhdr`` path (``.vmrk``/``.eeg`` alongside)."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        stem = f"sub-{subject:02d}/eeg/sub-{subject:02d}_task-oddball_eeg"
        if mirror_root is not None:
            return [str(Path(mirror_root) / f"{stem}.vhdr")]
        kwargs = {"path": path, "force_update": force_update, "verbose": verbose}
        paths = [
            dl.data_dl(f"{_S3_BASE}/{stem}{ext}", self.code, **kwargs)
            for ext in (".vhdr", ".vmrk", ".eeg")
        ]
        return paths[:1]

    def _get_single_subject_data(self, subject):
        raw = mne.io.read_raw_brainvision(
            self.data_path(subject)[0], preload=True, verbose=False
        )
        annotations = raw.annotations
        joins = np.unique(
            annotations.onset[
                np.isin(annotations.description, _JOIN_MARKERS) & (annotations.onset > 0)
            ]
        )
        trials = annotations[np.isin(annotations.description, list(_MARKERS))]
        trials.rename(_MARKERS)
        raw.set_annotations(trials)
        raw = stim_channels_with_selected_ids(raw, self.event_id)
        if joins.size:
            raw.set_annotations(
                raw.annotations
                + mne.Annotations(
                    joins, 0.0, "EDGE boundary", orig_time=raw.annotations.orig_time
                )
            )
        raw.set_montage("colin27_1005", on_missing="ignore", verbose=False)
        return {"0": {"0": raw}}
