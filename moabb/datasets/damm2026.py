"""Damm2026 finger motor imagery dataset (OpenNeuro ds008446)."""

import warnings
from itertools import product
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.channels import make_standard_montage

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
from moabb.datasets.utils import stim_channels_with_selected_ids


_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds008446"

# Two block-order conditions x two runs = four runs of one session.
_TASKS = ("random", "sequential")
_RUNS = (1, 2)

# events.tsv ``trial_type``: 3-7 finger cues; 1 (white screen) and 2 (fixation)
# are excluded.
_EVENTS = {"thumb": 3, "index": 4, "middle": 5, "ring": 6, "pinky": 7}
_CODE_TO_LABEL = {v: k for k, v in _EVENTS.items()}

# 62 scalp EEG channels (extended 10-20).
# fmt: off
_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF7", "AF5", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO2", "PO8", "O1", "Oz", "O2", "AF9", "AF10",
]
# fmt: on

# Reference electrodes and the flat trigger channel, dropped by the loader.
_NON_EEG = ["Ref1", "Ref2", "Marker"]


class Damm2026(OpenNeuroMirrorMixin, BaseDataset):
    """Random and sequential order finger motor imagery dataset [1]_.

    20 participants imagined moving one finger (thumb, index, middle, ring or
    pinky) per cue, with cues in a ``random`` or a ``sequential`` block order
    (counterbalanced). Each trial is a 3 s white screen (marker 1), a 3 s
    fixation cross (marker 2) and 6 s of imagery (markers 3-7). Two runs per
    condition of 75 trials each give four runs of one session. The loader
    keeps the 62 scalp EEG channels and drops Ref1/Ref2 and the flat
    ``Marker`` trigger channel.

    Notes
    -----
    The per-sample ``events.tsv`` ``onset`` column is scaled by
    ``1 / sampling_rate`` once too often; the true sample is recovered as
    ``round(onset * sfreq**2)`` and events are rebuilt from the collapsed
    marker blocks rather than through ``read_raw_bids``.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Damm, L. M., Jiang, D., & Demosthenous, A. (2026). Random and
           Sequential Order Finger Motor Imagery. OpenNeuro. Dataset.
           DOI: https://doi.org/10.18112/openneuro.ds008446.v1.0.1
    """

    nemar_id = "on008446"
    nemar_subject_template = "{subject:02d}"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            channel_types={"eeg": 62},
            montage="standard_1005",
            hardware="g.tec GmbH",
            reference="linked mastoids",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=20,
            health_status="healthy",
            gender={"female": 13, "male": 4, "non_binary": 3},
            age_min=19.0,
            age_max=57.0,
            handedness={"right": 18, "left": 2},
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=5,
            class_labels=list(_EVENTS.keys()),
            trial_duration=6.0,
            study_design=(
                "Five-class finger motor imagery (thumb/index/middle/ring/"
                "pinky) cued in random and sequential block orders, "
                "counterbalanced across participants."
            ),
            feedback_type="none",
            stimulus_type="visual finger cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds008446.v1.0.1",
            investigators=["Laura Marie Damm", "Dai Jiang", "Andreas Demosthenous"],
            institution="University College London",
            country="GB",
            data_url="https://openneuro.org/datasets/ds008446",
            publication_year=2026,
            funding=[
                "Engineering and Physical Sciences Research Council (EPSRC) grant "
                "EP/S022139/1"
            ],
            ethics_approval=[
                "UCL Humanities, Arts and Sciences Research Ethics Committee; "
                "Approval Number: 26907/001"
            ],
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=6.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=300,
            n_trials_per_class=dict.fromkeys(_EVENTS, 60),
            trials_context=(
                "20 subjects x 300 imagery trials (75 per run x 4 runs, 60 per finger)."
            ),
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_session"]),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=False,
        ),
        file_format="EDF (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Damm2026",
            interval=[0, 6],
            paradigm="imagery",
            doi="10.18112/openneuro.ds008446.v1.0.1",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the subject's four EDF paths (``_events.tsv`` sidecars alongside)."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        sub = f"sub-{subject:02d}"
        stems = [
            f"{sub}/eeg/{sub}_task-{t}_run-{r:02d}" for t, r in product(_TASKS, _RUNS)
        ]
        if mirror_root is not None:
            return [str(Path(mirror_root) / f"{stem}_eeg.edf") for stem in stems]
        kwargs = {"path": path, "force_update": force_update, "verbose": verbose}
        edf_paths = []
        for stem in stems:
            edf_paths.append(
                dl.data_dl(f"{_S3_BASE}/{stem}_eeg.edf", self.code, **kwargs)
            )
            dl.data_dl(f"{_S3_BASE}/{stem}_events.tsv", self.code, **kwargs)
        return edf_paths

    def _read_events(self, events_path, sfreq, n_times):
        """Rebuild finger-cue onsets from the per-sample ``events.tsv``.

        The true sample is ``round(onset * sfreq**2)``; consecutive identical
        markers are collapsed and only codes 3-7 are kept.
        """
        df = pd.read_csv(events_path, sep="\t")
        if df.empty:
            return np.empty((0, 3), dtype=int)
        codes = df["trial_type"].to_numpy()
        samples = np.rint(df["onset"].to_numpy() * sfreq * sfreq).astype(int)

        block_starts = np.r_[0, np.where(np.diff(codes) != 0)[0] + 1]
        events = []
        for start in block_starts:
            code = int(codes[start])
            if code not in _CODE_TO_LABEL:
                continue
            sample = int(samples[start])
            if 0 <= sample < n_times:
                events.append([sample, 0, code])
        return np.asarray(events, dtype=int).reshape(-1, 3)

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {run: raw}}`` with one run per task/run combination."""
        montage = make_standard_montage("colin27_1005")
        runs = {}
        edf_paths = self.data_path(subject)
        for run_index, (task, run) in enumerate(product(_TASKS, _RUNS)):
            edf_path = edf_paths[run_index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            events = self._read_events(
                edf_path.replace("_eeg.edf", "_events.tsv"),
                raw.info["sfreq"],
                raw.n_times,
            )
            raw.drop_channels(
                [
                    ch
                    for ch, kind in zip(raw.ch_names, raw.get_channel_types())
                    if kind == "stim"
                ]
            )
            raw.drop_channels(_NON_EEG, on_missing="ignore")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage(montage, on_missing="ignore", verbose=False)
            annotations = mne.annotations_from_events(
                events=events,
                sfreq=raw.info["sfreq"],
                event_desc=_CODE_TO_LABEL,
                orig_time=raw.info["meas_date"],
            )
            raw.set_annotations(annotations)
            runs[f"{run_index}{task}{run}"] = stim_channels_with_selected_ids(
                raw, self.event_id
            )
        return {"0": runs}
