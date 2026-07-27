"""Damm2026 finger motor imagery dataset (OpenNeuro ds008446)."""

import warnings

import mne
import numpy as np
import pandas as pd
from mne.channels import make_standard_montage

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


# OpenNeuro S3 mirror (public, no authentication required).
_OPENNEURO_ID = "ds008446"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# The two block-order conditions and their two runs each.
_TASKS = ("random", "sequential")
_RUNS = (1, 2)

# Marker legend (events.tsv ``trial_type`` column): the 5 finger MI cues.
# 1 = WhiteScreen and 2 = FocusCross are baseline/fixation and are excluded.
_EVENTS = {"thumb": 3, "index": 4, "middle": 5, "ring": 6, "pinky": 7}
_CODE_TO_LABEL = {v: k for k, v in _EVENTS.items()}

# 62 scalp EEG channels (extended 10-20). The recording additionally carries
# two reference electrodes (Ref1/Ref2) and one flat trigger channel (Marker),
# all dropped by the loader.
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

# Reference/trigger channels present in the EDF but not scalp EEG.
_NON_EEG = ["Ref1", "Ref2", "Marker"]


class Damm2026(BaseDataset):
    """Random and sequential order finger motor imagery dataset [1]_.

    **Dataset description**

    EEG recordings from 20 participants performing a five-class finger motor
    imagery paradigm. On each trial the participant is cued to imagine moving
    one of the five fingers of a single hand (thumb, index, middle, ring or
    pinky). Cues are presented in two block-order conditions:

    - ``random``: finger cues appear in a randomised order,
    - ``sequential``: finger cues appear in a fixed sequence.

    The starting condition was counterbalanced across participants. Each
    condition contributes two runs, giving four runs per subject that are
    exposed as four runs of a single session.

    Trial structure (per cue):

    - 3 s white screen (marker 1),
    - 3 s fixation cross (marker 2),
    - 6 s finger motor imagery (markers 3-7, one per finger).

    Each run holds 75 imagery trials (15 per finger), so a subject provides
    300 imagery trials in total (60 per finger).

    Recording used a 65-channel g.tec system at 512 Hz: 62 scalp EEG channels
    (extended 10-20), 2 reference electrodes (Ref1/Ref2, linked mastoids) and
    1 trigger channel (Marker). The loader keeps the 62 EEG channels and drops
    the reference and (flat) trigger channels.

    Notes
    -----
    The BIDS EDF files ship a per-sample ``events.tsv`` (one row per sample)
    whose ``onset`` column is scaled by ``1 / sampling_rate`` too many times;
    the true sample index is recovered as ``round(onset * sfreq**2)``. Events
    are therefore rebuilt from the collapsed marker blocks rather than through
    ``read_raw_bids``. The EDF ``Marker`` channel itself is flat and carries no
    triggers.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Damm, L. M., Jiang, D., & Demosthenous, A. (2026). Random and
           Sequential Order Finger Motor Imagery. OpenNeuro. Dataset.
           DOI: https://doi.org/10.18112/openneuro.ds008446.v1.0.1
    """

    nemar_id = "ds008446"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=62,
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
            gender={"female": 13, "male": 4, "unknown": 3},
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
            funding=["Engineering and Physical Sciences Research Council (EPSRC)"],
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
        """Download and return the EDF paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            The local paths to the subject's four EDF files. The matching
            ``_events.tsv`` sidecars are downloaded alongside them.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sub = f"sub-{subject:02d}"
        edf_paths = []
        for task in _TASKS:
            for run in _RUNS:
                base = f"{sub}/eeg/{sub}_task-{task}_run-{run:02d}"
                edf_url = f"{_S3_BASE}/{base}_eeg.edf"
                events_url = f"{_S3_BASE}/{base}_events.tsv"
                edf_path = dl.data_dl(edf_url, self.code, force_update=force_update)
                dl.data_dl(events_url, self.code, force_update=force_update)
                edf_paths.append(edf_path)
        return edf_paths

    def _read_events(self, events_path, sfreq, n_times):
        """Rebuild trial-onset events from the per-sample ``events.tsv``.

        The ``onset`` column is scaled by ``1 / sfreq`` twice, so the true
        sample index is ``round(onset * sfreq**2)``. Consecutive identical
        markers are collapsed and only the five finger cues (codes 3-7) are
        kept.
        """
        df = pd.read_csv(events_path, sep="\t")
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
        return np.asarray(events, dtype=int)

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"0": {run: mne.io.Raw}}`` with one run per task/run combination.
        """
        edf_paths = self.data_path(subject)
        montage = make_standard_montage("standard_1005")

        runs = {}
        run_index = 0
        for task in _TASKS:
            for run in _RUNS:
                edf_path = edf_paths[run_index]
                events_path = edf_path.replace("_eeg.edf", "_events.tsv")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

                events = self._read_events(events_path, raw.info["sfreq"], raw.n_times)

                drop = [ch for ch in _NON_EEG if ch in raw.ch_names]
                raw = raw.drop_channels(drop)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = raw.set_montage(montage, on_missing="ignore", verbose=False)

                annotations = mne.annotations_from_events(
                    events=events,
                    sfreq=raw.info["sfreq"],
                    event_desc=_CODE_TO_LABEL,
                    orig_time=raw.info["meas_date"],
                )
                raw = raw.set_annotations(annotations)

                run_key = f"{run_index}{task}{run}"
                runs[run_key] = stim_channels_with_selected_ids(raw, self.event_id)
                run_index += 1

        return {"0": runs}
