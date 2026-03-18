"""Lower limb motor imagery EEG dataset for stroke patients.

Liu, Gui, Yan, Wang, Gao, Han, Chen, Wu, and Ming (2025), Scientific Data.
DOI: 10.1038/s41597-025-04618-4
Data DOI: 10.6084/m9.figshare.27130299
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
from .utils import download_and_extract_subject_zip, stim_channels_with_selected_ids


log = logging.getLogger(__name__)

_ZENODO_RECORD = "18987384"
_DOI = "10.1038/s41597-025-04618-4"

_EVENTS = {
    "gait_imagery": 3,
    "rest": 9,
}

# Session labels in the BIDS output.
# Paradigm group (sub-01..15, sub-25): all 5 sessions.
# Longitudinal group (sub-16..24, sub-26, sub-27): only pre/post/follow.
_ALL_SESSIONS = ["pre", "mises", "miies", "post", "follow"]
_LONGITUDINAL_SESSIONS = ["pre", "post", "follow"]

# Subjects in the paradigm group (5 sessions).
_PARADIGM_SUBJECTS = set(range(1, 16)) | {25}

# Per-subject demographics from participants.tsv.
# fmt: off
_AGES = [
    40, 40, 48, 41, 59, 55, 33, 59, 54, 66,
    52, 38, 48, 65, 52, 48, 63, 54, 68, 51,
    39, 53, 64, 50, 55, 55, 41,
]
_SEXES = [
    "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
    "M", "M", "F", "M", "F", "F", "M", "M", "M", "M",
    "M", "M", "F", "M", "M", "M", "M",
]
# fmt: on


class Liu2025(BaseDataset):
    """Lower limb motor imagery dataset from Liu et al 2025.

    Dataset from *Lower limb motor imagery EEG dataset based on the
    multi-paradigm and longitudinal-training of stroke patients* [1]_.

    It contains EEG from 27 stroke patients recorded with a 64-channel
    NeuSen W (Neuracle) system at 1000 Hz. The task is binary: gait
    motor imagery vs idle state.

    Five experiment types (mapped to BIDS sessions):

    - **ses-pre**: Pre-treatment conventional MI (all 27 subjects)
    - **ses-mises**: MI with sequential electrical stimulation (16 subj)
    - **ses-miies**: MI with invariable electrical stimulation (16 subj)
    - **ses-post**: Post-treatment assessment (all 27 subjects)
    - **ses-follow**: Follow-up assessment (all 27 subjects)

    Each session has ~4 runs of 10 MI + 10 idle trials (20 per run).
    Trial structure: 3 s baseline, 1 s cue, 5 s MI/idle, rest.

    Event codes: 3 = gait MI onset (``gait_imagery``),
    9 = idle onset (``rest``). Only these two events are used for
    classification; codes 1-2 and 7-8 are baseline/instruction markers.

    Parameters
    ----------
    sessions : str or list of str, optional
        Which sessions to load. One or more of ``"pre"``, ``"mises"``,
        ``"miies"``, ``"post"``, ``"follow"``. Default: ``"pre"`` only.

    References
    ----------
    .. [1] Liu, Y., Gui, Z., Yan, D., Wang, Z., Gao, R., Han, N.,
           Chen, J., Wu, J., Ming, D. (2025). Lower limb motor imagery
           EEG dataset based on the multi-paradigm and longitudinal-training
           of stroke patients. Scientific Data, 12, 314.
           https://doi.org/10.1038/s41597-025-04618-4
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 60, "ecg": 1, "eog": 4},
            montage="standard_1020",
            hardware="NeuSen W (Neuracle, Inc.)",
            sensor_type="Ag/AgCl",
            filters={"notch": 50.0},
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=27,
            health_status="stroke (recovery phase)",
            gender={"male": 23, "female": 4},
            age_min=33.0,
            age_max=68.0,
            ages=_AGES,
            sexes=_SEXES,
            handedness="mixed",
            species="human",
            clinical_population="stroke",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=5.0,
            study_design=(
                "Binary gait MI vs idle. 27 stroke patients, "
                "5 experiment types (pre/MI-SES/MI-IES/post/follow). "
                "~4 runs x 20 trials per session."
            ),
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Yuan Liu",
                "Zhuolan Gui",
                "De Yan",
                "Zhuang Wang",
                "Ruisi Gao",
                "Ningxin Han",
                "Junying Chen",
                "Jialing Wu",
                "Dong Ming",
            ],
            institution="Tianjin University",
            country="CN",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
            publication_year=2025,
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=3,
        runs_per_session=4,
        tags=Tags(
            pathology=["Stroke"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=8640,
            trials_context=(
                "27 subjects x 3 default sessions (pre/post/follow) x "
                "~4 runs x ~20 trials = ~6480. Paradigm group subjects "
                "have 2 extra sessions (mises/miies)."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM"],
            feature_extraction=["CSP", "ERSP"],
            frequency_bands={
                "MI_features": [8.0, 25.0],
                "preprocessing": [3.0, 35.0],
            },
            spatial_filters=["CSP", "CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_subject", "cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation", "gait"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="BrainVision (.vhdr/.vmrk/.eeg)",
    )

    def __init__(self, sessions=None, subjects=None, selected_sessions=None):
        if sessions is None:
            _sel_sessions = ["pre"]
        elif isinstance(sessions, str):
            _sel_sessions = [sessions]
        else:
            _sel_sessions = list(sessions)

        for s in _sel_sessions:
            if s not in _ALL_SESSIONS:
                raise ValueError(f"session must be one of {_ALL_SESSIONS}, got {s!r}")

        # Map original session labels to MOABB-compatible keys (must start
        # with a digit to match ^[0-9]+[a-zA-Z0-9]*).
        self._session_labels = _sel_sessions
        moabb_keys = [f"{i}{s}" for i, s in enumerate(_sel_sessions)]

        super().__init__(
            subjects=list(range(1, 28)),
            sessions_per_subject=len(_sel_sessions),
            events=dict(_EVENTS),
            code="Liu2025",
            interval=[0, 5],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=selected_sessions,
        )
        # Override _selected_sessions with MOABB-compatible keys so that
        # get_data() session filtering matches keys from _get_single_subject_data.
        self._selected_sessions = moabb_keys

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        basepath = Path(self.data_path(subject))

        # Determine available sessions for this subject.
        if subject in _PARADIGM_SUBJECTS:
            available = _ALL_SESSIONS
        else:
            available = _LONGITUDINAL_SESSIONS

        sessions = {}
        for sess_idx, ses_label in enumerate(self._session_labels):
            if ses_label not in available:
                log.info(
                    "Subject %d has no session '%s' (longitudinal group), " "skipping.",
                    subject,
                    ses_label,
                )
                continue

            # Find all .vhdr files for this session.
            ses_dir = basepath / f"sub-{subject:02d}" / f"ses-{ses_label}" / "eeg"
            if not ses_dir.exists():
                log.warning("Session dir not found: %s", ses_dir)
                continue

            vhdr_files = sorted(ses_dir.glob("*.vhdr"))
            if not vhdr_files:
                log.warning("No .vhdr files in %s", ses_dir)
                continue

            runs = {}
            for run_idx, vhdr in enumerate(vhdr_files):
                try:
                    raw = mne.io.read_raw_brainvision(
                        str(vhdr), preload=True, verbose="ERROR"
                    )

                    # Map BrainVision stimulus annotations to event names.
                    # BV format: "Stimulus/S  3" -> "gait_imagery"
                    desc = raw.annotations.description.copy()
                    new_desc = []
                    for d in desc:
                        code = d.replace("Stimulus/S", "").strip()
                        if code == "3":
                            new_desc.append("gait_imagery")
                        elif code == "9":
                            new_desc.append("rest")
                        else:
                            new_desc.append(d)
                    raw.annotations.description = np.array(new_desc)

                    raw = stim_channels_with_selected_ids(raw, self.event_id)
                    runs[str(run_idx)] = raw
                except Exception as e:
                    log.warning("Failed to load %s: %s", vhdr.name, e)

            if runs:
                # MOABB requires session keys matching ^[0-9]+[a-zA-Z0-9]*
                sess_key = f"{sess_idx}{ses_label}"
                sessions[sess_key] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No data found for subject {subject} "
                f"sessions={self._selected_sessions}"
            )
        return sessions

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}, must be in {self.subject_list}")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"sub-{subject:02d}"

        if subj_dir.exists() and list(subj_dir.rglob("*.vhdr")) and not force_update:
            return str(data_dir)

        # Download per-subject ZIP from Zenodo and extract.
        url = f"https://zenodo.org/records/{_ZENODO_RECORD}/files/sub-{subject:02d}.zip"
        download_and_extract_subject_zip(url, sign, data_dir, path, force_update, verbose)

        return str(data_dir)
