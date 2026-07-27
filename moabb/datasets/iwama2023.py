"""High-density scalp EEG SMR-BMI motor imagery dataset (Iwama 2023, Dataset 1).

Iwama, S., Morishige, M., Kodama, M. et al. High-density scalp
electroencephalogram dataset during sensorimotor rhythm-based
brain-computer interfacing. Sci Data 10, 385 (2023).
DOI: 10.1038/s41597-023-02260-6
Data DOI: 10.18112/openneuro.ds004444.v1.0.1
"""

import logging
import tempfile
from pathlib import Path

import mne
import pandas as pd
import requests

from .base import BaseBIDSDataset
from .download import get_dataset_path
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
    Tags,
)
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# OpenNeuro dataset ID.
_OPENNEURO_ID = "ds004444"

# S3 base URL for direct download (no auth needed for OpenNeuro).
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# Marker codes in the ``value`` column of the BIDS events.tsv:
#   1 = rest      (baseline period, used as the "rest" class)
#   2 = ready     (preparation cue, not decoded)
#   3 = task      (kinesthetic motor imagery of the right hand, "right_hand")
#   4 = interval  (inter-trial interval, not decoded)
_EVENTS = {"right_hand": 3, "rest": 1}

# Maximum number of recorded sessions (ses-01 .. ses-16). Some subjects
# have fewer (sub-020 has 8, sub-030 has 9); missing sessions are skipped.
_MAX_SESSIONS = 16

# Per-session files that make up one BIDS recording.
_SESSION_SUFFIXES = [
    "eeg.edf",
    "events.tsv",
    "channels.tsv",
    "electrodes.tsv",
    "eeg.json",
]

# Root-level BIDS files needed for a valid dataset root.
_ROOT_FILES = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "task-smrbmi_eeg.json",
    "task-smrbmi_events.json",
]

_DOWNLOAD_ATTEMPTS = 3


class Iwama2023(BaseBIDSDataset):
    """High-density (128ch) SMR-BMI motor imagery dataset, Dataset 1 [1]_.

    This is *Dataset 1* of the BMI-HDEEG collection released with the data
    descriptor *High-density scalp electroencephalogram dataset during
    sensorimotor rhythm-based brain-computer interfacing* [1]_. It contains
    high-density EEG from 30 healthy participants recorded with a 128-channel
    Magstim EGI HydroCel Geodesic Sensor Net (GES400) at 1000 Hz, referenced
    to Cz with the ground placed on FCz. The recorded EDF carries 129 EEG
    channels in total: the 128 net electrodes (E1-E128) plus the Cz
    reference itself, which is stored as an additional channel.

    Participants performed a sensorimotor-rhythm (SMR) neurofeedback BCI task.
    Each trial consists of four phases marked in the events file:

    - **rest** (6 s): baseline / rest period (``value = 1``)
    - **ready** (1 s): preparation cue (``value = 2``)
    - **task** (6 s): kinesthetic motor imagery of right-hand/finger
      movement, with ERD-based neurofeedback (``value = 3``)
    - **interval** (8 s): inter-trial interval (``value = 4``)

    MOABB exposes the binary decoding problem **right_hand** (motor imagery,
    ``value = 3``) versus **rest** (``value = 1``). Each recorded session
    (``ses-01`` .. ``ses-16``) contains 20 trials. Most subjects have 16
    sessions; a few have fewer.

    The data is hosted on OpenNeuro (ds004444) in BIDS format (EDF files).

    .. note::
        The BIDS ``events.tsv`` files store the ``onset`` column in
        milliseconds (BIDS normally expects seconds); this loader rescales
        the onsets to seconds before building the annotations.

    References
    ----------
    .. [1] Iwama, S., Morishige, M., Kodama, M., Takahashi, Y., Hirose, R.,
           & Ushiba, J. (2023). High-density scalp electroencephalogram
           dataset during sensorimotor rhythm-based brain-computer
           interfacing. Scientific Data, 10, 385.
           https://doi.org/10.1038/s41597-023-02260-6
    """

    nemar_id = "ds004444"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=129,
            channel_types={"eeg": 129},
            montage="GSN-HydroCel-129",
            hardware="Magstim EGI GES400",
            cap_manufacturer="Magstim EGI",
            cap_model="HydroCel Geodesic Sensor Net",
            reference="Cz",
            ground="FCz",
            filters={"highpass": 0.1, "lowpass": 100, "notch": 50},
            line_freq=50.0,
            software="EGI NetStation",
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            health_status="healthy",
            species="human",
            age_min=18.0,
            age_max=27.0,
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=21.0,
            study_design=(
                "Sensorimotor-rhythm (SMR) neurofeedback BCI. Each trial: "
                "rest (6 s) -> ready (1 s) -> task (6 s kinesthetic motor "
                "imagery of right hand with ERD feedback) -> interval (8 s). "
                "20 trials per session, up to 16 sessions per subject."
            ),
            feedback_type="visual ERD neurofeedback",
            stimulus_type="visual instruction",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-023-02260-6",
            investigators=[
                "Seitaro Iwama",
                "Masumi Morishige",
                "Yoshikazu Takahashi",
                "Ryotaro Hirose",
                "Midori Kodama",
                "Junichi Ushiba",
            ],
            institution="Keio University",
            country="JP",
            data_url="https://openneuro.org/datasets/ds004444",
            publication_year=2023,
            license="CC0",
        ),
        sessions_per_subject=_MAX_SESSIONS,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand", "rest"],
            imagery_duration_s=6.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=20,
            trials_context=(
                "20 trials per session, up to 16 sessions per subject "
                "(2 classes: right-hand motor imagery vs rest)."
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-session", evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "neurofeedback", "rehabilitation"],
            environment="laboratory",
            online_feedback=True,
        ),
        file_format="EDF (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 31)),
            sessions_per_subject=_MAX_SESSIONS,
            events=dict(_EVENTS),
            code="Iwama2023",
            interval=[0, 6],
            paradigm="imagery",
            doi="10.1038/s41597-023-02260-6",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_path_search_params(self, subject):
        """Zero-padded subject numbers (sub-001, not sub-1); EDF files only."""
        out = {"extensions": [".edf"]}
        if subject is not None:
            out["subjects"] = f"{subject:03d}"
        return out

    def _get_single_subject_data(self, subject):
        """Read each session's EDF and attach corrected event annotations.

        The BIDS ``events.tsv`` onsets are in milliseconds, so they are
        rescaled to seconds before building the annotations (the default BIDS
        reader would misplace them).
        """
        bids_paths = self.bids_paths(subject)
        inv_events = {code: label for label, code in self.event_id.items()}

        sessions = {}
        for bids_path in bids_paths:
            edf_path = Path(bids_path.fpath)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

            events_tsv = edf_path.with_name(
                edf_path.name.replace("_eeg.edf", "_events.tsv")
            )
            annotations = self._build_annotations(events_tsv, inv_events)
            raw.set_annotations(annotations)

            try:
                raw.set_montage("GSN-HydroCel-129", match_case=False, on_missing="ignore")
            except Exception as exc:  # montage is optional; keep the data usable
                log.warning("Could not set montage for %s: %s", edf_path.name, exc)

            raw = stim_channels_with_selected_ids(raw, self.event_id)

            session = bids_path.session if bids_path.session is not None else "0"
            run = bids_path.run if bids_path.run is not None else "0"
            sessions.setdefault(session, {})[run] = raw

        return sessions

    @staticmethod
    def _build_annotations(events_tsv, inv_events):
        """Build MNE annotations from a BIDS events.tsv (onset in ms -> s)."""
        df = pd.read_csv(events_tsv, sep="\t")
        df = df[df["value"].isin(inv_events.keys())]
        onset_s = df["onset"].astype(float).to_numpy() / 1000.0
        duration_s = df["duration"].astype(float).to_numpy()
        description = df["value"].map(inv_events).to_numpy()
        return mne.Annotations(
            onset=onset_s, duration=duration_s, description=description
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the subject's BIDS files from OpenNeuro S3 and return the root."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        bids_root = Path(get_dataset_path("Iwama2023", path)) / "MNE-iwama2023-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        # A staged BIDS subject is complete enough to discover directly from
        # its EDF files.  In particular, subjects can have non-consecutive
        # session labels (sub-030 has ses-16 after ses-08), so probing S3 for
        # the next numerical session both misses valid data and can block an
        # otherwise offline load on a missing-object request.
        if not force_update and self._staged_subject_is_complete(bids_root, subject):
            log.info("Using locally staged Iwama2023 subject %03d", subject)
            return str(bids_root)

        self._download_root_files(bids_root, force_update)

        subj_str = f"sub-{subject:03d}"
        for ses in range(1, _MAX_SESSIONS + 1):
            ses_str = f"ses-{ses:02d}"
            base = f"{subj_str}/{ses_str}/eeg/{subj_str}_{ses_str}_task-smrbmi_"
            got_any = False
            for suffix in _SESSION_SUFFIXES:
                got_any |= self._download_file(bids_root, base + suffix, force_update)
            if not got_any and ses > 1:
                # No files for this session index: assume no more sessions exist.
                break

        return str(bids_root)

    @staticmethod
    def _staged_subject_edfs(bids_root, subject):
        """Return locally staged EDFs for one subject, including sparse sessions."""
        subject_dir = Path(bids_root) / f"sub-{subject:03d}"
        return sorted(subject_dir.glob("ses-*/eeg/*_eeg.edf"))

    @classmethod
    def _staged_subject_is_complete(cls, bids_root, subject):
        """Return whether every staged EDF has all required BIDS sidecars."""
        edfs = cls._staged_subject_edfs(bids_root, subject)
        if not edfs:
            return False
        for edf in edfs:
            prefix = edf.name.removesuffix("eeg.edf")
            if not all(
                (edf.parent / f"{prefix}{suffix}").is_file()
                for suffix in _SESSION_SUFFIXES
            ):
                return False
        return True

    def _download_root_files(self, bids_root, force_update):
        for rel_path in _ROOT_FILES:
            self._download_file(bids_root, rel_path, force_update)

    @staticmethod
    def _download_file(bids_root, rel_path, force_update) -> bool:
        """Download one file from OpenNeuro S3. Returns True if present locally."""
        local_path = bids_root / rel_path
        if local_path.exists() and not force_update:
            return True
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_S3_BASE}/{rel_path}"
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            temp_path = None
            response = None
            try:
                log.info(
                    "Downloading %s (attempt %d/%d) ...",
                    rel_path,
                    attempt,
                    _DOWNLOAD_ATTEMPTS,
                )
                response = requests.get(url, stream=True, timeout=300)
                if response.status_code == 404:
                    log.debug("Not found: %s (skipping)", url)
                    return False
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=local_path.parent, delete=False
                ) as fout:
                    temp_path = Path(fout.name)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            fout.write(chunk)
                temp_path.replace(local_path)
                return True
            except requests.RequestException as exc:
                if attempt == _DOWNLOAD_ATTEMPTS:
                    raise
                log.warning(
                    "Download of %s failed (%s); retrying (%d/%d)",
                    rel_path,
                    exc,
                    attempt,
                    _DOWNLOAD_ATTEMPTS,
                )
            finally:
                if response is not None:
                    response.close()
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        raise RuntimeError(f"Unreachable download retry state for {url}")
