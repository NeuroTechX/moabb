"""Iwama 2023 high-density SMR-BMI motor imagery dataset (OpenNeuro ds004444)."""

import logging
import tempfile
from pathlib import Path

import mne
import pandas as pd
import requests

from ._openneuro_mirror import OpenNeuroMirrorMixin
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

_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds004444"

# events.tsv ``value``: 1 rest, 2 ready, 3 task (right-hand MI), 4 interval.
_EVENTS = {"right_hand": 3, "rest": 1}

# ses-01 .. ses-16; some subjects have fewer (sub-020: 8, sub-030: 9).
_MAX_SESSIONS = 16

_SESSION_SUFFIXES = "eeg.edf events.tsv channels.tsv electrodes.tsv eeg.json".split()
_ROOT_FILES = (
    "dataset_description.json participants.tsv participants.json "
    "task-smrbmi_eeg.json task-smrbmi_events.json"
).split()
_DOWNLOAD_ATTEMPTS = 3


class Iwama2023(OpenNeuroMirrorMixin, BaseBIDSDataset):
    """High-density (128ch) SMR-BMI motor imagery dataset, Dataset 1 [1]_.

    Dataset 1 of the BMI-HDEEG collection: 30 healthy right-handed
    participants (25 males, 5 females), 128-channel EGI HydroCel net at
    1000 Hz (the EDF carries 129 EEG channels, the extra one being the Cz
    reference; CPz was the ground). Each trial of the SMR neurofeedback task
    is rest (6 s, ``value = 1``), ready (1 s), task (6 s kinesthetic right-hand
    motor imagery with ERD feedback, ``value = 3``) and interval (8 s). MOABB
    exposes **right_hand** vs **rest**; each session (``ses-01`` .. ``ses-16``,
    fewer for some subjects) has 20 trials. The paper describes, per day, a
    pre-evaluation block, 6 neurofeedback blocks and a post-evaluation block
    over two consecutive days (16 blocks, stored one EDF per block).

    .. note::
        The BIDS ``events.tsv`` onsets are in milliseconds; this loader
        rescales them to seconds. The EDF's own ``Status`` trigger channel is
        dropped because its codes do not follow ``events.tsv``.

    References
    ----------
    .. [1] Iwama, S., Morishige, M., Kodama, M., Takahashi, Y., Hirose, R.,
           & Ushiba, J. (2023). High-density scalp electroencephalogram
           dataset during sensorimotor rhythm-based brain-computer
           interfacing. Scientific Data, 10, 385.
           https://doi.org/10.1038/s41597-023-02260-6
    """

    nemar_id = "on004444"
    nemar_subject_template = "{subject:03d}"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 129},
            montage="GSN-HydroCel-129",
            hardware="Magstim EGI GES400",
            cap_manufacturer="Magstim EGI",
            cap_model="HydroCel Geodesic Sensor Net",
            reference="Cz",
            ground="CPz",
            filters={"highpass": 0.1, "lowpass": 100, "notch": 50},
            line_freq=50.0,
            software="EGI NetStation",
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            health_status="healthy",
            species="human",
            gender={"male": 25, "female": 5},
            handedness="right-handed",
            age_mean=21.23,
            age_std=2.2,
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
                "Midori Kodama",
                "Yoshikazu Takahashi",
                "Ryotaro Hirose",
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
        """Read each session's EDF with annotations from its ms-onset events.tsv."""
        inv_events = {code: label for label, code in self.event_id.items()}
        sessions = {}
        for bids_path in self.bids_paths(subject):
            edf_path = Path(bids_path.fpath)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            # The native ``Status`` code 1 marks rest onsets *and* ~1 s before
            # each task onset; kept, it would add 20 spurious rest epochs.
            raw.drop_channels(
                [
                    ch
                    for ch, kind in zip(raw.ch_names, raw.get_channel_types())
                    if kind == "stim"
                ]
            )

            events_tsv = edf_path.with_name(
                edf_path.name.replace("_eeg.edf", "_events.tsv")
            )
            df = pd.read_csv(events_tsv, sep="\t")
            df = df[df["value"].isin(inv_events.keys())]
            raw.set_annotations(
                mne.Annotations(
                    onset=df["onset"].astype(float).to_numpy() / 1000.0,
                    duration=df["duration"].astype(float).to_numpy(),
                    description=df["value"].map(inv_events).to_numpy(),
                )
            )
            try:
                raw.set_montage("GSN-HydroCel-129", match_case=False, on_missing="ignore")
            except Exception as exc:  # montage is optional; keep the data usable
                log.warning("Could not set montage for %s: %s", edf_path.name, exc)

            raw = stim_channels_with_selected_ids(raw, self.event_id)
            session = bids_path.session if bids_path.session is not None else "0"
            run = bids_path.run if bids_path.run is not None else "0"
            sessions.setdefault(session, {})[run] = raw
        return sessions

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the subject's BIDS files from OpenNeuro S3 and return the root."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        if mirror_root is not None:
            return mirror_root

        bids_root = Path(get_dataset_path("Iwama2023", path)) / "MNE-iwama2023-data"
        bids_root.mkdir(parents=True, exist_ok=True)
        subj_str = f"sub-{subject:03d}"

        # Session labels can be sparse (sub-030 has ses-16 after ses-08), so a
        # staged subject whose EDFs all have their sidecars is used as is
        # instead of probing S3 (which would also block offline loads).
        edfs = sorted((bids_root / subj_str).glob("ses-*/eeg/*_eeg.edf"))
        if (
            not force_update
            and edfs
            and all(
                (edf.parent / f"{edf.name.removesuffix('eeg.edf')}{suffix}").is_file()
                for edf in edfs
                for suffix in _SESSION_SUFFIXES
            )
        ):
            log.info("Using locally staged Iwama2023 subject %03d", subject)
            return str(bids_root)

        for rel_path in _ROOT_FILES:
            self._download_file(bids_root, rel_path, force_update)
        for ses in range(1, _MAX_SESSIONS + 1):
            ses_str = f"ses-{ses:02d}"
            base = f"{subj_str}/{ses_str}/eeg/{subj_str}_{ses_str}_task-smrbmi_"
            got_any = False
            for suffix in _SESSION_SUFFIXES:
                got_any |= self._download_file(bids_root, base + suffix, force_update)
            if not got_any and ses > 1:
                break  # no files for this session index: no more sessions
        return str(bids_root)

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
                    "Downloading %s (attempt %d/%d)",
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
                log.warning("Download of %s failed (%s); retrying", rel_path, exc)
            finally:
                if response is not None:
                    response.close()
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
