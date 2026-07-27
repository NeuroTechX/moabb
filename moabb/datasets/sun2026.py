"""Cross-Environment Multi-Paradigm Motor Imagery EEG dataset (OpenNeuro ds007221).

Sun, Wang, Pan, Cao, and Meng (2026).
Data DOI: 10.18112/openneuro.ds007221.v1.0.0
Hosted on OpenNeuro (ds007221), BIDS / BrainVision format.

This loader exposes the ``graz`` motor-imagery sub-dataset (the pure four-class
Graz paradigm, subjects 1-14). See the class docstring for why the other
sub-datasets bundled in ds007221 are out of scope.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import requests
from mne.channels import make_standard_montage
from mne_bids import BIDSPath, read_raw_bids

from .base import BaseDataset
from .download import get_dataset_path
from .metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# OpenNeuro dataset id and the published snapshot that carries the v1.0.0 DOI.
_OPENNEURO_ID = "ds007221"
_SNAPSHOT = "1.0.0"
# ds007221 object reads on the plain S3 mirror (s3.amazonaws.com/openneuro.org)
# return HTTP 403; the files are only served through the OpenNeuro "crn"
# snapshot endpoint, which 302-redirects to a short-lived signed S3 URL. The
# endpoint encodes the BIDS path with ":" instead of "/".
_CRN_BASE = (
    f"https://openneuro.org/crn/datasets/{_OPENNEURO_ID}/snapshots/{_SNAPSHOT}/files"
)

# The four Graz motor-imagery classes as they appear in the events.tsv
# ``trial_type`` column (the numeric ``value`` codes are reused with different
# meanings by the other, out-of-scope, sub-datasets, so the string labels are
# authoritative). read_raw_bids turns ``trial_type`` into annotations directly.
_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}

# The graz sub-dataset (subjects 1-14) has two sessions and, per session,
# six runs -- except a couple of sessions that have a seventh run. Runs are
# probed up to this bound and missing ones are skipped.
_SESSIONS = ("01", "02")
_MAX_RUNS = 8

# 64 scalp EEG channels (10-10 layout, incl. M1/M2 mastoids and CB1/CB2). The
# recording additionally carries 5 auxiliary channels that the BIDS
# channels.tsv mislabels as "EEG"; the loader retypes them (see _AUX_TYPES).
# fmt: off
_EEG_CH_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "M1",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]
# fmt: on

# Auxiliary channels stored alongside the scalp EEG, with the MNE type they
# should carry. With return_all_modalities=False (the default) MOABB keeps only
# EEG, so these are excluded from motor-imagery decoding.
_AUX_TYPES = {"HEO": "eog", "VEO": "eog", "EKG": "ecg", "EMG": "emg", "Trigger": "stim"}

# Root-level BIDS files needed for a valid dataset root.
_ROOT_FILES = ("dataset_description.json", "participants.tsv", "participants.json")

# Per-run sidecar suffixes (the big ``eeg.eeg`` binary is fetched separately,
# and only once the run's header is known to exist).
_RUN_SIDECARS = (
    "eeg.vhdr",
    "eeg.vmrk",
    "eeg.json",
    "channels.tsv",
    "events.tsv",
    "events.json",
)


class Sun2026(BaseDataset):
    """Cross-Environment Multi-Paradigm Motor Imagery EEG, Graz subset [1]_.

    Dataset from the *Cross-Environment Multi-Paradigm Motor Imagery EEG
    Dataset* by Sun et al., hosted on OpenNeuro as ``ds007221`` [1]_. The full
    OpenNeuro release bundles 84 subjects recorded across several recording
    environments (a shielded laboratory and a simulated hospital) and several
    paradigms: a pure Graz motor-imagery protocol, an SSMVEP-MI protocol, and
    hybrid protocols that pair motor imagery with video / SSVideo visual
    stimulation. Those protocols use heterogeneous class vocabularies (for
    example the SSMVEP-MI subjects mix ``*_MI`` with action-observation
    ``*_AO`` trials, and the hybrid subjects are binary left/right hand), so
    mixing them into a single motor-imagery loader would be unsound.

    **This loader therefore exposes only the** ``graz`` **sub-dataset**
    (subjects 1-14), which is a clean, uniform four-class Graz motor-imagery
    task:

    - ``left_hand``  : left-hand motor imagery,
    - ``right_hand`` : right-hand motor imagery,
    - ``feet``       : feet motor imagery,
    - ``rest``       : rest / no-imagery.

    Each subject was recorded in two sessions, each holding six runs (a couple
    of sessions have a seventh). A run contains 40 cued trials (10 per class).
    Per the dataset description each trial has a cue period (-2 to 0 s), a
    task / motor-imagery period (0 to 4 s), and a rest period, so the imagery
    interval is set to ``[0, 4]``.

    EEG was recorded with a Brain Products (BrainVision) system at 1000 Hz
    using 64 scalp electrodes (10-10 layout, including M1/M2 mastoids and
    CB1/CB2), referenced to the nose. Five auxiliary channels (HEO, VEO, EKG,
    EMG and a Trigger channel) are also stored; the BIDS ``channels.tsv``
    mislabels them as EEG, so the loader retypes them to EOG / ECG / EMG /
    stim and, with the default ``return_all_modalities=False``, they are
    dropped from decoding.

    Notes
    -----
    ds007221 objects are not served by the usual public S3 mirror (they return
    HTTP 403); this loader downloads through the OpenNeuro snapshot endpoint,
    which redirects to signed URLs. It is pinned to snapshot ``1.0.0`` (the
    version that carries the v1.0.0 DOI).

    Class labels are read from the per-run BIDS ``events.tsv`` ``trial_type``
    column via ``read_raw_bids`` (they are not inferred from acquisition
    order).

    Parameters
    ----------
    subjects : list of int | None
        Subset of subjects to expose. If ``None``, all 14 are used.
    sessions : list | None
        Subset of sessions to expose.

    References
    ----------
    .. [1] Sun, X., Wang, K., Pan, L., Cao, Y., & Meng, L. (2026).
           Cross-Environment Multi-Paradigm Motor Imagery EEG Dataset.
           OpenNeuro. https://doi.org/10.18112/openneuro.ds007221.v1.0.0
    """

    # nemar_id pending: OpenNeuro ds007221 (NEMAR mirror id not yet confirmed).
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64, "eog": 2, "ecg": 1, "emg": 1, "stim": 1},
            montage="standard_1005",
            hardware="Brain Products (BrainVision)",
            reference="nose",
            line_freq=50.0,
            sensors=list(_EEG_CH_NAMES),
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                emg_channels=1,
                other_physiological=["ecg", "trigger"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            gender={"male": 5, "female": 9},
            handedness={"right": 14},
            age_min=21.0,
            age_max=32.0,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=list(_EVENTS.keys()),
            trial_duration=4.0,
            study_design=(
                "Four-class Graz motor imagery (left hand / right hand / feet "
                "/ rest) recorded across two sessions and multiple recording "
                "environments (shielded laboratory and simulated hospital)."
            ),
            feedback_type="none",
            stimulus_type="visual Graz cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds007221.v1.0.0",
            investigators=[
                "Xinwei Sun",
                "Kun Wang",
                "Lincong Pan",
                "Yupei Cao",
                "Lin Meng",
            ],
            repository="OpenNeuro",
            data_url="https://openneuro.org/datasets/ds007221",
            country="CN",
            publication_year=2026,
            license="CC0",
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=6,
            trials_context=(
                "graz subset: 14 subjects x 2 sessions x ~6 runs (a couple of "
                "sessions have 7) x 40 trials/run (10 per class)."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            environment="laboratory / simulated hospital", online_feedback=False
        ),
        sessions_per_subject=2,
        runs_per_session=6,
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=2,
            events=dict(_EVENTS),
            code="Sun2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.18112/openneuro.ds007221.v1.0.0",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: Raw}}`` for one subject."""
        bids_paths = self.data_path(subject)

        sessions = {}
        for bids_path in bids_paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = read_raw_bids(bids_path=bids_path, verbose=False).load_data()

            # The channels.tsv marks HEO/VEO/EKG/EMG/Trigger as EEG; retype the
            # ones that are present so downstream picks use only scalp EEG.
            retype = {ch: t for ch, t in _AUX_TYPES.items() if ch in raw.ch_names}
            if retype:
                raw.set_channel_types(retype, verbose=False)

            # Electrode positions come from the CapTrak electrodes.tsv when
            # present; fall back to a standard montage if any EEG channel has
            # no coordinates so spatial filters still have positions.
            eeg_idx = [i for i, t in enumerate(raw.get_channel_types()) if t == "eeg"]
            if eeg_idx and any(
                np.isnan(raw.info["chs"][i]["loc"][:3]).any() for i in eeg_idx
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw.set_montage(
                        make_standard_montage("standard_1005"),
                        match_case=False,
                        on_missing="ignore",
                        verbose=False,
                    )

            session = bids_path.session
            run = bids_path.run
            sessions.setdefault(session, {})[run] = raw

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the subject's Graz BIDS files and return their BIDSPaths."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        bids_root = Path(get_dataset_path("Sun2026", path)) / "MNE-sun2026-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        for rel in _ROOT_FILES:
            self._download_file(bids_root, rel, force_update)

        subj = f"sub-{subject:02d}"
        bids_paths = []
        for ses in _SESSIONS:
            eeg_dir = f"{subj}/ses-{ses}/eeg"
            found_local_run = False
            # Per-session electrode positions (optional).
            for suffix in ("electrodes.tsv", "coordsystem.json"):
                self._download_file(
                    bids_root,
                    f"{eeg_dir}/{subj}_ses-{ses}_space-CapTrak_{suffix}",
                    force_update,
                )

            # A fully staged subject must load without probing OpenNeuro for a
            # hypothetical next run. Collect every complete local run first;
            # fall back to remote discovery only when no complete local session
            # exists or a locally present run is incomplete.
            local_session_paths = []
            incomplete_local_run = False
            for run in range(1, _MAX_RUNS + 1):
                stem = f"{subj}_ses-{ses}_task-graz_run-{run:02d}"
                required = [
                    bids_root / eeg_dir / f"{stem}_{suffix}" for suffix in _RUN_SIDECARS
                ]
                required.append(bids_root / eeg_dir / f"{stem}_eeg.eeg")
                if required[0].exists():
                    if not all(local_path.exists() for local_path in required):
                        incomplete_local_run = True
                        break
                    local_session_paths.append(
                        BIDSPath(
                            root=bids_root,
                            subject=f"{subject:02d}",
                            session=ses,
                            task="graz",
                            run=f"{run:02d}",
                            datatype="eeg",
                            suffix="eeg",
                            extension=".vhdr",
                        )
                    )
            if local_session_paths and not incomplete_local_run and not force_update:
                bids_paths.extend(local_session_paths)
                continue

            for run in range(1, _MAX_RUNS + 1):
                stem = f"{subj}_ses-{ses}_task-graz_run-{run:02d}"
                # Probe the run via its header; a missing header means the run
                # (and any later run in the session) does not exist.
                try:
                    header_present = self._download_file(
                        bids_root, f"{eeg_dir}/{stem}_eeg.vhdr", force_update
                    )
                except requests.RequestException:
                    # The released run count varies. If one or more complete
                    # local runs were found, an offline failure while probing
                    # run N+1 means there are no further locally staged runs;
                    # keep the valid local session instead of failing it.
                    if found_local_run and not force_update:
                        log.info(
                            "Stopping offline run discovery for %s session %s "
                            "after run %02d",
                            subj,
                            ses,
                            run - 1,
                        )
                        break
                    raise
                if not header_present:
                    break
                for suffix in _RUN_SIDECARS[1:]:
                    self._download_file(
                        bids_root, f"{eeg_dir}/{stem}_{suffix}", force_update
                    )
                self._download_file(bids_root, f"{eeg_dir}/{stem}_eeg.eeg", force_update)

                bids_paths.append(
                    BIDSPath(
                        root=bids_root,
                        subject=f"{subject:02d}",
                        session=ses,
                        task="graz",
                        run=f"{run:02d}",
                        datatype="eeg",
                        suffix="eeg",
                        extension=".vhdr",
                    )
                )
                found_local_run = True

        return bids_paths

    @staticmethod
    def _download_file(bids_root, rel_path, force_update):
        """Fetch one dataset-relative file from the OpenNeuro snapshot endpoint.

        Returns ``True`` when the file is present locally afterwards, ``False``
        when the endpoint reports it does not exist (HTTP 404).
        """
        local_path = Path(bids_root) / rel_path
        if local_path.exists() and not force_update:
            return True
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # The crn "files" endpoint uses ":" as the path separator.
        url = f"{_CRN_BASE}/{rel_path.replace('/', ':')}"
        log.info("Downloading %s ...", rel_path)
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 404:
            log.debug("Not found: %s (skipping)", rel_path)
            return False
        resp.raise_for_status()
        with open(local_path, "wb") as fout:
            for chunk in resp.iter_content(chunk_size=8192):
                fout.write(chunk)
        return True
