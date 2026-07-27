"""Intention-Action Conflict EEG-Hand Kinematics dataset (IACKD).

Cai, Fu, Wang, Lu, Guo, and Xu (2026).
Data DOI: 10.18112/openneuro.ds006840.v1.0.0
Hosted on OpenNeuro (ds006840), BIDS / BrainVision format.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from mne import Annotations
from mne.channels import make_standard_montage
from mne_bids import BIDSPath, read_raw_bids

from .base import BaseDataset
from .download import get_dataset_path
from .metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# OpenNeuro dataset ID and public S3 mirror (no auth required).
_OPENNEURO_ID = "ds006840"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# The two hand acquisitions and four runs each -> 8 EEG runs per subject.
_ACQUISITIONS = ("left", "right")
_RUNS = ("01", "02", "03", "04")

# 26 EEG channels (extended 10/20, Neuroscan) + 3 MISC (VEOG, HEOG, Trigger).
# fmt: off
_EEG_CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
# fmt: on
_MISC_CH_NAMES = ["VEOG", "HEOG", "Trigger"]

# Class label codes.
# Default (2-class): hand laterality from the events.tsv `hand` column.
_HAND_EVENTS = {"left_hand": 1, "right_hand": 2}
# 4-class: congruency x cued direction.
_CONFLICT_EVENTS = {
    "congruent_left": 1,
    "congruent_right": 2,
    "incongruent_left": 3,
    "incongruent_right": 4,
}


class Cai2026(BaseDataset):
    """Intention-Action Conflict EEG-Hand Kinematics dataset (IACKD).

    Dataset from *IACKD: Intention Action Conflict EEG-Hand Kinematics
    Dataset* by Cai et al., hosted on OpenNeuro as ``ds006840`` [1]_.

    Fifteen healthy subjects performed a unimanual ball-steering / reaching
    task. On each trial a visual color cue instructed a **congruent** (red
    ball) or **incongruent** (yellow ball) mapping between the required hand
    movement and the ball motion, and the ball had to be steered toward a
    target boundary in a cued **direction** (left / right). Each trial has a
    ``premove`` phase (~1 s of movement preparation) followed by an
    ``execution`` phase (variable duration) and a ``hit`` marker.

    EEG was recorded with a Compumedics Neuroscan system at 1024 Hz using 26
    scalp electrodes (extended 10/20) plus three auxiliary channels (VEOG,
    HEOG, Trigger), referenced to the common average. Each subject has two
    acquisitions (``acq-left`` = left-hand trials, ``acq-right`` =
    right-hand trials) of four runs each, giving eight runs total, all mapped
    to a single MOABB session.

    By default the loader epochs on the movement-preparation onset of each
    trial and labels it by **hand laterality** (2 classes: ``left_hand`` /
    ``right_hand``). Set ``four_class=True`` to instead label trials by the
    ``congruency x direction`` combination (4 classes).

    Parameters
    ----------
    four_class : bool
        If ``False`` (default), the two classes are left-hand vs right-hand.
        If ``True``, the four classes are congruency x direction
        (congruent/incongruent x left/right).
    subjects : list of int | None
        Subset of subjects to expose. If ``None``, all 15 are used.
    sessions : list | None
        Subset of sessions to expose.

    References
    ----------
    .. [1] Cai, M., Fu, R., Wang, Y., Lu, B., Guo, S., & Xu, F. (2026).
           IACKD: Intention Action Conflict EEG-Hand Kinematics Dataset.
           OpenNeuro. https://doi.org/10.18112/openneuro.ds006840.v1.0.0
    """

    # nemar_id pending: OpenNeuro ds006840 (NEMAR mirror id not yet confirmed).
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1024.0,
            n_channels=26,
            channel_types={"eeg": 26, "misc": 3},
            montage="standard_1020",
            hardware="Compumedics Neuroscan (32-ch)",
            reference="average",
            ground="n/a",
            software="PsychoPy 2024.1",
            filters="n/a",
            line_freq=50.0,
            sensors=list(_EEG_CH_NAMES),
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["vertical", "horizontal"],
                has_emg=False,
                other_physiological=["trigger"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=15, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events=dict(_HAND_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_HAND_EVENTS.keys()),
            study_design=(
                "Unimanual ball-steering/reaching under congruent vs "
                "incongruent visuomotor mapping (color-cued), with a cued "
                "movement direction (left/right). Premove (~1 s) then "
                "execution phase per trial."
            ),
            feedback_type="visual",
            stimulus_type="visual (colored ball + target boundary)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds006840.v1.0.0",
            investigators=[
                "Mengpu Cai",
                "Rongrong Fu",
                "Yaodong Wang",
                "Bin Lu",
                "Saiwei Guo",
                "Fangyao Xu",
            ],
            institution="Yanshan University",
            institution_address="Qinhuangdao, Hebei, China",
            institution_department="School of Electrical Engineering",
            country="CN",
            repository="OpenNeuro",
            data_url="https://openneuro.org/datasets/ds006840",
            publication_year=2026,
            license="CC0",
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=["left_hand", "right_hand"]
        ),
        data_structure=DataStructureMetadata(
            n_blocks=8,
            trials_context=(
                "2 acquisitions (left/right hand) x 4 runs per subject; "
                "per-trial premove + execution + hit markers."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=8,
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, four_class=False, subjects=None, sessions=None):
        events = dict(_CONFLICT_EVENTS) if four_class else dict(_HAND_EVENTS)
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events=events,
            code="Cai2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.18112/openneuro.ds006840.v1.0.0",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self.four_class = four_class

    def _trial_label(self, row):
        """Map an events.tsv (premove) row to a class label string."""
        if self.four_class:
            return f"{row['congruency']}_{row['direction']}"
        return f"{row['hand']}_hand"

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}} for one subject."""
        bids_paths = self.data_path(subject)

        runs = {}
        for idx, bids_path in enumerate(bids_paths):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = read_raw_bids(bids_path=bids_path, verbose=False).load_data()

            # Fall back to a standard montage when electrode positions are
            # missing so downstream spatial filters have coordinates.
            eeg_idx = [i for i, t in enumerate(raw.get_channel_types()) if t == "eeg"]
            if eeg_idx and any(
                np.isnan(raw.info["chs"][i]["loc"][:3]).any() for i in eeg_idx
            ):
                raw.set_montage(
                    make_standard_montage("standard_1020"), on_missing="ignore"
                )

            # Rebuild annotations from the events.tsv so trials are labelled by
            # the requested class scheme (hand laterality or congruency x
            # direction) rather than the movement-phase trial_type column.
            events_path = bids_path.copy().update(suffix="events", extension=".tsv")
            df = pd.read_csv(events_path.fpath, sep="\t")
            premove = df[df["trial_type"] == "premove"]
            onset = premove["onset"].to_numpy(dtype=float)
            descriptions = [self._trial_label(row) for _, row in premove.iterrows()]
            raw.set_annotations(
                Annotations(
                    onset=onset, duration=np.zeros(len(onset)), description=descriptions
                )
            )

            acq = bids_path.acquisition
            run = bids_path.run
            runs[f"{idx}acq{acq}run{run}"] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download subject files from OpenNeuro S3 and return BIDSPaths."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        bids_root = Path(get_dataset_path("Cai2026", path)) / "MNE-cai2026-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        # Root-level BIDS metadata required by mne_bids.
        self._download_files(bids_root, ["dataset_description.json"], force_update)

        subj_str = f"sub-{subject:02d}"
        bids_paths = []
        for acq in _ACQUISITIONS:
            for run in _RUNS:
                stem = f"{subj_str}_task-ihc_acq-{acq}_run-{run}"
                rel = [
                    f"{subj_str}/eeg/{stem}_eeg.vhdr",
                    f"{subj_str}/eeg/{stem}_eeg.eeg",
                    f"{subj_str}/eeg/{stem}_eeg.vmrk",
                    f"{subj_str}/eeg/{stem}_eeg.json",
                    f"{subj_str}/eeg/{stem}_channels.tsv",
                    f"{subj_str}/eeg/{stem}_events.tsv",
                ]
                # Electrode positions (one set per acquisition).
                rel += [
                    f"{subj_str}/eeg/{subj_str}_acq-{acq}_space-CapTrak_electrodes.tsv",
                    f"{subj_str}/eeg/{subj_str}_acq-{acq}_space-CapTrak_coordsystem.json",
                ]
                self._download_files(bids_root, rel, force_update)

                bids_paths.append(
                    BIDSPath(
                        root=bids_root,
                        subject=f"{subject:02d}",
                        task="ihc",
                        acquisition=acq,
                        run=run,
                        datatype="eeg",
                        suffix="eeg",
                        extension=".vhdr",
                        check=True,
                    )
                )

        return bids_paths

    @staticmethod
    def _download_files(bids_root, rel_paths, force_update):
        """Fetch a list of dataset-relative files from the OpenNeuro S3 mirror."""
        for rel_path in rel_paths:
            local_path = Path(bids_root) / rel_path
            if local_path.exists() and not force_update:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{_S3_BASE}/{rel_path}"
            log.info("Downloading %s ...", rel_path)
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code == 404:
                log.warning("Not found: %s (skipping)", url)
                continue
            resp.raise_for_status()
            with open(local_path, "wb") as fout:
                for chunk in resp.iter_content(chunk_size=8192):
                    fout.write(chunk)
