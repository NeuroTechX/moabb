"""Lioi-Perronnet 2020 EEG-fMRI motor imagery neurofeedback dataset (XP2).

Lioi, Cury, Perronnet, Mano, Bannier, Lecuyer, Barillot (2020), Scientific Data.
DOI: 10.1038/s41597-020-0498-3
Data DOI: 10.18112/openneuro.ds002338.v2.0.1 (OpenNeuro ds002338)
"""

import json
import logging
from pathlib import Path

import mne_bids
import numpy as np
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
_OPENNEURO_ID = "ds002338"

# S3 base URL for direct download (no auth needed for OpenNeuro).
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# The 17 XP2 subjects (non-contiguous IDs on OpenNeuro), index 1..17.
# fmt: off
_SUBJECT_IDS = [
    "xp201", "xp202", "xp203", "xp204", "xp205", "xp206", "xp207",
    "xp210", "xp211", "xp213", "xp216", "xp217", "xp218", "xp219",
    "xp220", "xp221", "xp222",
]
# fmt: on

# Neurofeedback task per subject (1-dimensional vs 2-dimensional NF).
# All subjects additionally have MIpre and MIpost pure-MI runs.
_SUBJECT_NF = {
    "xp201": "1dNF",
    "xp202": "1dNF",
    "xp203": "1dNF",
    "xp206": "1dNF",
    "xp211": "1dNF",
    "xp218": "1dNF",
    "xp219": "1dNF",
    "xp220": "1dNF",
    "xp222": "1dNF",
    "xp204": "2dNF",
    "xp205": "2dNF",
    "xp207": "2dNF",
    "xp210": "2dNF",
    "xp213": "2dNF",
    "xp216": "2dNF",
    "xp217": "2dNF",
    "xp221": "2dNF",
}

# 63 EEG channels from the BrainProducts BrainCap MR (Ch32=ECG is dropped
# from this list and handled as an auxiliary channel).
# fmt: off
_CH_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP9", "TP10",
    "POz", "F1", "F2", "C1", "C2", "P1", "P2", "AF3", "AF4", "FC3",
    "FC4", "CP3", "CP4", "PO3", "PO4", "F5", "F6", "C5", "C6", "P5",
    "P6", "AF7", "AF8", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "FT9",
    "FT10", "Fpz", "CPz",
]
# fmt: on

# events.tsv trial_type -> MOABB class label.
# Both the pure-MI ("Task-MI") and the neurofeedback ("Task-NF") blocks are the
# same kinesthetic right-hand motor imagery; "Rest" is the baseline block.
_TRIALTYPE_TO_LABEL = {"Task-MI": "right_hand", "Task-NF": "right_hand", "Rest": "rest"}

# Ordered per-subject runs: MIpre, 3 neurofeedback runs, MIpost.
# Each entry is (task_template, run_entity_or_None, run_key). "{nf}" is filled
# with the subject's neurofeedback task (1dNF or 2dNF).
_RUN_PLAN = [
    ("MIpre", None, "0MIpre"),
    ("{nf}", "01", "1NF01"),
    ("{nf}", "02", "2NF02"),
    ("{nf}", "03", "3NF03"),
    ("MIpost", None, "4MIpost"),
]

_DATASET_DESCRIPTION = {
    "Name": (
        "A multi-modal human neuroimaging dataset for data integration: "
        "simultaneous EEG and fMRI acquisition during a motor imagery "
        "neurofeedback task: XP2"
    ),
    "BIDSVersion": "1.2.0",
    "License": "CC0",
    "Authors": [
        "Giulia Lioi",
        "Claire Cury",
        "Lorraine Perronnet",
        "Marsel Mano",
        "Elise Bannier",
        "Anatole Lecuyer",
        "Christian Barillot",
    ],
    "DatasetDOI": "10.18112/openneuro.ds002338.v2.0.1",
}


class Lioi2020(BaseBIDSDataset):
    """Right-hand motor imagery EEG-fMRI neurofeedback dataset (XP2) [1]_.

    EEG recorded simultaneously with fMRI (only the EEG is exposed here) from
    17 healthy subjects performing kinesthetic motor imagery of their right
    hand, from the study on bimodal EEG-fMRI neurofeedback [2]_ (OpenNeuro
    ``ds002338``, experiment XP2).

    The paradigm is a block design alternating 20 s ``Rest`` and 20 s
    right-hand motor-imagery blocks. Each subject has five continuous EEG
    runs:

    - ``MIpre``   : motor imagery without feedback (before NF training)
    - three neurofeedback runs (``1dNF`` or ``2dNF`` depending on subject)
    - ``MIpost``  : motor imagery without feedback (after NF training)

    The mental task is identical across all runs (right-hand kinesthetic MI),
    so both the ``Task-MI`` and ``Task-NF`` blocks are mapped to the
    ``right_hand`` class and the ``Rest`` blocks to the ``rest`` class,
    yielding a 2-class (motor imagery vs. rest) problem.

    EEG was acquired with a 64-channel BrainProducts BrainCap MR (63 EEG + 1
    ECG) referenced to FCz at 5000 Hz inside the MR scanner. The ECG channel
    is dropped by default.

    The data are hosted on OpenNeuro in BIDS BrainVision format and downloaded
    directly from the OpenNeuro S3 mirror.

    Parameters
    ----------
    imagery_only : bool
        If True, expose only the two pure motor-imagery runs (``MIpre`` and
        ``MIpost``) that are identical across all subjects. If False
        (default), also expose the three neurofeedback runs.

    References
    ----------
    .. [1] Lioi, G., Cury, C., Perronnet, L., Mano, M., Bannier, E.,
           Lecuyer, A., & Barillot, C. (2020). Simultaneous EEG-fMRI during a
           neurofeedback task, a brain imaging dataset for multimodal data
           integration. Scientific Data, 7, 173.
           https://doi.org/10.1038/s41597-020-0498-3
    .. [2] Perronnet, L., Lecuyer, A., Mano, M., Bannier, E., Lotte, F.,
           Clerc, M., & Barillot, C. (2017). Unimodal versus bimodal EEG-fMRI
           neurofeedback of a motor imagery task. Frontiers in Human
           Neuroscience, 11, 193.
    """

    nemar_id = "ds002338"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=5000.0,
            n_channels=63,
            channel_types={"eeg": 63, "ecg": 1},
            montage="standard_1005",
            hardware="BrainProducts BrainAmp MR plus",
            cap_manufacturer="BrainProducts",
            cap_model="BrainCap MR",
            reference="FCz",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=17, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events={"rest": 1, "right_hand": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["rest", "right_hand"],
            trial_duration=20.0,
            study_design=(
                "Block design alternating 20 s rest and 20 s right-hand "
                "kinesthetic motor imagery, recorded simultaneously with fMRI. "
                "Five EEG runs per subject: MIpre, three neurofeedback runs "
                "(1dNF or 2dNF), and MIpost."
            ),
            feedback_type="EEG-fMRI",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-020-0498-3",
            investigators=[
                "Giulia Lioi",
                "Claire Cury",
                "Lorraine Perronnet",
                "Marsel Mano",
                "Elise Bannier",
                "Anatole Lecuyer",
                "Christian Barillot",
            ],
            institution="Univ Rennes, Inria, CNRS, Inserm, IRISA",
            country="FR",
            data_url="https://openneuro.org/datasets/ds002338",
            publication_year=2020,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=5,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand"],
            imagery_duration_s=20.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=8,
            trials_context="~8 right-hand MI blocks and ~8 rest blocks per run.",
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="within_subject", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["neurofeedback", "rehabilitation"],
            environment="MR scanner",
            online_feedback=True,
        ),
        data_processed=False,
        file_format="BrainVision (BIDS)",
    )

    def __init__(
        self,
        imagery_only=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        self.imagery_only = imagery_only
        super().__init__(
            subjects=list(range(1, len(_SUBJECT_IDS) + 1)),
            sessions_per_subject=1,
            events={"rest": 1, "right_hand": 2},
            code="Lioi2020",
            interval=[0, 20],
            paradigm="imagery",
            doi="10.1038/s41597-020-0498-3",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _sid(self, subject):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        return _SUBJECT_IDS[subject - 1]

    def _run_plan(self, subject):
        """Return the (task, run_entity, run_key) plan for a subject."""
        nf = _SUBJECT_NF[self._sid(subject)]
        plan = [(task.format(nf=nf), run, key) for task, run, key in _RUN_PLAN]
        if self.imagery_only:
            plan = [entry for entry in plan if "NF" not in entry[2]]
        return plan

    def _get_path_search_params(self, subject):
        out = {"extensions": [".vhdr"]}
        if subject is not None:
            out["subjects"] = self._sid(subject)
        return out

    @staticmethod
    def _sanitize_events_tsv(events_path):
        """Normalize a BIDS ``events.tsv`` so every row has the header's number
        of tab-separated columns.

        Some of the OpenNeuro ``task-MIpre`` / ``task-MIpost`` events files carry
        rows with a spurious trailing tab (an empty 5th field). mne-bids reads
        events with ``numpy.loadtxt``, which then raises "the number of columns
        changed from 4 to 5". This rewrites the file (idempotently) with a fixed
        column count and without blank lines, leaving the onset/duration/
        trial_type/stim_file values untouched. Runs offline on the already
        downloaded file.
        """
        events_path = Path(events_path)
        if not events_path.exists():
            return
        original = events_path.read_text(encoding="utf-8")
        lines = [ln for ln in original.splitlines() if ln != ""]
        if not lines:
            return
        ncol = len(lines[0].split("\t"))
        fixed = []
        for line in lines:
            fields = line.split("\t")
            if len(fields) > ncol:
                fields = fields[:ncol]  # drop spurious trailing empty column(s)
            elif len(fields) < ncol:
                fields += ["n/a"] * (ncol - len(fields))
            fixed.append("\t".join(fields))
        new = "\n".join(fixed) + "\n"
        if new != original:
            events_path.write_text(new, encoding="utf-8")

    def _get_single_subject_data(self, subject):
        """Load BrainVision runs and map block labels to MI-vs-rest classes."""
        bids_paths = self.bids_paths(subject)

        # Normalize each run's (task-level) events.tsv before mne-bids reads it,
        # so ragged rows do not break its numpy.loadtxt-based TSV reader.
        for bids_path in bids_paths:
            self._sanitize_events_tsv(
                Path(bids_path.root) / f"task-{bids_path.task}_events.tsv"
            )

        # Map each (task, run) BIDS entity pair to our descriptive run key.
        key_by_entity = {(task, run): key for task, run, key in self._run_plan(subject)}

        result = {}
        for bids_path in bids_paths:
            key = key_by_entity.get((bids_path.task, bids_path.run))
            if key is None:
                continue
            raw = mne_bids.read_raw_bids(
                bids_path, extra_params=self._get_read_extra_params(subject)
            )
            raw.load_data()

            # ECG is the only non-EEG sensor; type it and (optionally) drop it.
            if "ECG" in raw.ch_names:
                raw.set_channel_types({"ECG": "ecg"})

            # Remap block trial_types to our class labels.
            desc = raw.annotations.description.astype(np.dtype("<U32"))
            for trial_type, label in _TRIALTYPE_TO_LABEL.items():
                desc[desc == trial_type] = label
            raw.annotations.description = desc

            raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

            if not self.return_all_modalities:
                raw.pick("eeg")

            result.setdefault("0", {})[key] = stim_channels_with_selected_ids(
                raw, self.event_id
            )

        return result

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        """Download BIDS BrainVision data from OpenNeuro S3, return BIDS root."""
        sid = self._sid(subject)

        bids_root = Path(get_dataset_path("Lioi2020", path))
        bids_root = bids_root / "MNE-lioi2020-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        self._ensure_dataset_description(bids_root)

        subj_dir = bids_root / f"sub-{sid}" / "eeg"
        for task, run, _ in self._run_plan(subject):
            run_ent = f"_run-{run}" if run is not None else ""
            stem = f"sub-{sid}_task-{task}{run_ent}_eeg"
            for ext in (".vhdr", ".vmrk", ".eeg"):
                rel = f"sub-{sid}/eeg/{stem}{ext}"
                self._download_file(rel, subj_dir / f"{stem}{ext}", force_update)
            # Shared top-level sidecars needed by mne-bids (events + json).
            for shared in (f"task-{task}_events.tsv", f"task-{task}_eeg.json"):
                self._download_file(shared, bids_root / shared, force_update)

        return str(bids_root)

    @staticmethod
    def _download_file(rel_path, local_path, force_update):
        """Download a single OpenNeuro S3 object; skip missing (404) files."""
        local_path = Path(local_path)
        if local_path.exists() and not force_update:
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_S3_BASE}/{rel_path}"
        log.info("Downloading %s ...", rel_path)
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 404:
            log.warning("Not found: %s (skipping)", url)
            return
        resp.raise_for_status()
        with open(local_path, "wb") as fout:
            for chunk in resp.iter_content(chunk_size=8192):
                fout.write(chunk)

    @staticmethod
    def _ensure_dataset_description(bids_root):
        dd_path = Path(bids_root) / "dataset_description.json"
        if not dd_path.exists():
            with open(dd_path, "w") as f:
                json.dump(_DATASET_DESCRIPTION, f, indent=2)
