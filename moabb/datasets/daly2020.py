"""Daly2020 tempo-based BCMI motor imagery dataset (BCMI-MIdAS).

Daly et al. (2018), "A dataset recorded during development of a tempo-based
brain-computer music interface".
Data DOI: 10.18112/openneuro.ds002720.v1.0.1 (OpenNeuro ds002720)
"""

import json
import logging
from pathlib import Path

import numpy as np
import requests

from .base import BaseBIDSDataset
from .download import get_dataset_path
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# OpenNeuro dataset ID and S3 mirror (no auth needed for OpenNeuro).
_OPENNEURO_ID = "ds002720"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# Binary kinesthetic motor imagery classes (from events.tsv trial_type column):
# 1 = Lower alpha  -> right-hand ball-squeeze imagery (increase music tempo);
#     motor imagery produces mu/alpha event-related desynchronization, i.e.
#     it lowers alpha (verified against sub-01_task-run2_events.json and the
#     README's tempo-increase/ball-squeeze pairing).
# 2 = Raise alpha  -> relax (decrease music tempo)
_EVENTS = {"right_hand": 1, "relax": 2}

# Map raw numeric annotation descriptions ("1"/"2") to class names.
_VALUE_TO_NAME = {"1": "right_hand", "2": "relax"}

# 19 scalp EEG channels (international 10-20, FCz reference).
# fmt: off
_CH_NAMES = [
    "FP1", "FP2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]
# fmt: on

# 9 runs per subject; run 1 is calibration with an EMPTY events file.
_N_RUNS = 9
_CALIBRATION_RUN = 1

# Minimal BIDS dataset_description.json for mne_bids compatibility.
_DATASET_DESCRIPTION = {
    "Name": (
        "A dataset recorded during development of a tempo-based "
        "brain-computer music interface"
    ),
    "BIDSVersion": "1.0.2",
    "License": "CC0",
    "Authors": [
        "Ian Daly",
        "Nicoletta Nicolaou",
        "Duncan Williams",
        "Faustina Hwang",
        "Alexis Kirke",
        "Eduardo Miranda",
        "Slawomir J. Nasuto",
    ],
    "DatasetDOI": "10.18112/openneuro.ds002720.v1.0.1",
}


class Daly2020(BaseBIDSDataset):
    """Tempo-based BCMI motor imagery dataset from Daly et al. 2018 [1]_.

    Dataset from the BCMI-MIdAS project (*Brain-Computer Music Interface for
    Monitoring and Inducing Affective States*), recorded during development of
    a tempo-based brain-computer music interface at the University of Reading.

    18 healthy participants controlled the tempo of a piece of music via
    intentional alpha-band neurofeedback. To *increase* the tempo they
    kinaesthetically imagined squeezing a ball in their **right hand** (motor
    imagery causes mu/alpha event-related desynchronization, i.e. it *lowers*
    alpha), and to *decrease* the tempo they **relaxed** (which *raises*
    alpha). This gives a binary kinesthetic motor-imagery contrast:

    - **right_hand** (event 1, "Lower alpha"): right-hand ball-squeeze imagery
    - **relax** (event 2, "Raise alpha"): relaxation / rest

    EEG was recorded with 19 scalp electrodes (international 10-20 system,
    FCz reference) at 1000 Hz. The paradigm was split into 9 runs per subject.
    The first run is a calibration run whose events file is empty and is
    therefore **skipped**; runs 2-9 contain alternating 20 s binary trials and
    are exposed as runs ``"0"`` .. ``"7"`` of a single session ``"0"``.

    .. note::
       The README prose states 19 participants, but the released data contains
       only 18 subjects (``sub-01`` .. ``sub-18``); this loader uses 18.
       The machine-readable ``dataset_description.json`` licenses the data as
       CC0 (the README text mentions CC-BY-4.0).

    The data is hosted on OpenNeuro (``ds002720``) in BIDS EDF format.

    References
    ----------
    .. [1] Daly, I., Nicolaou, N., Williams, D., Hwang, F., Kirke, A.,
           Miranda, E., & Nasuto, S. J. (2018). A dataset recorded during
           development of a tempo-based brain-computer music interface.
           OpenNeuro ds002720.
           https://doi.org/10.18112/openneuro.ds002720.v1.0.1
    """

    nemar_id = "ds002720"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=19,
            channel_types={"eeg": 19},
            montage="standard_1020",
            reference="FCz",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=18, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=20.0,
            study_design=(
                "Tempo-based brain-computer music interface with alpha "
                "neurofeedback. Right-hand kinesthetic ball-squeeze imagery "
                "(lower alpha, increase tempo) vs relaxation (raise alpha, "
                "decrease tempo). 9 runs per subject (run 1 calibration, "
                "runs 2-9 alternating 20 s binary trials)."
            ),
            feedback_type="continuous",
            stimulus_type="auditory music tempo",
            stimulus_modalities=["audio"],
            primary_modality="audio",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.18112/openneuro.ds002720.v1.0.1",
            investigators=[
                "Ian Daly",
                "Nicoletta Nicolaou",
                "Duncan Williams",
                "Faustina Hwang",
                "Alexis Kirke",
                "Eduardo Miranda",
                "Slawomir J. Nasuto",
            ],
            institution="University of Reading",
            country="GB",
            data_url="https://openneuro.org/datasets/ds002720",
            publication_year=2018,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=8,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=20.0,
        ),
        file_format="EDF (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Daly2020",
            interval=[0, 20],
            paradigm="imagery",
            doi="10.18112/openneuro.ds002720.v1.0.1",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_path_search_params(self, subject):
        """Zero-padded subject numbers (sub-01, not sub-1); EDF only."""
        out = {"extensions": [".edf"]}
        if subject is not None:
            out["subjects"] = f"{subject:02d}"
        return out

    def _get_single_subject_data(self, subject):
        """Split the 9 BIDS ``task-runN`` files into runs of one session.

        The BIDS ``task`` entity encodes the run (``run1`` .. ``run9``); there
        is no BIDS session/run entity, so the default loader would collapse all
        files onto a single key. Here the calibration run (empty events) is
        dropped and the remaining runs are re-indexed ``"0"`` .. ``"7"``.
        """
        bids_paths = self.bids_paths(subject)

        def _run_number(bids_path):
            # task entity is "runN"; extract the trailing integer.  # codespell:ignore
            return int("".join(c for c in bids_path.task if c.isdigit()))

        ordered = sorted(bids_paths, key=_run_number)

        result = {}
        runs = {}
        run_idx = 0
        for bids_path in ordered:
            if _run_number(bids_path) == _CALIBRATION_RUN:
                continue  # calibration run has an empty events file
            raw = self._read_raw_bids(bids_path)

            # Remap numeric annotation descriptions ("1"/"2") to class names.
            desc = raw.annotations.description.astype(np.dtype("<25U"))
            for code, name in _VALUE_TO_NAME.items():
                desc[desc == code] = name
            raw.annotations.description = desc

            runs[str(run_idx)] = stim_channels_with_selected_ids(raw, self.event_id)
            run_idx += 1

        result["0"] = runs
        return result

    def _read_raw_bids(self, bids_path):
        import mne_bids

        raw = mne_bids.read_raw_bids(
            bids_path, extra_params=self._get_read_extra_params(None), verbose=False
        )
        raw.load_data(verbose=False)
        return raw

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the subject's BIDS files from OpenNeuro S3, return BIDS root."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        bids_root = Path(get_dataset_path("Daly2020", path)) / "MNE-daly2020-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        subj_str = f"sub-{subject:02d}"
        rel_files = []
        for run in range(1, _N_RUNS + 1):
            stem = f"{subj_str}/eeg/{subj_str}_task-run{run}"
            rel_files += [
                f"{stem}_eeg.edf",
                f"{stem}_eeg.json",
                f"{stem}_channels.tsv",
                f"{stem}_events.tsv",
                f"{stem}_events.json",
            ]

        for rel_path in rel_files:
            self._download_file(bids_root, rel_path, force_update)

        self._ensure_dataset_description(bids_root)
        return str(bids_root)

    @staticmethod
    def _download_file(bids_root, rel_path, force_update):
        local_path = bids_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and not force_update:
            return
        url = f"{_S3_BASE}/{rel_path}"
        log.info("Downloading %s ...", rel_path)
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            log.warning("Not found: %s (skipping)", url)
            return
        resp.raise_for_status()
        with open(local_path, "wb") as fout:
            for chunk in resp.iter_content(chunk_size=8192):
                fout.write(chunk)

    @staticmethod
    def _ensure_dataset_description(bids_root):
        dd_path = bids_root / "dataset_description.json"
        if not dd_path.exists():
            with open(dd_path, "w") as f:
                json.dump(_DATASET_DESCRIPTION, f, indent=2)
