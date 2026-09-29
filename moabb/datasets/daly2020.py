"""Daly2020 tempo-based BCMI motor imagery dataset (BCMI-MIdAS, OpenNeuro ds002720)."""

import logging
from pathlib import Path

import numpy as np

from ._openneuro_mirror import (
    OpenNeuroMirrorMixin,
    drop_native_stim,
    relabel_annotations,
    write_dataset_description,
)
from .base import BaseBIDSDataset
from .download import data_dl, get_dataset_path
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

_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds002720"

# events.tsv trial_type 1 = "Lower alpha": right-hand ball-squeeze imagery (mu/alpha
# ERD; verified against sub-01_task-run2_events.json and the README);
# 2 = "Raise alpha": relax.
_EVENTS = {"right_hand": 1, "relax": 2}
_VALUE_TO_NAME = {str(code): name for name, code in _EVENTS.items()}

# 19 scalp EEG channels (international 10-20, FCz reference).
_CH_NAMES = "FP1 FP2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2".split()

# 9 runs per subject; run 1 is calibration with an EMPTY events file.
_N_RUNS = 9
_CALIBRATION_RUN = 1
_RUN_SUFFIXES = "eeg.edf eeg.json channels.tsv events.tsv events.json".split()


class Daly2020(OpenNeuroMirrorMixin, BaseBIDSDataset):
    """Tempo-based BCMI motor imagery dataset from Daly et al. 2018 [1]_.

    Recorded at the University of Reading during development of a tempo-based
    brain-computer music interface (BCMI-MIdAS). 18 healthy participants
    raised the music tempo by imagining squeezing a ball in their right hand
    (**right_hand**, event 1, "Lower alpha") and lowered it by relaxing
    (**relax**, event 2, "Raise alpha"). Each subject has 9 runs of
    alternating 20 s trials; the calibration run 1 (empty events file) and the
    targetless runs sub-05/run2, sub-15/run3 and sub-17/runs2-3 are skipped,
    and the rest are exposed as runs ``"0"``, ``"1"``, ... of session ``"0"``.

    .. note::
       The README states 19 participants and CC-BY-4.0, but the release
       contains 18 subjects and ``dataset_description.json`` says CC0; this
       loader follows the release. The summary table reports the designed 72
       trials per class; retained counts can be lower.

    References
    ----------
    .. [1] Daly, I., Nicolaou, N., Williams, D., Hwang, F., Kirke, A.,
           Miranda, E., & Nasuto, S. J. (2018). A dataset recorded during
           development of a tempo-based brain-computer music interface.
           OpenNeuro ds002720.
           https://doi.org/10.18112/openneuro.ds002720.v1.0.1
    """

    nemar_id = "on002720"
    nemar_subject_template = "{subject:02d}"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
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
        """Split the BIDS ``task-runN`` files into runs of one session.

        The run lives in the BIDS ``task`` entity (no session/run entity), so
        the default loader would collapse all files onto one key.
        """

        def _run_number(bids_path):
            return int("".join(c for c in bids_path.task if c.isdigit()))

        ordered = sorted(self.bids_paths(subject), key=_run_number)
        run_numbers = [_run_number(p) for p in ordered]
        if len(run_numbers) != len(set(run_numbers)):
            raise ValueError("Daly2020 has duplicate task/run files")

        runs = {}
        for bids_path in ordered:
            if _run_number(bids_path) == _CALIBRATION_RUN:
                continue
            raw = drop_native_stim(self._read_raw_bids(bids_path))
            relabel_annotations(raw, _VALUE_TO_NAME)
            if not np.isin(raw.annotations.description, list(self.event_id)).any():
                log.warning(
                    "Skipping Daly2020 subject %02d %s: no motor-imagery events",
                    subject,
                    bids_path.task,
                )
                continue
            runs[str(len(runs))] = stim_channels_with_selected_ids(raw, self.event_id)

        if not runs:
            raise ValueError(
                f"Daly2020 subject {subject} has no usable motor-imagery runs."
            )
        return {"0": runs}

    def _read_raw_bids(self, bids_path):
        import mne_bids

        raw = mne_bids.read_raw_bids(
            bids_path, extra_params=self._get_read_extra_params(None), verbose=False
        )
        raw.load_data(verbose=False)
        return raw

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the subject's BIDS files from OpenNeuro S3, return BIDS root."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        if mirror_root is not None:
            return mirror_root

        bids_root = Path(get_dataset_path("Daly2020", path)) / "MNE-daly2020-data"
        bids_root.mkdir(parents=True, exist_ok=True)
        subj_str = f"sub-{subject:02d}"
        for run in range(1, _N_RUNS + 1):
            stem = f"{subj_str}/eeg/{subj_str}_task-run{run}"
            for suffix in _RUN_SUFFIXES:
                rel_path = f"{stem}_{suffix}"
                data_dl(
                    f"{_S3_BASE}/{rel_path}",
                    "Daly2020",
                    path=path,
                    force_update=force_update,
                    verbose=verbose,
                    fname=rel_path,
                )
        write_dataset_description(
            bids_root,
            "A dataset recorded during development of a tempo-based "
            "brain-computer music interface",
            "1.0.2",
            "10.18112/openneuro.ds002720.v1.0.1",
            self.METADATA.documentation.investigators,
        )
        return str(bids_root)
