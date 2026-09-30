"""Lioi XP1 EEG-fMRI motor imagery / neurofeedback dataset (OpenNeuro ds002336)."""

import logging
from pathlib import Path

import mne
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets._openneuro_mirror import OpenNeuroMirrorMixin
from moabb.datasets.base import BaseDataset
from moabb.datasets.lioi2020_xp2 import _AUTHORS, _CH_NAMES
from moabb.datasets.metadata.schema import (
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
from moabb.datasets.utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds002336"

# Motor-imagery-vs-rest runs in acquisition order; the ``task-motorloc``
# motor-execution localizer is excluded. MIpre/MIpost are absent for sub 1.
_TASKS = ["MIpre", "eegNF", "fmriNF", "eegfmriNF", "MIpost"]

# BrainVision block markers; R128 (fMRI volume) and the rest never match event_id.
_MARKER_TO_LABEL = {"Stimulus/S 99": "rest", "Stimulus/S  2": "right_hand"}

_EVENTS = {"rest": 1, "right_hand": 2}


class LioiXP1(OpenNeuroMirrorMixin, BaseDataset):
    """XP1 simultaneous EEG-fMRI motor imagery / neurofeedback dataset [1]_ [2]_.

    Ten healthy subjects performed right-hand kinaesthetic motor imagery
    inside an MR scanner in 20 s rest / task blocks. Five runs are exposed as
    runs of one session: ``MIpre`` and ``MIpost`` (no feedback, 5 task blocks
    each), ``eegNF``, ``fmriNF`` and ``eegfmriNF`` (neurofeedback, 10 task
    blocks each); the ``task-motorloc`` motor-execution localizer is
    excluded. Only the EEG is loaded (fMRI ignored); it is raw and still
    contains MR gradient and ballistocardiogram artifacts. Channel ``ECG`` is
    typed ``ecg``.

    .. note::
       The original study [2]_ reports that ``MI_pre``/``MI_post`` could not
       be acquired for two of the ten participants and that the EEG of those
       runs was lost for a third; the loader skips any run missing from the
       release (subject 1 is known to lack ``MIpre``/``MIpost``). Per-subject
       run and trial counts are therefore not uniform.

    References
    ----------
    .. [1] Lioi, G., Cury, C., Perronnet, L., Mano, M., Bannier, E.,
           Lecuyer, A., & Barillot, C. (2019). Simultaneous MRI-EEG during a
           motor imagery neurofeedback task: an open access brain imaging
           dataset for multi-modal data integration. bioRxiv 862375.
           https://doi.org/10.1101/862375
    .. [2] Perronnet, L., Lecuyer, A., Mano, M., Bannier, E., Lotte, F.,
           Clerc, M., & Barillot, C. (2017). Unimodal versus bimodal EEG-fMRI
           neurofeedback of a motor imagery task. Frontiers in Human
           Neuroscience, 11, 193. https://doi.org/10.3389/fnhum.2017.00193
    """

    nemar_id = "on002336"
    nemar_subject_template = "xp1{subject:02d}"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=5000.0,
            channel_types={"eeg": 63, "ecg": 1},
            montage="standard_1005",
            hardware="Brain Products MR-compatible 64-channel EEG",
            reference="FCz",
            ground="AFz",
            line_freq=50.0,
            sensors=list(_CH_NAMES),
            auxiliary_channels=AuxiliaryChannelsMetadata(other_physiological=["ecg"]),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"male": 8, "female": 2},
            age_min=19.0,
            age_max=39.0,
            species="human",
            ages=[25, 27, 25, 31, 39, 36, 19, 29, 27, 26],
            sexes=["male"] * 5 + ["female"] + ["male"] * 2 + ["female", "male"],
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=20.0,
            study_design=(
                "20 s block design alternating rest and right-hand kinaesthetic "
                "motor imagery, across MI-without-feedback and three "
                "neurofeedback (EEG, fMRI, bimodal) runs, acquired simultaneously "
                "with fMRI."
            ),
            feedback_type="neurofeedback",
            stimulus_type="visual (ball-to-target)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1101/862375",
            related_paper_dois=["10.3389/fnhum.2017.00193"],
            investigators=list(_AUTHORS),
            institution="Univ Rennes, Inria, CNRS, Inserm, IRISA",
            country="FR",
            data_url="https://openneuro.org/datasets/ds002336",
            publication_year=2019,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=5,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand"],
            imagery_duration_s=20.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=780,
            trials_context=(
                "Per subject: 3 NF runs x 10 MI blocks + 2 MI runs x 5 blocks "
                "= 40 right-hand MI blocks (30 for sub 1, missing MIpre/MIpost), "
                "each paired with a 20 s rest block. Total right-hand blocks 390, "
                "matched by 390 rest blocks -> 780 blocks. The original study "
                "reports MIpre/MIpost missing or lost for up to three subjects, so "
                "the released total may be lower."
            ),
        ),
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="LioiXP1",
            interval=[0, 20],
            paradigm="imagery",
            doi="10.1101/862375",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        sub = f"sub-{self._nemar_subject(subject)}"
        stems = [f"{sub}/eeg/{sub}_task-{task}_eeg" for task in _TASKS]
        if mirror_root is not None:
            vhdrs = [Path(mirror_root) / f"{stem}.vhdr" for stem in stems]
            return [str(vhdr) for vhdr in vhdrs if vhdr.is_file()]

        paths = []
        for stem in stems:
            # The .vhdr references its .eeg/.vmrk siblings by basename.
            try:
                dl.data_dl(
                    f"{_S3_BASE}/{stem}.eeg",
                    self.code,
                    path=path,
                    force_update=force_update,
                )
            except Exception as exc:  # noqa: BLE001
                # MIpre/MIpost are missing for some subjects (HTTP 404).
                log.info("Skipping absent run %s (%s)", stem, exc)
                continue
            for ext in (".vmrk", ".vhdr"):
                local = dl.data_dl(
                    f"{_S3_BASE}/{stem}{ext}",
                    self.code,
                    path=path,
                    force_update=force_update,
                )
            paths.append(local)
        return paths

    def _get_single_subject_data(self, subject):
        vhdr_paths = self.data_path(subject)
        runs = {}
        for idx, task in enumerate(_TASKS):
            match = [p for p in vhdr_paths if f"task-{task}_eeg.vhdr" in p]
            if not match:
                continue
            raw = mne.io.read_raw_brainvision(match[0], preload=True, verbose=False)
            if "ECG" in raw.ch_names:
                raw.set_channel_types({"ECG": "ecg"})
            raw.annotations.rename(
                {
                    k: v
                    for k, v in _MARKER_TO_LABEL.items()
                    if k in raw.annotations.description
                }
            )
            with mne.utils.use_log_level("error"):
                raw.set_montage(
                    make_standard_montage("colin27_1005"), on_missing="ignore"
                )
            runs[f"{idx}{task}"] = stim_channels_with_selected_ids(raw, self.event_id)
        return {"0": runs}
