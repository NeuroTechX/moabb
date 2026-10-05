"""P300 speller dataset from Yağan et al. 2023.

Yağan et al. (2023), Digital Signal Processing.
DOI: 10.1016/j.dsp.2023.103950
Data DOI: 10.17632/vyczny2r4w.1
"""

import logging
import re
import zipfile
from pathlib import Path

import mne

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
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

_DATASET_URL = (
    "https://data.mendeley.com/public-files/datasets/vyczny2r4w/"
    "files/d44782d1-9e3c-4cea-a556-2011fd9ab645/file_downloaded"
)

# BrainVision marker codes used in the .vmrk files (from the dataset README):
# S1-S8: intensification of columns 1-8; S9-S13: intensification of rows 1-5;
# S14: co-occurs with a row/column marker when the intensified row/column
# contains the target character; S15-S21: ISI/block/word/character boundaries.
_FLASH_CODES = [f"Stimulus/S {i:2d}" for i in range(1, 14)]
_TARGET_CODE = "Stimulus/S 14"

_N_BLOCKS = 10
_N_SUBJECTS = 18

_EVENTS = {"Target": 2, "NonTarget": 1}

# Confirmed from the released .vhdr headers: 1000 Hz, 32 EEG channels in µV
# (BrainVision IEEE_FLOAT_32), covering central and parieto-occipital areas.
_SFREQ = 1000.0
_HARDWARE = "BrainProducts actiCHamp"
# fmt: off
_CH_NAMES = [
    "FC5", "FC3", "FC1", "C5", "C3", "C1", "Cz", "CP5", "CP3", "CP1", "CPz",
    "P1", "PO7", "PO3", "O1", "FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2",
    "CP4", "CP6", "Pz", "P2", "POz", "PO4", "PO8", "Oz", "O2", "Fz",
]
# fmt: on


def _brainvision_overrides(vhdr_path):
    """Build the ``overrides`` for one released BrainVision header.

    Two released headers declare a ``MarkerFile`` belonging to another subject
    (``s12b10.vhdr`` -> ``s20b10.vmrk``, ``s8b7.vhdr`` -> ``s21b7.vmrk``), so the
    declared marker file is not in the archive. Point the reader at the block's
    own ``.vmrk`` explicitly rather than relying on the not-found fallback.
    Returns ``None`` when the header is self-consistent and needs no override.
    """
    vhdr_path = Path(vhdr_path)
    match = re.search(
        r"^MarkerFile=(.+)$", vhdr_path.read_text(errors="replace"), flags=re.MULTILINE
    )
    if match is None:
        return None
    declared = match.group(1).strip()
    if declared.lower() == "false" or (vhdr_path.parent / declared).exists():
        return None
    return {"marker_fname": vhdr_path.with_suffix(".vmrk").name}


class Yagan2023(BaseDataset):
    """P300 speller dataset from Yağan et al. 2023.

    Dataset from [1]_, distributed on Mendeley Data [2]_.

    The dataset contains EEG recordings from 18 healthy subjects performing a
    P300-based BCI speller task with the classic row-column paradigm on a
    5x8 character grid (5 rows, 8 columns). Each subject spelled 160
    characters across 10 blocks recorded in a single session; the EEG of each
    block is stored as one BrainVision file set (``s{subject}b{block}.eeg`` /
    ``.vhdr`` / ``.vmrk``).

    In each block, the rows and columns of the grid are intensified in random
    order for every spelled character. Markers S1-S8 and S9-S13 indicate
    column and row intensifications respectively; marker S14 is emitted
    together with an intensification marker when the intensified row or column
    contains the target character. This adapter labels each intensification
    as ``Target`` when it coincides with an S14 marker and as ``NonTarget``
    otherwise, which is the standard target/non-target split for P300
    spellers.

    Known recording issues documented by the authors in the dataset README:
    block 6 of subject 5 is cut during the last word (11 characters instead
    of 15), block 4 of subject 9 suffered a data overflow incident (the
    authors excluded it from their analyses; it is loaded here for
    completeness), and block 10 of subject 17 is cut during the last
    character.

    References
    ----------
    .. [1] M. Yağan, S. Musellim, S. S. Arslan, T. Çakar, N. Alp, and
       H. Ozkan, "A new benchmark dataset for P300 ERP-based BCI
       applications," Digital Signal Processing, vol. 135, p. 103950, 2023.
       DOI: 10.1016/j.dsp.2023.103950

    .. [2] M. Yağan et al., "A New Benchmark Dataset Towards Ubiquitous P300
       ERP-based BCI Applications," Mendeley Data, V1, 2022.
       DOI: 10.17632/vyczny2r4w.1

    Notes
    -----
    .. versionadded:: 1.8.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": 32},
            montage="10-20",
            hardware=_HARDWARE,
            sensors=_CH_NAMES,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(n_subjects=18, health_status="healthy"),
        experiment=ExperimentMetadata(
            paradigm="p300",
            events={"Target": 2, "NonTarget": 1},
            n_classes=2,
            class_labels=["target", "non-target"],
            stimulus_type="row-column speller",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            feedback_type="none",
            has_training_test_split=False,
            instructions=(
                "Subjects focused on a target character in a 5x8 grid while "
                "rows and columns were intensified in random order "
                "(copy-spelling, 160 characters in 10 blocks)."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.dsp.2023.103950",
            investigators=[
                "Mehmet Yağan",
                "Serkan Musellim",
                "Suayb S. Arslan",
                "Tuna Çakar",
                "Nihan Alp",
                "Huseyin Ozkan",
            ],
            senior_author="Huseyin Ozkan",
            institution="Sabanci University",
            country="TR",
            repository="Mendeley Data",
            data_url="https://doi.org/10.17632/vyczny2r4w.1",
            license="CC-BY-4.0",
            publication_year=2023,
            keywords=["P300", "ERP", "BCI", "speller", "row-column paradigm"],
        ),
        sessions_per_subject=1,
        runs_per_session=10,
        data_processed=False,
        file_format="brainvision",
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", n_targets=160
        ),
        data_structure=DataStructureMetadata(
            n_trials="160 characters per subject (10 blocks x 16 characters, "
            "with documented exceptions for s5b6, s9b4, s17b10)",
            trials_context=(
                "Each character involves repeated intensifications of all 13 "
                "rows/columns; 2 of the 13 intensifications per sequence "
                "contain the target character."
            ),
        ),
        signal_processing=SignalProcessingMetadata(),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_session"]),
        bci_application=BCIApplicationMetadata(
            environment="laboratory",
            online_feedback=False,
            applications=["communication"],
        ),
        tags=Tags(pathology=["healthy"], modality=["visual"], type=["erp"]),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, _N_SUBJECTS + 1)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="Yagan2023",
            interval=[-0.2, 1.0],
            paradigm="p300",
            doi="10.1016/j.dsp.2023.103950",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        zip_path = Path(
            dl.data_dl(
                _DATASET_URL,
                sign,
                path,
                force_update=force_update,
                verbose=verbose,
                fname="dataset.zip",
            )
        )
        extract_dir = data_dir / "extracted"
        if force_update or not extract_dir.is_dir():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        # Locate this subject's block files; ``s5b7(1).vhdr`` is a duplicate
        # export of ``s5b7`` (same data file) shipped in the archive, skip it.
        def _block_number(p):
            return int(re.search(r"b(\d+)\.vhdr$", p.name).group(1))

        vhdr_paths = sorted(
            (p for p in extract_dir.rglob(f"s{subject}b*.vhdr") if "(" not in p.name),
            key=_block_number,
        )
        if not vhdr_paths:
            raise FileNotFoundError(
                f"No BrainVision header files found for subject {subject} in {extract_dir}"
            )
        return [str(p) for p in vhdr_paths]

    @staticmethod
    def _get_single_run_data(file_path):
        """Load one block (BrainVision file set) and label Target/NonTarget flashes.

        Every row/column intensification (markers S1-S13) is relabeled
        ``Target`` if an S14 marker (target-character intensification) follows
        it within 10 ms (in the released files the S14 marker is stamped 2-3 ms
        after the associated flash), and ``NonTarget`` otherwise. All other
        markers (ISI, block/word/character boundaries) are dropped.
        """
        raw = mne.io.read_raw_brainvision(
            file_path,
            preload=True,
            overrides=_brainvision_overrides(file_path),
            verbose=False,
        )
        raw.set_montage(
            mne.channels.make_standard_montage("colin27_1020"),
            on_missing="ignore",
            verbose=False,
        )
        annots = raw.annotations

        target_onsets = sorted(
            o for o, d in zip(annots.onset, annots.description) if d == _TARGET_CODE
        )

        onsets, descriptions = [], []
        for onset, desc in zip(annots.onset, annots.description):
            if desc not in _FLASH_CODES:
                continue
            is_target = any(0.0 <= t - onset <= 0.010 for t in target_onsets)
            onsets.append(onset)
            descriptions.append("Target" if is_target else "NonTarget")

        if not onsets:
            raise ValueError(
                f"No row/column intensification markers found in {file_path}"
            )

        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=0.0, description=descriptions)
        )
        return stim_channels_with_selected_ids(raw, _EVENTS)

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        vhdr_paths = self.data_path(subject)
        runs = {}
        for block_idx, file_path in enumerate(vhdr_paths):
            try:
                runs[str(block_idx)] = self._get_single_run_data(file_path)
            except Exception as exc:  # documented corrupt/overflow blocks
                log.warning("Could not load %s: %s", file_path, exc)
        if not runs:
            raise FileNotFoundError(f"No loadable blocks found for subject {subject}")
        return {"0": runs}
