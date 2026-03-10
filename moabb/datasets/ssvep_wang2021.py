"""Combined SSVEP dataset: Single stimulus location for two inputs.

Wang et al. (2021), European Journal of Neuroscience.
DOI: 10.1111/ejn.15030
"""

import zipfile
from pathlib import Path

import mne

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from .utils import safe_extract_zip


_ZENODO_RECORD = "18873228"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"


class Wang2021Combined(BaseDataset):
    """Combined SSVEP dataset with single stimulus location for two inputs.

    Dataset from [1]_.

    This dataset uses a one-to-two SSVEP paradigm where spatially overlapping
    stimuli are presented at the same location. Users attend to one of four
    arrow directions (up, down, left, right), each flickering at a distinct
    frequency derived from the 85 Hz CRT refresh rate.

    8 subjects are available across two experiments:

    - Experiment 1 (S01-S05): Scheme 1 (moving dot-formed arrows)
    - Experiment 2 (S06-S08): Scheme 2 (space-based attention) and
      Scheme 3 (object-based attention)

    EEG was recorded at 1000 Hz with a 32-channel ANT Neuro system.
    Each trial consists of 500 ms fixation, 500 ms cue, and 5000 ms
    stimulation.

    Warnings
    --------
    The Zenodo archive used by this adapter contains only 8 of the 20
    subjects described in the paper. Subject numbering (S01-S08) is based
    on alphabetical order of the original subject name prefixes.

    References
    ----------
    .. [1] L. Wang, Z. Zhang, D. Han, Z. Zhang, Z. Liu, and W. Liu,
       "Single stimulus location for two inputs: A combined brain-computer
       interface based on Steady-State Visual Evoked Potential (SSVEP),"
       European Journal of Neuroscience, vol. 53, no. 3, pp. 861-875, 2021.
       DOI: 10.1111/ejn.15030
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1005",
            hardware="eego mylab (ANT Neuro)",
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="healthy",
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=4,
            trial_duration=5.0,
            stimulus_type="overlapping SSVEP arrows (CRT 85 Hz)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            study_design="One-to-two combined SSVEP with overlapping stimuli",
            task_type="covert_attention",
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1111/ejn.15030",
            investigators=[
                "Lu Wang",
                "Zhenhao Zhang",
                "Dan Han",
                "Zhijun Zhang",
                "Zhifang Liu",
                "Wei Liu",
            ],
            senior_author="Zhijun Zhang",
            institution="Shandong University",
            country="CN",
            repository="Zenodo",
            data_url=_ZENODO_BASE.rsplit("/files", 1)[0],
            license="CC BY 4.0",
            publication_year=2021,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[14.17, 12.14, 9.44, 7.73],
        ),
        data_structure=DataStructureMetadata(
            n_blocks=2,
        ),
        bci_application=BCIApplicationMetadata(
            environment="lab",
            online_feedback=False,
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        file_format="CNT",
    )

    _events = {
        "14.17": 1,  # Up arrow (85/6 Hz)
        "12.14": 2,  # Down arrow (85/7 Hz)
        "9.44": 3,  # Right arrow (85/9 Hz)
        "7.73": 4,  # Left arrow (85/11 Hz)
    }

    # Map .cnt trigger codes to frequency-based event labels
    _TRIGGER_MAP = {
        "11": "14.17",  # Direction 1 (up) -> 85/6 Hz
        "21": "12.14",  # Direction 2 (down) -> 85/7 Hz
        "31": "9.44",  # Direction 3 (right) -> 85/9 Hz
        "41": "7.73",  # Direction 4 (left) -> 85/11 Hz
    }

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events=self._events,
            code="Wang2021Combined",
            interval=[0.0, 5.0],
            paradigm="ssvep",
            doi="10.1111/ejn.15030",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject from .cnt files.

        Each subject has 2 .cnt files (runs). Data is read with MNE and
        returned as separate runs within a single session.

        Annotations are renamed from trigger codes (11, 21, 31, 41) to
        frequency strings (14.17, 12.14, 9.44, 7.73) matching _events.
        """
        cnt_files = self.data_path(subject)

        runs = {}
        for run_idx, cnt_path in enumerate(cnt_files):
            # read_raw_cnt is used instead of read_raw_ant because the ANT
            # reader's C library (libEep) segfaults on macOS.
            raw = mne.io.read_raw_cnt(cnt_path, preload=False, verbose=False)

            # Rename annotation descriptions from trigger codes to frequencies
            raw.annotations.rename(self._TRIGGER_MAP)

            runs[str(run_idx)] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subject_dir = data_dir / f"S{subject:02d}"

        # Return cached files if already extracted
        cnt_files = sorted(subject_dir.glob("*.cnt")) if subject_dir.is_dir() else []
        if cnt_files and not force_update:
            return [str(f) for f in cnt_files]

        # Download per-subject ZIP from Zenodo
        zip_name = f"S{subject:02d}.zip"
        url = f"{_ZENODO_BASE}/{zip_name}"
        zip_path = dl.data_dl(url, sign, path, force_update, verbose)

        # Extract .cnt files into subject directory
        subject_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            safe_extract_zip(zf, subject_dir)

        cnt_files = sorted(subject_dir.glob("*.cnt"))
        if not cnt_files:
            raise FileNotFoundError(
                f"No .cnt files found for subject {subject} in {subject_dir}"
            )
        return [str(f) for f in cnt_files]
