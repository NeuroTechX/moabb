"""Combined SSVEP dataset: Single stimulus location for two inputs.

Wang et al. (2021), European Journal of Neuroscience.
DOI: 10.1111/ejn.15030
"""

from pathlib import Path

import mne

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)
from .utils import FIGSHARE_DL_URL


_RAR_FILE_ID = "25000886"  # Raw data.rar


class Wang2021Combined(BaseDataset):
    """Combined SSVEP dataset with single stimulus location for two inputs.

    Dataset from [1]_.

    This dataset uses a one-to-two SSVEP paradigm where spatially overlapping
    stimuli are presented at the same location. Users attend to one of four
    arrow directions (up, down, left, right), each flickering at a distinct
    frequency derived from the 85 Hz CRT refresh rate.

    8 subjects are available in the Figshare archive across two experiments:
    - Experiment 1 (P1-P5): Scheme 1 (moving dot-formed arrows)
    - Experiment 2 (P6-P8): Scheme 2 (space-based attention) and
      Scheme 3 (object-based attention)

    EEG was recorded at 1000 Hz with a 32-channel ANT Neuro system.
    Each trial consists of 500 ms fixation, 500 ms cue, and 5000 ms
    stimulation.

    Warnings
    --------
    The data archive is in RAR format and requires the ``rarfile`` package
    for extraction (``pip install rarfile``). The ``unrar`` system utility
    must also be installed.

    The raw data on Figshare is stored as .cnt files (ANT Neuro format)
    inside nested RAR archives. The archive is labeled "(not all)" and
    may not contain complete data for all 20 subjects. Subject numbering
    is based on the alphabetical order of the subject name directories
    found after extraction.

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
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.12628471",
            license="CC BY 4.0",
            publication_year=2021,
        ),
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
            raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose=False)

            # Rename annotation descriptions from trigger codes to frequencies
            raw.annotations.rename(self._TRIGGER_MAP)

            runs[str(run_idx)] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = "WANG2021COMBINED"
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if .cnt files are already extracted
        cnt_files = self._find_subject_cnt_files(data_dir, subject)
        if cnt_files and not force_update:
            return cnt_files

        # Download the outer RAR archive
        rar_path = dl.data_dl(
            f"{FIGSHARE_DL_URL}{_RAR_FILE_ID}", sign, path, force_update, verbose
        )

        # Extract nested RARs
        try:
            import rarfile
        except ImportError:
            raise ImportError(
                "The Wang2021Combined dataset requires the 'rarfile' package. "
                "Install it with: pip install rarfile\n"
                "You also need the 'unrar' utility installed on your system."
            )

        data_dir.mkdir(parents=True, exist_ok=True)

        # Extract outer RAR (contains "Raw data/Exp1 (not all).rar" etc.)
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(data_dir)

        # Extract inner RARs
        raw_data_dir = data_dir / "Raw data"
        if raw_data_dir.is_dir():
            for inner_rar in raw_data_dir.glob("*.rar"):
                with rarfile.RarFile(str(inner_rar)) as rf:
                    rf.extractall(data_dir)

        # Find subject's .cnt files
        cnt_files = self._find_subject_cnt_files(data_dir, subject)
        if not cnt_files:
            raise FileNotFoundError(
                f"Could not find .cnt files for subject {subject} after "
                f"extraction. Available subjects may be limited -- the "
                f"Figshare archive is labeled '(not all)'."
            )

        return cnt_files

    def _find_subject_cnt_files(self, data_dir, subject):
        """Map numeric subject ID to extracted .cnt files.

        Subjects are mapped by alphabetical order of the unique subject
        name prefixes found in the extracted directories (Exp1 and Exp2).
        """
        all_cnt = sorted(data_dir.rglob("*.cnt"))
        if not all_cnt:
            return []

        # Extract unique subject names from filenames like "name_1_1.cnt"
        subject_names = set()
        for p in all_cnt:
            # Pattern: subjectname_scheme_run.cnt
            parts = p.stem.rsplit("_", 2)
            if len(parts) >= 2:
                subject_names.add(parts[0])

        subject_names = sorted(subject_names)

        if subject > len(subject_names):
            return []

        target_name = subject_names[subject - 1]
        cnt_files = sorted(
            [str(p) for p in all_cnt if p.stem.startswith(target_name + "_")]
        )
        return cnt_files
