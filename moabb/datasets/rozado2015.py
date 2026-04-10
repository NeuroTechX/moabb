"""Motor imagery BCI with pupillometry augmentation.

Rozado et al. (2015), PLOS ONE.
DOI: 10.1371/journal.pone.0121262
Data DOI: 10.7910/DVN/28932
"""

import importlib
import logging
from pathlib import Path

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

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
    SignalProcessingMetadata,
    Tags,
)
from .utils import extract_rar


log = logging.getLogger(__name__)

# Harvard Dataverse access API URLs for the two RAR archives.
# File IDs obtained from the Dataverse API for DOI:10.7910/DVN/28932.
_DATAVERSE_BASE = "https://dataverse.harvard.edu/api/access/datafile/"
_RAR_FILES = {
    "userdata1.rar": f"{_DATAVERSE_BASE}2531738",
    "userdata2.rar": f"{_DATAVERSE_BASE}2531739",
}

# Subjects 1-15 are in userdata1.rar, subjects 16-30 in userdata2.rar.
_SUBJECT_TO_RAR = {}
for _s in range(1, 16):
    _SUBJECT_TO_RAR[_s] = "userdata1.rar"
for _s in range(16, 31):
    _SUBJECT_TO_RAR[_s] = "userdata2.rar"


class Rozado2015(BaseDataset):
    """Motor imagery BCI dataset with pupillometry augmentation.

    Dataset from [1]_.

    This dataset contains 32-channel EEG recorded from 30 healthy subjects
    (15 female, 15 male, ages 15-61, mean 38) using a BioSemi ActiveTwo
    system at 512 Hz. The experiment investigates a two-class motor imagery
    BCI (left hand grasping vs. rest) augmented with pupil diameter
    measurements.

    Each subject performed 1 session with 2 experiments of 25 trials each
    (50 trials total, ~25 per class). Each experiment is loaded as a
    separate run.

    Trial structure (12 s total):

    - 0 s: auditory cue ("Left" or "Nothing")
    - 0-6 s: motor imagery or rest period
    - 6 s: auditory stop cue ("Stop")
    - 6-12 s: micro-break

    Data is stored as XDF (eXtensible Data Format) files inside two RAR
    archives on Harvard Dataverse. Loading requires the ``pyxdf`` library
    (install with ``pip install moabb[xdf]``) and a RAR extraction tool
    (``unar``, ``unrar``, or ``7z``).

    27 subjects were right-handed, 3 were left-handed.

    References
    ----------
    .. [1] D. Rozado, T. Duenser, and B. Gruen, "Improving the performance
       of an EEG-based motor imagery brain computer interface using task
       evoked changes in pupil diameter," PLoS ONE, vol. 10, no. 3,
       e0121262, 2015.
       DOI: 10.1371/journal.pone.0121262
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="biosemi32",
            hardware="BioSemi ActiveTwo",
            reference="CMS/DRL",
            line_freq=50.0,
            sensor_type="active",
            electrode_material="sintered Ag/AgCl",
            cap_manufacturer="BioSemi",
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            health_status="healthy",
            gender={"male": 15, "female": 15},
            age_mean=38.0,
            age_std=9.69,
            age_min=15,
            age_max=61,
            handedness={"right": 27, "left": 3},
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            events={"left_hand": 1, "rest": 2},
            n_classes=2,
            trial_duration=6.0,
            stimulus_type="auditory cue",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            feedback_type="none",
            study_design="Motor imagery with pupillometry augmentation",
            task_type="left hand grasping imagery vs rest",
            class_labels=["left_hand", "rest"],
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0121262",
            investigators=["David Rozado", "Andreas Duenser", "Ben Howell"],
            senior_author="David Rozado",
            institution="CSIRO",
            institution_department="Digital Productivity Flagship",
            country="AU",
            repository="Harvard Dataverse",
            data_url="https://doi.org/10.7910/DVN/28932",
            license="CC0 1.0",
            publication_year=2015,
            keywords=[
                "motor imagery",
                "BCI",
                "pupillometry",
                "EEG",
                "brain-computer interface",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left hand grasping", "rest"],
            cue_duration_s=0.0,
            imagery_duration_s=6.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=2,
            block_duration_s=300.0,
            trials_context=(
                "2 experiments of 25 trials each (50 trials total per "
                "subject). Each experiment is stored as one XDF file."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["CSP", "pupil_diameter"],
            frequency_bands={"bandpass": [8.0, 30.0]},
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold", cv_folds=10, evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(environment="lab", online_feedback=False),
        tags=Tags(pathology=["healthy"], modality=["auditory"], type=["motor_imagery"]),
        file_format="XDF",
        sessions_per_subject=1,
        runs_per_session=2,
    )

    # BioSemi 32-channel layout + stim channel
    # fmt: off
    _ch_names = [
        "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
        "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
        "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
        "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
    ]
    # fmt: on

    _events = {"left_hand": 1, "rest": 2}

    # Marker strings in XDF files -> event IDs
    _MARKER_MAP = {"left": 1, "nothing": 2}

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 31)),
            sessions_per_subject=1,
            events=self._events,
            code="Rozado2015",
            interval=[0.0, 6.0],
            paradigm="imagery",
            doi="10.1371/journal.pone.0121262",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject.

        Each XDF file corresponds to one experiment (run). The two
        experiments are returned as runs "0" and "1" within session "0".
        """
        xdf_files = self.data_path(subject)
        runs = {}
        for run_idx, xdf_path in enumerate(xdf_files):
            runs[str(run_idx)] = self._load_xdf_run(xdf_path)
        return {"0": runs}

    def _load_xdf_run(self, xdf_path):
        """Load one XDF file and return a Raw object.

        XDF files contain multiple streams. We extract:
        - The EEG stream (type='EEG', 32 channels, ~512 Hz)
        - The marker stream (type='Markers') with "left", "nothing", etc.

        Events are placed on a stim channel aligned to EEG timestamps.
        """
        try:
            pyxdf = importlib.import_module("pyxdf")
        except ImportError as exc:
            raise ImportError(
                "The 'pyxdf' package is required to load XDF data for "
                "Rozado2015. Install it with `pip install moabb[xdf]`."
            ) from exc

        streams, _ = pyxdf.load_xdf(str(xdf_path))

        # Find EEG and marker streams
        eeg_stream = None
        marker_stream = None
        for s in streams:
            stream_type = s["info"]["type"][0]
            if stream_type == "EEG":
                eeg_stream = s
            elif stream_type == "Markers":
                marker_stream = s

        if eeg_stream is None:
            raise ValueError(f"No EEG stream found in {xdf_path}")
        if marker_stream is None:
            raise ValueError(f"No Marker stream found in {xdf_path}")

        # EEG data: shape (n_samples, n_channels) -> transpose to (n_channels, n_samples)
        eeg_data = eeg_stream["time_series"].T
        eeg_ts = eeg_stream["time_stamps"]
        sfreq = float(eeg_stream["info"]["nominal_srate"][0])
        n_channels = eeg_data.shape[0]

        # Use only the first 32 channels if more are present
        if n_channels > 32:
            eeg_data = eeg_data[:32, :]
            n_channels = 32

        # If fewer than 32 channels, use what's available
        ch_names_used = self._ch_names[:n_channels]

        # Build stim channel from marker timestamps
        stim = np.zeros((1, eeg_data.shape[1]))
        for marker, ts in zip(marker_stream["time_series"], marker_stream["time_stamps"]):
            label = marker[0]
            if label in self._MARKER_MAP:
                sample_idx = np.searchsorted(eeg_ts, ts)
                if 0 <= sample_idx < eeg_data.shape[1]:
                    stim[0, sample_idx] = self._MARKER_MAP[label]

        # Scale to Volts: BioSemi raw 24-bit ADC counts, LSB = 31.25 nV
        data = np.concatenate([31.25e-9 * eeg_data, stim], axis=0)

        ch_types = ["eeg"] * n_channels + ["stim"]
        ch_names_full = list(ch_names_used) + ["stim"]
        info = create_info(ch_names_full, sfreq, ch_types)
        raw = RawArray(data=data, info=info, verbose=False)
        montage = make_standard_montage("biosemi32")
        raw.set_montage(montage, on_missing="ignore")
        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return list of XDF file paths for a subject.

        Downloads the appropriate RAR archive from Harvard Dataverse and
        extracts it if needed.

        Parameters
        ----------
        subject : int
            Subject number (1-30).
        path : str | None
            Location for data storage.
        force_update : bool
            Force re-download.
        update_path : None
            Unused, kept for API compatibility.
        verbose : bool | None
            Verbosity level.

        Returns
        -------
        list of str
            Paths to XDF files for this subject.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}. Must be in {self.subject_list}")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if already extracted
        xdf_files = self._find_subject_xdf(data_dir, subject)
        if xdf_files and not force_update:
            return xdf_files

        # Determine which RAR archive contains this subject
        rar_name = _SUBJECT_TO_RAR[subject]
        url = _RAR_FILES[rar_name]

        # Download the RAR archive
        rar_path = dl.data_dl(url, sign, path, force_update, verbose)

        # Extract the RAR archive
        extract_dir = data_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s to %s", rar_name, extract_dir)
        extract_rar(rar_path, extract_dir)

        xdf_files = self._find_subject_xdf(data_dir, subject)
        if not xdf_files:
            raise FileNotFoundError(
                f"No XDF files found for subject {subject} after extracting "
                f"{rar_name}. Searched in {data_dir}"
            )
        return xdf_files

    @staticmethod
    def _find_subject_xdf(data_dir, subject):
        """Find XDF files for a subject under data_dir.

        The RAR archives extract to a directory tree like::

            extracted/<archive_id>/<subject>/exp1/experiment.xdf
            extracted/<archive_id>/<subject>/exp2/experiment.xdf

        Each subject has two experiment directories (exp1 and exp2).
        """
        if not data_dir.is_dir():
            return []

        # Primary pattern: extracted/<archive_id>/<subject>/exp*/experiment.xdf
        xdf_files = sorted(data_dir.rglob(f"{subject}/exp*/experiment.xdf"))

        if not xdf_files:
            # Fallback: try zero-padded subject number
            xdf_files = sorted(data_dir.rglob(f"{subject:02d}/exp*/experiment.xdf"))

        # Deduplicate and sort
        xdf_files = sorted(set(xdf_files))
        return [str(f) for f in xdf_files]
