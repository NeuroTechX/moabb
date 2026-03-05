"""Single-flicker online SSVEP BCI for spatial navigation.

Chen et al. (2017), PLOS ONE.
DOI: 10.1371/journal.pone.0178385
"""

import importlib
import logging
import zipfile
from pathlib import Path

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from .utils import safe_extract_zip


log = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/580485/files/single%20flicker%20SSVEP%20BCI%20raw%20data.zip"


class Chen2017SingleFlicker(BaseDataset):
    """Single-flicker online SSVEP BCI dataset.

    Dataset from [1]_.

    This dataset uses a spatially coded SSVEP paradigm where a single white
    square flickers at 15 Hz in the center of the screen. Four non-flickering
    target squares are placed at the cardinal directions (N, E, W, S). The
    user gazes at one target, producing a distinct spatial topography of the
    15 Hz SSVEP response for each direction.

    The dataset contains 32-channel EEG recorded from 12 healthy subjects
    (7 female, 5 male, mean age 23.5, range 19-32) using a BioSemi ActiveTwo
    system.

    Two sessions are available per subject:

    - **Session "0" (training)**: Structured calibration data from ``.xdf``
      files recorded at 2048 Hz. Each subject has 2 runs of 100 trials
      (50 per direction, 200 total), with ~3.5 s per trial. Requires
      ``pyxdf`` (install with ``pip install moabb[xdf]``).
    - **Session "1" (online)**: Adaptive BCI game data from ``.mat`` files
      recorded at 512 Hz. Variable-length trials from approximately 16 game
      rounds per subject.

    Both sessions use the same BioSemi ActiveTwo cap with 32 EEG channels
    (A1-A32) and biosemi32 montage. The sampling rates differ between
    sessions (2048 Hz for training, 512 Hz for online).

    Warnings
    --------
    This paradigm uses a SINGLE flicker frequency (15 Hz) with spatially-coded
    directions. Standard frequency-based SSVEP analysis (CCA, FBCCA) will NOT
    work. Use broadband spatial features or classification approaches instead.

    References
    ----------
    .. [1] J. Chen, D. Zhang, A. K. Engel, Q. Gong, and A. Maye,
       "Application of a single-flicker online SSVEP BCI for spatial
       navigation," PLoS ONE, vol. 12, no. 5, e0178385, 2017.
       DOI: 10.1371/journal.pone.0178385
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=2048.0,
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
            n_subjects=12,
            health_status="healthy",
            gender={"male": 5, "female": 7},
            age_mean=23.5,
            age_min=19,
            age_max=32,
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=4,
            stimulus_type="single-flicker spatially coded",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            feedback_type="visual",
            study_design="Spatial navigation with single 15 Hz flicker",
            task_type="spatial navigation",
            class_labels=["north", "east", "west", "south"],
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0178385",
            investigators=[
                "Jingjing Chen",
                "Dan Zhang",
                "Andreas K. Engel",
                "Qin Gong",
                "Alexander Maye",
            ],
            senior_author="Alexander Maye",
            institution="University Medical Center Hamburg-Eppendorf",
            country="DE",
            repository="Zenodo",
            data_url="https://zenodo.org/records/580485",
            license="CC BY 4.0",
            publication_year=2017,
            institution_department="Department of Neurophysiology and Pathophysiology, University Medical Center Hamburg-Eppendorf",
            ethics_approval=["Ethics committee of the medical association, Hamburg"],
            funding=[
                "DFG TRR169/B1/Z2 Crossmodal Learning",
                "Landesforschungsfoerderung Hamburg CROSS FV25",
            ],
            keywords=[
                "SSVEP",
                "BCI",
                "spatial navigation",
                "single-flicker",
                "online BCI",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[15.0],
        ),
        bci_application=BCIApplicationMetadata(
            environment="lab",
            online_feedback=True,
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        file_format="XDF/MAT",
    )

    # BioSemi 32-channel layout
    # fmt: off
    _ch_names = [
        "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
        "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
        "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
        "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
        "stim",
    ]
    # fmt: on

    _events = {"north": 1, "east": 2, "west": 3, "south": 4}

    # ASCII class codes in .mat files -> event IDs
    _CLASS_MAP = {ord("N"): 1, ord("E"): 2, ord("W"): 3, ord("S"): 4}

    # Marker strings in .xdf files -> event IDs
    _XDF_MARKER_MAP = {"N": 1, "E": 2, "W": 3, "S": 4}

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=2,
            events=self._events,
            code="Chen2017SingleFlicker",
            interval=[0.0, 3.5],
            paradigm="ssvep",
            doi="10.1371/journal.pone.0178385",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject with training and online sessions.

        Session "0" (training): XDF calibration data at 2048 Hz (requires pyxdf).
        Session "1" (online): MAT game data at 512 Hz.
        """
        file_paths = self.data_path(subject)
        sessions = {}

        # Session "0": training data from .xdf files
        xdf_files = file_paths.get("xdf", [])
        if xdf_files:
            runs = {}
            for run_idx, xdf_path in enumerate(xdf_files):
                runs[str(run_idx)] = self._load_xdf_run(xdf_path)
            sessions["0"] = runs

        # Session "1": online data from .mat files
        mat_files = file_paths.get("mat", [])
        if mat_files:
            sessions["1"] = {"0": self._load_mat_data(subject, mat_files)}

        if not sessions:
            raise FileNotFoundError(f"No data files found for subject {subject}")

        return sessions

    def _load_xdf_run(self, xdf_path):
        """Load one XDF training run and return a Raw object.

        XDF files contain a 57-channel EEG stream at 2048 Hz and a Markers
        stream with direction labels ("N", "E", "W", "S", "0").
        Channels 1:33 (A1-A32) are extracted as EEG.
        """
        try:
            pyxdf = importlib.import_module("pyxdf")
        except ImportError as exc:
            raise ImportError(
                "The 'pyxdf' package is required to load XDF training data for "
                "Chen2017SingleFlicker. Install it with `pip install moabb[xdf]`."
            ) from exc

        streams, _ = pyxdf.load_xdf(str(xdf_path))

        # Find EEG and Marker streams
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

        # EEG: shape (n_samples, 57), select channels 1:33 (A1-A32)
        eeg_data = eeg_stream["time_series"][:, 1:33].T  # -> (32, n_samples)
        eeg_ts = eeg_stream["time_stamps"]
        sfreq = 2048

        # Build stim channel from marker timestamps
        stim = np.zeros((1, eeg_data.shape[1]))
        for marker, ts in zip(marker_stream["time_series"], marker_stream["time_stamps"]):
            label = marker[0]
            if label in self._XDF_MARKER_MAP:
                sample_idx = np.searchsorted(eeg_ts, ts)
                if 0 <= sample_idx < eeg_data.shape[1]:
                    stim[0, sample_idx] = self._XDF_MARKER_MAP[label]

        # Scale to Volts and build RawArray
        data = np.concatenate([1e-6 * eeg_data, stim], axis=0)

        ch_types = ["eeg"] * 32 + ["stim"]
        info = create_info(self._ch_names, sfreq, ch_types)
        raw = RawArray(data=data, info=info, verbose=False)
        montage = make_standard_montage("biosemi32")
        raw.set_montage(montage, on_missing="ignore")
        return raw

    def _load_mat_data(self, subject, mat_files):
        """Load all .mat game rounds for one subject and return a Raw object.

        Raw .mat files contain 57-channel BioSemi data: row 0 is trigger
        (Trig1), rows 1-32 are EEG (A1-A32), rows 33-56 are external
        channels.  Only the 32 EEG channels are kept.

        Class labels are ASCII codes: N=78, E=69, S=83, W=87.
        """
        n_channels = 32
        sfreq = 512

        all_trials = []
        for mat_file in mat_files:
            mat = loadmat(str(mat_file), squeeze_me=True)
            data_struct = mat["data"]

            # .item() needed to unwrap 0-d structured array
            trials = data_struct["trial"].item()
            classes = data_struct["class"].item()

            if not hasattr(trials, "__len__"):
                trials = [trials]
                classes = [classes]

            for trial_data, trial_class in zip(trials, classes):
                if trial_data.ndim == 1:
                    continue
                # trial_data shape: (57, n_samples) -- select rows 1:33 (A1-A32)
                eeg = trial_data[1:33, :]
                n_samples = eeg.shape[1]

                # De-mean
                eeg = eeg - eeg.mean(axis=1, keepdims=True)

                # Map ASCII class code to event ID; skip unknown codes
                event_id = self._CLASS_MAP.get(int(trial_class))
                if event_id is None:
                    continue

                # Build stim channel
                stim = np.zeros((1, n_samples))
                stim[0, 0] = event_id

                # Concatenate EEG (scaled to V) + stim
                trial_with_stim = np.concatenate([1e-6 * eeg, stim], axis=0)

                # Add buffer
                buff = np.zeros((n_channels + 1, 50))
                trial_with_stim = np.concatenate([buff, trial_with_stim, buff], axis=1)
                all_trials.append(trial_with_stim)

        if not all_trials:
            raise ValueError(f"No valid trials found for subject {subject}")

        # Concatenate all trials into continuous data
        log.info(
            "Trial data de-meaned and concatenated with a buffer"
            " to create continuous data"
        )
        continuous = np.concatenate(all_trials, axis=1)

        ch_types = ["eeg"] * n_channels + ["stim"]
        info = create_info(self._ch_names, sfreq, ch_types)
        raw = RawArray(data=continuous, info=info, verbose=False)
        montage = make_standard_montage("biosemi32")
        raw.set_montage(montage, on_missing="ignore")
        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if already extracted
        mat_files = sorted(data_dir.rglob(f"{subject}_*.mat"))
        xdf_files = sorted(data_dir.rglob(f"{subject}_*.xdf"))
        if (mat_files or xdf_files) and not force_update:
            return {
                "mat": [str(f) for f in mat_files],
                "xdf": [str(f) for f in xdf_files],
            }

        # Download the zip
        zip_path = dl.data_dl(ZENODO_URL, sign, path, force_update, verbose)

        # Extract both .mat and .xdf files
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            selected = [
                member
                for member in zf.infolist()
                if member.filename.endswith(".mat") or member.filename.endswith(".xdf")
            ]
            safe_extract_zip(zf, data_dir, members=selected)

        mat_files = sorted(data_dir.rglob(f"{subject}_*.mat"))
        xdf_files = sorted(data_dir.rglob(f"{subject}_*.xdf"))
        return {"mat": [str(f) for f in mat_files], "xdf": [str(f) for f in xdf_files]}
