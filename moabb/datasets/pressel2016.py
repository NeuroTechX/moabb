"""Imagined Speech Database (Spanish vowels and directional commands).

Pressel Coretto, Gareis, and Rufiner (2017), SIPAIM/SPIE Proceedings.
DOI: 10.1117/12.2255697
Data: Google Drive (Base de Datos Habla Imaginada).
"""

import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


log = logging.getLogger(__name__)

_SIGN = "pressel2016"
_SFREQ = 1024.0
_N_CHANNELS = 6

# fmt: off
_CH_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4"]
# fmt: on

# Stimulus codes (1-indexed) -> class labels.
# Codes 1-5: vowels, Codes 6-11: directional commands.
_STIMULUS_MAP = {
    1: "vowel_a",
    2: "vowel_e",
    3: "vowel_i",
    4: "vowel_o",
    5: "vowel_u",
    6: "arriba",
    7: "abajo",
    8: "adelante",
    9: "atras",
    10: "derecha",
    11: "izquierda",
}

# Modality codes: 1 = imagined speech, 2 = pronounced speech.
_MODALITY_IMAGINED = 1
_MODALITY_PRONOUNCED = 2

_GDRIVE_FILE_ID = "0By7apHbIp8ENZVBLRFVlSFhzbHc"
_GDRIVE_URL = (
    f"https://drive.google.com/uc?export=download&id={_GDRIVE_FILE_ID}"
    "&confirm=t"
    "&resourcekey=0-JVHv2UiRsxim41Wioro0EA"
)


class Pressel2016(BaseDataset):
    """Imagined Speech Database - Spanish vowels and commands.

    Dataset from Pressel Coretto, Gareis, and Rufiner [1]_.

    **Dataset Description**

    Fifteen Argentinian volunteers (7 female, 8 male, ages 24-28)
    performed two tasks: imagined speech and pronounced speech of
    11 stimuli (5 Spanish vowels: A, E, I, O, U; and 6 directional
    commands: arriba, abajo, adelante, atras, derecha, izquierda).

    EEG was recorded at 1024 Hz from 6 channels (F3, F4, C3, C4,
    P3, P4) using a Grass 8-18-36 amplifier with a DataTranslation
    DT9816 ADC. Signals were bandpass filtered at 2-45 Hz.

    Each trial is 4 seconds (4096 samples). Data is organized as a
    matrix where each row is a trial with 6*4096 = 24576 EEG samples
    concatenated, plus 3 label columns (modality, stimulus, artifact).

    By default, only imagined speech trials (modality=1) are loaded.
    Artifact-flagged trials (artifact=2) are excluded.

    Parameters
    ----------
    include_pronounced : bool
        If True, include pronounced speech trials as a second session.
        Default False (imagined speech only).

    References
    ----------
    .. [1] Pressel Coretto, G. A., Gareis, I. E., & Rufiner, H. L.
           (2017). Open access database of EEG signals recorded during
           imagined speech. 12th International Symposium on Medical
           Information Processing and Analysis (SIPAIM 2016), SPIE
           Proceedings, 10160.
           https://doi.org/10.1117/12.2255697
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1024.0,
            n_channels=6,
            channel_types={"eeg": 6},
            montage="standard_1020",
            hardware="Grass 8-18-36 amplifier + DataTranslation DT9816 ADC",
            sensors=list(_CH_NAMES),
            filters={"highpass": 2.0, "lowpass": 45.0},
            line_freq=50.0,
            sensor_type="EEG",
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"female": 7, "male": 8},
            age_min=24,
            age_max=28,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={v: k for k, v in _STIMULUS_MAP.items()},
            paradigm="imagery",
            n_classes=11,
            class_labels=list(_STIMULUS_MAP.values()),
            trial_duration=4.0,
            study_design=(
                "Cue-based imagined and pronounced speech of 5 Spanish "
                "vowels and 6 directional commands. Two modalities: "
                "imagined (silent) and pronounced (vocalized)."
            ),
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1117/12.2255697",
            investigators=[
                "German A. Pressel Coretto",
                "Ivan E. Gareis",
                "Hugo Leonardo Rufiner",
            ],
            institution="Universidad Nacional de Entre Rios",
            country="AR",
            publication_year=2017,
            license="Open access",
            contact_info=["germanpressel@gmail.com"],
            associated_paper_doi="10.1117/12.2255697",
            keywords=[
                "imagined speech",
                "EEG",
                "Spanish",
                "vowels",
                "directional commands",
                "open access database",
            ],
            description=(
                "Open access database of EEG signals recorded during "
                "imagined speech. 15 subjects, 6 channels, 11 classes "
                "(5 vowels + 6 directional commands). Presented at "
                "SIPAIM 2016. 83 citations."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=["Bandpass 2-45 Hz"],
            highpass_hz=2.0,
            lowpass_hz=45.0,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_STIMULUS_MAP.values()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="varies (~278-424 clean imagined per subject)",
            trials_context=(
                "15 subjects, ~50-70 trials per class before artifact "
                "rejection. 10-52% artifact rate across subjects."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    def __init__(self, include_pronounced=False, subjects=None, sessions=None):
        self._include_pronounced = include_pronounced
        n_sessions = 2 if include_pronounced else 1
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=n_sessions,
            events={v: k for k, v in _STIMULUS_MAP.items()},
            code="Pressel2016",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1117/12.2255697",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _load_eeg_mat(self, fpath, modality=None):
        """Load EEG .mat file and return Raw object.

        Parameters
        ----------
        fpath : str
            Path to the Sxx_EEG.mat file.
        modality : int or None
            If set, filter to only this modality (1=imagined, 2=pronounced).

        Returns
        -------
        raw : mne.io.RawArray
        """
        mat = loadmat(str(fpath), squeeze_me=False)

        eeg_matrix = mat["EEG"]  # (n_trials, 24579)
        n_samples_per_ch = 4096  # 4 seconds at 1024 Hz

        # Last 3 columns are labels: modality, stimulus, artifact.
        label_cols = eeg_matrix[:, -3:]
        eeg_data = eeg_matrix[:, :-3]  # (n_trials, 6*4096)

        modality_col = label_cols[:, 0].astype(int)
        stimulus_col = label_cols[:, 1].astype(int)
        artifact_col = label_cols[:, 2].astype(int)  # 1=clean, 2=artifact

        # Filter by modality and remove artifact trials (artifact_col==2).
        # BAD annotations are not reliably rejected by all paradigms, so
        # we exclude artifact trials at the data level.
        mask = np.ones(len(eeg_data), dtype=bool)
        if modality is not None:
            mask &= modality_col == modality
        mask &= artifact_col == 1  # keep only clean trials
        eeg_data = eeg_data[mask]
        stimulus_col = stimulus_col[mask]

        n_trials = eeg_data.shape[0]

        # Reshape all trials: (n_trials, 6*4096) -> (n_trials, 6, 4096).
        data = eeg_data.reshape(n_trials, _N_CHANNELS, n_samples_per_ch)

        raw = build_raw_from_epochs(
            data,
            list(_CH_NAMES),
            _SFREQ,
            stimulus_col,
            montage_name="standard_1020",
            buffer_samples=100,
        )

        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fpath = self.data_path(subject)
        sessions = {}

        # Session 0: Imagined speech.
        raw_imagined = self._load_eeg_mat(fpath, modality=_MODALITY_IMAGINED)
        sessions["0"] = {"0": raw_imagined}

        # Session 1: Pronounced speech (optional).
        if self._include_pronounced:
            raw_pronounced = self._load_eeg_mat(fpath, modality=_MODALITY_PRONOUNCED)
            sessions["1"] = {"0": raw_pronounced}

        return sessions

    def _subject_dir(self):
        path = dl.get_dataset_path(_SIGN, None)
        return Path(path) / f"MNE-{_SIGN}-data"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = self._subject_dir()
        base.mkdir(parents=True, exist_ok=True)

        mat_file = base / f"S{subject:02d}_EEG.mat"
        if mat_file.exists() and not force_update:
            return str(mat_file)

        # Check in zip file (manual download location).
        mne_data = Path(dl.get_dataset_path(_SIGN, path))
        zip_path = mne_data / "imagined_speech.zip"
        if zip_path.exists():
            with zipfile.ZipFile(str(zip_path)) as zf:
                target = (
                    f"Base de Datos Habla Imaginada/S{subject:02d}/S{subject:02d}_EEG.mat"
                )
                try:
                    with zf.open(target) as src:
                        mat_file.write_bytes(src.read())
                    return str(mat_file)
                except KeyError:
                    pass

        # Try alternate extracted location.
        mne_data = Path(dl.get_dataset_path(_SIGN, path))
        alt_extracted = (
            mne_data
            / "imagined_speech"
            / "Base de Datos Habla Imaginada"
            / f"S{subject:02d}"
            / f"S{subject:02d}_EEG.mat"
        )
        if alt_extracted.exists():
            shutil.copy2(str(alt_extracted), str(mat_file))
            return str(mat_file)

        # Attempt download via gdown (Google Drive).
        log.info("Downloading from Google Drive...")
        try:
            import gdown

            zip_dl = base / "imagined_speech.zip"
            gdown.download(id=_GDRIVE_FILE_ID, output=str(zip_dl), quiet=False)
            if zip_dl.exists():
                with zipfile.ZipFile(str(zip_dl)) as zf:
                    target = (
                        f"Base de Datos Habla Imaginada/S{subject:02d}/"
                        f"S{subject:02d}_EEG.mat"
                    )
                    with zf.open(target) as src:
                        mat_file.write_bytes(src.read())
                return str(mat_file)
        except ImportError:
            log.warning("gdown not installed. Install with: pip install gdown")
        except Exception as exc:
            log.warning("Google Drive download failed: %s", exc)

        if not mat_file.exists():
            raise FileNotFoundError(
                f"Could not find {mat_file}. Download imagined_speech.zip from "
                f"Google Drive and place in {mne_data}/"
            )
        return str(mat_file)
