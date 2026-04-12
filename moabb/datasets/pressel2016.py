"""Imagined Speech Database (Spanish vowels and directional commands).

Pressel Coretto, Gareis, and Rufiner (2017), SIPAIM/SPIE Proceedings.
DOI: 10.1117/12.2255697
Data: Zenodo mirror (`10.5281/zenodo.19502780`). The original
distribution is on Google Drive; the Zenodo record is a faithful
per-subject re-packaging for reliable automated download.
"""

from pathlib import Path

import mne
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
from .utils import build_raw_from_epochs, download_and_extract_subject_zip


def _speech_hed(label, unit="Word", modality="Visual-presentation"):
    """Build a HED tag string for an imagined-speech event."""
    return (
        f"(Sensory-event, Experimental-stimulus, {modality}), "
        f"(Agent-action, (Imagine, Speak, ({unit}, (Label/{label}))))"
    )


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

_ZENODO_RECORD = "19502780"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"


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

    .. figure:: /_static/paper_figures/Pressel2016.png
       :alt: Pressel2016 trial structure (Fig. 1 of the SPIE paper) —
             Ready interval (2 s) → Stimulus presentation (2 s) →
             Imagine/Pronounce interval (4 s) → Rest interval (4 s).
       :width: 100%

       Figure 1 of [1]_ — trial structure (Ready, Stimulus,
       Imagine/Pronounce, Rest). Reproduced from the author postprint
       at the sinc(i)/UNL institutional repository.

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
            hed_tags={
                # Vowels (phoneme-level)
                "vowel_a": _speech_hed("a", "Phoneme"),
                "vowel_e": _speech_hed("e", "Phoneme"),
                "vowel_i": _speech_hed("i", "Phoneme"),
                "vowel_o": _speech_hed("o", "Phoneme"),
                "vowel_u": _speech_hed("u", "Phoneme"),
                # Directional commands (word-level)
                "arriba": _speech_hed("arriba"),
                "abajo": _speech_hed("abajo"),
                "adelante": _speech_hed("adelante"),
                "atras": _speech_hed("atras"),
                "derecha": _speech_hed("derecha"),
                "izquierda": _speech_hed("izquierda"),
            },
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
            license="other-open",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
            repository="Zenodo",
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

        # Filter by modality; keep artifact trials and mark them via
        # BAD_artifact annotations on the resulting Raw so downstream
        # code can decide how to handle them rather than having them
        # silently dropped at load time.
        if modality is not None:
            mask = modality_col == modality
            eeg_data = eeg_data[mask]
            stimulus_col = stimulus_col[mask]
            artifact_col = artifact_col[mask]

        n_trials = eeg_data.shape[0]
        data = eeg_data.reshape(n_trials, _N_CHANNELS, n_samples_per_ch)

        raw = build_raw_from_epochs(
            data,
            list(_CH_NAMES),
            _SFREQ,
            stimulus_col,
            montage_name="standard_1020",
            buffer_samples=100,
        )

        bad_trial_indices = np.where(artifact_col == 2)[0]
        if bad_trial_indices.size:
            events = mne.find_events(raw, stim_channel="STI", verbose=False)
            onsets = events[bad_trial_indices, 0] / raw.info["sfreq"]
            durations = np.full(bad_trial_indices.size, n_samples_per_ch / _SFREQ)
            raw.set_annotations(
                raw.annotations
                + mne.Annotations(
                    onset=onsets, duration=durations, description="BAD_artifact"
                )
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

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        subj_dir = (
            Path(dl.get_dataset_path(_SIGN, path))
            / f"MNE-{_SIGN.lower()}-data"
            / f"S{subject:02d}"
        )
        mat_file = subj_dir / f"sub-{subject:02d}_eeg.mat"
        if mat_file.exists() and not force_update:
            return str(mat_file)

        url = f"{_ZENODO_BASE}/S{subject:02d}.zip"
        download_and_extract_subject_zip(
            url, _SIGN, subj_dir, path=path, force_update=force_update, verbose=verbose
        )
        if not mat_file.exists():
            raise FileNotFoundError(
                f"Expected {mat_file} after extracting {url}, but it is "
                f"missing. The Zenodo record may have been re-packaged."
            )
        return str(mat_file)
