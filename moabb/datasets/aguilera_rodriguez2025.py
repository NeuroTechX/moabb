"""Imagined Speech EEG dataset comparing paradigm designs.

Aguilera-Rodriguez et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-05926-5
Data DOI: 10.17632/57g8z63tmy.1
"""

import logging
import shutil
from pathlib import Path

import mne
import numpy as np

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


log = logging.getLogger(__name__)

_SIGN = "aguilerarodriguez2025"
_SFREQ = 500.0
_DOI = "10.1038/s41597-025-05926-5"

# mBrainTrain Smarting 24-channel layout (FCz reference, Fpz ground).
# Channel 1-24 in the EDF map to these names in order.
# fmt: off
_CH_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz", "AFz", "CPz", "POz", "M1", "M2",
]
# fmt: on

# OpenViBE annotation labels -> imagined speech words.
# The paper uses 4 Spanish directional words.
_ANNOT_MAP = {
    "OVTK_StimulationId_Label_01": "avanzar",
    "OVTK_StimulationId_Label_02": "retroceder",
    "OVTK_StimulationId_Label_03": "derecha",
    "OVTK_StimulationId_Label_04": "izquierda",
}

# Mendeley Data API for file listing.
_MENDELEY_API = "https://data.mendeley.com/api/datasets/57g8z63tmy/files?version=1"


class AguileraRodriguez2025(BaseDataset):
    """Imagined Speech EEG dataset comparing paradigm designs.

    Dataset from Aguilera-Rodriguez et al. [1]_, published in
    Scientific Data.

    **Dataset Description**

    Fifteen participants (8 male, 7 female, ages 18-27) performed
    imagined speech of four Spanish directional words: "avanzar"
    (advance), "retroceder" (backwards), "derecha" (right),
    "izquierda" (left).

    Two paradigms were used:

    - **Traditional** (session 0): Cue-based design built with
      OpenViBE. EEG stored as EDF files with annotation markers.
    - **Gamified** (session 1): Video-game (Pac-man maze) design
      built with Pygame/LSL. EEG stored as XDF files.

    EEG was recorded at 500 Hz with 24 channels using mBrainTrain
    Smarting (FCz reference, Fpz ground). Each paradigm has 120
    trials (30 per word).

    .. note::
        Only the traditional paradigm (EDF) is loaded by default.
        The gamified paradigm uses XDF format which requires ``pyxdf``.

    References
    ----------
    .. [1] Aguilera-Rodriguez, E., Cuevas-Romero, A., Mendoza-Franco, S.,
           Wornovitzky-Green, J., Rivera-Cerros, E., Villanueva-Cazares, D.,
           Munoz-Ubando, L. A., Ibarra-Zarate, D., & Alonso-Valerdi, L. M.
           (2025). An EEG-based Imagined Speech Database for comparing
           Paradigm Designs. Scientific Data, 12, 1644.
           https://doi.org/10.1038/s41597-025-05926-5
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=24,
            channel_types={"eeg": 24},
            hardware="mBrainTrain Smarting (Belgrade, Serbia)",
            reference="FCz",
            ground="Fpz",
            sensors=list(_CH_NAMES),
            line_freq=60.0,
            sensor_type="EEG",
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"male": 8, "female": 7},
            age_min=18,
            age_max=27,
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"avanzar": 1, "retroceder": 2, "derecha": 3, "izquierda": 4},
            paradigm="imagery",
            n_classes=4,
            class_labels=["avanzar", "retroceder", "derecha", "izquierda"],
            trial_duration=11.8,
            study_design=(
                "Comparison of traditional cue-based vs gamified (Pac-man) "
                "paradigms for imagined speech BCI. Traditional paradigm: "
                "visual+auditory cue with 5 beeps at T=1.4s rhythm, subject "
                "imagines speech for 7 repetitions, last 3 extracted for "
                "analysis. 2s rest between trials."
            ),
            stimulus_type="visual + auditory cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "Visual word cue + auditory beep at T=1.4s rhythm. "
                "Subject imagines pronouncing the word at each beep. "
                "Continues for 2 more repetitions after beeps stop."
            ),
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Edgar Aguilera-Rodriguez",
                "Alma Cuevas-Romero",
                "Santiago Mendoza-Franco",
                "Jonathan Wornovitzky-Green",
                "Eduardo Rivera-Cerros",
                "David Villanueva-Cazares",
                "Luis Alberto Munoz-Ubando",
                "David Ibarra-Zarate",
                "Luz Maria Alonso-Valerdi",
            ],
            institution="Tecnologico de Monterrey",
            institution_department="Escuela de Ingenieria y Ciencias",
            institution_address=(
                "Ave. Eugenio Garza Sada 2501, Monterrey, N.L., 64849, Mexico"
            ),
            country="MX",
            data_url="https://data.mendeley.com/datasets/57g8z63tmy/1",
            publication_year=2025,
            license="CC-BY-NC-ND-4.0",
            repository="Mendeley Data",
            senior_author="Luz Maria Alonso-Valerdi",
            associated_paper_doi="10.1038/s41597-025-05926-5",
            keywords=[
                "imagined speech",
                "EEG",
                "brain-computer interface",
                "gamified paradigm",
                "biomedical engineering",
                "Spanish",
            ],
            description=(
                "EEG-based imagined speech database comparing traditional "
                "cue-based and gamified (Pac-man) paradigms. 4 Spanish "
                "directional words. Ethics: CONBIOETICA-19-CEI-011-20161017. "
                "Paper reports 32.48% (traditional) and 35.65% (gamified) "
                "accuracy with Random Forest."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["avanzar", "retroceder", "derecha", "izquierda"],
            cue_duration_s=1.4,
            imagery_duration_s=9.8,
        ),
        data_structure=DataStructureMetadata(
            n_trials=1800,
            n_trials_per_class={
                "avanzar": 450,
                "retroceder": 450,
                "derecha": 450,
                "izquierda": 450,
            },
            trials_context=("15 subjects x 120 trials (30 per class). Session ~32 min."),
        ),
        data_processed=False,
        file_format="EDF",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events={"avanzar": 1, "retroceder": 2, "derecha": 3, "izquierda": 4},
            code="AguileraRodriguez2025",
            interval=[0, 4],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fpath = self.data_path(subject)

        raw = mne.io.read_raw_edf(fpath, preload=True, verbose="ERROR")

        # Rename channels from generic "Channel N" to actual electrode names.
        rename_map = {}
        for i, name in enumerate(_CH_NAMES):
            old_name = f"Channel {i + 1}"
            if old_name in raw.ch_names:
                rename_map[old_name] = name
        raw.rename_channels(rename_map)

        # Drop gyroscope channels if present.
        gyro_chs = [ch for ch in raw.ch_names if ch.startswith("Gyro")]
        if gyro_chs:
            raw.drop_channels(gyro_chs)

        # Set montage (M1/M2 won't be found in standard montage, which is fine).
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        # Remap annotation descriptions to word labels.
        if raw.annotations is not None and len(raw.annotations) > 0:
            new_desc = []
            for desc in raw.annotations.description:
                mapped = _ANNOT_MAP.get(desc)
                if mapped is not None:
                    new_desc.append(mapped)
                else:
                    new_desc.append("BAD_" + desc)
            raw.annotations.description = np.array(new_desc)

        return {"0": {"0": raw}}

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

        edf_file = base / f"S{subject}.edf"
        if edf_file.exists() and not force_update:
            return str(edf_file)

        # Check alternate locations (manual download).
        mne_data = Path(dl.get_dataset_path(_SIGN, path))
        alt_paths = [
            mne_data
            / "mendeley_imagined_speech_traditional_gamified"
            / "traditional_raw",
            mne_data / "imagined_speech_eeg_paradigms" / "raw_traditional",
        ]
        for alt in alt_paths:
            src = alt / f"S{subject}.edf"
            if src.exists():
                shutil.copy2(str(src), str(edf_file))
                return str(edf_file)

        # Download from Mendeley Data API.
        log.info("Downloading subject %d from Mendeley Data...", subject)
        import requests

        resp = requests.get(_MENDELEY_API, timeout=30)
        if resp.ok:
            target_name = f"S{subject}.edf"
            for finfo in resp.json():
                fname = finfo.get("filename", "")
                if fname == target_name:
                    download_url = finfo.get("content_details", {}).get(
                        "download_url", ""
                    )
                    if download_url:
                        downloaded = dl.data_dl(
                            download_url,
                            _SIGN,
                            path=path,
                            force_update=force_update,
                            verbose=verbose,
                        )
                        downloaded = Path(downloaded)
                        if downloaded != edf_file:
                            shutil.move(str(downloaded), str(edf_file))
                        return str(edf_file)

        if not edf_file.exists():
            raise FileNotFoundError(
                f"Could not find or download {edf_file}. "
                "Download manually from https://data.mendeley.com/datasets/57g8z63tmy/1"
            )
        return str(edf_file)
