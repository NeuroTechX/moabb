"""Imagined Speech EEG dataset comparing paradigm designs.

Aguilera-Rodriguez et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-05926-5
Data DOI: 10.17632/57g8z63tmy.1
"""

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


def _speech_hed(label):
    """Build a HED tag string for an imagined-speech event (visual + auditory)."""
    return (
        "(Sensory-event, Experimental-stimulus, "
        "Visual-presentation, Auditory-presentation), "
        f"(Agent-action, (Imagine, Speak, (Word, (Label/{label}))))"
    )


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

# Stable Mendeley Data download URLs for the traditional-paradigm EDFs
# (dataset 57g8z63tmy v1). Enumerated once via
# https://data.mendeley.com/api/datasets/57g8z63tmy/files?version=1
# so the loader doesn't need a runtime API call.
# fmt: off
_SUBJECT_URLS = {
    1: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/5eb0269f-05ba-48f0-811d-277a257e8832/file_downloaded",
    2: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/513909eb-42b6-463c-b06c-e544c768f70b/file_downloaded",
    3: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/20f6008b-b862-4d2e-8698-d09a63a90ebb/file_downloaded",
    4: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/605afb13-2f54-435e-827c-5dbfa55b17c1/file_downloaded",
    5: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/b86b9355-15b4-43a9-aa9e-03cfdcda89f2/file_downloaded",
    6: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/4a86115a-d742-4a2d-8bb7-6c5e72948878/file_downloaded",
    7: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/5c48b9ea-451b-485a-a286-f5985d0f46a3/file_downloaded",
    8: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/b8e9340d-20c4-47ad-a648-217bc9fd38d3/file_downloaded",
    9: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/1810c5ea-574a-4d07-96a6-b564c74af112/file_downloaded",
    10: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/78c96afb-0642-491a-9e27-b4bdc819cd01/file_downloaded",
    11: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/b17c723e-4016-4c86-8009-aaa5f7316c20/file_downloaded",
    12: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/5207d944-0754-4a66-b699-9eec86b5428d/file_downloaded",
    13: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/cd1b5bdf-d9d6-4143-ab9d-832c0f869c9d/file_downloaded",
    14: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/89dd25bc-a900-40a3-bc60-322b800eb472/file_downloaded",
    15: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/f6216d04-0176-44a2-a961-99d03df6a077/file_downloaded",
}
# fmt: on


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

    .. figure:: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-025-05926-5/MediaObjects/41597_2025_5926_Fig1_HTML.png
       :alt: AguileraRodriguez2025 trial structure — written word cue
             + 7 imagined-speech repetitions at T=1.4 s, then 2 s rest.
       :width: 100%

       Figure 1 of [1]_ (CC-BY-NC-ND-4.0). Recommended bandpass:
       1-100 Hz — see :class:`~moabb.paradigms.SpeechImagery`.

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
            hed_tags={
                "avanzar": _speech_hed("avanzar"),
                "retroceder": _speech_hed("retroceder"),
                "derecha": _speech_hed("derecha"),
                "izquierda": _speech_hed("izquierda"),
            },
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

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        downloaded = Path(
            dl.data_dl(
                _SUBJECT_URLS[subject],
                _SIGN,
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
        )
        if downloaded.suffix == ".edf":
            return str(downloaded)
        # Mendeley URLs end in "file_downloaded" (no extension), but
        # mne.io.read_raw_edf requires a .edf suffix. Expose a
        # same-inode .edf view via a hardlink so pooch's cache stays
        # intact and subsequent calls don't re-download.
        edf_path = downloaded.with_suffix(".edf")
        if not edf_path.exists():
            edf_path.hardlink_to(downloaded)
        return str(edf_path)
