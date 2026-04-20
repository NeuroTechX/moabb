"""Imagined Speech EEG dataset comparing paradigm designs.

Aguilera-Rodriguez et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-05926-5
Data DOI: 10.17632/57g8z63tmy.1
"""

from pathlib import Path

import mne
import numpy as np
from mne.utils import _soft_import

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
_CH_NAMES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "AFz",
    "CPz",
    "POz",
    "M1",
    "M2",
]

_CH_POSITIONS = {
    "Fp1": [-270.0, 860.0, 360.0],
    "Fp2": [270.0, 860.0, 360.0],
    "F3": [-470.0, 620.0, 800.0],
    "F4": [470.0, 620.0, 800.0],
    "C3": [-610.0, 0.0, 970.0],
    "C4": [610.0, 0.0, 970.0],
    "P3": [-470.0, -620.0, 800.0],
    "P4": [470.0, -620.0, 800.0],
    "O1": [-270.0, -860.0, 360.0],
    "O2": [270.0, -860.0, 360.0],
    "F7": [-670.0, 520.0, 360.0],
    "F8": [670.0, 520.0, 360.0],
    "T7": [-780.0, 0.0, 360.0],
    "T8": [780.0, 0.0, 360.0],
    "P7": [-670.0, -520.0, 360.0],
    "P8": [670.0, -520.0, 360.0],
    "Fz": [0.0, 670.0, 950.0],
    "Cz": [0.0, 0.0, 1200.0],
    "Pz": [0.0, -670.0, 950.0],
    "AFz": [0.0, 830.0, 690.0],
    "CPz": [0.0, -340.0, 1130.0],
    "POz": [0.0, -830.0, 690.0],
    "M1": [-730.0, -250.0, 0.0],
    "M2": [730.0, -250.0, 0.0],
}

_MONTAGE = mne.channels.make_dig_montage(ch_pos=_CH_POSITIONS, coord_frame="head")

# OpenViBE annotation labels -> imagined speech words.
_ANNOT_MAP = {
    "OVTK_StimulationId_Label_01": "avanzar",
    "OVTK_StimulationId_Label_02": "retroceder",
    "OVTK_StimulationId_Label_03": "derecha",
    "OVTK_StimulationId_Label_04": "izquierda",
}

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

_SUBJECT_URLS_GAMIFIED = {
    1: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/98b1e41d-ec87-4255-9703-7b4019d25ec9/file_downloaded",
    2: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/cf740440-926b-4836-9752-e3b49b8b99b3/file_downloaded",
    3: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/535cfba2-f69f-412d-9f67-e33e8554ef7a/file_downloaded",
    4: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/7d60a42d-2e71-4023-820d-6172d68fc582/file_downloaded",
    5: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/2c12579c-d365-4d20-ac54-c5eb017eea2c/file_downloaded",
    6: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/b37dbcd5-713c-4b7c-8063-6a46cb337579/file_downloaded",
    7: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/b54af8a4-8cfd-4844-879b-6f59a406097c/file_downloaded",
    8: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/144456f0-5c39-4f4a-9000-b4130e74cb4a/file_downloaded",
    9: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/04f108a8-319f-4fa4-87be-68b282d4afa7/file_downloaded",
    10: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/db236bd1-a0b2-4e7a-9f4f-8494b4f06f6a/file_downloaded",
    11: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/989e385a-0cbb-4c97-970a-5bbac0597a11/file_downloaded",
    12: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/fa4c950c-69bc-4683-b817-3e5ab4981bec/file_downloaded",
    13: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/8b36d21b-66e2-4571-bb06-4bbb3087736b/file_downloaded",
    14: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/485ccad2-e9a3-42fd-b329-c1930f7bd463/file_downloaded",
    15: "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files/646f20cd-37c0-45de-8a8b-aa8944231fbd/file_downloaded",
}


class AguileraRodriguez2025(BaseDataset):
    """Imagined Speech EEG dataset comparing paradigm designs.

    .. note::
        Session 2 (gamified paradigm) is distributed as XDF and requires the
        optional ``pyxdf`` dependency (install with ``pip install moabb[xdf]``).
        Session 1 (traditional paradigm, EDF) works without it — restrict with
        ``AguileraRodriguez2025(sessions=[1])``.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=24,
            channel_types={"eeg": 24},
            hardware="mBrainTrain Smarting",
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
                "analysis. 2s rest between trials. "
                "Gamified paradigm: The experiment started with a maze appearing on the screen, "
                "composed of white borders, progress dots, and checkpoints represented by cookies. "
                "The trial was divided into four steps: (1) movement decision, (2) imagined speech, "
                "(3) vocalized speech, and (4) character movement. The first three steps were "
                "performed each in period T=1.4s. The trial starts with white borders, the "
                "character standing still, and an auxiliary arrow pointing in the next correct "
                "direction. Based on the way the character was faced, the user must decide which "
                "word is adequate to achieve the goal (movement decision). Then, the maze borders "
                "turn green, indicating the participant must imagine pronouncing the previously "
                "decided word (imagined speech)."
            ),
            stimulus_type="visual + auditory cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
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
            publication_year=2025,
            license="CC-BY-NC-ND-4.0",
            repository="Mendeley Data",
            associated_paper_doi=_DOI,
            keywords=["imagined speech", "EEG", "gamified paradigm"],
            description="EEG-based imagined speech database comparing traditional cue-based and gamified paradigms.",
        ),
        sessions_per_subject=2,
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
            trials_context=("15 subjects x 120 trials (30 per class)."),
        ),
        data_processed=False,
        file_format="EDF",
    )

    def __init__(self, subjects=None, sessions=None):
        if sessions is None:
            sessions = [1, 2]
        self.sessions = sessions

        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=2,
            events={"avanzar": 1, "retroceder": 2, "derecha": 3, "izquierda": 4},
            code="AguileraRodriguez2025",
            interval=[0, 2],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = {}
        file_path_list = self.data_path(subject)

        for session in self.sessions:
            session_name = str(session)
            sessions[session_name] = {}
            fpath = file_path_list[self.sessions.index(session)]

            if session == 1:
                # Traditional (EDF)
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose="ERROR")
                rename_map = dict(zip(raw.ch_names, _CH_NAMES))
                gyro_chs = [ch for ch in raw.ch_names if ch.startswith("Gyro")]
                if gyro_chs:
                    raw.drop_channels(gyro_chs)
                raw.rename_channels(rename_map)
                raw.set_montage(_MONTAGE)

                if raw.annotations is not None and len(raw.annotations) > 0:
                    new_desc = []
                    for desc in raw.annotations.description:
                        mapped = _ANNOT_MAP.get(desc)
                        if mapped is not None:
                            new_desc.append(mapped)
                        else:
                            new_desc.append("BAD_" + desc)
                    raw.annotations.description = np.array(new_desc)

                sessions[session_name]["0"] = raw

            elif session == 2:
                # Gamified (XDF) — pyxdf is an optional dependency
                # (install via `pip install moabb[xdf]`).
                pyxdf = _soft_import(
                    "pyxdf",
                    "loading XDF gamified-paradigm data for AguileraRodriguez2025",
                )
                streams, _ = pyxdf.load_xdf(fpath)
                eeg_stream = None
                marker_stream = None
                for stream in streams:
                    if stream["info"]["type"][0] == "EEG":
                        eeg_stream = stream
                    elif stream["info"]["type"][0] == "Markers":
                        marker_stream = stream

                if eeg_stream is None or marker_stream is None:
                    raise RuntimeError(
                        f"EEG or Marker stream not found for subject {subject}"
                    )

                data_exp = eeg_stream["time_series"].T
                t_start = eeg_stream["time_stamps"][0]
                onsets = marker_stream["time_stamps"] - t_start
                descriptions = [str(t[0]) for t in marker_stream["time_series"]]

                new_onsets, new_descriptions = [], []
                for onset, descrp in zip(onsets, descriptions):
                    d_lower = descrp.lower()
                    if d_lower in ["avanzar", "derecha", "izquierda", "retroceder"]:
                        new_onsets.append(onset)
                        new_descriptions.append(d_lower)

                annot = mne.Annotations(
                    onset=new_onsets,
                    duration=[0] * len(new_onsets),
                    description=new_descriptions,
                )
                info = mne.create_info(_CH_NAMES, 500, "eeg")
                raw = mne.io.RawArray(data_exp, info)
                raw.set_montage(_MONTAGE)
                raw.set_annotations(annot)

                sessions[session_name]["0"] = raw

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for session in self.sessions:
            if session == 1:
                url = _SUBJECT_URLS[subject]
                ext = ".edf"
            elif session == 2:
                url = _SUBJECT_URLS_GAMIFIED[subject]
                ext = ".xdf"
            else:
                raise ValueError(f"Invalid session {session}")

            downloaded = Path(
                dl.data_dl(
                    url, _SIGN, path=path, force_update=force_update, verbose=verbose
                )
            )

            if downloaded.suffix == ext:
                paths.append(str(downloaded))
                continue

            final_path = downloaded.with_suffix(ext)
            if not final_path.exists():
                final_path.hardlink_to(downloaded)
            paths.append(str(final_path))

        return paths
