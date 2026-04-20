"""
Inner Speech EEG dataset (Nieto et al. 2022).
Scientific Data DOI: 10.1038/s41597-022-01147-2
OpenNeuro: ds003626
"""

import mne

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


_DOI = "10.1038/s41597-022-01147-2"
_OPENNEURO_URL = "https://s3.amazonaws.com/openneuro.org/ds003626/"
_SFREQ = 1024.0
_EVENTS = {
    "pronounced/up": 2131,
    "pronounced/down": 2132,
    "pronounced/right": 2133,
    "pronounced/left": 2134,
    "inner/up": 2231,
    "inner/down": 2232,
    "inner/right": 2233,
    "inner/left": 2234,
    "visualized/up": 2331,
    "visualized/down": 2332,
    "visualized/right": 2333,
    "visualized/left": 2334,
}


def _speech_hed(label, modality):
    """Build HED tags for Nieto 2022."""
    action = "Speak" if modality == "pronounced" else "Imagine"
    return (
        f"(Sensory-event, Experimental-stimulus, Visual-presentation), "
        f"(Agent-action, ({action}, Speak, (Word, (Label/{label}))))"
    )


class Nieto2022(BaseDataset):
    """Inner Speech EEG dataset (Nieto et al., 2022).

    Dataset containing 10 subjects performing four directional tasks (Up, Down,
    Right, Left) across three modalities: Inner Speech, Pronounced Speech,
    and Visualized Condition.

    Recorded using a BioSemi ActiveTwo system with 128 EEG channels and
    8 external EXG channels.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=136,
            channel_types={"eeg": 128, "emg": 8},
            montage="biosemi128",
            hardware="BioSemi ActiveTwo high resolution biopotential measuring system",
            software="ActiView",
            filters={"lowpass": 208.0},
            line_freq=50.0,
            sensor_type="EEG/EMG",
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"female": 4, "male": 6},
            age_min=24,
            age_max=56,
            handedness="right-handed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=_EVENTS,
            paradigm="imagery",
            n_classes=4,
            class_labels=["Arriba/Up", "Abajo/Down", "Derecha/Right", "Izquierda/Left"],
            trial_duration=4.5,
            study_design=(
                "Four mental tasks (up, down, right, left) performed in three conditions: "
                "inner speech, pronounced speech, and visualized condition. Each trial "
                "includes concentration (0.5s), cue (0.5s), action (2.5s), and relax (1s) intervals."
            ),
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            hed_tags={
                k: _speech_hed(k.split("/")[-1], k.split("/")[0]) for k in _EVENTS.keys()
            },
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Nicolás Nieto",
                "Victoria Peterson",
                "Hugo Leonardo Rufiner",
                "Juan Esteban Kamienkowski",
                "Ruben Spies",
            ],
            institution="CIMEC (UNL-CONICET) / sinc(i) (UNL-CONICET)",
            country="AR",
            publication_year=2022,
            license="CC-BY-4.0",
            data_url="https://openneuro.org/datasets/ds003626",
            repository="OpenNeuro",
            contact_info=["nnieto@sinc.unl.edu.ar"],
            associated_paper_doi=_DOI,
            keywords=["inner speech", "EEG", "BCI", "Spanish", "pronounced speech"],
            description=(
                "A multi-modal dataset for inner speech recognition containing EEG recordings "
                "from 10 subjects performing four directional tasks in three conditions: "
                "inner speech, pronounced speech, and visualized condition."
            ),
        ),
        data_structure=DataStructureMetadata(
            n_trials_per_class={
                "pronounced/up": 30,
                "pronounced/down": 30,
                "pronounced/right": 30,
                "pronounced/left": 30,
                "inner/up": 60,
                "inner/down": 60,
                "inner/right": 60,
                "inner/left": 60,
                "visualized/up": 30,
                "visualized/down": 30,
                "visualized/right": 30,
                "visualized/left": 30,
            },
            trials_context="More than 5600 total trials across all subjects and sessions.",
        ),
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["Arriba", "Abajo", "Derecha", "Izquierda"],
            imagery_duration_s=2.5,
        ),
        data_processed=False,
        file_format="BDF",
    )

    def __init__(self, subjects=None, sessions=None):
        if sessions is None:
            sessions = [1, 2, 3]
        self.sessions = sessions

        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=3,
            events=_EVENTS,  # Pass only the event names (strings)
            code="Nieto2022",
            interval=[1.0, 3.5],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self.events = _EVENTS

    def _get_single_subject_data(self, subject):
        """Read .bdf files and handle 136-channel montage with state-tracking triggers."""
        sessions = {}

        # 1. Define triggers and labels
        RUN_TRIGGERS = {21: "pronounced", 22: "inner", 23: "visualized"}
        DIRECTION_TRIGGERS = [31, 32, 33, 34]
        DIR_LABELS = {31: "up", 32: "down", 33: "right", 34: "left"}

        # Iterate through the 3 session files
        for session_idx in range(1, 4):
            session_name = str(session_idx)
            file_path = self.data_path(subject, session=session_idx)

            # Load the raw BioSemi BDF
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)

            # Find events from the specified stimulus channel
            events = mne.find_events(raw, verbose=False)

            # 2. State-tracking loop to find and rename valid trials
            current_run_prefix = None
            valid_onsets = []
            valid_descriptions = []

            for i in range(len(events)):
                event_code = events[i, 2]

                # Track the current modality state (21, 22, or 23)
                if event_code in RUN_TRIGGERS:
                    current_run_prefix = event_code

                # Combine triggers: Prefix + Direction (e.g., 22 + 31 = 2231)
                elif event_code in DIRECTION_TRIGGERS and current_run_prefix is not None:
                    modality_str = RUN_TRIGGERS[current_run_prefix]
                    direction_str = DIR_LABELS[event_code]

                    # Calculate onset in seconds
                    onset = (events[i, 0] - raw.first_samp) / raw.info["sfreq"]

                    # Append only the specific trial format: 'modality/direction'
                    valid_onsets.append(onset)
                    valid_descriptions.append(f"{modality_str}/{direction_str}")

            # 3. Apply the filtered annotations to the raw object
            raw.set_annotations(
                mne.Annotations(
                    onset=valid_onsets,
                    duration=[0.0] * len(valid_onsets),
                    description=valid_descriptions,
                )
            )

            # 4. Final configuration: Montage and EXG types
            montage = mne.channels.make_standard_montage("biosemi128")
            raw.set_montage(montage, on_missing="ignore")
            raw.drop_channels(["Status"])
            # Set EXG channels to 'emg'
            ch_types = {ch: "emg" for ch in raw.ch_names if "EXG" in ch}
            raw.set_channel_types(ch_types)

            sessions[session_name] = {"0": raw}

        return sessions

    def data_path(
        self,
        subject,
        session=None,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        subj_str = f"sub-{subject:02d}"
        ses_str = f"ses-{session:02d}"
        filename = f"{subj_str}_{ses_str}_task-innerspeech_eeg.bdf"
        url = f"{_OPENNEURO_URL}{subj_str}/{ses_str}/eeg/{filename}"
        return dl.data_dl(url, "NIETO2022", path=path, force_update=force_update)
