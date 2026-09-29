"""neuroTUM-BCI Cybathlon motor imagery dataset."""

import mne
import numpy as np
import yaml

from moabb.datasets import download as dl
from moabb.datasets._xdf import read_xdf
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)


# Base URL for the individual files of the Zenodo record.
NEUROTUMBCI_BASE = "https://zenodo.org/records/18087806/files/"

# Sessions available per subject (participant P1 has 5, P2 has 3).
NEUROTUMBCI_SESSIONS = {1: [1, 2, 3, 4, 5], 2: [1, 2, 3]}

# The 24 channel labels stored in the Smarting mobi EEG stream.
NEUROTUMBCI_CHANNELS = (
    "Fp1 Fp2 Fz F1 F2 F3 F4 FC1 FC2 FC3 FC4 Cz C1 C2 C3 C4 CPz CP1 CP2 Pz P3 P4 M1 M2"
).split()

# Cue-action strings (from the per-subject mapping file) to MOABB class labels.
NEUROTUMBCI_LABELS = {
    "REST": "rest",
    "LEFT HAND MI": "left_hand",
    "RIGHT HAND MI": "right_hand",
    "LEGS MI": "feet",
}


class neuroTUMBCI(BaseDataset):
    """Motor imagery dataset from the neuroTUM 2024 Cybathlon BCI [1]_, [2]_.

    Recorded by the neuroTUM student team (Technical University of Munich)
    while developing a mobile online BCI for the 2024 Cybathlon BCI race [2]_.
    Two pilots (one tetraplegic) wore a 24-channel Smarting mobi (mBrainTrain)
    streaming at 250 Hz over Bluetooth; P1 has five sessions, P2 three.

    In each arrow-cue trial a 3 s fixation cross is followed by a 1 s cue and a
    3 s black screen during which the pilot performs the cued task. Each pilot
    used three classes chosen from their own most distinguishable patterns, read
    from the per-subject ``mapping.yaml``: P1 rest / right hand / legs
    (``feet``), P2 rest / left hand / right hand. Epochs start at the
    black-screen marker; the separate calibration recordings are not loaded.

    References
    ----------
    .. [1] neuroTUM e.V. (2025). neuroTUM-BCI: Cybathlon Dataset. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.18087806

    .. [2] Tscherniak, I. W., Thiemann, N. C., McWhinnie-Fernandez, A., et al.
       (2026). Improving motor imagery decoding methods for an EEG-based mobile
       brain-computer interface in the context of the 2024 Cybathlon. arXiv.
       DOI: https://doi.org/10.48550/arXiv.2511.23384

    Notes
    -----
    XDF files are read with MOABB's built-in reader; no optional XDF
    dependency is required. Four labels describe the union of subject-specific
    three-class tasks, not a four-class task available for every participant.
    Subject 1 supplies five sessions and subject 2 supplies three; the catalog
    reports the minimum session count.

    .. versionadded:: 1.8
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            channel_types={"eeg": 24},
            montage="10-20",
            hardware="Smarting mobi (mBrainTrain), 24-channel, wireless (Bluetooth 2.1)",
            cap_manufacturer="mBrainTrain",
            cap_model="Smarting mobi",
            reference=None,
            ground=None,
            sensors=NEUROTUMBCI_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=False),
        ),
        participants=ParticipantMetadata(
            n_subjects=2,
            health_status="one tetraplegic pilot and one able-bodied participant",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["rest", "left_hand", "right_hand", "feet"],
            trial_duration=3.0,
            study_design="Arrow-cue paradigm with a subject-specific set of three "
            "mental tasks (rest, hand and/or leg motor imagery). Each trial: 3 s "
            "reset (fixation cross), 1 s directional cue, 3 s blackscreen execution, "
            "3 s reset.",
            feedback_type="none",
            stimulus_type="visual arrow/circle cues",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"rest": 1, "left_hand": 2, "right_hand": 3, "feet": 4},
            instructions="On a black screen the pilot performs the cued mental task "
            "for 3 s following a 1 s directional cue.",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.18087806",
            description="Mobile EEG motor imagery dataset from two pilots recorded "
            "during the development of the neuroTUM BCI system for the 2024 "
            "Cybathlon BCI race.",
            investigators=[
                "Isabel W. Tscherniak",
                "Niels C. Thiemann",
                "Ana McWhinnie-Fernandez",
                "Iustin Curcean",
                "Leon L. J. Jokinen",
                "Sadat Hodzic",
                "Thomas E. Huber",
                "Daniel Pavlov",
                "Manuel Methasani",
                "Pietro Marcolongo",
                "Glenn V. Krafczyk",
                "Oscar Osvaldo Soto Rivera",
                "Thien Le",
                "Flaminia Pallotti",
                "Enrico A. Fazzi",
            ],
            contact_info=["team@neurotum.com"],
            institution="Technical University of Munich / neuroTUM e.V.",
            institution_address="Munich, Germany",
            country="DE",
            data_url="https://doi.org/10.5281/zenodo.18087806",
            publication_year=2026,
            related_paper_dois=["10.48550/arXiv.2511.23384"],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=3,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        file_format="XDF",
        abstract="Two-pilot mobile EEG motor imagery dataset (24-channel Smarting "
        "mobi, 250 Hz) collected for the neuroTUM 2024 Cybathlon BCI race, using an "
        "arrow-cue paradigm with a subject-specific set of three classes (rest plus "
        "hand and/or leg motor imagery).",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=[1, 2],
            sessions_per_subject=3,
            events={"rest": 1, "left_hand": 2, "right_hand": 3, "feet": 4},
            code="neuroTUM-BCI",
            interval=(0, 3),
            paradigm="imagery",
            doi="10.5281/zenodo.18087806",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the subject's per-session XDF files, then its mapping file."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        names = [f"P{subject}_S{ses}_data.xdf" for ses in NEUROTUMBCI_SESSIONS[subject]]
        return [
            dl.data_dl(
                NEUROTUMBCI_BASE + name,
                self.code,
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
            for name in [*names, f"P{subject}_mapping.yaml"]
        ]

    def _load_mapping(self, mapping_path):
        """Parse a subject mapping file into ``{cue_string: moabb_label}``."""
        with open(mapping_path) as f:
            raw_map = yaml.safe_load(f)["mapping"]
        label_map = {}
        for cue, action in raw_map.items():
            action = str(action).strip().upper()
            if action in NEUROTUMBCI_LABELS:
                label_map[str(cue).strip()] = NEUROTUMBCI_LABELS[action]
        return label_map

    @staticmethod
    def _load_xdf(fpath):
        """Load an XDF file, tolerating recorder footer quirks."""
        streams, _ = read_xdf(fpath)
        eeg_stream = marker_stream = None
        for stream in streams:
            stype = stream["info"]["type"][0]
            if stype == "EEG":
                eeg_stream = stream
            elif stype == "Markers":
                marker_stream = stream
        if eeg_stream is None or marker_stream is None:
            raise RuntimeError(f"EEG or Marker stream not found in {fpath}")
        return eeg_stream, marker_stream

    def _get_single_subject_data(self, subject):
        """Return ``{session: {"0": Raw}}`` for every session of the subject."""
        *data_files, mapping_path = self.data_path(subject)
        label_map = self._load_mapping(mapping_path)
        info = mne.create_info(NEUROTUMBCI_CHANNELS, sfreq=250.0, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")

        sessions = {}
        for idx, fpath in enumerate(data_files):
            eeg_stream, marker_stream = self._load_xdf(fpath)

            # XDF stores the EEG in microvolts; MNE expects volts.
            data = np.asarray(eeg_stream["time_series"], dtype=float).T * 1e-6
            t_start = eeg_stream["time_stamps"][0]

            onsets, descriptions = [], []
            for stamp, sample in zip(
                marker_stream["time_stamps"], marker_stream["time_series"]
            ):
                cue = str(sample[0]).strip()
                if cue in label_map:
                    onsets.append(stamp - t_start)
                    descriptions.append(label_map[cue])

            raw = mne.io.RawArray(data, info.copy(), verbose=False)
            raw.set_montage(montage, on_missing="ignore", verbose=False)
            raw.set_annotations(mne.Annotations(onsets, 0.0, descriptions))
            sessions[str(idx)] = {"0": raw}
        return sessions
