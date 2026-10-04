"""MILimbEEG motor and motor-imagery limb dataset (Asanza et al., 2023)."""

import glob
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)

from .utils import download_and_extract_subject_zip


# Mendeley Data record 10.17632/x8psbz3f6x.2. The whole-record archive is served
# via the Mendeley public-api zip endpoint, which 302-redirects to a freshly
# signed S3 URL; the static cache-bucket URL is no longer publicly GETtable.
# Per-subject folders (``S1`` .. ``S60``) are not individually addressable.
MILIMBEEG_URL = "https://data.mendeley.com/public-api/zip/x8psbz3f6x/download/2"

# 16 dry electrodes (OpenBCI Cyton+Daisy) in channel order 1..16, read off the
# authors' 10-10 placement figure ``16_OpenBCI_electrodes_10_10system.png``.
CHANNELS = "FC5 F3 Fz F4 FC6 FC1 FC2 Cz T7 CP5 C3 CP1 CP2 C4 CP6 T8".split()

SFREQ = 125.0

# Task label -> event code, mirroring the authors' encoding (1..8).
EVENTS = {
    "beo": 1,  # baseline eyes open
    "clh": 2,  # closing left hand
    "crh": 3,  # closing right hand
    "dlf": 4,  # dorsal flexion left foot
    "plf": 5,  # plantar flexion left foot
    "drf": 6,  # dorsal flexion right foot
    "prf": 7,  # plantar flexion right foot
    "rest": 8,  # rest between tasks
}


class MILimbEEG(BaseDataset):
    """Motor and motor-imagery limb EEG dataset (MILimbEEG) [1]_ [2]_.

    **Dataset description**

    Over 8,680 four-second EEG recordings from 60 adult volunteers of
    Ecuadorian nationality (average age 36 years; 31 females, 29 males; three
    left-handed), recruited among ESPOL colleagues and patients of a
    neurosurgeon at the Hospital Luis Vernaza in Guayaquil, Ecuador. The cohort
    is not uniformly healthy: two participants have amputations (both upper
    limbs; right lower limb below the knee), one has hydrocephalus after a
    ventricular infarct and eighteen are post-COVID-19. EEG was acquired
    with a 16-channel OpenBCI Cyton+Daisy (dry electrodes, monopolar against a
    neutral electrode on both ear lobes) at 125 Hz, hardware band-pass filtered
    5-50 Hz with a 60 Hz notch. In each repetition (one for most subjects;
    only one subject performed up to four), participants first executed
    (``M``) and then imagined (``I``) hand closing and foot dorsal/plantar
    flexion, each task presented randomly up to five times per run, plus a
    baseline-eyes-open and rest trials. This loader exposes the
    **motor-imagery** files only, one session per repetition: the per-trial
    CSVs (microvolts, converted to volts) are concatenated into a continuous
    recording with a cue annotation at every trial onset and a non-rejecting
    ``EDGE boundary`` annotation at every join.

    .. warning::

        The Mendeley subject folders (``S1`` .. ``S60``) are not individually
        addressable; the whole-record archive is served through a
        session-signed URL and may require manual retrieval.

    References
    ----------

    .. [1] Asanza, V., Montoya, D., Lorente-Leyva, L. L., Peluffo-Ordonez,
       D. H., & Gonzalez, K. (2023). MILimbEEG: A dataset of EEG signals
       related to upper and lower limb execution of motor and motor imagery
       tasks. Data in Brief, 50, 109540.
       DOI: https://doi.org/10.1016/j.dib.2023.109540

    .. [2] Asanza, V., Montoya, D., Lorente-Leyva, L. L., Peluffo-Ordonez,
       D. H., & Gonzalez, K. (2022). MILimbEEG. Mendeley Data, V2.
       DOI: https://doi.org/10.17632/x8psbz3f6x.2

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=125.0,
            channel_types={"eeg": 16},
            montage="10-10",
            hardware="OpenBCI Cyton+Daisy",
            sensor_type="dry",
            electrode_type="dry",
            reference="neutral ear (monopolar)",
            ground=None,
            sensors=CHANNELS,
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=False, has_emg=False),
        ),
        participants=ParticipantMetadata(
            n_subjects=60,
            health_status="mixed",
            clinical_population=(
                "mostly healthy; 2 amputees (both upper limbs; right lower limb "
                "below the knee), 1 hydrocephalus after ventricular infarct, "
                "18 post-COVID-19"
            ),
            gender={"female": 31, "male": 29},
            age_mean=36.0,
            handedness={"right": 57, "left": 3},
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=8,
            class_labels=list(EVENTS.keys()),
            trial_duration=4.0,
            study_design=(
                "Upper- and lower-limb motor execution and motor imagery. Per "
                "repetition, participants first performed then imagined hand "
                "closing (left/right) and foot flexion (dorsal/plantar, "
                "left/right), with a baseline-eyes-open and rest periods."
            ),
            feedback_type="none",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events=EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.dib.2023.109540",
            description=(
                "Over 8,680 four-second EEG recordings from 60 volunteers "
                "performing and imagining upper- and lower-limb movements, "
                "recorded with a 16-channel OpenBCI Cyton+Daisy at 125 Hz."
            ),
            investigators=[
                "Victor Asanza",
                "Daniel Montoya",
                "Leandro L. Lorente-Leyva",
                "Diego H. Peluffo-Ordonez",
                "Kleber Gonzalez",
            ],
            institution="Escuela Superior Politecnica del Litoral (ESPOL)",
            country="EC",
            data_url="https://doi.org/10.17632/x8psbz3f6x.2",
            publication_year=2023,
            keywords=[
                "motor imagery",
                "motor execution",
                "EEG",
                "brain-computer interface",
                "upper limb",
                "lower limb",
                "OpenBCI",
            ],
            license="CC-BY-4.0",
            repository="Mendeley Data",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["mixed"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=True,
            preprocessing_steps=["hardware band-pass 5-50 Hz", "60 Hz notch"],
            highpass_hz=5.0,
            lowpass_hz=50.0,
            bandpass=[5.0, 50.0],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=list(EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            trials_context=(
                "Per repetition and per activity type (execution/imagery): 1 "
                "baseline-eyes-open, 5 trials each of 6 limb movements, and 31 "
                "rest trials (62 files); only the imagery files are loaded here."
            )
        ),
        file_format="CSV",
        contributing_labs=["ESPOL", "Hospital General Luis Vernaza"],
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 60 + 1)),
            sessions_per_subject=1,
            events=EVENTS,
            code="MILimbEEG",
            interval=[0.0, 4.0 - 1 / SFREQ],
            paradigm="imagery",
            doi="10.1016/j.dib.2023.109540",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        extract_dir = (
            Path(dl.get_dataset_path(self.code, path))
            / f"MNE-{self.code.lower()}-data"
            / "MILimbEEG"
        )
        if force_update or not extract_dir.exists():
            download_and_extract_subject_zip(
                MILIMBEEG_URL, self.code, extract_dir, path, force_update, verbose
            )

        # Subject folders may sit at the archive root or under a single wrapper
        # directory; resolve both layouts.
        candidates = [
            extract_dir / f"S{subject}",
            *(Path(p) for p in glob.glob(str(extract_dir / "*" / f"S{subject}"))),
        ]
        subject_dir = next((c for c in candidates if c.is_dir()), candidates[0])
        return [str(subject_dir)]

    def _read_trial(self, csv_file):
        """Read one per-trial CSV as a (n_channels, n_times) array in volts.

        Files hold an optional leading sample-index column plus the 16 electrode
        columns, x time-sample rows, in microvolts; the Mendeley v2 export ships
        a header row where the first field is empty and the remaining fields are
        the integer column indices 0..15. Peeking at the raw first line and
        trying to parse every field as a float is enough to detect that header
        (and the all-string header variant some older exports used) without
        depending on pandas' type inference.
        """
        with open(csv_file) as fh:
            first_line = fh.readline()
        try:
            [float(x) for x in first_line.rstrip("\r\n").split(",")]
            header = None
        except ValueError:
            header = 0
        frame = pd.read_csv(csv_file, header=header)
        data = frame.to_numpy(dtype=float)
        if data.shape[1] != len(CHANNELS) and data.shape[0] == len(CHANNELS):
            data = data.T
        if data.shape[1] == len(CHANNELS) + 1:
            data = data[:, 1:]
        elif data.shape[1] != len(CHANNELS):
            raise ValueError("Expected 16 EEG columns and optional sample index")
        if data.shape[0] != round(4 * SFREQ):
            raise ValueError("Expected a complete four-second stored trial")
        return data.T * 1e-6

    def _get_single_subject_data(self, subject):
        subject_dir = Path(self.data_path(subject)[0])
        # Motor-imagery files only: S<n>R<r>I<z>_<rep>.csv (excludes execution "M"),
        # grouped by repetition R<r> -> one session per repetition.
        sessions_files = {}
        for f in sorted(glob.glob(str(subject_dir / f"S{subject}R*I*_*.csv"))):
            rep = Path(f).stem.split("R", 1)[1].split("I", 1)[0]
            sessions_files.setdefault(rep, []).append(f)

        code_to_name = {v: k for k, v in EVENTS.items()}
        info = mne.create_info(
            ch_names=CHANNELS + ["STI 014"],
            sfreq=SFREQ,
            ch_types=["eeg"] * len(CHANNELS) + ["stim"],
        )
        sessions = {}
        for sess_idx, rep in enumerate(sorted(sessions_files)):
            segments, events, cursor = [], [], 0
            for f in sessions_files[rep]:
                # Task code: the digits right after the first "I" of the stem.
                label = re.match(r"\d*", Path(f).stem.split("I", 1)[1]).group()
                code = int(label) if label else None
                if code not in code_to_name:
                    continue
                trial = self._read_trial(f)
                stim = np.zeros((1, trial.shape[1]))
                stim[0, 0] = code
                segments.append(np.vstack([trial, stim]))
                events.append([cursor, 0, code])
                cursor += trial.shape[1]
            if not segments:
                continue

            raw = mne.io.RawArray(np.concatenate(segments, axis=1), info, verbose=False)
            raw.set_montage("colin27_1020", on_missing="ignore", verbose=False)
            events = np.asarray(events)
            raw.set_annotations(
                mne.annotations_from_events(
                    events, sfreq=SFREQ, event_desc=code_to_name, verbose=False
                )
                + mne.Annotations(events[1:, 0] / SFREQ, 0.0, "EDGE boundary")
            )
            sessions[str(sess_idx)] = {"0": raw}
        return sessions
