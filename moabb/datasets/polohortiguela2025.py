"""Polo-Hortiguela 2025 lower-limb motor imagery dataset."""

from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)

from .utils import download_and_extract_subject_zip


# Zenodo record 14672334 -- one zip per subject and per exoskeleton condition.
POLOHORTIGUELA2025_BASE = "https://zenodo.org/api/records/14672334/files/{fname}/content"

# Conditions (exoskeleton models) map to sessions.
CONDITIONS = {"0static": "STATIC", "1motion": "MOTION"}

# Row order of ``data_EEG``: 28 EEG (rows 1-28), 4 EOG (29-32), 3 inertial (33-35).
EEG_CHANNELS = (
    "AF3 F3 Fz FC3 FC1 FCz C5 C3 C1 Cz CP3 CP1 CPz P3 Pz PO3 "
    "AF4 F4 FC2 FC4 C2 C4 C6 CP2 CP4 P4 POz PO4"
).split()
EOG_CHANNELS = ["VU", "VD", "HR", "HL"]
INERTIAL_CHANNELS = ["AX", "AY", "AZ"]

# Sample-wise task codes stored in ``task_EEG``.  The condition is encoded in
# the tens digit: STATIC uses 211/311, while MOTION uses 221/321.  In both
# cases the 30-second relaxation phase is the ``21`` code and the 28-second
# motor-imagery phase is the ``31`` code.
REST_CODES = (211, 221)
MI_CODES = (311, 321)

SFREQ = 250.0


class PoloHortiguela2025(BaseDataset):
    """Motor imagery of ankle dorsiflexion/plantarflexion dataset [1]_.

    Six participants alternated kinesthetic motor imagery of ankle
    dorsiflexion/plantarflexion of the dominant foot with relaxation while
    wearing a low-cost ankle exoskeleton, in an open-loop (no control) protocol
    with auditory cues. The Zenodo description records a single recording
    session per participant; the two exoskeleton models are exposed here as
    MOABB sessions: ``static`` (the exoskeleton stays still) and ``motion`` (it
    performs plantar/dorsal flexion). Each session holds 11 continuous
    repetitions (15 s baseline, 15 s rest, 28 s motor imagery, 15 s rest, 5 s
    return); the loader annotates the ``rest`` and ``motor_imagery`` phases from
    the sample-wise task codes, accepting both the STATIC and MOTION code
    variants, and converts EEG/EOG from microvolts to volts. Signals were
    recorded at 250 Hz with 28 EEG, 4 EOG and 3 inertial channels.

    References
    ----------

    .. [1] Polo-Hortiguela, C., Ortiz, M., Ianez, E., & Azorin, J. M. (2025).
       EEG Signal Dataset During Dorsiflexion and Plantar Flexion Movements.
       Zenodo. https://doi.org/10.5281/zenodo.14672334

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    nemar_id = "nm000302"

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"eeg": 28, "eog": 4, "misc": 3},
            montage="standard_1005",
            reference=None,
            ground=None,
            sensors=EEG_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=4,
                eog_type=["vertical", "vertical", "horizontal", "horizontal"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=6, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["rest", "motor_imagery"],
            trial_duration=4.0,
            study_design="Kinesthetic motor imagery of ankle dorsiflexion/plantarflexion alternated with relaxation, with a static or motion lower-limb exoskeleton.",
            stimulus_type="auditory",
            stimulus_modalities=["audio"],
            synchronicity="cue-based",
            mode="offline",
            events={"rest": 1, "motor_imagery": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.14672334",
            description="Open-loop EEG dataset of lower-limb (ankle dorsiflexion/plantarflexion) kinesthetic motor imagery versus relaxation from six healthy participants, recorded with a static and a motion exoskeleton model.",
            investigators=[
                "Cristina Polo-Hortiguela",
                "Mario Ortiz",
                "Eduardo Ianez",
                "Jose M. Azorin",
            ],
            institution="Universidad Miguel Hernandez de Elche",
            country="ES",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.14672334",
            license="CC-BY-4.0",
            publication_year=2025,
            funding=[
                "PID2021-124111OB-C31 (MICIU/AEI/10.13039/501100011033, ERDF EU)",
                "PRE2022-103336 (MICIU/AEI/10.13039/501100011033)",
                "ValgrAI (Generalitat Valenciana, European Union)",
                "Neurokit (ICAR)",
            ],
        ),
        sessions_per_subject=2,
        runs_per_session=11,
        sessions=list(CONDITIONS.keys()),
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
        file_format="MAT",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 6 + 1)),
            sessions_per_subject=2,
            events={"rest": 1, "motor_imagery": 2},
            code="PoloHortiguela2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.14672334",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the extracted static and motion folders of a single subject."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        data_dir = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        paths = []
        for condition in CONDITIONS.values():
            stem = f"B{subject:02d}_S1_{condition}"
            if force_update or not (data_dir / stem).exists():
                # Every Zenodo "/content" URL would be cached as "content"; name it.
                download_and_extract_subject_zip(
                    POLOHORTIGUELA2025_BASE.format(fname=f"{stem}.zip"),
                    self.code,
                    data_dir,
                    path,
                    force_update,
                    verbose,
                    fname=f"{stem}.zip",
                    redownload_corrupted=True,
                )
            paths.append(str(data_dir / stem))
        return paths

    def _make_raw(self, mat_file):
        """Build a continuous mne.Raw from one repetition .mat file."""
        mat = loadmat(mat_file, struct_as_record=False, squeeze_me=True)["session"]
        # data_EEG is (35, n_samples) in microvolts; the inertial rows stay as-is.
        data = np.array(mat.data_EEG, dtype=float)
        data[:32, :] *= 1e-6
        info = mne.create_info(
            EEG_CHANNELS + EOG_CHANNELS + INERTIAL_CHANNELS,
            SFREQ,
            ["eeg"] * 28 + ["eog"] * 4 + ["misc"] * 3,
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage("colin27_1005", on_missing="ignore", verbose=False)

        task = np.asarray(mat.task_EEG).ravel()
        onsets, durations, descriptions = [], [], []
        for codes, label in ((REST_CODES, "rest"), (MI_CODES, "motor_imagery")):
            edges = np.diff(np.r_[0, np.isin(task, codes).astype(int), 0])
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                onsets.append(s / SFREQ)
                durations.append((e - s) / SFREQ)
                descriptions.append(label)
        raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        sessions = {}
        for session, folder in zip(CONDITIONS, self.data_path(subject)):
            mat_files = sorted(Path(folder).glob("*.mat"))
            # Skip a condition with no repetitions rather than an empty session.
            if mat_files:
                sessions[session] = {
                    str(run): self._make_raw(f) for run, f in enumerate(mat_files)
                }
        return sessions
