"""Polo-Hortiguela 2025 lower-limb motor imagery dataset."""

import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

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


# Zenodo record 14672334 -- one zip per subject and per exoskeleton condition.
POLOHORTIGUELA2025_BASE = "https://zenodo.org/api/records/14672334/files/{fname}/content"

# Conditions (exoskeleton models) map to sessions.
CONDITIONS = {"0static": "STATIC", "1motion": "MOTION"}

# 28 EEG channel labels, in the row order of ``data_EEG`` (rows 1-28).
EEG_CHANNELS = [
    "AF3",
    "F3",
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "C5",
    "C3",
    "C1",
    "Cz",
    "CP3",
    "CP1",
    "CPz",
    "P3",
    "Pz",
    "PO3",
    "AF4",
    "F4",
    "FC2",
    "FC4",
    "C2",
    "C4",
    "C6",
    "CP2",
    "CP4",
    "P4",
    "POz",
    "PO4",
]
# 4 EOG channels (rows 29-32) and 3 inertial/accelerometer channels (rows 33-35).
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

    .. admonition:: Dataset summary

        ================  =======  =======  ==========  =================  ============  ===============  ===========
        Name              #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate    #Sessions
        ================  =======  =======  ==========  =================  ============  ===============  ===========
        PoloHortiguela2025      6       28           2                 22             4s           250Hz            2
        ================  =======  =======  ==========  =================  ============  ===============  ===========

    Open-loop EEG dataset [1]_ recorded from six healthy participants while they
    performed kinesthetic motor imagery of ankle movements (dorsiflexion and
    plantarflexion) alternated with relaxation, combined with a lower-limb
    exoskeleton. Two exoskeleton models were tested and are exposed here as two
    sessions: a **static** model, in which the exoskeleton stays stationary, and
    a **motion** model, in which the exoskeleton assists the participant by
    performing the plantarflexion/dorsiflexion movements.

    EEG was recorded with 28 electrodes over the sensorimotor cortex plus 4 EOG
    and 3 inertial (accelerometer) channels at 250 Hz. Each subject and model
    contains 11 repetitions (runs). Every repetition is a continuous recording
    following the sequence: 15 s baseline, 15 s relaxation (rest), 28 s motor
    imagery, 15 s relaxation (rest), 5 s exoskeleton return. This loader exposes
    a 2-class problem, ``rest`` (relaxation) versus ``motor_imagery``, extracting
    the onsets of the relaxation and motor-imagery phases as annotations.

    References
    ----------

    .. [1] Polo-Hortiguela, C., Ortiz, M., Ianez, E., & Azorin, J. M. (2025).
       EEG Signal Dataset During Dorsiflexion and Plantar Flexion Movements.
       Zenodo. https://doi.org/10.5281/zenodo.14672334

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            n_channels=28,
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
            institution="Universitas Miguel Hernandez de Elche",
            country="ES",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.14672334",
            license="CC-BY-4.0",
            publication_year=2025,
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

    def _zip_fname(self, subject, condition):
        return f"B{subject:02d}_S1_{condition}.zip"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return, for a single subject, the local folders of the extracted
        static and motion zips (one entry per condition)."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for condition in CONDITIONS.values():
            fname = self._zip_fname(subject, condition)
            url = POLOHORTIGUELA2025_BASE.format(fname=fname)
            path_zip = Path(
                dl.data_dl(url, self.code, path=path, force_update=force_update)
            )
            # The Zenodo "/content" endpoint names every downloaded file
            # "content", so the extracted folder must be derived from the
            # archive's own top-level directory (the zip name without ".zip",
            # e.g. "Bxx_S1_STATIC"), NOT from path_zip.stem (which is "content").
            extract_dir = path_zip.parent / Path(fname).stem  # .../Bxx_S1_STATIC
            if not extract_dir.is_dir():
                archive_matches = False
                try:
                    with z.ZipFile(path_zip, "r") as zip_ref:
                        # Only extract when the cached archive actually holds this
                        # condition's folder (the shared "content" name can leave
                        # a different condition's archive cached here).
                        archive_matches = any(
                            n.startswith(extract_dir.name + "/")
                            for n in zip_ref.namelist()
                        )
                        if archive_matches:
                            zip_ref.extractall(path_zip.parent)
                except BadZipFile:
                    warnings.warn(
                        "Corrupted zip file detected, re-downloading...", stacklevel=2
                    )
                    path_zip.unlink(missing_ok=True)

                # Every Zenodo URL ends in "/content", so data_dl can return a
                # valid cached zip for another subject or condition. Refresh
                # when the archive does not contain this expected top-level
                # folder, rather than silently returning an empty session.
                if not archive_matches:
                    path_zip = Path(
                        dl.data_dl(url, self.code, path=path, force_update=True)
                    )
                    with z.ZipFile(path_zip, "r") as zip_ref:
                        if not any(
                            n.startswith(extract_dir.name + "/")
                            for n in zip_ref.namelist()
                        ):
                            raise FileNotFoundError(
                                f"{path_zip} does not contain {extract_dir.name}"
                            )
                        zip_ref.extractall(path_zip.parent)
            paths.append(str(extract_dir))
        return paths

    def _make_raw(self, mat_file):
        """Build a continuous mne.Raw from one repetition .mat file."""
        mat = loadmat(mat_file, struct_as_record=False, squeeze_me=True)["session"]

        # data_EEG is (35, n_samples): 28 EEG + 4 EOG + 3 inertial, in microvolts.
        data = np.asarray(mat.data_EEG, dtype=float)
        ch_names = EEG_CHANNELS + EOG_CHANNELS + INERTIAL_CHANNELS
        ch_types = ["eeg"] * 28 + ["eog"] * 4 + ["misc"] * 3

        # microvolts -> volts for the EEG/EOG channels (mne expects SI units).
        data = data.copy()
        data[:32, :] *= 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

        # Build annotations from the sample-wise task code channel.
        task = np.asarray(mat.task_EEG).ravel()
        onsets, durations, descriptions = [], [], []
        for codes, label in ((REST_CODES, "rest"), (MI_CODES, "motor_imagery")):
            # The archive uses a condition-specific code (STATIC/MOTION) for
            # the same protocol phase.  Match both variants rather than
            # silently emitting targetless MOTION raws.
            mask = np.isin(task, codes)
            if not mask.any():
                continue
            # Segment starts: where the mask turns on.
            starts = np.where(np.diff(np.concatenate(([0], mask.astype(int)))) == 1)[0]
            ends = np.where(np.diff(np.concatenate((mask.astype(int), [0]))) == -1)[0]
            for s, e in zip(starts, ends):
                onsets.append(s / SFREQ)
                durations.append((e - s + 1) / SFREQ)
                descriptions.append(label)

        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )
        raw.set_annotations(annotations)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        condition_dirs = self.data_path(subject)  # [static_dir, motion_dir]
        sessions = {}
        for session_key, extract_dir in zip(CONDITIONS.keys(), condition_dirs):
            extract_dir = Path(extract_dir)
            mat_files = sorted(extract_dir.glob("*.mat"))
            runs = {}
            for run_idx, mat_file in enumerate(mat_files):
                runs[str(run_idx)] = self._make_raw(mat_file)
            # Skip a condition with no data on disk (the Zenodo "/content"
            # naming can leave only one condition cached); an empty session
            # would otherwise trigger "No objects to concatenate" downstream.
            if runs:
                sessions[session_key] = runs
        return sessions
