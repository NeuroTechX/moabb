"""Lu2026 chronic-stroke upper-limb motor imagery dataset."""

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
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


# HS Stroke dataset, figshare 10.6084/m9.figshare.30747806 (source EEG .mat files)
LU2026_SOURCEDATA_URL = "https://ndownloader.figshare.com/files/66772196"

# The 11 EEG channels recorded (order as stored in the .mat files)
LU2026_CHANNELS = ["FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4"]

LU2026_SFREQ = 256.0
# Each trial is 17 s: preparation (0-9 s), motor imagery (9-13 s), feedback (13-17 s)
LU2026_MI_ONSET_S = 9.0
LU2026_TRIAL_LEN_S = 17.0


class Lu2026(BaseDataset):
    """Motor imagery dataset from chronic stroke patients [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials/class      Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Lu2026          14       11           2                ~50            4s           256Hz               20
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    The HS Stroke dataset [1]_ contains motor imagery (MI) EEG recordings from
    14 chronic stroke patients performing left- and right-hand upper-limb
    (wrist-extension) motor imagery. Each participant completed 20 recording
    sessions over roughly 20 days, and each session comprises several runs. In
    every trial the participant went through three stages: a 9 s preparation
    (relaxation with sensory input), a 4 s motor imagery phase in which they
    imagined extending the left or right wrist, and a 4 s visual feedback phase
    (17 s per trial in total). Left- and right-hand trials are interleaved
    within each run.

    EEG was recorded at 256 Hz from 11 electrodes over the sensorimotor cortex
    (FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP4) placed according to the
    international 10-20 system, referenced to the linked mastoids (M1-M2 average)
    with the ground on the medial frontal cortex. Data were collected at the
    Department of Rehabilitation, Huashan Hospital, Fudan University.

    The signal files are stored in an EEG-BIDS tree
    (``sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-motorimagery_run-NN_eeg.mat``). Each
    MATLAB file holds ``EEGdata`` of shape (channels, samples, trials),
    ``EEGdatalabel`` (trial labels, 1 = left hand, 2 = right hand) and
    ``configuration_channel`` (the selected-channel mask). This loader
    reconstructs one continuous recording per run and annotates the motor
    imagery onset of every trial.

    References
    ----------

    .. [1] Lu, R., Luo, J., Zhong, S.-h., & Gao, T. (2026). HS Stroke dataset:
       an upper-limb motor imagery EEG dataset of chronic stroke patients.
       Scientific Data. DOI: https://doi.org/10.1038/s41597-026-07742-x

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=11,
            channel_types={"eeg": 11},
            montage="10-20",
            reference="linked mastoids (M1-M2 average)",
            ground="medial frontal cortex",
            sensors=list(LU2026_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="chronic stroke patients",
            clinical_population="chronic stroke patients (upper-limb hemiparesis)",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            study_design="Left- and right-hand (wrist-extension) upper-limb motor imagery. "
            "Each trial: 9 s preparation, 4 s motor imagery, 4 s visual feedback.",
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-026-07742-x",
            description="Upper-limb motor imagery EEG dataset of 14 chronic stroke "
            "patients performing left- and right-hand wrist-extension imagery over "
            "20 sessions, with paired clinical/neurophysiological measures "
            "(FMA, sEMG, MEPs).",
            investigators=["Rongrong Lu", "Jian Luo", "Sheng-hua Zhong", "Tianhao Gao"],
            institution="Huashan Hospital, Fudan University",
            institution_department="Department of Rehabilitation",
            country="CN",
            data_url="https://doi.org/10.6084/m9.figshare.30747806",
            publication_year=2026,
            license="CC-BY-4.0",
            repository="Figshare",
        ),
        sessions_per_subject=20,
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Motor Imagery"]),
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None, **kwargs):
        self.events = {"left_hand": 1, "right_hand": 2}
        super().__init__(
            subjects=list(range(1, 14 + 1)),
            sessions_per_subject=20,
            events=self.events,
            code="Lu2026",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.1038/s41597-026-07742-x",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the .mat file paths for a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            Sorted list of paths to the subject's per-run ``.mat`` files.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download the source-data archive (shared across all subjects)
        path_zip = Path(
            dl.data_dl(LU2026_SOURCEDATA_URL, self.code, force_update=force_update)
        )
        path_folder = path_zip.parent

        sub = f"sub-{subject:02d}"

        # Extract the archive once
        if not (path_folder / sub).is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                path_zip.unlink(missing_ok=True)
                path_zip = Path(
                    dl.data_dl(LU2026_SOURCEDATA_URL, self.code, force_update=True)
                )
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        subject_files = sorted((path_folder / sub).glob("ses-*/eeg/*_eeg.mat"))
        return [str(p) for p in subject_files]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{session: {run: mne.io.Raw}}`` for the subject.
        """
        file_list = self.data_path(subject)

        # Group run files by their session directory (ses-01, ses-02, ...)
        sessions_files = {}
        for file_path in file_list:
            session_name = Path(file_path).parts[-3]
            sessions_files.setdefault(session_name, []).append(file_path)

        sessions = {}
        for session_idx, session_name in enumerate(sorted(sessions_files)):
            runs = {}
            for run_idx, file_path in enumerate(sorted(sessions_files[session_name])):
                runs[str(run_idx)] = self._load_run(file_path)
            sessions[str(session_idx)] = runs

        return sessions

    def _load_run(self, file_path):
        """Build one continuous :class:`mne.io.Raw` from a run ``.mat`` file."""
        mat = loadmat(file_path, verify_compressed_data_integrity=False)

        # EEGdata: (n_channels, n_samples, n_trials) -> (n_trials, n_channels, n_samples)
        eeg = np.asarray(mat["EEGdata"], dtype=np.float64).transpose(2, 0, 1)
        labels = np.asarray(mat["EEGdatalabel"]).ravel().astype(int)

        n_trials, n_channels, n_samples = eeg.shape

        # Concatenate trials into a single continuous recording
        continuous = np.concatenate(list(eeg), axis=1)

        # Stim channel: mark each trial's motor-imagery onset with the class code
        stim = np.zeros((1, continuous.shape[1]), dtype=np.float64)
        mi_offset = int(round(LU2026_MI_ONSET_S * LU2026_SFREQ))
        for trial_idx, label in enumerate(labels):
            stim[0, trial_idx * n_samples + mi_offset] = int(label)

        data = np.concatenate([continuous, stim], axis=0)

        ch_names = list(LU2026_CHANNELS) + ["STI"]
        ch_types = ["eeg"] * n_channels + ["stim"]
        info = mne.create_info(ch_names=ch_names, sfreq=LU2026_SFREQ, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage("standard_1020", match_case=False, verbose=False)
        return raw
