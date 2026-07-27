"""Pan2025 cross-session motor imagery dataset."""

import numpy as np
import scipy.io as sio
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

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


# Harvard Dataverse "Cross-Session Motor Imagery EEG dataset" (doi:10.7910/DVN/GH74ZG).
# Each subject has two session files (D1, D2). Files carry no persistent id, so
# they are addressed by their numeric Dataverse datafile id.
PAN2025_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# subject -> {session_index: datafile_id}, resolved from the Dataverse files API.
PAN2025_FILE_IDS = {
    1: {0: 11704410, 1: 11704404},
    2: {0: 11704420, 1: 11704417},
    3: {0: 11704412, 1: 11704421},
    4: {0: 11704402, 1: 11704418},
    5: {0: 11704405, 1: 11704407},
    6: {0: 11704419, 1: 11704411},
    7: {0: 11704413, 1: 11704409},
    8: {0: 11704415, 1: 11704406},
    9: {0: 11704414, 1: 11704416},
    10: {0: 11704403, 1: 11704408},
}


class Pan2025(BaseDataset):
    """Cross-session motor imagery dataset from Pan et al. 2025.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Pan2025         10       28           2                 90            4s           250 Hz            2
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings from 10 healthy subjects performing a
    cued left- vs right-hand motor imagery task across two experimental sessions
    (recorded on two separate days, D1 and D2), designed to study cross-session
    variability in brain-computer interface applications.

    Each session provides 180 trials (90 left-hand, 90 right-hand). Signals were
    recorded from 28 EEG channels covering the sensorimotor cortex (FC, C, CP and
    P rows of the 10-10 system) at a sampling rate of 250 Hz. Each stored epoch
    spans -1.5 s to 4.0 s relative to the imagery cue (1375 samples); the motor
    imagery task window is 0 to 4 s.

    The data are distributed as MATLAB ``.mat`` files (one per subject and session)
    on Harvard Dataverse, each containing a ``data`` array of shape
    ``(n_channels, n_samples, n_trials)``, an integer ``label`` vector (1 =
    left hand, 2 = right hand), and an ``Info`` struct with the channel names
    (``chaninfo``), the sampling rate (``fs``) and the epoch period.

    References
    ----------

    .. [1] Pan, Lincong (2025). Cross-Session Motor Imagery EEG dataset.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/GH74ZG

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=28,
            channel_types={"eeg": 28},
            montage="10-10",
            reference=None,
            ground=None,
            sensors=[
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
            ],
        ),
        participants=ParticipantMetadata(n_subjects=10, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 90, "right_hand": 90},
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/GH74ZG",
            description="Cross-session motor imagery EEG dataset from 10 subjects "
            "performing cued left- vs right-hand motor imagery across two sessions.",
            investigators=["Lincong Pan"],
            institution="Tianjin University",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/GH74ZG",
            publication_year=2025,
            license="CC0-1.0",
            repository="Harvard Dataverse",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10 + 1)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2},
            code="Pan2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.7910/DVN/GH74ZG",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of a single subject's two session files."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for session in (0, 1):
            file_id = PAN2025_FILE_IDS[subject][session]
            url = f"{PAN2025_BASE_URL}{file_id}"
            local = dl.data_dl(url, self.code, path, force_update, verbose)
            paths.append(local)
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        file_paths = self.data_path(subject)

        sessions = {}
        for session_idx, file_path in enumerate(file_paths):
            raw = self._mat_to_raw(file_path)
            sessions[str(session_idx)] = {"0": raw}
        return sessions

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one ``.mat`` session file into a continuous :class:`mne.io.RawArray`.

        Epoched trials ``(n_channels, n_samples, n_trials)`` are concatenated along
        time; a stim channel marks each trial's cue onset (t = 0) with its class
        code (1 = left hand, 2 = right hand).
        """
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        info_struct = mat["Info"]
        sfreq = float(info_struct.fs)
        period = np.asarray(info_struct.period, dtype=float)
        ch_names = [str(c) for c in np.atleast_1d(info_struct.chaninfo)]

        data = np.asarray(mat["data"], dtype=float)  # (n_channels, n_samples, n_trials)
        labels = np.atleast_1d(np.asarray(mat["label"]).ravel()).astype(int)

        n_channels, n_samples, n_trials = data.shape
        # Concatenate trials along time: (n_channels, n_trials * n_samples).
        cont = np.transpose(data, (0, 2, 1)).reshape(n_channels, n_trials * n_samples)
        # Convert from microvolts to volts.
        cont = cont * 1e-6

        # Sample offset of the cue (t = 0) within each epoch.
        cue_offset = int(round((0.0 - period[0]) * sfreq))

        stim = np.zeros((1, cont.shape[1]))
        for trial_idx, label in enumerate(labels[:n_trials]):
            onset = trial_idx * n_samples + cue_offset
            stim[0, onset] = label

        full = np.vstack([cont, stim])
        mne_info = create_info(
            ch_names=list(ch_names) + ["STI 014"],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels + ["stim"],
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
