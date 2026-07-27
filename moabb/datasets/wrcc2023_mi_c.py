"""WRCC2023 MI-C three-class motor imagery dataset (World Robot Contest 2023)."""

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


# "MI-C dataset of the BCI competition WRCC2023" (doi:10.7910/DVN/G8FBHH), the
# three-class motor-imagery track of the World Robot Contest 2023 BCI-Controlled
# Robot Contest. One MATLAB v5 ``.mat`` file per subject (subject1.mat ..
# subject8.mat). Files carry no persistent id, so they are addressed by their
# numeric Harvard Dataverse datafile id.
WRCC2023_MI_C_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# subject -> Dataverse datafile id, resolved from the dataset files API.
WRCC2023_MI_C_FILE_IDS = {
    1: 10358845,
    2: 10358849,
    3: 10358843,
    4: 10358842,
    5: 10358844,
    6: 10358847,
    7: 10358846,
    8: 10358848,
}

# 59 EEG channels (indices 1-59) of the Neuracle NeuSen W 64-channel system used
# in the World Robot Contest MI tracks, in acquisition order. The ``.mat`` files
# store only the numeric signal array with no channel names; these names/order
# follow the identical WRC/SHU Neuracle 59-EEG montage documented for this rig.
WRCC2023_MI_C_CHANNELS = [
    "Fpz",
    "Fp1",
    "Fp2",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "Fz",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "FCz",
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "FC5",
    "FC6",
    "FT7",
    "FT8",
    "Cz",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "T7",
    "T8",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "TP7",
    "TP8",
    "Pz",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "POz",
    "PO3",
    "PO4",
    "PO5",
    "PO6",
    "PO7",
    "PO8",
    "Oz",
    "O1",
    "O2",
]

# Class codes stored in the per-trial ``label`` array (1 x 90, int32).
WRCC2023_MI_C_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3}


class WRCC2023_MI_C(BaseDataset):
    """Three-class motor imagery dataset from the World Robot Contest 2023 (MI-C).

    .. admonition:: Dataset summary

        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        WRCC2023_MI_C      8       59           3                 30            4s          1000 Hz            1
        ===========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings distributed as the "MI-C" (three-class
    motor imagery) dataset of the BCI competition held at the 2023 World Robot
    Contest / BCI-Controlled Robot Contest. Eight participants performed a cued
    three-class motor-imagery task: imagined left-hand movement, imagined
    right-hand movement, and imagined bipedal (foot) movement. Two of the
    participants are stroke patients and the remainder are healthy individuals.

    Each subject provides 90 trials (30 left hand, 30 right hand, 30 feet). Trials
    are stored pre-epoched as a 4 s window (4000 samples at 1000 Hz) over 59 EEG
    channels arranged according to the international 10-20/10-10 system (channels
    1-59 of a Neuracle 64-channel montage; the discarded channels 60-64 carried
    ECG/EOG). Trial order is randomised; the class of every trial is given by a
    data-borne integer ``label`` array, not by acquisition order.

    The data are distributed as MATLAB ``.mat`` files (one per subject) on Harvard
    Dataverse, each containing a ``data`` array of shape
    ``(n_trials, n_channels, n_samples)`` = ``(90, 59, 4000)`` in volts, an integer
    ``label`` vector of length 90 (1 = left hand, 2 = right hand, 3 = feet) and a
    scalar ``fs`` giving the 1000 Hz sampling rate. Signals are raw (DC-coupled,
    unfiltered): a slowly varying per-channel offset of a few millivolts rides
    under the microvolt-scale EEG.

    References
    ----------

    .. [1] WRCC2023 (2024). MI-C dataset of the BCI competition WRCC2023.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/G8FBHH

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=59,
            channel_types={"eeg": 59},
            montage="standard_1005",
            reference=None,
            ground=None,
            hardware="Neuracle NeuSen W 64-channel wireless EEG (channels 1-59 EEG)",
            line_freq=50.0,
            sensors=list(WRCC2023_MI_C_CHANNELS),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            species="homo sapiens",
            health_status="mixed: healthy individuals and stroke patients (2 stroke)",
            clinical_population="stroke",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trials_per_class={"left_hand": 30, "right_hand": 30, "feet": 30},
            synchronicity="cue-based",
            mode="offline",
            events=dict(WRCC2023_MI_C_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/G8FBHH",
            description="Three-class (left hand, right hand, feet) motor imagery "
            "EEG dataset from the 2023 World Robot Contest BCI competition (MI-C "
            "track); 8 subjects including 2 stroke patients, 90 trials each.",
            institution="World Robot Contest (BCI-Controlled Robot Contest)",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/G8FBHH",
            publication_year=2024,
            license="CC0-1.0",
            repository="Harvard Dataverse",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        file_format="mat",
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 8 + 1)),
            sessions_per_subject=1,
            events=dict(WRCC2023_MI_C_EVENTS),
            code="WRCC2023-MI-C",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.7910/DVN/G8FBHH",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path of a single subject's ``.mat`` file."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        file_id = WRCC2023_MI_C_FILE_IDS[subject]
        url = f"{WRCC2023_MI_C_BASE_URL}{file_id}"
        local = dl.data_dl(url, self.code, path, force_update, verbose)
        return [local]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        file_path = self.data_path(subject)[0]
        raw = self._mat_to_raw(file_path)
        return {"0": {"0": raw}}

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one subject ``.mat`` file into a continuous :class:`mne.io.RawArray`.

        The stored epochs ``(n_trials, n_channels, n_samples)`` are concatenated
        along time; a stim channel marks each trial's cue onset (t = 0) with its
        class code (1 = left hand, 2 = right hand, 3 = feet).
        """
        mat = sio.loadmat(file_path, squeeze_me=True)
        sfreq = float(np.atleast_1d(mat["fs"]).ravel()[0])

        data = np.asarray(mat["data"], dtype=float)  # (n_trials, n_channels, n_samples)
        labels = np.atleast_1d(np.asarray(mat["label"]).ravel()).astype(int)

        n_trials, n_channels, n_samples = data.shape
        # Concatenate trials along time: (n_channels, n_trials * n_samples).
        # Values are already in volts (raw, DC-coupled), so no unit rescaling.
        cont = np.transpose(data, (1, 0, 2)).reshape(n_channels, n_trials * n_samples)
        # Prepend one zero sample so the first trial's cue is not at sample 0,
        # where mne.find_events cannot detect a rising edge (would drop trial 1).
        pad = 1
        cont = np.concatenate([np.zeros((n_channels, pad)), cont], axis=1)

        stim = np.zeros((1, cont.shape[1]))
        for trial_idx, label in enumerate(labels[:n_trials]):
            onset = pad + trial_idx * n_samples
            stim[0, onset] = label

        full = np.vstack([cont, stim])
        mne_info = create_info(
            ch_names=list(WRCC2023_MI_C_CHANNELS) + ["STI 014"],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels + ["stim"],
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
