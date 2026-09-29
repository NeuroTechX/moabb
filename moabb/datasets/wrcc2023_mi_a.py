"""WRCC2023 MI-A three-class motor imagery dataset (World Robot Contest)."""

import numpy as np
import scipy.io as sio
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


# "MI-A dataset of the BCI competition WRCC2023" on Harvard Dataverse
# (doi:10.7910/DVN/J9JFES), the motor-imagery track of the 2023 World Robot
# Contest (WRCC) BCI Controlled Robot Contest. Files carry no persistent id, so
# they are addressed by their numeric Dataverse datafile id (one .mat per subject).
WRCC2023_MI_A_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# subject -> datafile id, resolved from the Dataverse files API for doi:10.7910/DVN/J9JFES.
WRCC2023_MI_A_FILE_IDS = {
    1: 10358829,
    2: 10358821,
    3: 10358827,
    4: 10358828,
    5: 10358824,
    6: 10358822,
    7: 10358825,
    8: 10358823,
    9: 10358826,
}

# 59 EEG channel names of the Neuracle 64-channel 10-10 cap used across the
# World Robot Contest BCI datasets (channels 1-59 EEG, 60 ECG, 61-64 EOG). The
# stored .mat files do not carry channel names; this canonical order is the same
# one used by the companion WRC/Neuracle dataset :class:`Yang2025` (WBCIC-SHU).
# fmt: off
WRCC2023_MI_A_CHANNELS = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]
# fmt: on

# Class code -> MOABB event name (data-borne in the per-trial ``label`` vector).
# 1 = left-hand grasping, 2 = right-hand grasping, 3 = foot hooking (both feet).
WRCC2023_MI_A_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3}


class WRCC2023_MI_A(BaseDataset):
    """WRCC2023 MI-A three-class motor imagery dataset.

    .. admonition:: Dataset summary

        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        WRCC2023_MI_A       9       59           3                 30            4s           1000 Hz            1
        ===========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings from 9 subjects performing a
    three-class motor imagery task, released as the "MI-A" dataset of the
    Brain-Computer Interface competition of the 2023 World Robot Contest
    (WRCC2023). The three imagined movements are left-hand grasping,
    right-hand grasping and foot (bipedal) hooking. Seven of the nine subjects
    are healthy; two are stroke patients (their identities are not disclosed by
    the depositor).

    Each subject provides 90 trials (30 left hand, 30 right hand, 30 feet).
    Signals were recorded with a 64-channel Neuracle wireless EEG system, of
    which channels 1-59 are EEG (10-10 montage), channel 60 is ECG and channels
    61-64 are EOG; only the 59 EEG channels are distributed here. The sampling
    rate is 1000 Hz and each stored epoch spans the 4 s motor-imagery period
    (4000 samples), so the imagery cue sits at t = 0 relative to every epoch.

    The data are distributed as MATLAB v5 ``.mat`` files (one per subject) on
    Harvard Dataverse, each containing a ``data`` array of shape
    ``(n_trials, n_channels, n_samples) = (90, 59, 4000)``, an integer ``label``
    vector of length 90 (1 = left hand, 2 = right hand, 3 = feet) and the
    sampling rate ``fs`` (1000). Per-trial class labels are therefore data-borne
    in the ``label`` array.

    Sample amplitudes are stored in volts. As in the companion MI-B and MI-C
    tracks, the raw DC-coupled signals include slowly varying per-channel
    offsets that are removed by the analysis paradigm's band-pass filtering.

    This dataset shares the Neuracle 64-channel hardware and the WRC
    left/right/foot motor-imagery paradigm with, but is a distinct recording
    from, :class:`Yang2025` (WBCIC-SHU, 62 subjects, BDF/BIDS on Figshare).

    References
    ----------

    .. [1] WRCC2023 (2024). MI-A dataset of the BCI competition WRCC2023.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/J9JFES

    Notes
    -----

    Each stored trial is exposed as a separate run to avoid filtering across
    discontinuities. The inclusive epoch endpoint is 3.999 s (4000 samples),
    without synthetic zero padding. Channel order and physical calibration
    follow the reconciled source loaders and still require source verification.

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 59},
            montage="10-10",
            hardware="Neuracle 64-channel wireless EEG",
            reference=None,
            ground=None,
            sensors=WRCC2023_MI_A_CHANNELS,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="mixed",
            clinical_population="stroke (2 of 9 subjects); remainder healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trials_per_class={"left_hand": 30, "right_hand": 30, "feet": 30},
            trial_duration=4.0,
            study_design="Three-class motor imagery (left-hand grasping, "
            "right-hand grasping, foot hooking). 90 trials per subject, 30 per "
            "class, each stored as a 4 s imagery epoch.",
            feedback_type="none",
            stimulus_type="cue",
            synchronicity="cue-based",
            mode="offline",
            events=dict(WRCC2023_MI_A_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/J9JFES",
            description="MI-A three-class (left hand, right hand, feet) motor "
            "imagery EEG dataset from the 2023 World Robot Contest BCI competition.",
            institution="World Robot Contest (WRCC2023) BCI competition",
            country="CN",
            data_url="https://doi.org/10.7910/DVN/J9JFES",
            publication_year=2024,
            license="CC0-1.0",
            repository="Harvard Dataverse",
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "World Robot Contest",
                "WRCC2023",
                "left hand",
                "right hand",
                "feet",
                "stroke",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=90,
        file_format="MAT (v5)",
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet"],
            imagery_duration_s=4.0,
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9 + 1)),
            sessions_per_subject=1,
            events=dict(WRCC2023_MI_A_EVENTS),
            code="WRCC2023-MI-A",
            interval=[0, 3.999],
            paradigm="imagery",
            doi="10.7910/DVN/J9JFES",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path of a single subject's ``.mat`` file."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        file_id = WRCC2023_MI_A_FILE_IDS[subject]
        url = f"{WRCC2023_MI_A_BASE_URL}{file_id}"
        local = dl.data_dl(url, self.code, path, force_update, verbose)
        return local

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        file_path = self.data_path(subject)
        if isinstance(file_path, list):
            file_path = file_path[0]
        raw = self._mat_to_raw(file_path)
        # Expose each stored trial as its own run: no synthetic padding or
        # filtering across unrelated trial boundaries.
        samples = 4000
        runs = {}
        for i in range(raw.n_times // samples):
            run = RawArray(
                raw.get_data(start=i * samples, stop=(i + 1) * samples),
                raw.info.copy(),
                verbose=False,
            )
            run.set_annotations(Annotations([0], [0], [raw.annotations.description[i]]))
            runs[str(i)] = run
        return {"0": runs}

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one subject ``.mat`` file into a continuous :class:`mne.io.RawArray`.

        The stored epochs ``(n_trials, n_channels, n_samples)`` are concatenated
        along time; an annotation marks each trial's motor-imagery cue onset
        (t = 0, i.e. the start of every 4 s epoch) with its class code
        (1 = left hand, 2 = right hand, 3 = feet).
        """
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

        sfreq = float(np.asarray(mat["fs"]).ravel()[0])
        if sfreq != 1000:
            raise ValueError("Expected WRCC sampling rate of 1000 Hz")
        data = np.asarray(mat["data"], dtype=float)
        labels = np.atleast_1d(np.asarray(mat["label"]).ravel())

        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3 or data.shape[1:] != (59, 4000):
            raise ValueError("Expected stored WRCC trials of shape (n_trials, 59, 4000)")
        if len(labels) != len(data) or not np.isin(labels, [1, 2, 3]).all():
            raise ValueError("WRCC labels must match trials and use codes 1, 2, 3")
        n_trials, n_channels, n_samples = data.shape
        if n_channels != len(WRCC2023_MI_A_CHANNELS):
            raise ValueError(
                f"WRCC2023_MI_A: expected {len(WRCC2023_MI_A_CHANNELS)} channels, "
                f"got {n_channels} in {file_path}"
            )

        # (n_trials, n_channels, n_samples) -> (n_channels, n_trials, n_samples)
        data = np.transpose(data, (1, 0, 2))
        # Concatenate trials along time. Values are already in volts, so no
        # unit rescaling is applied.
        cont = data.reshape(n_channels, n_trials * n_samples)
        # Annotation events retain sample-zero cues without a stim channel.
        annotations = Annotations(
            onset=np.arange(n_trials) * n_samples / 1000,
            duration=np.zeros(n_trials),
            description=[{1: "left_hand", 2: "right_hand", 3: "feet"}[v] for v in labels],
        )

        full = cont
        mne_info = create_info(
            ch_names=list(WRCC2023_MI_A_CHANNELS),
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        raw.set_annotations(annotations)
        return raw
