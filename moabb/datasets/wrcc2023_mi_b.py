"""WRCC2023 MI-B three-class motor imagery dataset (BCI competition)."""

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
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.wrcc2023_mi_a import WRCC2023_MI_A_CHANNELS


# Harvard Dataverse "MI-B dataset of the BCI competition WRCC2023"
# (doi:10.7910/DVN/HFNWRX). Nine subjects, one MATLAB v5 ".mat" file each. Files
# carry no persistent id, so they are addressed by their numeric Dataverse
# datafile id (resolved from the Dataverse files API).
WRCC2023_MI_B_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"

# subject -> datafile id (file label subject<N>.mat on Harvard Dataverse).
WRCC2023_MI_B_FILE_IDS = {
    1: 10358839,
    2: 10358833,
    3: 10358840,
    4: 10358836,
    5: 10358838,
    6: 10358834,
    7: 10358835,
    8: 10358837,
    9: 10358832,
}

# Sampling rate and channel names are not stored in the file. The World Robot
# Contest (WRCC) BCI motor-imagery series records with a Neuracle 64-channel
# system, keeping the same 59 EEG channels used by the companion MI-A and MI-C
# tracks at 1000 Hz. The stored 4000-sample epochs are therefore 4.0 s long.
WRCC2023_MI_B_SFREQ = 1000.0

# Class codes stored in the ``label`` variable. The dataset description lists the
# three motor-imagery categories in the order left hand, right hand, bipedal
# (both feet); the codes 1/2/3 follow that order and the standard MI convention.
WRCC2023_MI_B_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3}


class WRCC2023_MI_B(BaseDataset):
    """Three-class motor imagery dataset (MI-B) of the BCI competition WRCC2023.

    .. admonition:: Dataset summary

        =============  =======  =======  ==========  =================  ============  ===============  ===========
        Name             #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =============  =======  =======  ==========  =================  ============  ===============  ===========
        WRCC2023_MI_B        9       59           3                 30            4s           1000 Hz            1
        =============  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Electroencephalography (EEG) recordings released as the "MI-B" motor-imagery
    dataset of the BCI competition held at the 2023 World Robot Contest (WRCC2023).
    Nine subjects each performed a cued three-class motor-imagery task: imagining
    left-hand movement, right-hand movement and bipedal (both-feet) movement. Two
    of the nine subjects are stroke patients; the remaining seven are healthy.

    Each subject file contains 90 trials (30 per class, interleaved) of 59-channel
    EEG. The signals are stored as a single ``data`` array of shape
    ``(n_trials, n_channels, n_samples) = (90, 59, 4000)`` together with an integer
    ``label`` vector (1 = left hand, 2 = right hand, 3 = feet). Each epoch spans
    4000 samples; at the 1000 Hz sampling rate of the WRCC motor-imagery recording
    setup this corresponds to a 4.0 s trial. The recordings are DC-coupled and
    stored in volts (per-channel electrode DC offsets of a few millivolts are
    present and are removed by the band-pass filtering of the analysis paradigm).

    The data are distributed as MATLAB (v5) ``.mat`` files (one per subject) on
    Harvard Dataverse. Per-trial class labels are data-borne: they live in the
    ``label`` variable of every subject file.

    References
    ----------

    .. [1] WRCC2023 (2024). MI-B dataset of the BCI competition WRCC2023.
       Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/HFNWRX

    Notes
    -----

    The ``.mat`` files do not store channel names, channel positions or the
    sampling rate. The loader applies the documented WRCC/Neuracle 59-channel
    acquisition order shared by the companion MI-A and MI-C tracks, and takes
    the sampling rate from the same 1000 Hz recording configuration.

    Each stored trial is exposed as a separate run to avoid filtering across
    discontinuities. The inclusive epoch endpoint is 3.999 s (4000 samples),
    without synthetic zero padding. Channel order and physical calibration
    follow the reconciled source loaders and still require source verification.

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=WRCC2023_MI_B_SFREQ,
            channel_types={"eeg": 59},
            montage="standard_1005",
            reference=None,
            ground=None,
            hardware="Neuracle 64-channel EEG (59 EEG channels retained)",
            sensors=list(WRCC2023_MI_A_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="mixed",
            clinical_population="stroke (2 of 9 subjects); remaining subjects healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trials_per_class={"left_hand": 30, "right_hand": 30, "feet": 30},
            trial_duration=4.0,
            study_design="Cued three-class motor imagery (left hand, right hand, "
            "both feet). Each subject file holds 90 interleaved trials, 30 per class.",
            synchronicity="cue-based",
            mode="offline",
            events=dict(WRCC2023_MI_B_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.7910/DVN/HFNWRX",
            description="MI-B three-class (left hand, right hand, feet) motor imagery "
            "EEG dataset of the BCI competition at the 2023 World Robot Contest; "
            "9 subjects (2 stroke, 7 healthy).",
            investigators=["WRCC2023"],
            country="CN",
            data_url="https://doi.org/10.7910/DVN/HFNWRX",
            publication_year=2024,
            license="CC0-1.0",
            repository="Harvard Dataverse",
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "World Robot Contest",
                "left hand",
                "right hand",
                "feet",
                "stroke",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=90,
        file_format="MAT (v5)",
        tags=Tags(
            pathology=["stroke", "healthy"], modality=["Motor"], type=["Motor Imagery"]
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9 + 1)),
            sessions_per_subject=1,
            events=dict(WRCC2023_MI_B_EVENTS),
            code="WRCC2023-MI-B",
            interval=[0, 3.999],
            paradigm="imagery",
            doi="10.7910/DVN/HFNWRX",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path of a single subject's ``.mat`` file."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        file_id = WRCC2023_MI_B_FILE_IDS[subject]
        url = f"{WRCC2023_MI_B_BASE_URL}{file_id}"
        return dl.data_dl(url, self.code, path, force_update, verbose)

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

        The epoched trials ``data`` of shape ``(n_trials, n_channels, n_samples)``
        are concatenated along time; an annotation marks each trial's cue onset
        (t = 0) with its class code (1 = left hand, 2 = right hand, 3 = feet) read
        from the data-borne ``label`` variable.
        """
        mat = sio.loadmat(file_path, squeeze_me=True)
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
                f"WRCC2023_MI_B: expected {len(WRCC2023_MI_A_CHANNELS)} channels, "
                f"got {n_channels} in {file_path}"
            )

        # (n_trials, n_channels, n_samples) -> (n_channels, n_trials * n_samples).
        # The stored signals are already in volts (DC-coupled recording), so no
        # unit rescaling is applied.
        cont = np.transpose(data, (1, 0, 2)).reshape(n_channels, n_trials * n_samples)
        # Annotation events retain sample-zero cues without a stim channel.
        annotations = Annotations(
            onset=np.arange(n_trials) * n_samples / 1000,
            duration=np.zeros(n_trials),
            description=[{1: "left_hand", 2: "right_hand", 3: "feet"}[v] for v in labels],
        )

        full = cont
        mne_info = create_info(
            ch_names=list(WRCC2023_MI_A_CHANNELS),
            sfreq=WRCC2023_MI_B_SFREQ,
            ch_types=["eeg"] * n_channels,
        )
        raw = RawArray(data=full, info=mne_info, verbose=False)
        raw.set_montage(
            make_standard_montage("standard_1005"), on_missing="ignore", verbose=False
        )
        raw.set_annotations(annotations)
        return raw
