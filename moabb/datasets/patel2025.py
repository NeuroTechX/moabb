"""Longitudinal ALS motor imagery EEG dataset (Patel et al. 2025)."""

import logging

import numpy as np
import scipy.io as sio
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)


# UCL Research Data Repository (Figshare), article 28156016 v1.
# doi:10.5522/04/28156016. Per-subject processed .mat files, addressed by their
# numeric Figshare file id (ndownloader.figshare.com/files/{id}). The subject
# numbers are those used by the authors (non-contiguous: they are a subset of a
# larger clinical cohort).
PATEL2025_BASE_URL = "https://ndownloader.figshare.com/files/"

# real subject id -> Figshare file id, verified against
# https://api.figshare.com/v2/articles/28156016 (2026-07).
PATEL2025_FILE_IDS = {
    1: 51526682,
    2: 51526688,
    5: 51526691,
    9: 51526694,
    21: 51526697,
    31: 51526700,
    34: 51526706,
    39: 51526709,
}

# 19 EEG channels (10-20 system), in the column order of the stored matrices,
# followed by 3 EOG channels. Casing follows the MNE ``standard_1005`` montage
# so electrode positions resolve. The dataset description lists the EEG order as
# FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2.
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]
EOG_CHANNELS = ["EOG1", "EOG2", "EOG3"]

# Class field name in the MATLAB struct -> event code.
_FIELD_TO_CODE = {"L": 1, "R": 2, "Re": 3}
_EVENTS = {"left_hand": 1, "right_hand": 2, "rest": 3}

# Sampling rate (Hz). Not stored in the .mat files; established from the data
# (rest segments are pinned at 256 samples = 1.0 s) and the authors' own code
# (``5 * fs`` samples for a 5 s imagery segment, matching the ~1288-sample
# median of the imagery trials).
SFREQ = 256.0

# Trials are variable length (rest ~1 s, imagery up to ~10 s). Each trial is
# written into a fixed 2 s (512-sample) block: longer trials are cropped, shorter
# trials (rest) are zero-padded at the tail. The tail padding falls outside the
# 1 s default analysis window, so it is never read by the default epoching, and
# no epoch can bleed into a neighbouring trial.
_BLOCK = int(round(SFREQ * 2.0))  # 512


class Patel2025(BaseDataset):
    """Longitudinal ALS motor imagery EEG dataset (Patel et al. 2025).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Patel2025        8       19           3               ~160            1s           256 Hz            1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    EEG recordings from eight patients with amyotrophic lateral sclerosis (ALS)
    performing a three-class motor imagery task: imagined left-hand grasping,
    imagined right-hand grasping, and rest. The data were originally collected by
    Geronimo, Simmons and Schiff [2]_ at the Penn State Hershey Medical Center ALS
    Clinic and are redistributed here in processed form by Patel et al. on the UCL
    Research Data Repository [1]_.

    Each participant completed four BCI sessions over one to two months (a
    calibration run followed by feedback runs controlling a cursor by motor
    imagery). Signals were recorded with two g.USBamp amplifiers via BCI2000 from
    19 electrodes placed according to the international 10-20 system (Fp1, Fp2, F7,
    F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2), referenced to
    linked earlobes with a ground at FPz, plus three EOG electrodes for artifact
    monitoring. The sampling rate is 256 Hz.

    The distributed ``.mat`` files each hold a single ``Subject<n>`` struct array
    of trials; every element carries three fields, ``L`` (left-hand imagery), ``R``
    (right-hand imagery) and ``Re`` (rest), each a ``samples x 22`` matrix (19 EEG
    then 3 EOG channels). The class of every trial is therefore data-borne: it is
    the name of the struct field the trial is stored under. Roughly 160 trials per
    class per subject are provided (the exact count varies slightly by subject).

    Trials have variable length (rest is about 1 s; imagery trials extend up to
    about 10 s of cursor-control feedback). This loader writes each trial into a
    fixed 2 s frame (cropping longer imagery trials, zero-padding the tail of the
    ~1 s rest trials) and marks its onset on a stim channel. The default analysis
    interval is the first 1 s, which lies within the real signal of every trial;
    the four original sessions are pooled in the processed data and are exposed
    here as a single session.

    .. warning::

        The distributed amplitudes are in the dataset's native (uncalibrated)
        units; this loader applies the conventional ``1e-6`` scaling. Absolute
        voltage values are therefore not physiological, but this does not affect
        band-power / spatial-filter motor-imagery decoding, which is invariant to
        a global scale.

    References
    ----------

    .. [1] Patel, R., Jiang, D., Bryson, B., Carlson, T., Demosthenous, A., &
           Geronimo, A. (2025). Longitudinal ALS EEG Dataset for Motor Imagery
           Studies. University College London.
           https://doi.org/10.5522/04/28156016

    .. [2] Geronimo, A., Simmons, Z., & Schiff, S. J. (2016). Performance
           predictors of brain-computer interfaces in patients with amyotrophic
           lateral sclerosis. Journal of Neural Engineering, 13(2), 026002.
           https://doi.org/10.1088/1741-2560/13/2/026002

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            n_channels=22,
            channel_types={"eeg": 19, "eog": 3},
            sensors=EEG_CHANNELS + EOG_CHANNELS,
            montage="standard_1005",
            reference="linked earlobes",
            ground="FPz",
            hardware="two g.USBamp (g.tec)",
            software="BCI2000",
            impedance_threshold_kohm=10.0,
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=True, eog_channels=3),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="patients",
            clinical_population="ALS",
            age_min=45.5,
            age_max=74.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="left_right_hand_rest",
            events=dict(_EVENTS),
            n_classes=3,
            class_labels=["left_hand", "right_hand", "rest"],
            trials_per_class={"left_hand": 160, "right_hand": 160, "rest": 160},
            synchronicity="synchronous",
            mode="offline",
            feedback_type="visual",
            study_design=(
                "Three-class motor imagery (imagined left- vs right-hand grasping "
                "vs rest) with cursor feedback, recorded longitudinally in ALS "
                "patients over four sessions across one to two months."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.5522/04/28156016",
            description=(
                "Longitudinal three-class motor imagery EEG (left hand, right "
                "hand, rest) from eight ALS patients; processed redistribution of "
                "the Geronimo et al. 2016 recordings."
            ),
            investigators=[
                "Rishan Patel",
                "Dai Jiang",
                "Barney Bryson",
                "Tom Carlson",
                "Andreas Demosthenous",
                "Andrew Geronimo",
            ],
            institution="University College London",
            country="GB",
            repository="Figshare",
            data_url="https://doi.org/10.5522/04/28156016",
            publication_year=2025,
            license="CC-BY-4.0",
            related_paper_dois=["10.1088/1741-2560/13/2/026002"],
            ethics_approval=["Penn State University IRB protocol PRAMSO40647EP"],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["ALS"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=["left_hand", "right_hand", "rest"]
        ),
        data_structure=DataStructureMetadata(
            n_trials="~480 per subject (~160 per class)",
            n_trials_per_class={"left_hand": 160, "right_hand": 160, "rest": 160},
            trials_context=(
                "Per-subject Subject<n> struct array; each element has L/R/Re "
                "fields (one trial each). Counts vary slightly across subjects."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["cursor_control"], environment="clinical", online_feedback=True
        ),
        data_processed=True,
        file_format="MAT",
    )

    def __init__(self):
        super().__init__(
            subjects=sorted(PATEL2025_FILE_IDS.keys()),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Patel2025",
            interval=[0, 1],
            paradigm="imagery",
            doi="10.5522/04/28156016",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path of a single subject's ``.mat`` file."""
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}. Valid: {self.subject_list}")

        url = f"{PATEL2025_BASE_URL}{PATEL2025_FILE_IDS[subject]}"
        return dl.data_dl(url, self.code, path, force_update, verbose)

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as ``{session: {run: Raw}}``."""
        file_path = self.data_path(subject)
        raw = self._mat_to_raw(file_path)
        return {"0": {"0": raw}}

    @staticmethod
    def _mat_to_raw(file_path):
        """Load one subject ``.mat`` file into a continuous :class:`mne.io.RawArray`.

        Every ``L`` / ``R`` / ``Re`` trial is written into a fixed 2 s block
        (cropped or tail-zero-padded), the blocks are concatenated along time, and
        a stim channel marks each block onset with the trial's class code
        (1 = left hand, 2 = right hand, 3 = rest).
        """
        mat = sio.loadmat(file_path, squeeze_me=False, struct_as_record=False)
        data_keys = [k for k in mat if not k.startswith("__")]
        if not data_keys:
            raise ValueError(f"No data variable found in {file_path}")
        struct = np.asarray(mat[data_keys[0]]).ravel()

        n_eeg = len(EEG_CHANNELS)
        n_ch = n_eeg + len(EOG_CHANNELS)  # 22

        blocks = []
        codes = []
        n_skipped = 0
        for element in struct:
            for field, code in _FIELD_TO_CODE.items():
                trial = np.asarray(getattr(element, field), dtype=float)
                if trial.ndim != 2 or trial.shape[0] == 0 or trial.shape[1] != n_ch:
                    n_skipped += 1
                    continue
                seg = trial.T  # (n_ch, n_samples)
                block = np.zeros((n_ch, _BLOCK))
                m = min(seg.shape[1], _BLOCK)
                block[:, :m] = seg[:, :m]
                blocks.append(block)
                codes.append(code)

        if n_skipped:
            log.warning(
                "Patel2025: skipped %d trial(s) in %s with an unexpected shape "
                "(empty or channel count != %d).",
                n_skipped,
                file_path,
                n_ch,
            )

        if not blocks:
            raise ValueError(f"No usable trials found in {file_path}")

        # Convert from the dataset's native units to volts (nominal scaling).
        # A one-sample lead pad keeps the first trial's onset off sample 0, so
        # ``mne.find_events`` (used downstream by MOABB) detects its rising edge.
        cont = np.concatenate(blocks, axis=1) * 1e-6
        n_lead = 1
        cont = np.pad(cont, ((0, 0), (n_lead, 0)))

        stim = np.zeros((1, cont.shape[1]))
        for block_idx, code in enumerate(codes):
            stim[0, n_lead + block_idx * _BLOCK] = code

        full = np.vstack([cont, stim])
        info = create_info(
            ch_names=EEG_CHANNELS + EOG_CHANNELS + ["STI 014"],
            sfreq=SFREQ,
            ch_types=["eeg"] * n_eeg + ["eog"] * len(EOG_CHANNELS) + ["stim"],
        )
        raw = RawArray(data=full, info=info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw
