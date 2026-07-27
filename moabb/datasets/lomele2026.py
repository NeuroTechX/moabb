"""Lomele2026 reach-to-grasp-lift motor execution dataset."""

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
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
from .utils import build_raw_from_epochs


_SFREQ = 1000.0

# The 62 scalp EEG electrodes followed by the two EOG channels, in the exact
# row order of ``data.trials(i).eeg`` (verified against s_7.mat).
# fmt: off
_CH_NAMES = [
    "Iz", "O2", "Oz", "O1", "PO8", "PO4", "POz", "PO3", "PO7", "P8",
    "P6", "P4", "P2", "Pz", "P1", "P3", "P5", "P7", "TP10", "TP8",
    "CP6", "CP4", "CP2", "CPz", "CP1", "CP3", "CP5", "TP7", "TP9", "T8",
    "C6", "C4", "C2", "Cz", "C1", "C3", "C5", "T7", "FT8", "FC6",
    "FC4", "FC2", "FCz", "FC1", "FC3", "FC5", "FT7", "F8", "F6", "F4",
    "F2", "Fz", "F1", "F3", "F5", "F7", "AF4", "AFz", "AF3", "Fp2",
    "Fpz", "Fp1", "VEOG", "HEOG",
]
# fmt: on

_EOG_CHANNELS = ("VEOG", "HEOG")

# Grip type indexed by ``data.trials(i).task`` (1, 2 or 3), per the paper's
# Data Records: 1 = precision grip (thumb-index), 2 = unconventional grip
# (thumb-ring finger), 3 = whole-hand power grasp.
_EVENTS = {"precision": 1, "unconventional": 2, "power": 3}

# Stable OSF file GUIDs for each subject's s_<n>.mat (node rsv4z, osfstorage).
# fmt: off
_OSF_URLS = {
    1: "https://osf.io/download/5bfne/",
    2: "https://osf.io/download/xjv2s/",
    3: "https://osf.io/download/t9cwb/",
    4: "https://osf.io/download/3dey2/",
    5: "https://osf.io/download/fg3sb/",
    6: "https://osf.io/download/wkghu/",
    7: "https://osf.io/download/kw8dq/",
    8: "https://osf.io/download/k9eyr/",
    9: "https://osf.io/download/6z5th/",
    10: "https://osf.io/download/gp8jy/",
    11: "https://osf.io/download/r34wv/",
    12: "https://osf.io/download/c3pzq/",
    13: "https://osf.io/download/f52dp/",
    14: "https://osf.io/download/h7u2d/",
}
# fmt: on


class Lomele2026(BaseDataset):
    """High-density EEG/EMG reach-to-grasp-lift motor execution dataset [1]_.

    **Dataset description**

    Synchronized high-density EEG and multi-muscle surface EMG recorded from
    14 healthy participants (13 right-handed, 1 left-handed) performing
    visually cued object prehension with a custom sensorized grasping box.
    On each trial the subject reaches for, grasps and lifts an object using
    one of three grip types:

    - **precision** grip (thumb-index opposition, PG),
    - **unconventional** grip (thumb-ring finger opposition, UG),
    - **power** whole-hand grasp (WH).

    This is a *motor execution* task (overt reach-to-grasp-lift movement), not
    motor imagery; it is exposed here through MOABB's ``imagery`` paradigm.

    EEG was acquired at 1000 Hz with a BrainAmp DC amplifier (Brain Products)
    over 62 scalp electrodes of the international 10/20 system plus 2 EOG
    electrodes at the outer canthi (64 channels total). EMG (not returned by
    this loader) was recorded from 13 upper-limb muscles with wireless
    WavePlus sensors (Cometa) and downsampled to 1000 Hz.

    Each trial is a 9 s epoch spanning -2 s to +7 s around the go signal
    (LED-on), which always falls at sample 2000. A custom sensorized box
    timestamps three behavioural events per trial as frame indices:
    ``LEDon_event_frame`` (go signal), ``touch_event_frame`` (object contact)
    and ``lift_event_frame`` (maximum displacement). The event used to epoch
    here is the LED-on go signal. Every subject contributes 90 trials balanced
    across the three grip types (30 per class); roughly half are flagged
    ``event_complete_trial == 0`` (touch/lift markers missing) but retain the
    full EEG and grip label, so all trials are kept for grip-type decoding.

    The data is delivered as one MATLAB v5 file per subject (``s_<n>.mat``,
    subjects 1-14) hosted on OSF, each containing a single ``data`` struct with
    fields ``subject``, ``info`` (``fs``, ``eeg_channels``, ``emg_channels``,
    ``tasks``), ``baseline`` (>=5 min eyes-open resting EEG, not returned) and
    ``trials``.

    References
    ----------

    .. [1] Lomele, G., Lencioni, T., D'Ambrosio, S., Comanducci, A.,
       Lucchetti, F., Marzegan, A., Derchi, C. C., Garzonio, S., Atzori, T.,
       Castiglioni, P., Fornia, L., Ferrarin, M., & Rabuffetti, M. (2026).
       High-Density EEG and Multi-Muscle EMG Dataset during Object Prehension
       with a sensorized Grasping Box in Humans. Scientific Data.
       DOI: https://doi.org/10.1038/s41597-026-07242-y

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=64,
            channel_types={"eeg": 62, "eog": 2},
            montage="10-20",
            hardware="BrainAmp DC amplifier (Brain Products)",
            sensor_type="Ag/AgCl",
            line_freq=50.0,
            sensors=_CH_NAMES,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["vertical", "horizontal"],
                has_emg=True,
                emg_channels=13,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            handedness={"right": 13, "left": 1},
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["precision", "unconventional", "power"],
            trials_per_class={"precision": 30, "unconventional": 30, "power": 30},
            trial_duration=9.0,
            study_design=(
                "Visually cued reach-to-grasp-lift of an object with a sensorized "
                "grasping box, using one of three grip types (precision, "
                "unconventional, whole-hand power). Trials aligned to the LED-on "
                "go signal; a dynamic phase (reach, grasp, lift) is followed by an "
                "isometric holding phase."
            ),
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events=_EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-026-07242-y",
            description=(
                "High-density EEG (62 ch) and 13-muscle EMG recorded during "
                "cued object prehension (reach-to-grasp-lift) with three grip "
                "types in 14 healthy adults."
            ),
            investigators=[
                "Giulia Lomele",
                "Tiziana Lencioni",
                "Sasha D'Ambrosio",
                "Angela Comanducci",
                "Francesca Lucchetti",
                "Alberto Marzegan",
                "Chiara Camilla Derchi",
                "Stefano Garzonio",
                "Tiziana Atzori",
                "Paolo Castiglioni",
                "Luca Fornia",
                "Maurizio Ferrarin",
                "Marco Rabuffetti",
            ],
            country="IT",
            repository="OSF",
            data_url="https://osf.io/rsv4z/",
            publication_year=2026,
            license="CC0-1.0",
            keywords=[
                "motor execution",
                "reach to grasp",
                "prehension",
                "grip type",
                "EEG",
                "EMG",
                "high-density EEG",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["motor_execution"]),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=False,
            notes=(
                "Released as 9 s epochs (-2 s to +7 s about LED-on) at 1000 Hz. "
                "EMG downsampled from 2000 Hz to 1000 Hz."
            ),
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["precision", "unconventional", "power"],
        ),
        data_structure=DataStructureMetadata(
            n_trials=90,
            n_trials_per_class={"precision": 30, "unconventional": 30, "power": 30},
            trials_context=(
                "90 trials per subject, 30 per grip type. Each trial is a 9 s "
                "epoch (64 EEG x 9000 samples) aligned to LED-on at sample 2000."
            ),
        ),
        file_format="MAT",
        data_processed=False,
        abstract=(
            "An open dataset of synchronized high-density EEG and multi-muscle "
            "EMG acquired during visually guided object prehension with a "
            "sensorized grasping box, covering three grip types (precision, "
            "unconventional and whole-hand power) in 14 healthy participants."
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14 + 1)),
            sessions_per_subject=1,
            events=_EVENTS,
            code="Lomele2026",
            interval=(0, 5),
            paradigm="imagery",
            doi="10.1038/s41597-026-07242-y",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")
        url = _OSF_URLS[subject]
        local = dl.data_dl(
            url, self.code, path=path, force_update=force_update, verbose=verbose
        )
        return [local]

    def _get_single_subject_data(self, subject):
        file_path = self.data_path(subject)[0]
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        data = mat["data"]

        trials = np.atleast_1d(data.trials)
        eeg = np.stack([np.asarray(t.eeg, dtype=float) for t in trials], axis=0)
        event_ids = np.array([int(t.task) for t in trials], dtype=int)

        # LED-on go signal (frame index, 1-based in MATLAB) is the epoch event.
        onset_sample = int(trials[0].LEDon_event_frame) - 1

        ch_types = ["eog" if ch in _EOG_CHANNELS else "eeg" for ch in _CH_NAMES]

        raw = build_raw_from_epochs(
            eeg,
            _CH_NAMES,
            _SFREQ,
            event_ids,
            montage_name="standard_1005",
            ch_types=ch_types,
            scale=1e-6,
            onset_sample=onset_sample,
        )
        return {"0": {"0": raw}}
