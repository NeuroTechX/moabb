"""HYGRIP hybrid dynamic grip-force dataset (Ortega et al., 2020)."""

import mne
import numpy as np

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


# Single standalone HDF5 file (~4.6 GB) hosted on figshare.
HYGRIP_URL = "https://ndownloader.figshare.com/files/22837760"

# Anonymised subject IDs used as top-level HDF5 group keys.
SUBJECT_KEYS = list("ABCDEFGHIJKLMN")

# EEG: 24 channels arranged in two 5x5 grids centred on C3/C4 (custom, non 10-20).
EEG_CH_NAMES = [f"EEG{i:02d}" for i in range(24)]

# HDF5 event label -> MOABB event code. Rows with label -1 (relax/begin/end)
# are not trial onsets and are dropped.
EVENT_CODE = {0: 1, 1: 2}  # 0 = left-hand, 1 = right-hand

# The HDF5 root stores an ``eeg_units`` attribute; the published file records
# the EEG in millivolts (verified: per-sample amplitudes ~0.01 in stored units,
# i.e. ~10 uV, consistent with scalp EEG). MNE expects SI volts, so the samples
# must be rescaled. This table maps the documented unit string to the factor
# that converts to volts. Defaults to millivolts if the attribute is absent.
_UNIT_TO_VOLTS = {
    "v": 1.0,
    "volt": 1.0,
    "volts": 1.0,
    "mv": 1e-3,
    "milivolt": 1e-3,  # codespell:ignore
    "milivolts": 1e-3,  # codespell:ignore
    "millivolt": 1e-3,
    "millivolts": 1e-3,
    "uv": 1e-6,
    "microvolt": 1e-6,
    "microvolts": 1e-6,
}


class HYGRIP2020(BaseDataset):
    """Hybrid Dynamic Grip (HYGRIP) dataset [1]_.

    .. admonition:: Dataset summary

        ================ ======= ======= =========== =============== =============== ===========
        Name             #Subj   #Chan   #Classes    #Trials/class   Trials length   Sampling
        ================ ======= ======= =========== =============== =============== ===========
        HYGRIP2020       14      24      2           10-13           21 s            1000 Hz
        ================ ======= ======= =========== =============== =============== ===========

    **Dataset description**

    HYGRIP [1]_ is a full-stack neurobehavioural dataset recorded from 14
    right-handed healthy volunteers (anonymised IDs ``A`` to ``N``) performing a
    uni-manual dynamic grip-force task within 25-50 % of each hand's maximum
    voluntary contraction. Subjects gripped a force sensor with either the left
    or the right hand (pseudo-randomised), each trial consisting of 10
    consecutive contraction (1.55 s) / relaxation (0.55 s) cycles cued visually
    from a ``Go`` onset (21 s of active grip). Each subject performed between 10
    and 13 trials per hand.

    This is a motor *execution* task (actual grip force); it is exposed here
    under the ``imagery`` paradigm so it is discoverable by the left/right-hand
    decoding paradigm.

    Only the EEG modality is loaded here. The recording additionally provides
    fNIRS, EMG, grip force, EOG and breathing signals in the same HDF5 file; see
    the reference for their organisation. The published EEG is 24 channels
    (12 per hemisphere) placed between the fNIRS source-detector optodes in two
    5x5 grids centred on C3 and C4, referenced to Cz, and down-sampled to
    1000 Hz (50 Hz and 12.5 Hz notch, high-pass above 1 Hz). Because the montage
    is custom (not a standard 10-20 layout) channels are named ``EEG00``..
    ``EEG23`` and no standard montage is set.

    The whole dataset is distributed as a single standalone HDF5 file
    (``hygrip.h5``) organised in three levels: root attributes hold the
    per-modality sampling frequencies (e.g. ``eeg_sfreq``) and grid layout; one
    group per subject (keys ``A``..``N``); and within each subject one subgroup
    per modality (``eeg``, ``emg``, ``frc``, ``oi1``/``oi2``, ``eog``, ``brt``)
    holding a ``(channels, time)`` array and an ``events`` attribute. The
    ``events`` attribute is an ``(n_events, 2)`` array of ``(time_s, label)``
    where label 0 = left hand, 1 = right hand and -1 = other markers
    (relax / begin / end).

    References
    ----------

    .. [1] Ortega, P., Zhao, T., & Faisal, A. A. (2020). HYGRIP: Full-Stack
       Characterization of Neurobehavioral Signals (fNIRS, EEG, EMG, Force, and
       Breathing) During a Bimanual Grip Force Control Task. Frontiers in
       Neuroscience, 14, 919. DOI: https://doi.org/10.3389/fnins.2020.00919

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=24,
            channel_types={"eeg": 24},
            montage="custom (two 5x5 grids centred on C3/C4)",
            reference="Cz",
            hardware="EEG electrodes interleaved with fNIRS optodes",
            sensors=EEG_CH_NAMES,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=1,
                has_emg=True,
                emg_channels=4,
                other_physiological=["grip force", "fNIRS", "breathing"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            handedness="right",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="motor execution",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=21.0,
            study_design="Uni-manual dynamic grip-force task at 25-50% MVC; 10 "
            "contraction (1.55 s) / relaxation (0.55 s) cycles per trial cued "
            "from a Go onset. Left/right hand pseudo-randomised.",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2020.00919",
            description="Hybrid Dynamic Grip (HYGRIP) full-stack neurobehavioural "
            "dataset (fNIRS, EEG, EMG, force, breathing) during a uni-manual "
            "dynamic grip-force control task; 14 right-handed subjects, "
            "left vs right hand.",
            investigators=["Pablo Ortega", "Tong Zhao", "A. Aldo Faisal"],
            senior_author="A. Aldo Faisal",
            institution="Imperial College London",
            institution_department="Brain and Behaviour Lab",
            country="GB",
            data_url="https://doi.org/10.6084/m9.figshare.12383639.v1",
            publication_year=2020,
            license="CC-BY-4.0",
            repository="Figshare",
            keywords=[
                "grip force",
                "motor execution",
                "EEG",
                "fNIRS",
                "EMG",
                "brain-computer interface",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Execution"]),
        file_format="HDF5",
        data_processed=True,
        contributing_labs=["Brain and Behaviour Lab, Imperial College London"],
        n_contributing_labs=1,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14 + 1)),
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="HYGRIP2020",
            interval=[0, 21],
            paradigm="imagery",
            doi="10.3389/fnins.2020.00919",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path to the single HDF5 file backing all subjects.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            A single-element list with the path to ``hygrip.h5``.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        h5_path = dl.data_dl(
            HYGRIP_URL, self.code, path=path, force_update=force_update, verbose=verbose
        )
        return [str(h5_path)]

    def _get_single_subject_data(self, subject):
        """Return the EEG data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"0": {"0": mne.io.RawArray}}`` for the single session/run.
        """
        import h5py

        h5_path = self.data_path(subject)[0]
        key = SUBJECT_KEYS[subject - 1]

        with h5py.File(h5_path, "r") as ds:
            sfreq = float(ds.attrs["eeg_sfreq"])
            eeg = np.asarray(ds[f"{key}/eeg"][:], dtype=np.float64)
            events_attr = np.asarray(ds[f"{key}/eeg"].attrs["events"])
            units_attr = ds.attrs.get(
                "eeg_units",
                b"milivolts",  # codespell:ignore
            )

        # The published EEG is stored in millivolts (per the file's ``eeg_units``
        # attribute); MNE expects SI volts, so rescale accordingly. Fall back to
        # millivolts if the attribute is missing or unrecognised.
        if isinstance(units_attr, bytes):
            units_attr = units_attr.decode("ascii", "ignore")
        units_key = str(units_attr).strip().lower()
        scale = _UNIT_TO_VOLTS.get(units_key, 1e-3)
        eeg = eeg * scale

        # eeg is (channels, time); guard against a transposed layout.
        if eeg.shape[0] != len(EEG_CH_NAMES) and eeg.shape[-1] == len(EEG_CH_NAMES):
            eeg = eeg.T

        ch_names = list(EEG_CH_NAMES) + ["STI"]
        ch_types = ["eeg"] * len(EEG_CH_NAMES) + ["stim"]

        stim = np.zeros((1, eeg.shape[-1]))
        for row in events_attr:
            label = int(round(float(row[1])))
            if label not in EVENT_CODE:
                continue
            onset = int(round(float(row[0]) * sfreq))
            if 0 <= onset < stim.shape[-1]:
                stim[0, onset] = EVENT_CODE[label]

        data = np.concatenate([eeg, stim], axis=0)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)

        return {"0": {"0": raw}}
