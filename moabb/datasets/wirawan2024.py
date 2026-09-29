"""Wirawan2024 MIMED Motor Imagery / Motor Execution dataset."""

import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

import mne
import numpy as np
import scipy.io as sio

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


# Direct download URL of the "Motor Imagery.zip" archive of the Mendeley
# record 10.17632/zs25xxjkm9.3 (version 3). It contains only the motor
# imagery .mat files (~19.5 MB), avoiding the ~1.2 GB full-record download.
WIRAWAN2024_MI_URL = (
    "https://data.mendeley.com/public-files/datasets/zs25xxjkm9/files/"
    "3dc41eb1-8999-4e0d-a77a-0394b53ba04e/file_downloaded"
)

# The 14 Emotiv EPOC X electrodes, in the channel order stored in each .mat.
WIRAWAN2024_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

# Scenario sub-folders inside the archive, each mapped to the single class
# it records. Inspection of the distributed .mat files shows they carry only
# ``joined_data`` (the four imagery repetitions), plus ``Fs``, ``channel`` and
# ``subject`` metadata -- there is no per-repetition activity label. The
# up/down split within a scenario is therefore not recoverable from the data,
# so each scenario folder is treated as one reliable class: the left-hand,
# right-hand and trunk (stand/sit) imagery.
WIRAWAN2024_SCENARIOS = {
    "Left Hand Up-Down Imagine": "left_hand",
    "Right Hand Up-Down Imagine": "right_hand",
    "Stand Up-Down Imagine": "trunk",
}

# 3 s baseline recorded before each imagery period, at 128 Hz.
WIRAWAN2024_SFREQ = 128
WIRAWAN2024_BASELINE_SAMPLES = 384


class Wirawan2024(BaseDataset):
    """Motor Imagery MIMED dataset from Wirawan et al. 2024 [1]_.

    **Dataset description**

    The MIMED (Motor Imagery and Motor Execution Dataset) was recorded from
    30 healthy students from the Bali region of Indonesia using an Emotiv
    EPOC X 14-channel wireless headset sampled at 128 Hz. The 14 electrodes
    follow the international 10-20 system: AF3, F7, F3, FC5, T7, P7, O1, O2,
    P8, T8, FC6, F4, F8, AF4.

    Participants performed six activities, both as motor execution and as
    motor imagery: raising the right hand, lowering the right hand, raising
    the left hand, lowering the left hand, standing and sitting. Only the
    motor imagery recordings are loaded here. The imagery recordings are
    distributed as three scenario files per subject (one folder per scenario),
    each containing four recording blocks across two days. Each block has a
    3 s (384-sample) baseline period; the event marker is placed at the onset
    of the imagery period, so that the 3 s baseline precedes the epoched
    window.

    The distributed ``.mat`` files carry only the recorded signals
    (``joined_data``) together with ``Fs``, ``channel`` and ``subject``
    metadata; they do **not** store a per-repetition activity label. The
    within-scenario up/down (or stand/sit) split is therefore not recoverable
    from the data. The loader consequently exposes the reliable, folder-borne
    3-class task in which each scenario folder is a single class:
    ``left_hand`` (1) from "Left Hand Up-Down Imagine", ``right_hand`` (2) from
    "Right Hand Up-Down Imagine" and ``trunk`` (3) from "Stand Up-Down
    Imagine". Each block contributes one four-second analysis window after the baseline
    (four windows per class, twelve in total), not an invented cue sequence.
    The two recording days are separate sessions, with six runs per day.

    The signals are stored in Emotiv raw units (micro-volts with a DC offset
    of roughly 4200 uV) and are rescaled to volts on load.

    Notes
    -----
    An earlier revision derived a 6-class up/down labelling from the
    acquisition order (even repetition -> raise, odd -> lower). That mapping is
    not carried by the ``.mat`` files and was dropped; only the folder-level
    3-class labelling is data-borne.

    References
    ----------

    .. [1] Wirawan, I. M. A., et al. (2024). Acquisition and processing of Motor Imagery and Motor
       Execution Dataset (MIMED) for six movement activities. Data in Brief, 56, 110833.
       DOI: https://doi.org/10.1016/j.dib.2024.110833

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            channel_types={"eeg": 14},
            montage="standard_1020",
            hardware="Emotiv EPOC X",
            reference="CMS/DRL (P3/P4)",
            ground="DRL",
            sensor_type="saline felt (Ag/AgCl)",
            sensors=list(WIRAWAN2024_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=30, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "trunk"],
            events={"left_hand": 1, "right_hand": 2, "trunk": 3},
            stimulus_type="video",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.dib.2024.110833",
            related_paper_dois=["10.17632/zs25xxjkm9.3"],
            description=(
                "MIMED: motor imagery and motor execution EEG dataset of six "
                "activities recorded from 30 subjects with an Emotiv EPOC X "
                "14-channel headset at 128 Hz."
            ),
            investigators=[
                "I Made Agus Wirawan",
                "Dechrit Maneetham",
                "I Gede Mahendra Darmawiguna",
                "Arnon Niyomphol",
                "Pakornkiat Sawetmethikul",
                "Padma Nyoman Crisnapati",
                "Yamin Thwe",
                "Ni Nyoman Mestri Agustini",
            ],
            country="ID",
            data_url="https://doi.org/10.17632/zs25xxjkm9.3",
            publication_year=2024,
            license="CC-BY-4.0",
            repository="Mendeley Data",
        ),
        sessions_per_subject=2,
        runs_per_session=6,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="MAT (converted from EDF)",
        abstract=(
            "The MIMED dataset provides EEG recordings of motor imagery and "
            "motor execution for six activities (raising and lowering each "
            "hand, standing and sitting) from 30 subjects, acquired with a "
            "14-channel Emotiv EPOC X headset at 128 Hz. The distributed "
            "imagery .mat files carry no per-repetition activity label, so the "
            "loader exposes the reliable folder-level 3-class task "
            "(left_hand / right_hand / trunk)."
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 30 + 1)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2, "trunk": 3},
            code="Wirawan2024",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1016/j.dib.2024.110833",
        )

    def _extract_root(self, path=None, force_update=False, verbose=None):
        """Download the Motor Imagery archive and return its extraction root."""
        path_zip = Path(
            dl.data_dl(WIRAWAN2024_MI_URL, self.code, path, force_update, verbose)
        )
        path_folder = path_zip.parent
        marker = path_folder / "Motor Imagery"

        if force_update or not marker.is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(
                    dl.data_dl(WIRAWAN2024_MI_URL, self.code, path, True, verbose)
                )
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        return path_folder

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of scenario file paths for a single subject.

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
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list of str
            One path per scenario file for the subject.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_folder = self._extract_root(path, force_update, verbose)
        sub = f"P{subject:02d}"
        subject_paths = []
        for scenario in WIRAWAN2024_SCENARIOS:
            subject_paths.append(
                str(path_folder / "Motor Imagery" / scenario / f"{sub}.mat")
            )
        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            Two recording-day sessions, each with six independently filtered
            scenario blocks annotated with their folder-level class.
        """
        file_paths = self.data_path(subject)
        if len(set(file_paths)) != len(file_paths):
            raise ValueError("Duplicate MIMED scenario files")

        if len(file_paths) != len(WIRAWAN2024_SCENARIOS):
            raise ValueError("Expected one file for each of the three scenarios")
        sessions = {"0": {}, "1": {}}
        for scenario_idx, (scenario_file, scenario) in enumerate(
            zip(file_paths, WIRAWAN2024_SCENARIOS)
        ):
            mat = sio.loadmat(scenario_file)
            if float(np.asarray(mat["Fs"]).ravel()[0]) != WIRAWAN2024_SFREQ:
                raise ValueError("Expected MIMED sampling rate of 128 Hz")
            joined = mat["joined_data"]
            if joined.shape != (1, 4):
                raise ValueError("Expected four MIMED recording blocks per scenario")
            for trial_idx in range(4):
                trial = np.asarray(joined[0, trial_idx], dtype=float)
                if trial.ndim != 2 or trial.shape[1] != 14 or trial.shape[0] < 897:
                    raise ValueError("Malformed or truncated MIMED recording block")
                info = mne.create_info(WIRAWAN2024_CHANNELS, 128, "eeg")
                raw = mne.io.RawArray(trial.T * 1e-6, info, verbose=False)
                raw.set_montage("standard_1020")
                raw.set_annotations(
                    mne.Annotations([3.0], [0.0], [WIRAWAN2024_SCENARIOS[scenario]])
                )
                # The paper identifies blocks 1-2 as day one and 3-4 as
                # day two. Separate runs prevent filtering across blocks.
                sessions[str(trial_idx // 2)][str(scenario_idx * 2 + trial_idx % 2)] = raw
        return sessions
