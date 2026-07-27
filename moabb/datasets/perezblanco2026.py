"""Perez-Blanco 2026 wrist-motion motor-execution EEG-EMG dataset."""

import warnings
import zipfile as z
from pathlib import Path

import mne
import numpy as np
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids


# Figshare article hosting the BIDS data (data DOI 10.6084/m9.figshare.29666735).
PEREZBLANCO2026_ARTICLE_ID = "29666735"

# The eight EEG electrodes (10-10, motor-cortex coverage) followed by the eight
# bipolar EMG channels, in the fixed order the BCI2000 -> EDF converter writes
# them. The EDF stores the sixteen data channels first, then seven BCI2000
# "state" channels (trial/target stamps and cursor coordinates).
# fmt: off
_EEG_NAMES = ["C3", "C1", "Cz", "C2", "C4", "CP3", "CPz", "CP4"]
_EMG_NAMES = ["M1L", "M1R", "M2L", "M2R", "M3L", "M3R", "M4L", "M4R"]
# fmt: on

# Class labels come from the BCI2000 ``CurrentTarget`` state (the author
# validation code maps 1=Flexion, 2=Extension, 3=Radial dev., 4=Ulnar dev.).
_EVENTS = {"flexion": 1, "extension": 2, "radial_deviation": 3, "ulnar_deviation": 4}
_TARGET_TO_CODE = {1: 1, 2: 2, 3: 3, 4: 4}
_CODE_TO_NAME = {v: k for k, v in _EVENTS.items()}


class PerezBlanco2026(BaseDataset):
    """Wrist-motion (4-direction pointing) motor-execution dataset [1]_.

    **Dataset description**

    EEG, EMG and wrist-kinematic data from 45 healthy participants performing a
    cursor-control wrist-pointing task with the *Biomech Wrist*, a 3-DoF wrist
    rehabilitation exoskeleton worn on the right forearm. Wrist flexion-extension
    moved the cursor horizontally and radial-ulnar deviation moved it vertically.
    On each 10 s trial a target appeared in one of four cardinal directions and
    the participant moved the cursor to it, yielding four balanced movement
    classes: **flexion, extension, radial deviation, ulnar deviation**.

    Each trial follows the sequence: 3 s fixation, 2 s target preview (all four
    targets shown), 2.5 s movement execution (one target shown), and 2.5 s
    return-to-center. Participants completed at least 8 experimental runs of
    40 trials (10 trials per movement per run), preceded and followed by
    resting-state baselines; the released run count varies by subject (e.g.
    sub-01 has 9 task runs, sub-02 has 10), so :meth:`data_path` loads every
    ``task-*_run-*`` file present rather than assuming a fixed count. Signals
    were acquired with a g.tec g.USBamp (serial UB-2016.05.01)
    at 512 Hz: 8 EEG channels (C3, C1, Cz, C2, C4, CP3, CPz, CP4; reference on the
    right ear, ground AFz) and 8 bipolar EMG channels over four forearm muscles.
    Data were recorded with BCI2000 in ``.dat`` format and converted to EDF for
    BIDS distribution.

    The trigger information is carried inside the EDF as BCI2000 state channels.
    This loader places one event at each movement-execution onset (the
    ``TrialInitMovStamp`` rising edge, 5 s into the trial) and reads the movement
    class from the ``CurrentTarget`` state at that sample. The default analysis
    interval ``[0, 2.5]`` s therefore spans the 2.5 s movement-execution phase.

    By default only the EEG channels are returned; set
    ``return_all_modalities=True`` to additionally return the 8 EMG channels.

    Parameters
    ----------
    subjects : list of int | None
        The subjects to load. If None, all 45 subjects are used.
    sessions : list of str | None
        The sessions to load. If None, the single session is used.
    return_all_modalities : bool
        If True, also return the 8 EMG channels alongside the EEG channels.

    References
    ----------
    .. [1] Perez-Blanco, J. G., Antelis-Ortiz, J. M., Hernandez-Rojas, L. G., &
           Lizarraga-Torreblanca, H. (2026). An EEG-EMG-kinematics dataset of
           wrist movements with a rehabilitation exoskeleton. Scientific Data.
           DOI: https://doi.org/10.1038/s41597-026-07287-z
           Data: https://doi.org/10.6084/m9.figshare.29666735

    .. versionadded:: 1.2.1
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=8,
            channel_types={"eeg": 8, "emg": 8},
            montage="10-10",
            hardware="g.tec g.USBamp (serial UB-2016.05.01)",
            cap_manufacturer="g.tec",
            cap_model="g.GAMMAcap",
            reference="right ear",
            ground="AFz",
            filters={"bandpass": [0.1, 200.0], "notch": 60.0},
            line_freq=60.0,
            sensors=list(_EEG_NAMES),
            auxiliary_channels=AuxiliaryChannelsMetadata(has_emg=True, emg_channels=8),
        ),
        participants=ParticipantMetadata(
            n_subjects=45,
            health_status="healthy",
            gender={"female": 26, "male": 19},
            age_min=20.0,
            age_max=83.0,
            handedness={"right": 40, "left": 5},
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=list(_EVENTS.keys()),
            trial_duration=10.0,
            study_design=(
                "4-direction wrist-pointing task with a 3-DoF wrist exoskeleton "
                "(Biomech Wrist). Flexion-extension controls horizontal cursor "
                "motion, radial-ulnar deviation controls vertical cursor motion. "
                "8 runs x 40 trials (10 per movement per run)."
            ),
            feedback_type="continuous visual",
            stimulus_type="visual target",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-026-07287-z",
            investigators=[
                "Jorge German Perez Blanco",
                "Javier Mauricio Antelis Ortiz",
                "Luis Guillermo Hernandez Rojas",
                "Hector Lizarraga Torreblanca",
            ],
            institution="Tecnologico de Monterrey",
            institution_address="Monterrey, Nuevo Leon, Mexico",
            country="MX",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.29666735",
            publication_year=2026,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=8,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=2.5,
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
            online_feedback=True,
        ),
        data_structure=DataStructureMetadata(
            n_trials=320,
            trials_context=(
                "45 subjects x >=8 runs x 40 trials (10 per class per run); "
                "the exact run count varies by subject (e.g. 9 for sub-01, "
                "10 for sub-02), so per-subject totals exceed this minimum"
            ),
            n_trials_per_class={
                "flexion": 80,
                "extension": 80,
                "radial_deviation": 80,
                "ulnar_deviation": 80,
            },
        ),
        file_format="EDF (BIDS, converted from BCI2000 .dat)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 46)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="PerezBlanco2026",
            interval=[0, 2.5],
            paradigm="imagery",
            doi="10.1038/s41597-026-07287-z",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the EDF file paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn't exist, the "~/mne_data" directory is used. If the
            dataset is not found under the given path, the data will be
            automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A sorted list of the subject's run EDF file paths.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sub = f"sub-{subject:02d}"
        dataset_root = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )

        # Figshare downloads are stored below ``files/<file-id>`` and the
        # subject zip is extracted next to that file. Once the EDFs exist, do
        # not query the Figshare API merely to rediscover the zip id: compute
        # nodes may be offline while the complete extracted subject is local.
        if not force_update:
            for path_folder in (dataset_root / "files", dataset_root):
                eeg_dir = path_folder / sub / "eeg"
                edf_files = sorted(eeg_dir.glob(f"{sub}_task-*_run-*_eeg.edf"))
                if edf_files:
                    return [str(f) for f in edf_files]

        # Resolve the per-subject zip download URL from the Figshare article.
        filelist = dl.fs_get_file_list(PEREZBLANCO2026_ARTICLE_ID)
        file_id = dl.fs_get_file_id(filelist)
        zip_name = f"{sub}.zip"
        if zip_name not in file_id:
            raise ValueError(f"{zip_name} not found in Figshare article")
        url = f"https://ndownloader.figshare.com/files/{file_id[zip_name]}"

        path_zip = Path(dl.data_dl(url, self.code, path=path, force_update=force_update))
        path_folder = path_zip.parent

        eeg_dir = path_folder / sub / "eeg"
        if not eeg_dir.is_dir() or force_update:
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path_folder)

        edf_files = sorted(eeg_dir.glob(f"{sub}_task-*_run-*_eeg.edf"))
        return [str(f) for f in edf_files]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"0": {run_str: mne.io.Raw}}`` for the subject.
        """
        runs = {}
        for run_idx, edf_path in enumerate(self.data_path(subject)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(edf_path, verbose=False).load_data(
                    verbose=False
                )

            raw = self._build_run(raw)
            if raw is None:
                continue
            runs[str(run_idx)] = raw

        return {"0": runs}

    def _build_run(self, raw):
        """Rename channels, extract movement events, and return a clean run."""
        # The EDF carries 16 data channels followed by 7 BCI2000 state channels;
        # rename the data channels by position and locate the state channels by
        # their (possibly truncated) EDF labels.
        rename = {
            raw.ch_names[i]: name
            for i, name in enumerate(_EEG_NAMES + _EMG_NAMES)
            if i < len(raw.ch_names)
        }
        raw.rename_channels(rename)

        target_ch = self._find_channel(raw, "CurrentTarget")
        movonset_ch = self._find_channel(raw, "TrialInitMovStam")
        if target_ch is None or movonset_ch is None:
            return None

        target = raw.get_data(picks=[target_ch])[0]
        movonset = raw.get_data(picks=[movonset_ch])[0]

        # Movement-execution onsets are the rising edges of TrialInitMovStamp.
        binary = (movonset > 0.5).astype(int)
        onsets = np.where(np.diff(binary) == 1)[0] + 1

        events = []
        for sample in onsets:
            tval = int(round(target[sample]))
            if tval in _TARGET_TO_CODE:
                events.append([int(sample), 0, _TARGET_TO_CODE[tval]])
        if not events:
            return None
        events = np.array(events, dtype=int)

        # Keep only EEG (+ EMG when requested); drop the BCI2000 state channels.
        keep = list(_EEG_NAMES)
        if self.return_all_modalities:
            keep = keep + _EMG_NAMES
        keep = [ch for ch in keep if ch in raw.ch_names]
        raw = raw.pick(keep)

        ch_types = {ch: "emg" for ch in _EMG_NAMES if ch in raw.ch_names}
        if ch_types:
            raw.set_channel_types(ch_types)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage(
                make_standard_montage("standard_1005"), on_missing="ignore", verbose=False
            )

        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=_CODE_TO_NAME
        )
        raw = raw.set_annotations(annotations)

        return stim_channels_with_selected_ids(raw, self.event_id)

    @staticmethod
    def _find_channel(raw, prefix):
        """Return the first channel name matching ``prefix`` (EDF may truncate)."""
        for ch in raw.ch_names:
            if ch.startswith(prefix) or prefix.startswith(ch):
                return ch
        return None
