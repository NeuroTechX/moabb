"""Motor imagery and overt spatial attention EEG-BCI dataset.

Forenzo and He (2023), IEEE Trans. Biomed. Eng.
DOI: 10.1109/TBME.2023.3298957
Data DOI: 10.1184/R1/23677098
"""

import logging
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
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

# KiltHub (CMU Figshare) download URLs per subject.
_FIGSHARE_BASE = "https://ndownloader.figshare.com/files/"
_FILE_IDS = {
    1: 41612202,
    2: 41612205,
    3: 41612334,
    4: 41612337,
    5: 41612352,
    6: 41612376,
    7: 41612382,
    8: 41612409,
    9: 41612421,
    10: 41612460,
    11: 41612535,
    12: 41612562,
    13: 41612574,
    14: 41612586,
    15: 41612721,
    16: 41612736,
    17: 41612748,
    18: 41612757,
    19: 41612760,
    20: 41612907,
    21: 41612913,
    22: 41612949,
    23: 41613000,
    24: 41613033,
    25: 41613144,
}

# MI target codes from the paper.
# 1D tasks: 1=left, 2=right (LR axis); 3=up, 4=down (UD axis)
# 2D tasks: 1=left, 2=right, 3=up, 4=down
_MI_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
}

_SFREQ = 1000.0


class Forenzo2023(BaseDataset):
    """Motor imagery + spatial attention dataset from Forenzo & He 2023.

    Dataset from the article *Integrating simultaneous motor imagery and
    spatial attention for EEG-BCI control* [1]_.

    It contains EEG data from 25 subjects recorded with a 64-channel
    Neuroscan system across 5 sessions on different days. Multiple task
    conditions were tested:

    - **MI**: Motor imagery only (left/right hand, 1D)
    - **OSA**: Overt spatial attention only
    - **MIOSA**: Combined MI + spatial attention

    By default, only the **MI** task (left-right axis) runs are loaded,
    yielding a standard 2-class left/right hand motor imagery dataset.

    Each MI run contains 5 trials of 60 seconds each (continuous pursuit
    paradigm with cursor control).

    Parameters
    ----------
    task : str
        Which task to load: ``"MI"`` (default), ``"OSA"``, ``"MIOSA"``,
        ``"MIOSA1"``, or ``"MIOSA2"``.
    axis : str
        Which control axis: ``"LR"`` (default) or ``"UD"``.

    References
    ----------
    .. [1] Forenzo, D., & He, B. (2024). Integrating simultaneous motor
           imagery and spatial attention for EEG-BCI control. IEEE Trans.
           Biomed. Eng., 71(1), 282-294.
           https://doi.org/10.1109/TBME.2023.3298957
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="standard_1005",
            hardware="Neuroscan Quik-Cap 64-ch, SynAmps 2/RT",
            sensor_type="Ag/AgCl",
            reference="between Cz and CPz",
            filters={"lowpass": 200, "notch_hz": 60},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"female": 10, "male": 15},
            age_mean=25.5,
            handedness="right-handed (24 of 25)",
            bci_experience="mixed (19 naive, 6 experienced)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_MI_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=60.0,
            study_design=(
                "5-session BCI study with motor imagery (MI), "
                "overt spatial attention (OSA), and combined (MIOSA) tasks"
            ),
            feedback_type="cursor",
            stimulus_type="continuous pursuit",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TBME.2023.3298957",
            investigators=["Dylan Forenzo", "Bin He"],
            institution="Carnegie Mellon University",
            country="US",
            data_url="https://kilthub.cmu.edu/articles/dataset/23677098",
            publication_year=2023,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=5,
        runs_per_session=3,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            imagery_duration_s=60.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=1875,
            trials_context="25 subjects x 5 sessions x 3 MI runs x 5 trials",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="MAT",
    )

    def __init__(self, task="MI", axis="LR", subjects=None, sessions=None):
        self.task = task
        self.axis = axis
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=5,
            events=dict(_MI_EVENTS),
            code="Forenzo2023",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1109/TBME.2023.3298957",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        subj_dir = base / f"Subject{subject:02d}"
        if not subj_dir.exists():
            subj_dir = base

        sessions = {}
        # Find all .mat files matching this task and axis
        mat_files = sorted(subj_dir.rglob("*.mat"))

        # Group by session
        session_files = {}
        for mf in mat_files:
            mat = loadmat(
                str(mf), squeeze_me=True, variable_names=["task", "axis", "session"]
            )
            file_task = str(mat.get("task", ""))
            file_axis = str(mat.get("axis", ""))
            file_sess = int(mat.get("session", 0))

            if file_task == self.task and file_axis == self.axis:
                session_files.setdefault(file_sess, []).append(mf)

        for sess_num, files in sorted(session_files.items()):
            runs = {}
            for run_idx, mf in enumerate(sorted(files)):
                raw = self._load_mat_run(mf)
                runs[str(run_idx)] = raw
            sessions[str(sess_num - 1)] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No {self.task}/{self.axis} data for subject {subject} in {base}"
            )
        return sessions

    def _load_mat_run(self, mat_path):
        """Load a single run .mat file into MNE Raw."""
        mat = loadmat(str(mat_path), squeeze_me=True)

        data = mat["data"]  # (channels x timepoints)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Get channel labels if available
        labels = mat.get("labels", None)
        if labels is not None:
            if hasattr(labels, "tolist"):
                ch_names = [str(ch).strip() for ch in labels.tolist()]
            else:
                ch_names = [str(labels)]
        else:
            ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

        # Get targets and events
        targets = mat.get("targets", np.array([]))
        events_struct = mat.get("event", None)

        fs = float(mat.get("fs", _SFREQ))

        # Build info
        ch_types = ["eeg"] * len(ch_names) + ["stim"]
        info = mne.create_info(
            ch_names=ch_names + ["STI"],
            ch_types=ch_types,
            sfreq=fs,
        )

        # Build stim channel from events
        stim = np.zeros((1, data.shape[1]))
        if events_struct is not None and hasattr(events_struct, "__len__"):
            for i, ev in enumerate(
                events_struct if hasattr(events_struct, "__iter__") else [events_struct]
            ):
                latency = int(getattr(ev, "latency", 0) * fs / 1000)
                target = int(targets[i]) if i < len(targets) else 1
                if 0 <= latency < data.shape[1]:
                    stim[0, latency] = target

        # Scale to volts
        if np.abs(data).max() > 1e-3:
            data = data * 1e-6

        full_data = np.concatenate([data, stim], axis=0)
        raw = mne.io.RawArray(data=full_data, info=info, verbose=False)

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Forenzo2023", path)
        basepath = Path(path) / "MNE-forenzo2023-data"
        basepath.mkdir(parents=True, exist_ok=True)

        subj_dir = basepath / f"Subject{subject:02d}"
        if subj_dir.exists() and list(subj_dir.glob("*.mat")):
            return str(basepath)

        # Download per-subject ZIP from KiltHub.
        file_id = _FILE_IDS.get(subject)
        if file_id is None:
            raise ValueError(f"No download URL for subject {subject}")

        url = f"{_FIGSHARE_BASE}{file_id}"
        zip_name = f"Subject{subject:02d}.zip"
        zip_path = basepath / zip_name

        if not zip_path.exists():
            log.info("Downloading Forenzo2023 subject %d (~3.8 GB)...", subject)
            dl_path = dl.data_dl(
                url,
                "Forenzo2023",
                path=str(basepath),
                force_update=force_update,
                verbose=verbose,
            )
            # Rename downloaded file to expected location.
            dl_path = Path(dl_path)
            if dl_path != zip_path:
                dl_path.rename(zip_path)

        # Extract
        if zip_path.exists() and not subj_dir.exists():
            subj_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(subj_dir))

        return str(basepath)
