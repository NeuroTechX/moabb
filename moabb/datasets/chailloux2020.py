"""Single-Option P300-BCI dataset.

Chailloux Peguero, Mendoza-Montoya, and Antelis (2020), Sensors.
DOI: 10.3390/s20247198
Data DOI: 10.18112/openneuro.ds003190.v1.0.1
"""

import logging
from pathlib import Path

import mne
import numpy as np

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_OPENNEURO_ID = "ds003190"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# 8 EEG channel names (from channels.tsv, normalized to standard_1020 case).
_CH_NAMES = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

# Channel name mapping from the raw BrainVision files.
_CH_RENAME = {"FZ": "Fz", "PZ": "Pz"}

# Runs for each task.  task-ctos has 2 runs (5F, 5H); task-cnos has 5 runs.
_CNOS_RUNS = ["4", "6", "7", "8", "9"]
_CTOS_RUNS = ["5F", "5H"]


def _subject_str(subject):
    """BIDS subject label: sub-01 .. sub-019."""
    if subject < 10:
        return f"sub-0{subject}"
    return f"sub-0{subject}"


class Chailloux2020(BaseDataset):
    """P300 BCI dataset from Chailloux Peguero et al 2020.

    Dataset from the paper [1]_.

    **Dataset Description**

    Nineteen healthy volunteers participated in three sessions
    (10-21 days apart) of a P300-based BCI using a single-option
    interface. Two stimulation types were tested: Standard Flash (SF)
    and Cartoon Face (CF).  Each session contains 5 runs of varying
    grid sizes (4-9 symbols) plus 1-2 runs comparing flash types.

    EEG was recorded at 256 Hz from 8 electrodes (Fz, Cz, P3, Pz,
    P4, PO7, PO8, Oz) using a g.USBamp amplifier. Target/NonTarget
    events are derived from per-block target markers embedded in the
    stimulus annotations.

    Parameters
    ----------
    task : str
        Which task to load: ``"cnos"`` (number-of-symbols runs,
        default), ``"ctos"`` (type-of-stimulus runs), or ``"all"``.

    References
    ----------
    .. [1] Chailloux Peguero, J. D., Mendoza-Montoya, O., &
           Antelis, J. M. (2020). Single-Option P300-BCI Performance
           Is Affected by Visual Stimulation Conditions. Sensors,
           20(24), 7198.
           https://doi.org/10.3390/s20247198
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="standard_1020",
            hardware="g.USBamp (g.tec)",
            sensor_type="g.SCARABEO (passive)",
            reference="right earlobe",
            ground="AFz",
            sensors=list(_CH_NAMES),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=19,
            health_status="healthy",
            gender={"male": 11, "female": 8},
            age_mean=25.0,
            age_min=19,
            age_max=33,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=1.0,
            study_design=(
                "P300 single-option interface; 2 stimulation types "
                "(Standard Flash, Cartoon Face); 5 grid sizes (4-9 symbols)"
            ),
            feedback_type="none",
            stimulus_type="flash / cartoon face",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3390/s20247198",
            investigators=[
                "J. David Chailloux Peguero",
                "Omar Mendoza-Montoya",
                "Javier M. Antelis",
            ],
            institution="Tecnologico de Monterrey",
            country="MX",
            publication_year=2020,
            data_url="https://openneuro.org/datasets/ds003190",
            license="CC0",
        ),
        sessions_per_subject=3,
        runs_per_session=7,
        tags=Tags(pathology=["Healthy"], modality=["ERP"], type=["P300"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", isi_ms=75.0, soa_ms=150.0
        ),
        data_structure=DataStructureMetadata(
            n_trials="varies", trials_context="~198 flashes per block, ~6 blocks per run"
        ),
        data_processed=False,
        file_format="BrainVision",
    )

    def __init__(self, task="cnos", subjects=None, sessions=None):
        if task not in ("cnos", "ctos", "all"):
            raise ValueError(f"task must be 'cnos', 'ctos', or 'all', got {task!r}")
        self._task = task

        super().__init__(
            subjects=list(range(1, 20)),
            sessions_per_subject=3,
            events={"Target": 2, "NonTarget": 1},
            code="Chailloux2020",
            interval=[0, 1],
            paradigm="p300",
            doi="10.3390/s20247198",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        self.data_path(subject)
        base = self._subject_base(subject)
        sessions = {}

        for ses_idx in range(1, 4):
            ses_str = f"ses-{ses_idx:02d}"
            ses_dir = base / ses_str / "eeg"
            if not ses_dir.exists():
                continue

            runs = {}
            run_labels = self._run_labels()
            for run_label in run_labels:
                for task_name in self._task_names():
                    vhdr = ses_dir / (
                        f"{_subject_str(subject)}_{ses_str}_"
                        f"task-{task_name}_run-{run_label}_eeg.vhdr"
                    )
                    if not vhdr.exists():
                        continue
                    try:
                        raw = self._load_run(str(vhdr))
                    except Exception:
                        log.warning("Failed to load %s, skipping.", vhdr)
                        continue
                    runs[str(len(runs))] = raw

            if runs:
                sessions[str(ses_idx - 1)] = runs

        return sessions

    def _task_names(self):
        if self._task == "cnos":
            return ["cnos"]
        elif self._task == "ctos":
            return ["ctos"]
        return ["cnos", "ctos"]

    def _run_labels(self):
        if self._task == "cnos":
            return _CNOS_RUNS
        elif self._task == "ctos":
            return _CTOS_RUNS
        return _CNOS_RUNS + _CTOS_RUNS

    @staticmethod
    def _load_run(vhdr_path):
        """Load a BrainVision file and build Target/NonTarget stim channel."""
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

        # Fix channel names and types.
        rename = {ch: _CH_RENAME.get(ch, ch) for ch in raw.ch_names}
        raw.rename_channels(rename)
        eeg_chs = [ch for ch in raw.ch_names if ch in _CH_NAMES]
        non_eeg = [ch for ch in raw.ch_names if ch not in _CH_NAMES]
        if non_eeg:
            raw.drop_channels(non_eeg)
        raw.set_channel_types(dict.fromkeys(eeg_chs, "eeg"))
        raw.set_montage("standard_1020", on_missing="warn")

        # Extract events from annotations.
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Build stim channel using per-block target markers.
        stim_data = np.zeros(raw.n_times)
        current_target = None

        # Sort events by time.
        order = np.argsort(events[:, 0])
        events = events[order]

        # Reverse map: event_id value -> annotation string.
        id_to_desc = {v: k for k, v in event_id.items()}

        for sample, _, eid in events:
            desc = id_to_desc.get(eid, "")
            # BrainVision annotations are prefixed with "trigger/" by MNE
            # (e.g. "trigger/201" instead of "201").  Strip the prefix so
            # the numeric matching logic below works correctly.
            if "/" in desc:
                desc = desc.split("/")[-1]
            # Target marker: "10X" where X is the target symbol.
            if desc.startswith("10") and len(desc) == 3 and desc[2:].isdigit():
                current_target = desc[2:]
                continue
            # Block markers (201, 202, 203, 200).
            if desc in ("200", "201", "202", "203"):
                continue
            # Stimulus flash: single digit 1-9.
            if desc.isdigit() and len(desc) <= 2 and current_target is not None:
                if desc == current_target:
                    stim_data[sample] = 2  # Target
                else:
                    stim_data[sample] = 1  # NonTarget

        # Add stim channel.
        stim_info = mne.create_info(["STI"], raw.info["sfreq"], ["stim"])
        stim_raw = mne.io.RawArray(stim_data[np.newaxis], stim_info, verbose=False)
        raw.add_channels([stim_raw], force_update_info=True)

        return raw

    def _subject_base(self, subject):
        path = dl.get_dataset_path("Chailloux2020", None)
        return Path(path) / "MNE-chailloux2020-data" / _subject_str(subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = self._subject_base(subject)
        subj_str = _subject_str(subject)

        import requests as _requests

        for ses_idx in range(1, 4):
            ses_str = f"ses-{ses_idx:02d}"
            ses_dir = base / ses_str / "eeg"

            run_labels = self._run_labels()
            for task_name in self._task_names():
                for run_label in run_labels:
                    prefix = f"{subj_str}_{ses_str}_task-{task_name}_run-{run_label}"
                    for ext in [".vhdr", ".vmrk", ".eeg"]:
                        fname = f"{prefix}_eeg{ext}"
                        local = ses_dir / fname
                        if local.exists() and not force_update:
                            continue
                        url = f"{_S3_BASE}/{subj_str}/{ses_str}/eeg/{fname}"
                        ses_dir.mkdir(parents=True, exist_ok=True)
                        log.info("Downloading %s ...", fname)
                        try:
                            resp = _requests.get(url, stream=True, timeout=120)
                            if resp.status_code == 404:
                                continue
                            resp.raise_for_status()
                            with open(local, "wb") as fout:
                                for chunk in resp.iter_content(chunk_size=8192):
                                    fout.write(chunk)
                        except Exception as e:
                            log.warning("Download failed for %s: %s", fname, e)

        return str(base)
