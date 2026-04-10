"""Home appliance control P300 BCI dataset.

Lee, Kim, Heo, et al. (2024), Frontiers in Human Neuroscience.
DOI: 10.3389/fnhum.2024.1320457
Data: https://github.com/jml226/Home-Appliance-Control-Dataset
"""

import logging
from functools import partialmethod
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

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/jml226/Home-Appliance-Control-Dataset/main"
)
_DOI = "10.3389/fnhum.2024.1320457"
_SIGN = "lee2024erp"

# fmt: off
_31_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT9", "FC5", "FC1", "FC2", "FC6", "FT10",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
_25_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4",
    "CP5", "CP1", "CP2", "CP6",
    "P3", "Pz", "P4",
    "O1", "Oz", "O2",
]
# fmt: on

# Per-experiment configuration.
_EXPERIMENT_CONFIGS = {
    "TV": {
        "dir_name": "TV",
        "n_subjects": 30,
        "n_classes": 4,
        "n_eeg": 31,
        "ch_names": _31_CH_NAMES,
        "zero_pad": True,
        "has_training": True,
        "training_combined": False,
        "demographics": {
            "gender": {"male": 23, "female": 7},
            "age_mean": 21.63,
            "age_std": 2.31,
        },
    },
    "DL": {
        "dir_name": "Doorlock",
        "n_subjects": 15,
        "n_classes": 4,
        "n_eeg": 31,
        "ch_names": _31_CH_NAMES,
        "zero_pad": True,
        "has_training": True,
        "training_combined": True,
        "demographics": {
            "gender": {"male": 12, "female": 3},
            "age_mean": 22.87,
            "age_std": 2.07,
        },
    },
    "EL": {
        "dir_name": "ElectricLight",
        "n_subjects": 15,
        "n_classes": 4,
        "n_eeg": 31,
        "ch_names": _31_CH_NAMES,
        "zero_pad": True,
        "has_training": True,
        "training_combined": True,
        "demographics": {
            "gender": {"male": 10, "female": 5},
            "age_mean": 22.13,
            "age_std": 2.20,
        },
    },
    "BS": {
        "dir_name": "BluetoothSpeaker",
        "n_subjects": 14,
        "n_classes": 6,
        "n_eeg": 31,
        "ch_names": _31_CH_NAMES,
        "zero_pad": True,
        "has_training": False,
        "training_combined": False,
        "demographics": {
            "gender": {"male": 9, "female": 5},
            "age_mean": 22.64,
            "age_std": 3.08,
        },
    },
    "AC": {
        "dir_name": "AirConditioner",
        "n_subjects": 10,
        "n_classes": 4,
        "n_eeg": 25,
        "ch_names": _25_CH_NAMES,
        "zero_pad": False,
        "has_training": True,
        "training_combined": True,
        "demographics": {
            "gender": {"male": 6, "female": 4},
            "age_mean": 22.40,
            "age_std": 2.59,
        },
    },
}


def _make_metadata(experiment):
    """Build DatasetMetadata for a given experiment."""
    config = _EXPERIMENT_CONFIGS[experiment]
    return DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=config["n_eeg"],
            channel_types={"eeg": config["n_eeg"]},
            montage="standard_1020",
            hardware="actiCHamp (Brain Products)",
            sensor_type="active",
            reference="linked mastoids",
            sensors=list(config["ch_names"]),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=config["n_subjects"],
            health_status="healthy",
            species="human",
            **config.get("demographics", {}),
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=1.0,
            study_design=(
                f"P300 BCI for {experiment} home appliance control; "
                f"{config['n_classes']}-class oddball; LCD display"
            ),
            feedback_type="visual",
            stimulus_type="flash",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Jongmin Lee",
                "Minju Kim",
                "Dojin Heo",
                "Jongsu Kim",
                "Min-Ki Kim",
                "Taejun Lee",
                "Jongwoo Park",
                "HyunYoung Kim",
                "Minho Hwang",
                "Laehyun Kim",
                "Sung-Phil Kim",
            ],
            institution="Ulsan National Institute of Science and Technology",
            country="KR",
            publication_year=2024,
            data_url="https://github.com/jml226/Home-Appliance-Control-Dataset",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        tags=Tags(pathology=["Healthy"], modality=["ERP"], type=["P300"]),
        bci_application=BCIApplicationMetadata(
            applications=["home_appliance_control"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", soa_ms=750.0
        ),
        data_structure=DataStructureMetadata(
            n_trials="50 training + 30 testing blocks per subject",
            trials_context="per_subject",
        ),
        data_processed=False,
        file_format="MATLAB",
    )


class Lee2024(BaseDataset):
    """P300 BCI dataset for home appliance control from Lee et al 2024.

    Dataset from the paper [1]_.

    **Dataset Description**

    Eighty-four subjects performed ERP-based BCI tasks controlling
    five types of home appliances: television (TV, 30 subjects),
    door lock (DL, 15), electric light (EL, 15), Bluetooth speaker
    (BS, 14), and air conditioner (AC, 10).

    EEG was recorded at 500 Hz using actiCHamp (Brain Products) with
    31 channels (LCD experiments) or 25 channels (AR experiment).
    Each subject completed 50 training blocks and 30 testing blocks.

    This base class should not be instantiated directly. Use the
    experiment-specific subclasses (Lee2024_TV, Lee2024_DL, etc.).

    References
    ----------
    .. [1] Lee, J., Kim, M., Heo, D., et al. (2024). A comprehensive
           dataset for home appliance control using ERP-based BCIs
           with the application of inter-subject transfer learning.
           Frontiers in Human Neuroscience, 18, 1320457.
           https://doi.org/10.3389/fnhum.2024.1320457
    """

    def __init__(self, experiment, subjects=None, sessions=None):
        self._experiment = experiment
        config = _EXPERIMENT_CONFIGS[experiment]
        self.METADATA = _make_metadata(experiment)

        super().__init__(
            subjects=list(range(1, config["n_subjects"] + 1)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code=f"Lee2024-{experiment}",
            interval=[0, 1],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        self.data_path(subject)
        config = _EXPERIMENT_CONFIGS[self._experiment]

        base = self._subject_dir(subject)
        runs = {}

        # Load testing blocks (all experiments have these).
        for block_idx in range(1, 31):
            mat_path = self._block_path(base, subject, "Testing", block_idx, config)
            if not mat_path.exists():
                continue
            try:
                raw = self._load_block(mat_path, config)
                if raw is not None:
                    runs[str(len(runs))] = raw
            except Exception:
                log.warning("Failed to load %s, skipping.", mat_path)

        # Load per-block training files if available.
        if config["has_training"] and not config["training_combined"]:
            for block_idx in range(1, 51):
                mat_path = self._block_path(base, subject, "Training", block_idx, config)
                if not mat_path.exists():
                    continue
                try:
                    raw = self._load_block(mat_path, config)
                    if raw is not None:
                        runs[str(len(runs))] = raw
                except Exception:
                    log.warning("Failed to load %s, skipping.", mat_path)

        # Load combined training file if applicable.
        if config["has_training"] and config["training_combined"]:
            subj_str = self._subj_str(subject, config)
            combined_path = base / f"{subj_str}_Training.mat"
            if combined_path.exists():
                try:
                    raw = self._load_combined(combined_path, config)
                    if raw is not None:
                        runs[str(len(runs))] = raw
                except Exception:
                    log.warning("Failed to load %s, skipping.", combined_path)

        return {"0": runs} if runs else {}

    def _subject_dir(self, subject):
        config = _EXPERIMENT_CONFIGS[self._experiment]
        path = dl.get_dataset_path(_SIGN, None)
        base = Path(path) / f"MNE-{_SIGN}-data" / config["dir_name"]
        subj_str = self._subj_str(subject, config)
        return base / f"Dat_{subj_str}"

    @staticmethod
    def _subj_str(subject, config):
        if config["zero_pad"]:
            return f"sub{subject:02d}"
        return f"sub{subject}"

    @staticmethod
    def _block_path(base, subject, phase, block_idx, config):
        subj_str = Lee2024._subj_str(subject, config)
        return base / f"{subj_str}_{phase}{block_idx}.mat"

    @staticmethod
    def _load_block(mat_path, config):
        """Load a single-block .mat file and return Raw."""
        data = loadmat(str(mat_path), squeeze_me=True)

        # Per-block files use sig_vec / trigger.
        if "sig_vec" in data:
            signals = np.array(data["sig_vec"], dtype=np.float64)
            trigger = np.array(data["trigger"], dtype=np.float64).ravel()
        elif "sig" in data:
            signals = np.array(data["sig"], dtype=np.float64)
            trigger = np.array(data["trig"], dtype=np.float64).ravel()
        else:
            log.warning("Unrecognized .mat structure in %s", mat_path)
            return None

        if signals.ndim == 1:
            return None

        # Ensure (n_channels, n_samples).
        n_eeg = config["n_eeg"]
        if signals.shape[0] != n_eeg and signals.shape[1] == n_eeg:
            signals = signals.T

        return Lee2024._build_raw(signals, trigger, config)

    @staticmethod
    def _load_combined(mat_path, config):
        """Load a combined training .mat file."""
        data = loadmat(str(mat_path), squeeze_me=True)

        if "sig" in data:
            signals = np.array(data["sig"], dtype=np.float64)
            trigger = np.array(data["trig"], dtype=np.float64).ravel()
        elif "sig_vec" in data:
            signals = np.array(data["sig_vec"], dtype=np.float64)
            trigger = np.array(data["trigger"], dtype=np.float64).ravel()
        else:
            return None

        n_eeg = config["n_eeg"]
        if signals.shape[0] != n_eeg and signals.shape[1] == n_eeg:
            signals = signals.T

        return Lee2024._build_raw(signals, trigger, config)

    @staticmethod
    def _build_raw(signals, trigger, config):
        """Build MNE Raw from EEG signals and trigger vector."""
        sfreq = 500.0
        n_eeg = config["n_eeg"]
        ch_names = list(config["ch_names"])

        signals = signals * 1e-6  # uV -> V

        # Extract Target vs NonTarget from trigger.
        stim_channel = np.zeros(signals.shape[1])

        # Find block boundaries: 11=start, 12=stim_start, 13=end.
        trig = trigger.astype(int)
        block_starts = np.where(trig == 11)[0]
        stim_starts = np.where(trig == 12)[0]

        for bs in block_starts:
            # Find the next stim_start after this block_start.
            ss_candidates = stim_starts[stim_starts > bs]
            if len(ss_candidates) == 0:
                continue
            ss = ss_candidates[0]

            # Target class is the trigger value between block_start and stim_start.
            between = trig[bs + 1 : ss]
            target_vals = between[(between >= 1) & (between <= 6)]
            if len(target_vals) == 0:
                continue
            target_class = target_vals[0]

            # Find the block end.
            end_candidates = np.where(trig[ss:] == 13)[0]
            if len(end_candidates) == 0:
                block_end = len(trig)
            else:
                block_end = ss + end_candidates[0]

            # Label flashes within the stimulation period.
            for i in range(ss, block_end):
                if 1 <= trig[i] <= 6:
                    if trig[i] == target_class:
                        stim_channel[i] = 2  # Target
                    else:
                        stim_channel[i] = 1  # NonTarget

        ch_names_full = ch_names + ["STI"]
        ch_types = ["eeg"] * n_eeg + ["stim"]
        all_data = np.vstack([signals, stim_channel[np.newaxis]])

        info = mne.create_info(ch_names_full, sfreq, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)
        raw.set_montage("standard_1020", on_missing="warn")

        # Convert stim channel events to annotations for BIDS compatibility.
        events = mne.find_events(raw, stim_channel="STI", verbose=False)
        if len(events) > 0:
            annot = mne.annotations_from_events(
                events, sfreq, event_desc={1: "NonTarget", 2: "Target"}
            )
            raw.set_annotations(annot)
        else:
            # Some experiments (AC, EL) have no events in stim channel.
            # Add a minimal annotation so BIDS export doesn't fail.
            raw.set_annotations(
                mne.Annotations(onset=[0], duration=[0], description=["stimulus"])
            )

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        config = _EXPERIMENT_CONFIGS[self._experiment]
        subj_dir = self._subject_dir(subject)
        subj_str = self._subj_str(subject, config)

        import requests as _requests

        files_to_dl = []

        # Testing blocks.
        for i in range(1, 31):
            files_to_dl.append(f"{subj_str}_Testing{i}.mat")

        # Training blocks (per-block).
        if config["has_training"] and not config["training_combined"]:
            for i in range(1, 51):
                files_to_dl.append(f"{subj_str}_Training{i}.mat")

        # Combined training.
        if config["has_training"] and config["training_combined"]:
            files_to_dl.append(f"{subj_str}_Training.mat")

        # Calibration signal.
        files_to_dl.append("cal_sig.mat")

        subj_dir.mkdir(parents=True, exist_ok=True)

        for fname in files_to_dl:
            local = subj_dir / fname
            if local.exists() and not force_update:
                continue
            dir_name = config["dir_name"]
            url = f"{_GITHUB_RAW}/{dir_name}/Dat_{subj_str}/{fname}"
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

        return str(subj_dir)


class Lee2024_TV(Lee2024):
    """Television control experiment (30 subjects, 4 classes, 31 EEG ch)."""

    __init__ = partialmethod(Lee2024.__init__, "TV")


class Lee2024_DL(Lee2024):
    """Door lock control experiment (15 subjects, 4 classes, 31 EEG ch)."""

    __init__ = partialmethod(Lee2024.__init__, "DL")


class Lee2024_EL(Lee2024):
    """Electric light control experiment (15 subjects, 4 classes, 31 EEG ch)."""

    __init__ = partialmethod(Lee2024.__init__, "EL")


class Lee2024_BS(Lee2024):
    """Bluetooth speaker experiment (14 subjects, 6 classes, 31 EEG ch)."""

    __init__ = partialmethod(Lee2024.__init__, "BS")


class Lee2024_AC(Lee2024):
    """Air conditioner control experiment (10 subjects, 4 classes, 25 EEG ch)."""

    __init__ = partialmethod(Lee2024.__init__, "AC")
