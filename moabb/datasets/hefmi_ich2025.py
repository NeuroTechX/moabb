"""Hybrid EEG-fNIRS motor imagery dataset for intracerebral hemorrhage.

Shi, Chen, et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-06100-7
Data DOI: 10.6084/m9.figshare.28955456.v4
"""

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


log = logging.getLogger(__name__)

# Figshare article ID and API endpoint.
_FIGSHARE_ARTICLE = 28955456
_FIGSHARE_API = f"https://api.figshare.com/v2/articles/{_FIGSHARE_ARTICLE}"

# Event codes (from the paper and MakeDatasetFromRaw.m).
_EVENTS = {"left_hand": 1, "right_hand": 2}

# 32 EEG channel names (from mne_analysis.py in the code.zip).
# fmt: off
_CH_NAMES = [
    "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
    "C4", "T7", "T8", "FC2", "FC5", "FC6", "Pz", "CP5",
    "PO3", "PO4", "Oz", "Fp2", "Fp1", "Fz", "F3", "F4",
    "F7", "F8", "P3", "P4", "P7", "P8", "O1", "O2",
]
# fmt: on

_SFREQ = 256.0

# Pre-built manifest mapping MOABB subject IDs to Figshare file IDs.
# Subjects 1-17 are healthy controls, 18-37 are ICH patients.
# Each subject has 1-6 sessions. File naming: {session}_epo.mat.
# This manifest was built by sorting EEG epoch files by Figshare file ID
# and grouping by session-1 boundaries.
# NOTE: This manifest may need updating if Figshare article version changes.
_MANIFEST = None  # Populated lazily from Figshare API.


class HefmiIch2025(BaseDataset):
    """Hybrid EEG-fNIRS MI dataset for ICH from Shi et al 2025.

    Dataset from the article *HEFMI-ICH: a hybrid EEG-fNIRS motor
    imagery dataset for brain-computer interface in intracerebral
    hemorrhage* [1]_.

    It contains EEG data from 37 subjects (17 healthy controls +
    20 ICH patients) recorded with 32-channel EEG at 256 Hz.
    The motor imagery task involves left and right hand grasping.

    Each session has 30 trials (15 left, 15 right). Trial structure:
    2 s cue + 10 s MI + 15 s rest. Sessions vary from 1 to 6 per
    subject.

    The data is pre-epoched in .mat format on Figshare.
    Only the EEG portion is loaded (fNIRS is excluded).

    Parameters
    ----------
    group : str
        Which subject group to load: ``"all"`` (default, subjects
        1-37), ``"healthy"`` (subjects 1-17), or ``"patient"``
        (subjects 18-37).

    References
    ----------
    .. [1] Shi, J., Chen, D., et al. (2025). HEFMI-ICH: a hybrid
           EEG-fNIRS motor imagery dataset for brain-computer
           interface in intracerebral hemorrhage. Scientific Data.
           https://doi.org/10.1038/s41597-025-06100-7
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="biosemi32",
            hardware="g.HIamp (g.tec medical engineering GmbH)",
            filters={},
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=37,
            health_status="mixed (17 healthy, 20 ICH patients)",
            gender={"female": 8, "male": 29},
            age_min=20.0,
            age_max=65.0,
            species="human",
            handedness="right-handed",
            clinical_population="intracerebral hemorrhage (ICH)",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=27.0,
            study_design=(
                "2-class hand MI (left/right grasping) for ICH "
                "rehabilitation. 17 healthy + 20 ICH patients, "
                "1-6 sessions per subject."
            ),
            feedback_type="none",
            stimulus_type="directional arrow + auditory beep",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-06100-7",
            investigators=[
                "Jian Shi",
                "Danyang Chen",
                "Xingwei Zhao",
                "Zhixian Zhao",
                "Shengjie Li",
                "Yeguang Xu",
                "Tao Ding",
                "Zheng Zhu",
                "Peng Zhang",
                "Qing Ye",
                "Yingxin Tang",
                "Ping Zhang",
                "Bo Tao",
                "Zhouping Tang",
            ],
            institution="Huazhong University of Science and Technology",
            country="CN",
            data_url="https://figshare.com/articles/dataset/28955456",
            publication_year=2025,
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy", "Stroke"],
            modality=["Motor"],
            type=["Clinical", "Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=10.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=3330,
            trials_context=("37 subjects x ~3 sessions x 30 trials = ~3330"),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "FBCSP+SVM", "EEGBaseNet", "TF+SVM"],
            feature_extraction=["CSP", "FBCSP", "time-frequency features"],
            frequency_bands={"preprocessing": [0.5, 30.0]},
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold", cv_folds=5, evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"], environment="clinical", online_feedback=False
        ),
        data_processed=True,
        file_format="MAT (pre-epoched)",
    )

    def __init__(
        self, group="all", subjects=None, sessions=None, *, return_all_modalities=False
    ):
        self.group = group

        if group == "healthy":
            subj_list = list(range(1, 18))
        elif group == "patient":
            subj_list = list(range(18, 38))
        elif group == "all":
            subj_list = list(range(1, 38))
        else:
            raise ValueError(
                f"group must be 'all', 'healthy', or 'patient', got {group!r}"
            )

        super().__init__(
            subjects=subj_list,
            sessions_per_subject=3,
            events=dict(_EVENTS),
            code="HefmiIch2025",
            interval=[0, 10],
            paradigm="imagery",
            doi="10.1038/s41597-025-06100-7",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))

        # Find epoch files for this subject.
        manifest = self._get_manifest(base)
        if subject not in manifest:
            raise FileNotFoundError(f"No EEG epoch files found for subject {subject}")

        import requests

        sessions = {}
        for sess_idx, (file_id, file_name) in enumerate(manifest[subject]):
            mat_path = base / file_name
            if not mat_path.exists():
                # Download this specific file directly from Figshare.
                url = f"https://ndownloader.figshare.com/files/{file_id}"
                log.info("Downloading %s (%s) ...", file_name, url)
                resp = requests.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                with open(mat_path, "wb") as fout:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fout.write(chunk)

            if not mat_path.exists():
                log.warning("Missing %s for subject %d", file_name, subject)
                continue

            try:
                raw = self._load_epoch_mat(mat_path)
                sessions[str(sess_idx)] = {"0": raw}
            except Exception as e:
                log.warning("Failed to load %s: %s", file_name, e)

        if not sessions:
            raise FileNotFoundError(f"No loadable data for subject {subject}")
        return sessions

    def _load_epoch_mat(self, mat_path):
        """Load a pre-epoched MAT file and reconstruct continuous Raw."""
        mat = loadmat(str(mat_path), squeeze_me=True)

        # Navigate the struct: epo.X, epo.y, epo.fs
        epo = mat.get("epo", mat)
        if hasattr(epo, "dtype") and epo.dtype.names:
            X = epo["X"].item()  # (samples, channels, trials)
            y = epo["y"].item().ravel()  # (trials,) -- 0=left, 1=right
            fs = float(epo["fs"].item()) if "fs" in epo.dtype.names else _SFREQ
        else:
            X = mat.get("X", mat.get("data"))
            y = mat.get("y", mat.get("labels", np.array([]))).ravel()
            fs = float(mat.get("fs", _SFREQ))

        if X.ndim != 3:
            raise ValueError(f"Expected 3D epoch array, got shape {X.shape}")

        n_samples, n_ch, n_trials = X.shape
        ch_names = (
            list(_CH_NAMES[:n_ch])
            if n_ch <= len(_CH_NAMES)
            else [f"EEG{i + 1}" for i in range(n_ch)]
        )

        # Transpose from (samples, channels, trials) to (trials, channels, samples).
        epochs = X.transpose(2, 1, 0)

        # y: 0=left, 1=right -> MOABB: 1=left, 2=right
        event_ids = y.astype(int) + 1

        # MI onset is at 12s into each epoch (from MakeDatasetFromRaw.m).
        return build_raw_from_epochs(
            epochs,
            ch_names,
            fs,
            event_ids,
            "standard_1005",
            buffer_samples=int(0.5 * fs),
            onset_sample=int(12 * fs),
        )

    def _get_manifest(self, basepath):
        """Build subject-to-file mapping from Figshare API."""
        global _MANIFEST
        if _MANIFEST is not None:
            return _MANIFEST

        import json

        manifest_path = basepath / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                _MANIFEST = {int(k): v for k, v in json.load(f).items()}
            return _MANIFEST

        # Fetch file list from Figshare API.
        log.info("Fetching HEFMI-ICH file manifest from Figshare...")
        try:
            import requests

            files = []
            page = 1
            while True:
                resp = requests.get(
                    f"{_FIGSHARE_API}/files",
                    params={"page": page, "page_size": 100},
                    timeout=30,
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                files.extend(batch)
                page += 1

        except Exception as e:
            log.warning("Could not fetch Figshare manifest: %s", e)
            _MANIFEST = {}
            return _MANIFEST

        # Filter to EEG epoch files (~42 MB each).
        # The dataset has both EEG (~42 MB) and fNIRS (~28 MB) epoch files.
        # Use size > 40 MB to select EEG only (all EEG files are 40.9-43.3 MB).
        eeg_files = [
            (f["id"], f["name"], f["size"])
            for f in files
            if f["name"].endswith("_epo.mat") and f["size"] > 40_000_000
        ]
        eeg_files.sort(key=lambda x: x[0])  # Sort by file ID.

        # Group into subjects: session "1_epo.mat" starts a new subject.
        manifest = {}
        subj_idx = 0
        for file_id, name, _size in eeg_files:
            if name == "1_epo.mat":
                subj_idx += 1
            if subj_idx not in manifest:
                manifest[subj_idx] = []
            # Use sequential session index for unique local filenames.
            sess_num = len(manifest[subj_idx]) + 1
            local_name = f"sub{subj_idx:02d}_{sess_num}_epo.mat"
            manifest[subj_idx].append((file_id, local_name))

        # Save manifest for future use.
        try:
            basepath.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

        _MANIFEST = manifest
        return _MANIFEST

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("HefmiIch2025", path)
        basepath = Path(path) / "MNE-hefmiich2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        return str(basepath)
