"""Sixth-finger and affected-hand motor imagery EEG dataset (stroke patients).

Wang, Liu, Huang, Wu, Huang, Gui, Li, Guo and Ming (2026).
A multi-paradigm and longitudinal EEG dataset including the "sixth-finger"
and "affected-hand" motor imagery of stroke patients.
Data DOI: 10.6084/m9.figshare.30509165
"""

import logging
import warnings
import zipfile
from pathlib import Path

import mne
import numpy as np

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
    SignalProcessingMetadata,
    Tags,
)
from .utils import safe_extract_zip, stim_channels_with_selected_ids


log = logging.getLogger(__name__)

_DOI = "10.6084/m9.figshare.30509165"

# Figshare article 30509165 (v11). The processed archive holds one EEGLAB
# EEG struct (saved as a v7.3/HDF5 .mat file) per subject/session/run, already
# band-pass filtered, average-referenced and ICA-cleaned. The raw archive
# ("sourcedata.zip", file id 64877880) is not used by this loader.
_FIGSHARE_DL_URL = "https://ndownloader.figshare.com/files/"
_PROCESSED_ZIP_ID = "64870068"  # processeddata.zip (~3.5 GB)

# Data-borne per-trial markers in EEG.event.type (see the dataset's events.json):
#   1 = beginning of the maintaining-rest period
#   2 = beginning of the Affected-hand motor imagery paradigm
#   3 = beginning of the Sixth-finger motor imagery paradigm
# Only the two discrete motor-imagery classes are kept (rest is excluded).
_EVENTS = {"affected_hand": 2, "sixth_finger": 3}
_MARKER_TO_LABEL = {"2": "affected_hand", "3": "sixth_finger"}

# Longitudinal acquisition phases, mapped to MOABB session keys (which must
# start with an integer index).
_ALL_PHASES = ["pre", "post", "follow"]

# Subjects without a follow-up phase (only pre/post), per participants.tsv.
_NO_FOLLOW = {5, 17, 18, 20, 21}

# Per-subject demographics from participants.tsv (24 subjects).
# fmt: off
_AGES = [
    48, 64, 64, 34, 62, 52, 77, 52, 68, 38,
    55, 57, 66, 55, 55, 60, 41, 64, 50, 66,
    54, 45, 55, 56,
]
_SEXES = [
    "M", "F", "M", "M", "F", "F", "M", "M", "M", "M",
    "M", "M", "M", "F", "M", "F", "M", "F", "M", "M",
    "M", "M", "M", "M",
]
# fmt: on


class Wang2026(BaseDataset):
    """Sixth-finger / affected-hand motor imagery dataset from Wang et al 2026.

    Dataset from *A multi-paradigm and longitudinal EEG dataset including the
    "sixth-finger" and "affected-hand" motor imagery of stroke patients* [1]_.

    EEG was recorded from 24 stroke patients with a 60-channel NeuSen W
    (Neuracle) system (10-20 layout, average reference, AFz ground). Two
    discrete motor-imagery paradigms were cued in an interleaved design:

    - **affected_hand**: motor imagery of the stroke-affected hand (marker 2)
    - **sixth_finger**: motor imagery of a novel virtual "sixth finger"
      (marker 3)

    Each run interleaves 10 affected-hand and 10 sixth-finger imagery cues with
    20 rest cues; rest (marker 1) is excluded here so the task is the binary
    comparison between the two imagery paradigms (the paper reports ~85-86%
    accuracy with CSP+SVM / CSP+LDA). The recording covers the longitudinal
    stages of rehabilitation, mapped to MOABB sessions:

    - **0pre**  : pre-training assessment (all 24 subjects)
    - **1post** : post-training assessment (all 24 subjects)
    - **2follow**: follow-up assessment (19 subjects; subjects 5, 17, 18, 20
      and 21 have only pre/post)

    Each session has 4 runs. Every subject belongs to one of two protocol
    groups tagged ``B`` or ``T`` in the source file names (14 ``B`` and 10
    ``T`` subjects); both groups use the same two imagery classes and the tag
    is preserved only as documentation.

    The loader reads the distributed **processed** files, which contain 43 EEG
    channels (a 10-10 subset) after the authors removed channels during
    preprocessing, even though the acquisition montage has 60 EEG channels.
    Class labels are decoded from the ``EEG.event.type`` markers embedded in
    each file (not inferred from acquisition order).

    Parameters
    ----------
    subjects : list of int, optional
        Subjects to load. Defaults to all 24.

    References
    ----------
    .. [1] Wang, Z., Liu, Y., Huang, S., Wu, W., Huang, H., Gui, Z., Li, Z.,
           Guo, J., & Ming, D. (2026). A multi-paradigm and longitudinal EEG
           dataset including the "sixth-finger" and "affected-hand" motor
           imagery of stroke patients. figshare.
           https://doi.org/10.6084/m9.figshare.30509165
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=43,
            channel_types={"eeg": 43},
            montage="standard_1020",
            hardware="NeuSen W (Neuracle, Inc.)",
            sensor_type="Ag/AgCl",
            reference="average",
            ground="AFz",
            filters={"notch": 50.0},
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=24,
            health_status="stroke (recovery phase)",
            gender={"male": 18, "female": 6},
            age_min=34.0,
            age_max=77.0,
            ages=_AGES,
            sexes=_SEXES,
            species="human",
            clinical_population="stroke",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=4.0,
            study_design=(
                "Binary motor imagery: affected-hand MI vs sixth-finger MI. "
                "24 stroke patients, longitudinal phases (pre/post/follow-up), "
                "4 runs per phase, 10 trials per class per run."
            ),
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Zhuang Wang",
                "Yuan Liu",
                "Shuaifei Huang",
                "Wenlai Wu",
                "Huimin Huang",
                "Zhuolan Gui",
                "Zhaoqi Li",
                "Jun Guo",
                "Dong Ming",
            ],
            institution="Tianjin University",
            country="CN",
            data_url="https://doi.org/10.6084/m9.figshare.30509165",
            repository="Figshare",
            publication_year=2026,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=3,
        runs_per_session=4,
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=5360,
            trials_context=(
                "10 affected-hand + 10 sixth-finger MI trials per run, 4 runs "
                "per phase. 19 subjects have 3 phases and 5 have 2, giving "
                "19*3*4*20 + 5*2*4*20 = 5360 imagery trials."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "CSP+LDA"], feature_extraction=["CSP"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=True,
        file_format="MAT (EEGLAB struct, v7.3/HDF5)",
    )

    def __init__(self, subjects=None):
        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=3,
            events=dict(_EVENTS),
            code="Wang2026",
            interval=[0, 4],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
        )

    def _available_phases(self, subject):
        """Return the acquisition phases present for a subject."""
        if subject in _NO_FOLLOW:
            return ["pre", "post"]
        return list(_ALL_PHASES)

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: mne.io.Raw}}`` for one subject."""
        basepath = Path(self.data_path(subject))
        subj_dir = basepath / f"sub-{subject:02d}"

        sessions = {}
        for phase_idx, phase in enumerate(self._available_phases(subject)):
            phase_dir = subj_dir / phase
            if not phase_dir.exists():
                log.warning("Phase dir not found: %s", phase_dir)
                continue

            mat_files = sorted(phase_dir.glob(f"sub-{subject:02d}_*_{phase}_*_eeg.mat"))
            if not mat_files:
                log.warning("No .mat files in %s", phase_dir)
                continue

            runs = {}
            for run_idx, mat in enumerate(mat_files):
                try:
                    raw = self._read_eeglab_mat(mat)
                    raw = stim_channels_with_selected_ids(raw, self.event_id)
                    runs[str(run_idx)] = raw
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("Failed to load %s: %s", mat.name, exc)

            if runs:
                sessions[f"{phase_idx}{phase}"] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No data found for subject {subject} under {subj_dir}"
            )
        return sessions

    def _read_eeglab_mat(self, file_path):
        """Read one EEGLAB ``EEG`` struct (v7.3/HDF5 .mat) into an MNE Raw.

        The processed files store continuous data in ``EEG.data`` with the
        cue markers embedded in ``EEG.event`` (``type`` and ``latency``).
        Only the two motor-imagery markers are annotated; ``rest`` and
        EEGLAB ``boundary`` markers are dropped.
        """
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading the Wang2026 v7.3 .mat files requires h5py (`pip install h5py`)."
            ) from exc

        with h5py.File(str(file_path), "r") as f:
            eeg = f["EEG"]
            sfreq = float(np.array(eeg["srate"]).ravel()[0])
            # h5py returns the MATLAB (n_channels, n_times) array transposed.
            data = np.array(eeg["data"]).T
            n_channels = data.shape[0]
            ch_names = self._resolve_channel_names(f, eeg, n_channels)
            onsets_s, descriptions = self._resolve_mi_events(f, eeg, sfreq)

        # EEGLAB data are in microvolts; MNE expects volts.
        data = data.astype("float64") * 1e-6
        info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage("standard_1005", on_missing="ignore")

        raw.set_annotations(
            mne.Annotations(
                onset=onsets_s, duration=np.zeros(len(onsets_s)), description=descriptions
            )
        )
        return raw

    @staticmethod
    def _resolve_channel_names(f, eeg, n_channels):
        """Read channel labels from ``EEG.chanlocs.labels`` when available."""
        try:
            labels_ref = eeg["chanlocs"]["labels"]
            names = []
            for ref in np.array(labels_ref).ravel():
                chars = np.array(f[ref]).ravel()
                names.append("".join(chr(int(c)) for c in chars))
            if len(names) == n_channels:
                return names
        except Exception:  # pragma: no cover - defensive
            pass
        return [f"EEG{i + 1:03d}" for i in range(n_channels)]

    @staticmethod
    def _resolve_mi_events(f, eeg, sfreq):
        """Return (onsets_seconds, labels) for the two MI markers only."""
        type_ref = np.array(eeg["event"]["type"]).ravel()
        lat_ref = np.array(eeg["event"]["latency"]).ravel()
        onsets_s = []
        descriptions = []
        for tref, lref in zip(type_ref, lat_ref):
            chars = np.array(f[tref]).ravel()
            try:
                marker = "".join(chr(int(c)) for c in chars)
            except Exception:  # pragma: no cover - defensive
                continue
            label = _MARKER_TO_LABEL.get(marker)
            if label is None:
                continue
            latency = float(np.array(f[lref]).ravel()[0])  # 1-based sample index
            onsets_s.append((latency - 1.0) / sfreq)
            descriptions.append(label)
        return np.asarray(onsets_s, dtype=float), descriptions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}, must be in {self.subject_list}")

        base = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        base.mkdir(parents=True, exist_ok=True)
        subj_dir = base / f"sub-{subject:02d}"

        if subj_dir.exists() and list(subj_dir.rglob("*_eeg.mat")) and not force_update:
            return str(base)

        # Download the whole processed archive once (it bundles every subject),
        # then extract it into the dataset directory.
        url = f"{_FIGSHARE_DL_URL}{_PROCESSED_ZIP_ID}"
        local_zip = dl.data_dl(
            url, self.code, path=path, force_update=force_update, verbose=verbose
        )
        with zipfile.ZipFile(str(local_zip), "r") as zf:
            safe_extract_zip(zf, base)

        return str(base)
