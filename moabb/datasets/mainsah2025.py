"""Mainsah2025 BigP3BCI dataset.

# License: BSD (3-clause)
"""

import logging
import re
import warnings
from pathlib import Path

import mne
import numpy as np
from mne.io import read_raw_edf

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
    PreprocessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

BASE_URL = "https://physionet.org/files/bigp3bci/1.0.0/"

# URL for the SHA256SUMS.txt file used as manifest
_MANIFEST_URL = BASE_URL + "SHA256SUMS.txt"

# Studies and their subject counts
# Study letter -> (n_subjects, has_ALS, grid_size, site)
_STUDIES = {
    "A": (13, False, "9x8", "Duke"),
    "B": (18, True, "6x6", "ETSU"),
    "C": (19, False, "9x8", "Duke"),
    "D": (17, False, "9x8", "Duke"),
    "E": (8, False, "9x8", "Duke"),
    "F": (10, True, "9x8", "Mixed"),
    "G": (20, False, "9x8", "Duke"),
    "H": (16, False, "9x8", "Duke"),
    "I": (13, False, "9x8", "Duke"),
    "J": (20, False, "6x6", "ETSU"),
    "K": (5, False, "9x8", "Duke"),
    "L": (11, True, "6x6", "ETSU"),
    "M": (21, False, "9x8", "Duke"),
    "N": (8, True, "6x6", "ETSU"),
    "O": (18, False, "9x8", "ETSU"),
    "P": (19, False, "9x8", "ETSU"),
    "Q": (36, False, "9x8", "ETSU"),
    "R": (20, False, "9x8", "ETSU"),
    "S1": (10, False, "9x8", "ETSU"),
    "S2": (24, False, "9x8", "ETSU"),
}

# 16 EEG channels used in ETSU studies (subset of Duke's 32)
_ETSU_CHANNELS = [
    "F3",
    "Fz",
    "F4",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP3",
    "CP4",
    "P3",
    "Pz",
    "P4",
    "PO7",
    "PO8",
    "Oz",
]

# 32 EEG channels used in Duke studies
_DUKE_CHANNELS = _ETSU_CHANNELS + [
    "FP1",
    "FP2",
    "F7",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "CPz",
    "P7",
    "P5",
    "PO3",
    "POz",
    "PO4",
    "O1",
    "O2",
]


def _build_subject_map():
    """Build a mapping from global subject number to (study, local_subject_num).

    Returns a dict: {1: ("A", 1), 2: ("A", 2), ..., 326: ("S2", 24)}
    """
    subject_map = {}
    global_idx = 1
    for study, (n_subjects, *_) in _STUDIES.items():
        for local_idx in range(1, n_subjects + 1):
            subject_map[global_idx] = (study, local_idx)
            global_idx += 1
    return subject_map


_SUBJECT_MAP = _build_subject_map()
_N_SUBJECTS = len(_SUBJECT_MAP)


def _study_subject_id(study, local_num):
    """Return the subject folder name, e.g., 'A_01' or 'S1_03'."""
    return f"{study}_{local_num:02d}"


def _parse_manifest(manifest_text):
    """Parse SHA256SUMS.txt into a dict of {subject_id: [relative_paths]}.

    The manifest has lines like:
        <sha256hash>  bigP3BCI-data/StudyA/A_01/SE001/Train/CB/A_01_SE001_CB_Train01.edf

    Returns a dict mapping subject_id (e.g., "A_01") to a list of relative
    paths under bigP3BCI-data/ (e.g., "StudyA/A_01/SE001/Train/CB/A_01_...edf").
    """
    subject_files = {}
    for line in manifest_text.strip().splitlines():
        line = line.strip()
        if not line or not line.endswith(".edf"):
            continue
        # Split hash and path
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        fpath = parts[1].strip()
        # Remove "bigP3BCI-data/" prefix if present
        if fpath.startswith("bigP3BCI-data/"):
            fpath = fpath[len("bigP3BCI-data/") :]

        # Extract subject_id from path: Study{X}/{subject_id}/...
        path_parts = fpath.split("/")
        if len(path_parts) < 3:
            continue
        subject_id = path_parts[1]  # e.g., "A_01", "S1_03"
        subject_files.setdefault(subject_id, []).append(fpath)

    return subject_files


class Mainsah2025(BaseDataset):
    """BigP3BCI: P300-based BCI dataset from Mainsah et al. 2025.

    **Dataset description**

    BigP3BCI [1]_ is a large, diverse, machine-learning-ready P300-based
    Brain-Computer Interface dataset curated from 20 visual P300 speller
    studies conducted at Duke University and East Tennessee State University.

    The dataset includes 326 participants across 20 studies (A through S2),
    including 47 participants with ALS (studies B, F, L, N). EEG was recorded
    using g.tec g.USBamp amplifiers at 256 Hz.

    Studies used two grid layouts for the P300 speller:

    - 6x6 grid (36 characters): studies B, J, L, N
    - 9x8 grid (72 characters): all other studies

    Multiple stimulus paradigms were used across studies: Row-Column (RC),
    Checkerboard (CB), Random (RD), Performance-Based (PB), Adaptive (AD),
    and variants. Each recording includes calibration (training) and test
    phases.

    Duke studies used 32-channel caps with 0.1-60 Hz bandpass, while ETSU
    studies used 16-channel caps with 0.5-30 Hz bandpass. The number of
    EEG channels varies by study (16 or 32).

    Data is in EDF+ format with IEEE P2731 BCI annotation channels encoding
    stimulus events via ``StimulusType`` (0=non-target, 1=target) and
    ``StimulusBegin`` (0=off, 1=on).

    .. warning::
        This dataset is 44.6 GB uncompressed. Files are downloaded
        per-subject on demand from PhysioNet. A manifest file
        (SHA256SUMS.txt, ~1.5 MB) is downloaded once on first use to
        discover the file listing.

    Parameters
    ----------
    subjects : list of int, optional
        List of subject numbers to load (1 to 326). If None, all subjects.
    sessions : list, optional
        List of sessions to load. If None, all sessions.

    References
    ----------
    .. [1] Mainsah, B., Fleeting, C., Balmat, T., Sellers, E., & Collins, L.
       (2025). bigP3BCI: An Open, Diverse and Machine Learning Ready
       P300-based Brain-Computer Interface Dataset (version 1.0.0).
       PhysioNet. https://doi.org/10.13026/0byy-ry86
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=32,
            channel_types={"eeg": 32},
            sensors=_DUKE_CHANNELS,
            sensor_type="eeg",
            reference="right mastoid",
            ground="left mastoid",
            hardware="g.tec g.USBamp biosignal amplifiers",
            software="BCI2000",
            filters="0.1-60 Hz (Duke) / 0.5-30 Hz (ETSU)",
            line_freq=60.0,
            montage="standard_1020",
            impedance_threshold_kohm=10.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
                eog_channels=0,
                eog_type=None,
                has_emg=False,
                emg_channels=0,
                other_physiological=None,
            ),
            cap_manufacturer="Electro-Cap International / g.tec",
            cap_model=None,
            electrode_type="passive gel / active dry (g.Sahara)",
            electrode_material="Ag-AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=_N_SUBJECTS,
            health_status="mixed",
            gender=None,
            age_mean=None,
            age_std=None,
            age_min=None,
            age_max=None,
            ages=None,
            handedness=None,
            clinical_population="ALS (studies B, F, L, N)",
            bci_experience="mixed",
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="P300 speller",
            events={"Target": 1, "NonTarget": 0},
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trials_per_class=None,
            trial_duration=None,
            tasks=["copy spelling (calibration)", "free spelling (test)"],
            study_design="within-subject",
            study_domain="P300 speller BCI",
            feedback_type="visual (test phase only)",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online (test) / offline (calibration)",
            has_training_test_split=True,
            instructions=(
                "Focus on target character while subsets of characters " "are illuminated"
            ),
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "software": "BCI2000",
                "paradigms": "RC, CB, RD, PB, AD and variants",
                "grids": "6x6 (36 chars) or 9x8 (72 chars)",
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.13026/0byy-ry86",
            description=(
                "BigP3BCI: An Open, Diverse and Machine Learning Ready "
                "P300-based Brain-Computer Interface Dataset. 20 visual P300 "
                "speller studies from Duke University and East Tennessee State "
                "University, including able-bodied and ALS participants."
            ),
            investigators=[
                "Boyla Mainsah",
                "Chance Fleeting",
                "Thomas Balmat",
                "Eric Sellers",
                "Leslie Collins",
            ],
            institution="Duke University / East Tennessee State University",
            country="US",
            repository="PhysioNet",
            data_url="https://physionet.org/content/bigp3bci/1.0.0/",
            license="CC-BY-4.0",
            publication_year=2025,
            senior_author="Leslie Collins",
            contact_info=None,
            associated_paper_doi=None,
            funding=None,
            institution_address=None,
            institution_department=None,
            ethics_approval=None,
            acknowledgements=None,
            how_to_acknowledge=(
                "Please cite: Mainsah, B., Fleeting, C., Balmat, T., "
                "Sellers, E., & Collins, L. (2025). bigP3BCI (version 1.0.0). "
                "PhysioNet. https://doi.org/10.13026/0byy-ry86"
            ),
            keywords=[
                "P300",
                "BCI",
                "speller",
                "ERP",
                "ALS",
                "brain-computer interface",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=None,
        sessions=None,
        contributing_labs=["Duke University", "East Tennessee State University"],
        n_contributing_labs=2,
        data_processed=False,
        file_format="EDF+",
        external_links={
            "source": "https://physionet.org/content/bigp3bci/1.0.0/",
        },
        tags=Tags(
            pathology=["Healthy", "ALS"],
            modality=["visual"],
            type=["EEG", "P300", "BCI", "speller"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            preprocessing_steps=None,
            highpass_hz=None,
            lowpass_hz=None,
            bandpass=None,
            notch_hz=None,
            filter_type=None,
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes=(
                "Hardware bandpass differs by site: " "Duke 0.1-60 Hz, ETSU 0.5-30 Hz"
            ),
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=None,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=None,
            imagery_tasks=None,
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        data_structure=DataStructureMetadata(
            n_trials=None,
            n_trials_per_class=None,
            n_blocks=None,
            block_duration_s=None,
            trials_context=(
                "Each session has calibration (Train) and test (Test) phases "
                "with variable number of conditions per study."
            ),
        ),
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, _N_SUBJECTS + 1)),
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=0),
            code="Mainsah2025",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.13026/0byy-ry86",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self._manifest_cache = None

    @staticmethod
    def _get_study_info(subject):
        """Return (study_letter, local_subject_num) for a global subject number."""
        return _SUBJECT_MAP[subject]

    def _get_manifest(self, base_path):
        """Download and parse the SHA256SUMS.txt manifest.

        Returns a dict mapping subject_id to list of relative file paths.
        The manifest is cached in memory after first load.
        """
        if self._manifest_cache is not None:
            return self._manifest_cache

        manifest_path = base_path / "SHA256SUMS.txt"
        dl.download_if_missing(str(manifest_path), _MANIFEST_URL, warn_missing=False)

        manifest_text = manifest_path.read_text(encoding="utf-8")
        self._manifest_cache = _parse_manifest(manifest_text)
        return self._manifest_cache

    def _download_subject_files(self, subject, base_path, force_update=False):
        """Download all EDF files for a subject using the manifest.

        Uses SHA256SUMS.txt as a manifest to discover exact file paths,
        then downloads each file individually from PhysioNet.
        """
        study, local_num = self._get_study_info(subject)
        subject_id = _study_subject_id(study, local_num)

        manifest = self._get_manifest(base_path)
        file_list = manifest.get(subject_id, [])

        if not file_list:
            raise FileNotFoundError(
                f"No files found in manifest for subject {subject} "
                f"(Study {study}, {subject_id}). "
                "The manifest may be outdated; try force_update=True."
            )

        downloaded_files = []

        for relative_path in file_list:
            url = BASE_URL + "bigP3BCI-data/" + relative_path
            local_file = base_path / relative_path

            if local_file.exists() and not force_update:
                downloaded_files.append(local_file)
                continue

            try:
                dl.download_if_missing(str(local_file), url, warn_missing=False)
                if local_file.exists():
                    downloaded_files.append(local_file)
            except Exception as e:
                log.warning("Failed to download %s: %s", relative_path, e)

        return sorted(downloaded_files)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number (1 to 326).
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of Path
            Paths to the subject's EDF files.
        """
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject {subject}. "
                f"Valid subjects: {self.subject_list[0]}-{self.subject_list[-1]}"
            )

        sign = "BIGP3BCI"
        base_path = Path(dl.get_dataset_path(sign, path)) / "MNE-bigp3bci-data"

        # Check if files already exist locally
        study, local_num = self._get_study_info(subject)
        subject_id = _study_subject_id(study, local_num)
        local_subject_path = base_path / f"Study{study}" / subject_id
        existing_files = sorted(local_subject_path.rglob("*.edf"))
        if existing_files and not force_update:
            return existing_files

        # Download using manifest
        downloaded = self._download_subject_files(subject, base_path, force_update)
        if not downloaded:
            raise FileNotFoundError(
                f"No EDF files found for subject {subject} "
                f"(Study {study}, {subject_id}). "
                "Check your internet connection or PhysioNet availability."
            )
        return downloaded

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The global subject number (1 to 326).

        Returns
        -------
        dict
            Dictionary: {"session_id": {"run_id": mne.io.Raw}}
        """
        file_paths = self.data_path(subject)
        study, local_num = self._get_study_info(subject)

        sessions = {}
        run_counter = {}  # per-session counter for unique run indices

        for fpath in file_paths:
            fpath = Path(fpath)

            # Parse session, paradigm, phase, and file number from filename
            # Pattern: {subj_id}_SE{NNN}_{paradigm}_{Phase}{NN}.edf
            # e.g., A_01_SE001_CB_Train01.edf or S1_03_SE002_CB_Test06.edf
            match = re.search(
                r"_SE(\d+)_([A-Za-z]+)_(Train|Test)(\d+)\.edf$",
                fpath.name,
                re.IGNORECASE,
            )
            if not match:
                log.warning("Skipping file with unexpected name: %s", fpath.name)
                continue

            se_num = match.group(1)
            paradigm = match.group(2).lower()
            phase = match.group(3).lower()
            file_num = match.group(4)

            session_key = str(int(se_num) - 1)
            # Use a unique sequential index per session
            idx = run_counter.get(session_key, 0)
            run_counter[session_key] = idx + 1
            run_key = f"{idx}{paradigm}{phase}{file_num}"

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = read_raw_edf(str(fpath), preload=True, verbose="ERROR")
            except Exception as e:
                log.warning("Failed to read %s: %s", fpath.name, e)
                continue

            # Extract stimulus events from data channels before picking EEG
            raw = self._extract_events_and_pick_eeg(raw)
            if raw is None:
                log.warning("Skipping %s: no valid channels/events.", fpath.name)
                continue

            # Store in sessions dict
            if session_key not in sessions:
                sessions[session_key] = {}
            sessions[session_key][run_key] = raw

        if not sessions:
            raise ValueError(
                f"No valid data found for subject {subject} "
                f"(Study {study}, {_study_subject_id(study, local_num)})"
            )

        return sessions

    def _extract_events_and_pick_eeg(self, raw):
        """Extract stimulus events from data channels and pick EEG channels.

        The BigP3BCI EDF+ files encode events as data channels (not EDF+
        annotations). The key channels are:

        - StimulusBegin: binary (0/1), rising edge marks stimulus onset
        - StimulusType: binary (0/1), 0=non-target, 1=target

        This method:
        1. Reads StimulusBegin/StimulusType to create MNE annotations
        2. Picks all EEG channels (prefixed with ``EEG_``) and strips prefix
        3. Sets montage

        Returns None if the file lacks required channels.
        """
        sfreq = raw.info["sfreq"]

        # Check for stimulus channels
        if "StimulusBegin" not in raw.ch_names:
            return None
        if "StimulusType" not in raw.ch_names:
            return None

        # Read stimulus channels
        stim_begin = raw.get_data(picks=["StimulusBegin"])[0]
        stim_type = raw.get_data(picks=["StimulusType"])[0]

        # Find rising edges of StimulusBegin
        diff = np.diff(stim_begin)
        onset_samples = np.where(diff > 0.5)[0] + 1

        if len(onset_samples) == 0:
            return None

        # Build annotations
        onset_times = onset_samples / sfreq + raw.first_time
        durations = np.zeros(len(onset_samples))
        descriptions = np.where(stim_type[onset_samples] > 0.5, "Target", "NonTarget")

        annotations = mne.Annotations(
            onset=onset_times, duration=durations, description=descriptions
        )

        # Identify all EEG channels (prefixed with "EEG_")
        ch_rename = {}
        eeg_channels = []
        for ch_name in raw.ch_names:
            if ch_name.startswith("EEG_"):
                clean = ch_name[4:]  # strip "EEG_" prefix
                ch_rename[ch_name] = clean
                eeg_channels.append(clean)

        if not eeg_channels:
            return None

        # Rename and pick EEG channels
        raw.rename_channels(ch_rename)
        raw.pick(eeg_channels)
        raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

        # Set annotations
        raw.set_annotations(annotations)

        # Set montage
        try:
            raw.set_montage(
                mne.channels.make_standard_montage("standard_1020"),
                on_missing="warn",
            )
        except Exception:
            pass

        return raw
