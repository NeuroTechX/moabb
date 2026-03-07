"""Kumar2024 Multi-Session Longitudinal Motor Imagery Dataset.

Kumar, Alawieh, Racz, Fakhreddine, and Millan (2024).
"Transfer learning promotes acquisition of individual BCI skills."
DOI: 10.1093/pnasnexus/pgae076
Data DOI: 10.5281/zenodo.10694880
"""

import logging
import re
import warnings
import zipfile
from pathlib import Path

import mne
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
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

ZENODO_URL = (
    "https://zenodo.org/api/records/10694880/files/Online_Offline_Race.zip/content"
)

# 22 EEG channel names from the .locs file and the paper (10-10 system)
_EEG_CHANNELS = [
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "C3",
    "Cz",
    "C4",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
]


class Kumar2024(BaseDataset):
    """Multi-session longitudinal motor imagery dataset from Kumar et al. 2024.

    Dataset from [1]_ [2]_.

    This dataset contains EEG recordings from 18 healthy, BCI-naive participants
    (7 female, 11 male, age 23.22 +/- 3.59 years) performing left-hand and
    right-hand motor imagery over 6 sessions conducted on separate days.

    Session 1 was an offline calibration session with 4 bar-feedback runs.
    Sessions 2-6 were online sessions consisting of bar-feedback runs with
    continuous visual feedback followed by car racing games. In each bar-feedback
    run, subjects performed 20 trials (10 left-hand, 10 right-hand MI) in
    pseudo-random order.

    For MOABB, only bar-feedback runs are included (car racing runs are excluded).
    Session 2 (online session 1) contains 4 bar runs, and sessions 3-6
    (online sessions 2-5) each contain 3 bar runs.

    EEG was recorded at 512 Hz using an ANT Neuro eego mylab system with 22
    EEG electrodes positioned according to the international 10-10 system
    (reference: CPz, ground: AFz), plus 3 EOG channels. Data is stored in
    GDF (General Data Format) files.

    The two transfer learning training protocols used were:
    - Generic Recentering (GR): unsupervised domain adaptation (subjects 1-9)
    - Personally Assisted Recentering (PAR): supervised recalibration (subjects 10-18)

    Trial structure (bar task):
    - Fixation cross: 1.0 s
    - Cue presentation: 1.5 s
    - MI + visual feedback: up to 5 s (offline) or 7 s (online)
    - Result display: 2.0 s
    - Inter-trial rest: 1.5 s

    References
    ----------
    .. [1] S. Kumar, H. Alawieh, F. S. Racz, R. Fakhreddine, and
       J. del R. Millan, "Transfer learning promotes acquisition of individual
       BCI skills," PNAS Nexus, vol. 3, no. 3, p. pgae076, 2024.
       DOI: 10.1093/pnasnexus/pgae076

    .. [2] S. Kumar, H. Alawieh, F. S. Racz, R. Fakhreddine, and
       J. del R. Millan, "Multi-Session longitudinal MI training dataset,"
       Zenodo, 2024. DOI: 10.5281/zenodo.10694880

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=22,
            channel_types={"eeg": 22, "eog": 3},
            montage="standard_1020",
            hardware="ANT Neuro eego mylab",
            cap_manufacturer="ANT Neuro",
            cap_model="waveguard EEG cap",
            sensor_type="EEG",
            reference="CPz",
            ground="AFz",
            sensors=_EEG_CHANNELS,
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=3,
                eog_type=["horizontal", "vertical"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            health_status="healthy",
            gender={"female": 7, "male": 11},
            age_mean=23.22,
            age_std=3.59,
            bci_experience="naive",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            events={"left_hand": 1, "right_hand": 2},
            trials_per_class={"left_hand": 10, "right_hand": 10},
            trial_duration=5.0,
            study_design=(
                "Longitudinal BCI training with inter-subject transfer learning. "
                "Subjects performed left/right hand MI with bar-feedback and car "
                "racing tasks across 6 sessions on separate days. Two groups: "
                "Generic Recentering (GR, N=9) and Personally Assisted Recentering "
                "(PAR, N=9)."
            ),
            feedback_type="continuous visual",
            stimulus_type="visual cue and bar feedback",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            synchronicity="cue-based",
            has_training_test_split=False,
            instructions=(
                "Subjects were instructed to mentally rehearse the kinesthetics, "
                "not the visualization, of left- or right-hand movements without "
                "overtly causing any muscle contraction."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/pnasnexus/pgae076",
            associated_paper_doi="10.1093/pnasnexus/pgae076",
            description=(
                "Multi-session longitudinal MI training dataset with 18 BCI-naive "
                "subjects over 6 sessions. Demonstrates that inter-subject transfer "
                "learning from a single expert promotes acquisition of individual "
                "BCI skills via unsupervised domain adaptation."
            ),
            investigators=[
                "Satyam Kumar",
                "Hussein Alawieh",
                "Frigyes Samuel Racz",
                "Rawan Fakhreddine",
                "Jose del R. Millan",
            ],
            senior_author="Jose del R. Millan",
            contact_info=[
                "satyam.kumar@utexas.edu",
                "jose.millan@austin.utexas.edu",
            ],
            institution="The University of Texas at Austin",
            institution_address="Austin, TX, USA",
            country="US",
            repository="Zenodo",
            data_url="https://zenodo.org/records/10694880",
            publication_year=2024,
            funding=[],
            ethics_approval=["The University of Texas at Austin (Protocol 2020-03-0073)"],
            keywords=[
                "motor imagery",
                "brain-computer interface",
                "EEG",
                "transfer learning",
                "domain adaptation",
                "Riemannian geometry",
                "longitudinal training",
                "BCI skill acquisition",
            ],
            license="CC-BY-4.0",
        ),
        sessions_per_subject=6,
        runs_per_session=4,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor Imagery"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            notes=(
                "Raw EEG signals recorded in GDF format. "
                "For analysis, signals were bandpass filtered at 8-30 Hz "
                "using a second-order Butterworth filter."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Riemannian MDM", "CSP+LDA"],
            feature_extraction=[
                "Covariance matrices",
                "Riemannian geometry",
                "CSP",
            ],
            frequency_bands={
                "mu_beta": [8.0, 30.0],
            },
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_subject"],
            cv_method="longitudinal sessions",
        ),
        performance={
            "GR_NKV_start": 0.2636,
            "GR_NKV_end": 0.4694,
            "PAR_NKV_start": 0.4045,
            "PAR_NKV_end": 0.6802,
        },
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "neurofeedback"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=1.5,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=20,
            n_trials_per_class={"left_hand": 10, "right_hand": 10},
            n_blocks=6,
            trials_context=(
                "20 trials per run (10 left, 10 right). "
                "Session 1: 4 offline runs (80 trials). "
                "Session 2: 4 online bar runs (80 trials). "
                "Sessions 3-6: 3 online bar runs each (60 trials). "
                "Total bar-feedback trials per subject: 400."
            ),
        ),
        file_format="GDF",
        data_processed=False,
        abstract=(
            "Subject training is crucial for acquiring brain-computer interface "
            "(BCI) control. Here, we show that a decoder trained on the data of "
            "a single expert is readily transferable to inexperienced users via "
            "domain adaptation techniques allowing calibration-free BCI training. "
            "We introduce two real-time frameworks: Generic Recentering (GR) "
            "through unsupervised adaptation and Personally Assisted Recentering "
            "(PAR) that extends GR by employing supervised recalibration. We "
            "evaluated our frameworks on 18 healthy naive subjects over five "
            "online sessions, who operated a synchronous bar task and a car "
            "racing game. Our frameworks promoted subjects' ability to acquire "
            "individual BCI skills."
        ),
        methodology=(
            "18 BCI-naive subjects participated in 6 sessions (1 offline + 5 "
            "online) on separate days. Each session comprised bar-feedback MI "
            "runs and car racing games. Bar runs had 20 trials (10 per class) "
            "with cue-based left/right hand MI. EEG recorded at 512 Hz with "
            "22 EEG + 3 EOG channels using ANT Neuro eego mylab. Two groups: "
            "GR (N=9, unsupervised) and PAR (N=9, supervised recalibration). "
            "Features: covariance matrices in 8-30 Hz band classified with "
            "Riemannian MDM decoder."
        ),
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=6,
            events=dict(left_hand=1, right_hand=2),
            code="Kumar2024",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1093/pnasnexus/pgae076",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the path to the extracted ZIP for this dataset.

        Downloads the single ZIP from Zenodo and extracts it if needed.

        Parameters
        ----------
        subject : int
            Subject number (1-18).
        path : None | str
            Storage location override.
        force_update : bool
            Re-download even if local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        Path
            Path to the extracted dataset root directory.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        zip_path = Path(dl.data_dl(ZENODO_URL, sign, path, force_update, verbose))
        extract_dir = zip_path.parent / "Online_Offline_Race"

        if not extract_dir.is_dir():
            log.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_path.parent)

        return extract_dir

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-18).

        Returns
        -------
        dict
            ``{session_str: {run_str: Raw}}`` with only bar-feedback runs.
        """
        extract_dir = self.data_path(subject)

        # Discover subject folder -- handle common naming patterns:
        #   S01, S1, sub01, sub1, Subject01, etc.
        subject_dir = self._find_subject_dir(extract_dir, subject)
        if subject_dir is None:
            raise FileNotFoundError(
                f"Could not find directory for subject {subject} under {extract_dir}"
            )

        # Discover session directories inside the subject folder.
        # Expected: session 1 = offline, sessions 2-6 = online
        session_dirs = self._find_session_dirs(subject_dir)

        sessions = {}
        for sess_idx, sess_path in sorted(session_dirs.items()):
            runs = self._load_bar_runs(sess_path, sess_idx)
            if runs:
                sessions[str(sess_idx)] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No bar-feedback GDF files found for subject {subject} "
                f"under {subject_dir}"
            )

        return sessions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_subject_dir(root, subject):
        """Locate the folder for a given subject under *root*.

        Tries several common naming patterns (case-insensitive) to be robust
        to the actual ZIP layout.
        """
        patterns = [
            f"S{subject:02d}",
            f"S{subject}",
            f"sub-{subject:02d}",
            f"sub{subject:02d}",
            f"Subject{subject:02d}",
            f"Subject{subject}",
        ]
        # First, check direct children of root
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name in patterns:
                return child
        # Try case-insensitive matching
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.lower() in [p.lower() for p in patterns]:
                return child
        # If root has a single subdirectory, recurse one level
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return Kumar2024._find_subject_dir(subdirs[0], subject)
        # Try all subdirectories (e.g., the ZIP nests under Online_Offline_Race/)
        for sd in subdirs:
            result = Kumar2024._find_subject_dir_shallow(sd, subject)
            if result is not None:
                return result
        return None

    @staticmethod
    def _find_subject_dir_shallow(root, subject):
        """Non-recursive single-level check for subject directory."""
        patterns = [
            f"S{subject:02d}",
            f"S{subject}",
            f"sub-{subject:02d}",
            f"sub{subject:02d}",
            f"Subject{subject:02d}",
            f"Subject{subject}",
        ]
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.lower() in [p.lower() for p in patterns]:
                return child
        return None

    @staticmethod
    def _find_session_dirs(subject_dir):
        """Discover session directories.

        Returns a dict mapping 0-based session index to Path.
        Handles multiple conventions:
        - Subdirectories named Offline, Online1..Online5
        - Subdirectories named Session1..Session6, ses-01..ses-06
        - GDF files directly in the subject folder (single-session fallback)
        - Separate Offline/ and Online/ directories
        """
        children = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
        child_names_lower = {d.name.lower(): d for d in children}

        sessions = {}

        # Pattern 1: Offline + Online1..Online5 (or online_1, etc.)
        offline_dir = child_names_lower.get("offline")
        if offline_dir is not None:
            sessions[0] = offline_dir
            for i in range(1, 6):
                for key in [f"online{i}", f"online_{i}", f"online{i:02d}"]:
                    d = child_names_lower.get(key)
                    if d is not None:
                        sessions[i] = d
                        break
            if sessions:
                return sessions

        # Pattern 2: Session1..Session6 or ses-01..ses-06
        for i in range(6):
            for key in [
                f"session{i + 1}",
                f"session{i + 1:02d}",
                f"ses-{i + 1:02d}",
                f"ses{i + 1:02d}",
            ]:
                d = child_names_lower.get(key)
                if d is not None:
                    sessions[i] = d
                    break
        if sessions:
            return sessions

        # Pattern 3: numbered directories (1, 2, ..., 6 or 01, 02, ..., 06)
        for i in range(6):
            for key in [str(i + 1), f"{i + 1:02d}"]:
                d = child_names_lower.get(key)
                if d is not None:
                    sessions[i] = d
                    break
        if sessions:
            return sessions

        # Pattern 4: All GDF files directly in subject_dir
        gdf_files = sorted(subject_dir.glob("*.gdf"))
        if not gdf_files:
            gdf_files = sorted(subject_dir.glob("*.GDF"))
        if gdf_files:
            # Single flat directory -- organize into a pseudo-session map
            # based on file naming convention.
            return Kumar2024._organize_flat_gdfs(gdf_files)

        # Pattern 5: arbitrary subdirectories with GDF files
        for child in children:
            gdfs = sorted(child.glob("*.gdf")) + sorted(child.glob("*.GDF"))
            if gdfs:
                sessions[len(sessions)] = child
        return sessions

    @staticmethod
    def _organize_flat_gdfs(gdf_files):
        """Group flat GDF files into sessions by naming convention.

        Expects names like ``offline_run1.gdf``, ``online1_run1.gdf``, or
        ``run01.gdf``, ``run05.gdf``, etc.
        """
        sessions = {}
        # Try to detect Offline/Online naming in filenames
        offline_files = []
        online_buckets = {}
        unmatched = []

        for f in gdf_files:
            name = f.stem.lower()
            if "race" in name:
                continue  # skip race files
            if "offline" in name:
                offline_files.append(f)
            elif "online" in name:
                # Extract online session number
                m = re.search(r"online[_\s]*(\d+)", name)
                if m:
                    idx = int(m.group(1))
                    online_buckets.setdefault(idx, []).append(f)
                else:
                    online_buckets.setdefault(1, []).append(f)
            else:
                unmatched.append(f)

        if offline_files:
            sessions[0] = offline_files
        for idx in sorted(online_buckets):
            sessions[idx] = online_buckets[idx]

        if not sessions and unmatched:
            # Fall back: group by order
            sessions[0] = unmatched

        # Convert list-of-files to a sentinel Path for _load_bar_runs
        # We store as dict mapping sess_idx -> list(Path)
        return sessions

    def _load_bar_runs(self, sess_path_or_files, sess_idx):
        """Load bar-feedback GDF files from a session directory.

        Parameters
        ----------
        sess_path_or_files : Path or list[Path]
            Either a directory or a list of GDF file paths.
        sess_idx : int
            0-based session index (used to skip race files).

        Returns
        -------
        dict
            ``{run_str: Raw}``
        """
        if isinstance(sess_path_or_files, list):
            gdf_files = sorted(sess_path_or_files)
        else:
            gdf_files = sorted(sess_path_or_files.glob("*.gdf"))
            if not gdf_files:
                gdf_files = sorted(sess_path_or_files.glob("*.GDF"))

        # Filter out race files
        bar_files = [f for f in gdf_files if "race" not in f.stem.lower()]

        if not bar_files:
            return {}

        montage = make_standard_montage("standard_1020")
        runs = {}
        for run_idx, gdf_path in enumerate(bar_files):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)

            # Pick only EEG channels (first 22 in file), drop EOG
            if not self.return_all_modalities:
                # Keep only the 22 EEG channels
                eeg_picks = raw.ch_names[:22]
                raw.pick(eeg_picks)

            # Rename channels to standard 10-10 names
            n_eeg = min(len(raw.ch_names), 22)
            rename_map = {}
            for i in range(n_eeg):
                if raw.ch_names[i] != _EEG_CHANNELS[i]:
                    rename_map[raw.ch_names[i]] = _EEG_CHANNELS[i]
            if rename_map:
                raw.rename_channels(rename_map)

            # Set channel types for any remaining EOG channels
            for ch in raw.ch_names:
                if ch not in _EEG_CHANNELS:
                    try:
                        raw.set_channel_types({ch: "eog"})
                    except (ValueError, KeyError):
                        pass

            # Set montage
            raw.set_montage(montage, on_missing="ignore")

            # Map GDF annotations: 769 -> left_hand, 770 -> right_hand
            raw.annotations.rename({"769": "left_hand", "770": "right_hand"})

            runs[str(run_idx)] = raw

        return runs
