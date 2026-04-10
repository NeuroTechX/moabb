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
                has_eog=True, eog_channels=3, eog_type=["horizontal", "vertical"]
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
            contact_info=["satyam.kumar@utexas.edu", "jose.millan@austin.utexas.edu"],
            institution="The University of Texas at Austin",
            institution_address="Austin, TX, USA",
            country="US",
            repository="Zenodo",
            data_url="https://zenodo.org/records/10694880",
            publication_year=2024,
            funding=["Coleman Fung Foundation", "Sinclair Foundation"],
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
        runs_per_session=3,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
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
            feature_extraction=["Covariance matrices", "Riemannian geometry", "CSP"],
            frequency_bands={"mu_beta": [8.0, 30.0]},
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

    # Map MOABB subject IDs (1-18) to the actual subject numbers in the ZIP.
    # GR group: subjects 1-9 (MOABB 1-9), PAR group: subjects 11-19 (MOABB 10-18)
    _MOABB_TO_RAW = {i: i for i in range(1, 10)}
    _MOABB_TO_RAW.update({i: i + 1 for i in range(10, 19)})  # 10->11, ..., 18->19

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=6,
            events={"left_hand": 1, "right_hand": 2},
            code="Kumar2024",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1093/pnasnexus/pgae076",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
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
        extract_dir : :class:`pathlib.Path`
            Path to the extracted dataset root directory.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        zip_path = Path(dl.data_dl(ZENODO_URL, sign, path, force_update, verbose))
        # The ZIP extracts Offline/, Online/, Race/ directly into the parent dir
        extract_dir = zip_path.parent

        # Check if already extracted by looking for the Offline directory
        if not (extract_dir / "Offline").is_dir():
            log.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        return extract_dir

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1-18, MOABB convention).

        Returns
        -------
        dict
            ``{session_str: {run_str: Raw}}`` with only bar-feedback runs.
        """
        extract_dir = self.data_path(subject)
        raw_subj = self._MOABB_TO_RAW[subject]

        # Determine which group directory to look in
        group = "GR" if raw_subj <= 9 else "PAR"

        # --- Session 1: Offline ---
        # Offline dir naming: Subject_01_Offline or Subject_11_Offline
        offline_subj_name = f"Subject_{raw_subj:02d}_Offline"
        offline_subj_dir = extract_dir / "Offline" / group / offline_subj_name

        # Offline session directory: Subject_01_Session_001_Offline
        # PAR subjects may use 3-digit padding: Subject_011_Session_001_Offline
        offline_sess_dir = self._find_session_subdir(
            offline_subj_dir, raw_subj, 1, "Offline"
        )

        sessions = {}
        if offline_sess_dir is not None and offline_sess_dir.is_dir():
            runs = self._load_bar_runs_from_dir(offline_sess_dir)
            if runs:
                sessions["0"] = runs

        # --- Sessions 2-6: Online ---
        # Online dir naming may differ: Subject_01_Online or Subject_011_Online
        online_subj_dir = self._find_online_subject_dir(
            extract_dir / "Online" / group, raw_subj
        )

        if online_subj_dir is not None and online_subj_dir.is_dir():
            for sess_num in range(2, 7):
                sess_dir = self._find_session_subdir(
                    online_subj_dir, raw_subj, sess_num, "Online"
                )
                if sess_dir is not None and sess_dir.is_dir():
                    runs = self._load_bar_runs_from_dir(sess_dir)
                    if runs:
                        sessions[str(sess_num - 1)] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No bar-feedback GDF files found for subject {subject} "
                f"(raw ID {raw_subj}) under {extract_dir}"
            )

        return sessions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_online_subject_dir(online_group_dir, raw_subj):
        """Locate the subject directory under Online/<group>/.

        Handles inconsistent zero-padding: Subject_01_Online vs Subject_011_Online.
        """
        if not online_group_dir.is_dir():
            return None
        # Try common patterns
        for pattern in [
            f"Subject_{raw_subj:02d}_Online",
            f"Subject_{raw_subj:03d}_Online",
            f"Subject_{raw_subj}_Online",
        ]:
            candidate = online_group_dir / pattern
            if candidate.is_dir():
                return candidate
        # Fallback: search by prefix
        for child in sorted(online_group_dir.iterdir()):
            if child.is_dir():
                m = re.match(r"Subject_0*(\d+)_Online", child.name)
                if m and int(m.group(1)) == raw_subj:
                    return child
        return None

    @staticmethod
    def _find_session_subdir(parent_dir, raw_subj, sess_num, suffix):
        """Locate a session subdirectory under a subject directory.

        Handles naming like:
        - Subject_01_Session_001_Offline
        - Subject_011_Session_002_Online
        """
        if parent_dir is None or not parent_dir.is_dir():
            return None
        # Try common patterns
        for subj_fmt in [f"{raw_subj:02d}", f"{raw_subj:03d}", str(raw_subj)]:
            pattern = f"Subject_{subj_fmt}_Session_{sess_num:03d}_{suffix}"
            candidate = parent_dir / pattern
            if candidate.is_dir():
                return candidate
        # Fallback: search for session number in directory names
        for child in sorted(parent_dir.iterdir()):
            if child.is_dir():
                m = re.search(r"Session_0*(\d+)", child.name)
                if m and int(m.group(1)) == sess_num:
                    return child
        return None

    def _load_bar_runs_from_dir(self, sess_dir):
        """Load all GDF files from a session directory as bar-feedback runs.

        Parameters
        ----------
        sess_dir : pathlib.Path
            Directory containing GDF files for one session.

        Returns
        -------
        dict
            ``{run_str: Raw}``
        """
        gdf_files = sorted(sess_dir.glob("*.gdf"))
        if not gdf_files:
            gdf_files = sorted(sess_dir.glob("*.GDF"))

        if not gdf_files:
            return {}

        montage = make_standard_montage("standard_1020")
        runs = {}
        for run_idx, gdf_path in enumerate(gdf_files):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)

            # The file has 26 channels: 22 EEG + 3 EOG (sens1-3) + Status
            # Rename EEG channels to standard 10-10 names (file has uppercase:
            # FZ->Fz, CZ->Cz, PZ->Pz, POZ->POz)
            rename_map = {}
            for i, ch in enumerate(raw.ch_names[:22]):
                if ch != _EEG_CHANNELS[i]:
                    rename_map[ch] = _EEG_CHANNELS[i]
            if rename_map:
                raw.rename_channels(rename_map)

            # Set EOG channel types for sens1-3
            eog_chs = {ch: "eog" for ch in raw.ch_names[22:] if ch.startswith("sens")}
            if eog_chs:
                raw.set_channel_types(eog_chs)

            # Drop non-EEG channels only when return_all_modalities is False
            if not self.return_all_modalities:
                raw.pick(list(_EEG_CHANNELS))

            # Set montage
            raw.set_montage(montage, on_missing="ignore")

            # GDF headers have malformed physical_min/max; data is in
            # microvolts but MNE reads it as volts.  Scale EEG channels.
            eeg_idx = mne.pick_types(raw.info, eeg=True)
            raw._data[eeg_idx] *= 1e-6

            # Map GDF event annotations: 769 -> left_hand, 770 -> right_hand
            raw.annotations.rename({"769": "left_hand", "770": "right_hand"})

            runs[str(run_idx)] = raw

        return runs
