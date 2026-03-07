"""SHU Multi-session Motor Imagery dataset from Ma et al 2022.

https://doi.org/10.1038/s41597-022-01647-1
"""

import logging
import os
import zipfile
from pathlib import Path

import mne
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
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

FIGSHARE_DL_URL = "https://ndownloader.figshare.com/files/"

# EDF archive on Figshare (password-protected)
_EDF_ZIP_FILE_ID = 36728991

# Figshare file IDs for individual events TSV files (freely downloadable)
# Naming on Figshare uses underscores: sub-NNN_ses-NN_task_motorimagery_events.tsv
# These will be placed at the BIDS-correct path:
#   sub-NNN/ses-NN/eeg/sub-NNN_ses-NN_task-motorimagery_events.tsv
_EVENTS_FILE_IDS = {
    (1, 1): 34166184,
    (1, 2): 34166187,
    (1, 3): 34166190,
    (1, 4): 34166193,
    (1, 5): 34166196,
    (2, 1): 34166211,
    (2, 2): 34166199,
    (2, 3): 34166202,
    (2, 4): 34166205,
    (2, 5): 34166208,
    (3, 1): 34166214,
    (3, 2): 34166217,
    (3, 3): 34166220,
    (3, 4): 34166223,
    (3, 5): 34166226,
    (4, 1): 34166229,
    (4, 2): 34166232,
    (4, 3): 34166235,
    (4, 4): 34166238,
    (4, 5): 34166241,
    (5, 1): 34166244,
    (5, 2): 34166247,
    (5, 3): 34166250,
    (5, 4): 34166253,
    (5, 5): 34166256,
    (6, 1): 34166259,
    (6, 2): 34166262,
    (6, 3): 34166265,
    (6, 4): 34166268,
    (6, 5): 34166271,
    (7, 1): 34166274,
    (7, 2): 34166277,
    (7, 3): 34166280,
    (7, 4): 34166283,
    (7, 5): 34166286,
    (8, 1): 34166289,
    (8, 2): 34166292,
    (8, 3): 34166295,
    (8, 4): 34166298,
    (8, 5): 34166301,
    (9, 1): 34166304,
    (9, 2): 34166307,
    (9, 3): 34166310,
    (9, 4): 34166313,
    (9, 5): 34166316,
    (10, 1): 34166319,
    (10, 2): 34166322,
    (10, 3): 34166325,
    (10, 4): 34166328,
    (10, 5): 34166331,
    (11, 1): 34166334,
    (11, 2): 34166337,
    (11, 3): 34166340,
    (11, 4): 34166343,
    (11, 5): 34166346,
    (12, 1): 34166349,
    (12, 2): 34166364,
    (12, 3): 34166352,
    (12, 4): 34166355,
    (12, 5): 34166358,
    (13, 1): 34166361,
    (13, 2): 34166367,
    (13, 3): 34166370,
    (13, 4): 34166373,
    (13, 5): 34166376,
    (14, 1): 34166379,
    (14, 2): 34166382,
    (14, 3): 34166385,
    (14, 4): 34166388,
    (14, 5): 34166391,
    (15, 1): 34166394,
    (15, 2): 34166397,
    (15, 3): 34166400,
    (15, 4): 34166403,
    (15, 5): 34166406,
    (16, 1): 34166409,
    (16, 2): 34166412,
    (16, 3): 34166415,
    (16, 4): 34166418,
    (16, 5): 34166421,
    (17, 1): 34166424,
    (17, 2): 34166427,
    (17, 3): 34166430,
    (17, 4): 34166433,
    (17, 5): 34166436,
    (18, 1): 34166439,
    (18, 2): 34166442,
    (18, 3): 34166445,
    (18, 4): 34166448,
    (18, 5): 34166451,
    (19, 1): 34166454,
    (19, 2): 34166457,
    (19, 3): 34166460,
    (19, 4): 34166463,
    (19, 5): 34166466,
    (20, 1): 34166469,
    (20, 2): 34166472,
    (20, 3): 34166475,
    (20, 4): 34166478,
    (20, 5): 34166481,
    (21, 1): 34166484,
    (21, 2): 34166487,
    (21, 3): 34166490,
    (21, 4): 34166493,
    (21, 5): 34166496,
    (22, 1): 34166499,
    (22, 2): 34166502,
    (22, 3): 34166505,
    (22, 4): 34166508,
    (22, 5): 34166511,
    (23, 1): 34166514,
    (23, 2): 34166517,
    (23, 3): 34166520,
    (23, 4): 34166523,
    (23, 5): 34166526,
    (24, 1): 34166529,
    (24, 2): 34166532,
    (24, 3): 34166535,
    (24, 4): 34166538,
    (24, 5): 34166541,
    (25, 1): 34166544,
    (25, 2): 34166547,
    (25, 3): 34166550,
    (25, 4): 34166553,
    (25, 5): 34166556,
}

# 32 EEG channels (from BIDS channels.tsv)
_CHANNELS = [
    "FP1",
    "FP2",
    "FZ",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "CZ",
    "C3",
    "C4",
    "T3",
    "T4",
    "A1",
    "A2",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "PZ",
    "P3",
    "P4",
    "T5",
    "T6",
    "PO3",
    "PO4",
    "OZ",
    "O1",
    "O2",
]


class Ma2022(BaseDataset):
    """Motor Imagery dataset from Ma et al 2022.

    **Dataset description**

    The SHU multi-session motor imagery dataset [1]_ was collected from
    25 healthy subjects (13 male, 12 female, aged 20--24) at Shanghai
    University. Each subject completed 5 independent sessions with 2--3
    day intervals, yielding 500 trials per subject (100 per session,
    50 left-hand and 50 right-hand motor imagery).

    The experiment followed a three-stage trial design:

    - **0--2 s**: Rest / preparation period (subjects may rest, blink,
      or make small body movements)
    - **2--4 s**: Prompt stage (a hand animation indicates the upcoming
      task direction)
    - **4--8 s**: Motor imagery execution (subjects imagine left- or
      right-hand movement according to the animated prompt)

    EEG was recorded with 32 electrodes placed according to the
    international 10--20 system at 250 Hz sampling rate using a
    unipolar reference at M1 and ground at AFz. The power line
    frequency is 50 Hz.

    .. admonition:: Password required
       :class: warning

       The EDF data files on Figshare are inside a **password-protected**
       ZIP archive. To use this dataset, you must:

       1. Contact the authors at yangbanghua@shu.edu.cn to request the
          password.
       2. Set the environment variable ``MA2022_PASSWORD`` to the password
          before loading the data, e.g.::

              import os
              os.environ["MA2022_PASSWORD"] = "your_password_here"

       If the password is not set, the dataset will raise a ``RuntimeError``
       when attempting to load data.

    References
    ----------

    .. [1] Ma J, Yang B, Qiu W, Li Y, Gao S, Xia X (2022) A large EEG
           dataset for studying cross-session variability in motor imagery
           brain-computer interface. Scientific Data 9, 531.
           https://doi.org/10.1038/s41597-022-01647-1

    Notes
    -----

    .. versionadded:: 1.3.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1020",
            sensor_type="EEG",
            hardware=None,
            reference="unipolar, placed on M1",
            ground="AFz",
            software=None,
            filters=None,
            sensors=_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"male": 13, "female": 12},
            age_min=20,
            age_max=24,
            bci_experience="novice",
        ),
        experiment=ExperimentMetadata(
            events={"left_hand": 1, "right_hand": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 50, "right_hand": 50},
            trial_duration=8.0,
            study_design=(
                "Two-class motor imagery (left hand, right hand) with visual "
                "hand-animation cues. Each trial: 0-2s rest/preparation, "
                "2-4s prompt (hand animation), 4-8s MI execution."
            ),
            feedback_type="none",
            stimulus_type="visual hand animation",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            instructions=(
                "Subject sat in front of a computer monitor. Each trial started "
                "with a 2s rest period, followed by a 2s hand animation prompt "
                "indicating left or right hand, then 4s of motor imagery "
                "execution corresponding to the prompted direction."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-022-01647-1",
            investigators=[
                "Jun Ma",
                "Banghua Yang",
                "Wenzheng Qiu",
                "Yunzhe Li",
                "Shouwei Gao",
                "Xinxing Xia",
            ],
            institution="Shanghai University",
            institution_department="School of Mechatronic Engineering and Automation",
            country="CN",
            institution_address="Shanghai, China",
            data_url="https://doi.org/10.6084/m9.figshare.19228725",
            publication_year=2022,
            senior_author="Banghua Yang",
            contact_info=["yangbanghua@shu.edu.cn"],
            funding=[
                "National Natural Science Foundation of China (61971283, 62171283)",
                "Shanghai Municipal Science and Technology Major Project (2021SHZDZX)",
            ],
            ethics_approval=["Ethics Committee of Shanghai University"],
            keywords=[
                "motor imagery",
                "brain-computer interface",
                "cross-session",
                "multi-session",
                "EEG",
                "BCI",
            ],
            license="CC-BY-4.0",
            repository="Figshare",
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "EEGNet", "deep ConvNet"],
            feature_extraction=["CSP", "deep learning features"],
            frequency_bands={
                "mu": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "analyzed_range": [8.0, 30.0],
            },
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_session", "cross_session"],
        ),
        performance={
            "within_session_mean_accuracy": 65.0,
            "cross_session_mean_accuracy": 60.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=100,
            trials_context="per_session",
            n_trials_per_class={"left_hand": 50, "right_hand": 50},
        ),
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=5,
            events=dict(left_hand=1, right_hand=2),
            code="Ma2022",
            # MI period is 4-8s from trial start; events mark each 4s trial
            # so interval [0, 4] captures the full MI epoch
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-022-01647-1",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_password(self):
        """Retrieve the ZIP password from the environment variable."""
        password = os.environ.get("MA2022_PASSWORD")
        if password is None:
            raise RuntimeError(
                "The Ma2022 dataset requires a password to extract the EDF data files. "
                "Please contact the authors at yangbanghua@shu.edu.cn to request the "
                "password, then set the environment variable MA2022_PASSWORD before "
                "loading data:\n\n"
                "    import os\n"
                '    os.environ["MA2022_PASSWORD"] = "your_password_here"\n'
            )
        return password

    def _get_dataset_dir(self, path=None):
        """Return the base directory for storing downloaded data."""
        path = Path(dl.get_dataset_path(self.code, path))
        dataset_dir = path / f"MNE-{self.code.lower()}-data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _download_and_extract_edf(self, dataset_dir, force_update=False, verbose=None):
        """Download and extract the password-protected EDF ZIP archive."""
        edf_dir = dataset_dir / "edf_files"
        marker = edf_dir / ".extraction_complete"

        if marker.exists() and not force_update:
            return edf_dir

        # Download the ZIP file
        url = f"{FIGSHARE_DL_URL}{_EDF_ZIP_FILE_ID}"
        zip_path = dl.data_dl(
            url, self.code, path=None, force_update=force_update, verbose=verbose
        )

        password = self._get_password()

        log.info("Extracting edf_files.zip (password-protected) to %s", edf_dir)
        edf_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(edf_dir, pwd=password.encode("utf-8"))

        # Mark extraction as complete
        marker.touch()
        return edf_dir

    def _download_events_tsv(self, subject, session, dataset_dir, verbose=None):
        """Download the events TSV file for a given subject and session.

        Returns the path to the downloaded file.
        """
        file_id = _EVENTS_FILE_IDS.get((subject, session))
        if file_id is None:
            raise ValueError(
                f"No events file found for subject {subject}, session {session}"
            )
        url = f"{FIGSHARE_DL_URL}{file_id}"
        return dl.data_dl(url, self.code, path=None, verbose=verbose)

    def _find_edf_file(self, edf_dir, subject, session):
        """Locate the EDF file for a given subject and session.

        The EDF ZIP may contain files in various directory structures.
        We search for the expected BIDS filename pattern.
        """
        sub_label = f"sub-{subject:03d}"
        ses_label = f"ses-{session:02d}"

        # Expected BIDS path inside the extracted directory
        # Try multiple patterns since ZIP internal structure may vary
        patterns = [
            # Standard BIDS: sub-NNN/ses-NN/eeg/sub-NNN_ses-NN_task-motorimagery_eeg.edf
            edf_dir
            / sub_label
            / ses_label
            / "eeg"
            / f"{sub_label}_{ses_label}_task-motorimagery_eeg.edf",
            # Flat structure
            edf_dir / f"{sub_label}_{ses_label}_task-motorimagery_eeg.edf",
            # Non-standard underscore variant
            edf_dir
            / sub_label
            / ses_label
            / "eeg"
            / f"{sub_label}_{ses_label}_task_motorimagery_eeg.edf",
            edf_dir / f"{sub_label}_{ses_label}_task_motorimagery_eeg.edf",
        ]

        for p in patterns:
            if p.exists():
                return p

        # Fallback: search recursively
        for edf_path in edf_dir.rglob(f"*{sub_label}*{ses_label}*_eeg.edf"):
            return edf_path

        raise FileNotFoundError(
            f"Could not find EDF file for subject {subject}, session {session} "
            f"in {edf_dir}. Searched patterns:\n" + "\n".join(f"  {p}" for p in patterns)
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        dataset_dir = self._get_dataset_dir(path)
        edf_dir = self._download_and_extract_edf(dataset_dir, force_update, verbose)

        paths = []
        for session in range(1, 6):
            edf_path = self._find_edf_file(edf_dir, subject, session)
            paths.append(str(edf_path))

        return paths

    def _get_single_subject_data(self, subject):
        """Return the data for a single subject.

        Returns
        -------
        dict
            Dictionary of {session_label: {run_label: mne.io.Raw}}.
        """
        dataset_dir = self._get_dataset_dir()
        edf_dir = self._download_and_extract_edf(dataset_dir)

        sessions = {}
        for ses_idx in range(1, 6):
            session_label = str(ses_idx)

            # Find and read the EDF file
            edf_path = self._find_edf_file(edf_dir, subject, ses_idx)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

            # Download and read the events TSV
            events_tsv_path = self._download_events_tsv(subject, ses_idx, dataset_dir)
            events_df = pd.read_csv(events_tsv_path, sep="\t")

            # Create annotations from events TSV
            # The events TSV has onset in milliseconds and sample indices
            # value: 1 = left_hand, 2 = right_hand
            # trial_type: "left" or "right"
            annotations = self._create_annotations(events_df, raw.info["sfreq"])
            raw.set_annotations(annotations)

            sessions[session_label] = {"0": raw}

        return sessions

    @staticmethod
    def _create_annotations(events_df, sfreq):
        """Create MNE Annotations from the events TSV DataFrame.

        The events TSV from this dataset has columns:
        onset, duration, trial_type, response_time, sample, value

        - onset: in milliseconds (based on the events.json description)
        - duration: in milliseconds
        - trial_type: "left" or "right"
        - value: 1 (left) or 2 (right)

        However, looking at the actual data, onset values are sample indices
        (e.g., 1, 1001, 2001, ...) with 1000 sample spacing at 250 Hz = 4s.
        The "sample" column matches onset, confirming these are sample indices.
        We convert to seconds using the sampling frequency.

        We map trial_type to MOABB standard event names:
        - "left" -> "left_hand"
        - "right" -> "right_hand"
        """
        _trial_type_map = {"left": "left_hand", "right": "right_hand"}

        # Convert onset from samples to seconds
        # onset values are 1-based sample indices (Matlab convention)
        onsets_seconds = (events_df["onset"].values - 1) / sfreq

        # Convert duration from samples to seconds
        durations_seconds = events_df["duration"].values / sfreq

        # Map trial types to MOABB event names
        descriptions = [
            _trial_type_map.get(str(tt).strip(), str(tt).strip())
            for tt in events_df["trial_type"]
        ]

        annotations = mne.Annotations(
            onset=onsets_seconds,
            duration=durations_seconds,
            description=descriptions,
        )
        return annotations
