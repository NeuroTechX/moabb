import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np
import requests
from mne.channels import make_standard_montage
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from tqdm import tqdm

from moabb.datasets.base import BaseDataset


BRAINFORM_URL = "https://zenodo.org/api/records/17225966/draft/files/BIDS.zip/content"


class RomaniBF2025ERP(BaseDataset):
    """
    MOABB class for BrainForm event-related potentials (ERP) dataset.

    The BrainForm dataset [1]_ is a dataset collected using a serious game
    for brain-computer interface (BCI) training and data collection. It includes
    EEG recordings from multiple subjects performing a ERP task on 10 unique stimuli.
    The dataset is organized in BIDS format and contains calibration and inference sessions.

    Each subject performed two calibration sessions, one with a checkerboard texture on the targets, the other one with
    a grain texture. Each calibration was followed by an inference session where the subject played the game. The game
    consisted of two separate tasks: in the first one, the subject had to hit moving aliens using the color matched
    target; in the second one, they need to follow a randomized sequence of colors by hitting the corresponding targets
    to unlock a door.

    Calibration sessions consisted of 60 trials on a single training target, for a total of 600 events (60 trials × 10
    unique targets). This means that by default, the data is unbalanced, with 60 target events and 540 non-target events
    per session.
    In inference sessions, the number of events varied depending on the subject's performance

    A total of 16 subjects took an optional free-play run after the main protocol, where they could choose their favorite
    texture and play the game again. The extra sessions include a calibration and can be included by setting the `extra_runs`
    parameter. The full protocol is described in [1]_.
    A study on cross-subject decoding using this dataset is presented in [2]_.

    A total of 2 subjects (15 and 18) did not complete the full protocol and are excluded by default.

    For many subjects, multiple calibration attempts were not successful, but they can be included by setting the
    `exclude_failed` parameter to False.

    Inference sessions can be included along with calibration by setting the `include_inference` parameter to True, but
    the triggers only indicate the stimulus onset and not ground truth labels.

    You can test the dataset with the following code:

    dataset = RomaniBF2025ERP(include_inference=True, exclude_failed=False)
    subject_sessions = dataset._get_single_subject_data(2)
    for ses, runs in subject_sessions.items():
        print(ses, list(runs.keys()))

    If all sessions are included, for each subject the output will look like this:
        ses-cb ['1calibration', '2inference']
        ses-grain ['1calibration', '2inference']
        ses-grainExtra ['1calibration', '2inference']
        ses-grainFailed0 ['1calibration', '2inference']
        ses-grainFailed1 ['1calibration', '2inference']
        ses-grainFailed2 ['1calibration', '2inference']

    Recording details:
        - EEG signals were recorded using a g.tec Unicorn with a sampling rate of 250 Hz and conductive gel applied.

        - Data were collected in Trento, Italy, where the power line frequency is 50 Hz.

        - EEG was recorded from 8 scalp electrodes according to the international 10--20 system:
          "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"
          EEG signals were referenced to the right mastoid and grounded to the left mastoid.

    References
    ----------

    [1] M. Romani, D. Zanoni, E. Farella, and L. Turchet, “BrainForm: a Serious Game for BCI Training and Data Collection,”
    Oct. 14, 2025, arXiv: arXiv:2510.10169. doi: 10.48550/arXiv.2510.10169.

    [2] M. Romani, F. Paissan, A. Fossà, and E. Farella, “Explicit modelling of subject dependency in BCI decoding,”
    Sept. 27, 2025, arXiv: arXiv:2509.23247. doi: 10.48550/arXiv.2509.23247.


    """

    _EVENTS = {"Target": 1, "NonTarget": 2}
    _CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    _SFREQ = 250

    def __init__(
        self,
        data_folder: str = None,
        subjects: Optional[List[int]] = None,
        exclude_subjects: Optional[List[int]] = None,
        calibration_length: int = 60,
        interval: tuple = [-0.1, 1.0],
        extra_runs: bool = True,
        include_inference: bool = False,
        exclude_failed: bool = True,
        rescale: int = 1e6,
        montage: str = "standard_1020",
    ):
        """
        Initialize the Brainform MOABB dataset.

        Parameters:
        -----------
        data_folder : str, optional
            Path to the Brainform dataset folder. If None, will download
            and extract the dataset to a temporary directory. You can provide the path to the original dataset
            or to a new dataset collected with BrainForm and converted to BIDS format.
        subjects : List[int], optional
            List of subject indices to include. If None, include all subjects.
        exclude_subjects : List[int], optional
            List of subject indices to exclude. By default, 15 and 18 are excluded since they did not complete the protocol.
        calibration_length : int
            Number of calibration trials per target.
        interval : tuple
            Time interval for epoching (in seconds).
        extra_runs : bool
            Whether to include extra runs session.
        include_inference : bool
            Whether to include inference data along with calibration.
        exclude_failed : bool
            Whether to exclude failed calibration sessions.
        rescale : int
            Factor to rescale the EEG data.
        montage : str
            Montage to use for EEG channel locations.
        """
        # Handle data folder - download if not provided
        if data_folder is None:
            self.data_folder = self._download_and_extract_dataset()
            self._is_temp_dir = True
        else:
            self.data_folder = data_folder
            self._is_temp_dir = False

        self.n_targets = 10  # Fixed for BrainForm dataset, 10 unique targets
        self.calibration_length = calibration_length
        self.extra_runs = extra_runs
        self.include_inference = include_inference
        self.exclude_failed = exclude_failed
        self.rescale = rescale
        self.montage = montage

        if subjects is None:
            # Discover subjects from BIDS structure
            subjects = self._discover_subjects()
        if exclude_subjects is None:
            exclude_subjects = [15, 18]
        if exclude_subjects is not None:
            subjects = [s for s in subjects if s not in exclude_subjects]

        if calibration_length <= 0:
            raise ValueError("calibration_length must be positive")
        if interval[0] >= interval[1]:
            raise ValueError("interval must be [start, end] with start < end")

        events = {"Target": 1, "NonTarget": 2}

        super().__init__(
            subjects=subjects,
            sessions_per_subject=2,
            events=events,
            code="RomaniBF2025ERP",
            interval=interval,
            paradigm="p300",
            doi="10.48550/arXiv.2510.10169",
        )

    def _download_and_extract_dataset(self) -> str:
        """
        Download and extract the Brainform dataset.

        Returns:
        --------
        str
            Path to the extracted dataset folder
        """
        # Create a cache directory in user's home
        cache_dir = Path.home() / ".cache" / "brainform_dataset"
        cache_dir.mkdir(parents=True, exist_ok=True)

        zip_path = cache_dir / "BIDS.zip"
        extract_path = cache_dir / "BIDS"

        # Check if already downloaded and extracted
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"Using cached dataset at: {extract_path}")
            return str(extract_path)

        # Download the zip file
        print(f"Downloading Brainform dataset from {BRAINFORM_URL}...")
        try:
            response = requests.get(BRAINFORM_URL, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc="Downloading"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Download complete. Extracting to {extract_path}...")

            # Extract the zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_dir)

            print("Extraction complete!")

            # Optionally remove the zip file to save space
            zip_path.unlink()

            return str(extract_path)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ) -> str:
        """
        Return the path to the dataset.
        Required abstract method from BaseDataset.

        Parameters
        ----------
        subject : int
            Subject identifier
        path : str, optional
            Not used for this dataset
        force_update : bool
            Not used for this dataset
        update_path : bool
            Not used for this dataset
        verbose : bool
            Not used for this dataset

        Returns
        -------
        str
            Path to the dataset folder
        """
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Dataset folder not found: {self.data_folder}")
        return self.data_folder

    def data_url(
        self,
        subject: str,
        path: str,
        force_update: bool = False,
        update_path: bool = None,
        verbose: bool = None,
    ) -> List[str]:
        """
        Return download URLs for the dataset.
        Required abstract method from BaseDataset.

        For local datasets, return empty list.

        Parameters:
        -----------
        subject : str
            Subject identifier
        path : str
            Path to store data
        force_update : bool
            Whether to force update
        update_path : bool
            Whether to update path
        verbose : bool
            Whether to be verbose

        Returns:
        --------
        List[str]
            List of URLs (empty for local datasets)
        """
        return [BRAINFORM_URL]

    def _get_single_subject_data(self, subject: int | str):
        sessions = {}
        if isinstance(subject, str):
            subject_label = subject
        else:
            subject_label = f"P{subject:02d}"  # e.g., sub-P02
        subject_folder = os.path.join(self.data_folder, f"sub-{subject_label}")

        if not os.path.exists(subject_folder):
            print(f"Subject folder {subject_folder} not found.")
            return {}

        # Loop through sessions (e.g. ses-cb, ses-grain, ses-grainExtra)
        for ses_name in sorted(os.listdir(subject_folder)):
            ses_path = os.path.join(subject_folder, ses_name)
            if not os.path.isdir(ses_path) or not ses_name.startswith("ses-"):
                continue

            if self.exclude_failed:
                if ses_name.__contains__("Failed"):
                    print(f"Excluding failed session: {ses_name}")
                    continue

            eeg_dir = os.path.join(ses_path, "eeg")
            if not os.path.exists(eeg_dir):
                continue

            # Look for EEG recording
            edf_files = [f for f in os.listdir(eeg_dir) if f.endswith(".edf")]
            if len(edf_files) == 0:
                print(f"No EEG EDF found for {ses_name}")
                continue

            ses_name = ses_name.replace("ses-", "")
            bids_path = BIDSPath(
                subject=subject_label,
                session=ses_name,
                task="ERP",
                datatype="eeg",
                extension=".edf",
                root=self.data_folder,
            )

            try:
                raw = read_raw_bids(bids_path=bids_path, verbose=False)
                raw.load_data()

                # Set montage
                montage = mne.channels.make_standard_montage(self.montage)
                raw.set_montage(montage)

                # Extract events
                events, event_id = mne.events_from_annotations(raw)

                # Sort by event time
                events = events[np.argsort(events[:, 0])]

                if event_id:
                    target_keys = [k for k in event_id.keys() if "target" in k.lower()]
                    n_targets = (
                        len(target_keys) if target_keys else len(np.unique(events[:, 2]))
                    )
                else:
                    n_targets = len(np.unique(events[:, 2]))

                n_calib_events = self.calibration_length * n_targets

                if len(events) < n_calib_events:
                    print(f"Warning: not enough events in {ses_name}")
                    continue

                # Split calibration vs inference by event count
                calib_end_sample = events[n_calib_events - 1, 0]
                calib_raw = raw.copy().crop(
                    tmin=0, tmax=calib_end_sample / raw.info["sfreq"]
                )
                infer_raw = raw.copy().crop(tmin=calib_end_sample / raw.info["sfreq"])

                sessions[ses_name] = {
                    "1calibration": calib_raw,
                }
                if self.include_inference:
                    sessions[ses_name]["2inference"] = infer_raw

            except Exception as e:
                print(f"Error reading {bids_path}: {e}")

        return sessions

    def is_valid(self, subject: int = None) -> bool:
        """
        Check if the dataset is valid.

        Parameters
        ----------
        subject : int, optional
            Subject to check. If None, check overall dataset validity.

        Returns
        -------
        bool
            True if dataset is valid
        """
        if subject is not None:
            subject_label = f"P{subject:02d}"
            subject_path = os.path.join(self.data_folder, f"sub-{subject_label}")
            return os.path.exists(subject_path) and os.path.isdir(subject_path)
        else:
            # Check if dataset folder exists and has at least one subject
            if not os.path.exists(self.data_folder):
                return False
            try:
                subjects = self._discover_subjects()
                return len(subjects) > 0
            except Exception:
                return False

    def _load_session_data(self, subject: str, session: str) -> Dict[str, mne.io.Raw]:
        """
        Load data for a specific subject and session.
        Reads BIDS-formatted EEG data and splits calibration/inference
        based on calibration_length × n_targets (inferred from events).

        Parameters
        ----------
        subject : str
            e.g. "P02"
        session : str
            e.g. "cb"

        Returns
        -------
        Dict[str, mne.io.Raw]
            e.g. {'1calibration': raw_cal, '2inference': raw_inf}
        """
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task="ERP",
            datatype="eeg",
            extension=".edf",
            root=self.data_folder,
        )

        if not bids_path.fpath.exists():
            print(f"No EDF file for sub-{subject}, ses-{session}")
            return {}

        try:
            # Load the raw EEG data
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            raw.load_data()

            # Set montage (10–20 system)
            montage = make_standard_montage(self.montage)
            raw.set_montage(montage)

            # Read events from annotations (preferred)
            events, event_id = mne.events_from_annotations(raw)
            events = events[np.argsort(events[:, 0])]  # sort by onset

            # --- Infer number of unique targets automatically ---
            # Assume event codes: "Target" and "NonTarget"
            if event_id:
                n_targets = sum("Target" in k or "target" in k for k in event_id.keys())
                if n_targets == 0:
                    # fallback: infer from event values
                    unique_vals = np.unique(events[:, 2])
                    n_targets = len(unique_vals)
            else:
                unique_vals = np.unique(events[:, 2])
                n_targets = len(unique_vals)

            # Compute how many events belong to calibration
            n_calib_events = self.calibration_length * n_targets
            total_events = len(events)
            if total_events < n_calib_events:
                print(
                    f"Warning: {subject} {session} has only {total_events} events (needs {n_calib_events})"
                )
                n_calib_events = total_events // 2  # fallback heuristic

            # Split calibration and inference based on event sample index
            calib_end_sample = events[n_calib_events - 1, 0]
            fs = raw.info["sfreq"]

            raw_cal = raw.copy().crop(tmin=0, tmax=calib_end_sample / fs)
            data = {"1calibration": raw_cal}

            if self.include_inference:
                raw_inf = raw.copy().crop(tmin=calib_end_sample / fs)
                data["2inference"] = raw_inf

            return data

        except Exception as e:
            print(f"Error loading sub-{subject}, ses-{session}: {e}")
            return {}

    def _get_single_run_data(self, subject, run):
        """
        Return data for a single run of a single subject.
        Required by MOABB's BaseDataset.

        Parameters
        ----------
        subject : int
            Subject number
        run : str
            Run identifier (e.g., 'ses-cb/1calibration')

        Returns
        -------
        raw : mne.io.Raw
            Raw data for the run
        """
        # Parse the run string to extract session and run type
        parts = run.split("/")
        if len(parts) == 2:
            session_name, run_type = parts
        else:
            raise ValueError(f"Invalid run format: {run}")

        subject_label = f"P{subject:02d}"
        bids_path = BIDSPath(
            subject=subject_label,
            session=session_name.replace("ses-", ""),
            task="ERP",
            datatype="eeg",
            extension=".edf",
            root=self.data_folder,
        )

        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()

        # Set montage
        montage = make_standard_montage(self.montage)
        raw.set_montage(montage)

        # Split calibration/inference if needed
        events, event_id = mne.events_from_annotations(raw)
        events = events[np.argsort(events[:, 0])]

        n_targets = len([k for k in event_id.keys() if "target" in k.lower()])
        if n_targets == 0:
            n_targets = len(np.unique(events[:, 2]))

        n_calib_events = self.calibration_length * n_targets
        calib_end_sample = events[n_calib_events - 1, 0]
        fs = raw.info["sfreq"]

        if run_type == "1calibration":
            raw = raw.crop(tmin=0, tmax=calib_end_sample / fs)
        elif run_type == "2inference":
            raw = raw.crop(tmin=calib_end_sample / fs)

        return raw

    def _discover_subjects(self) -> List[str]:
        """
        Discover available subjects from the BIDS structure.

        Returns
        -------
        subjects : List[str]
            e.g. ["P01", "P02", "P03"]
        """
        subject_labels = get_entity_vals(root=self.data_folder, entity_key="subject")
        subjects = []

        for subj in subject_labels:
            subj_path = os.path.join(self.data_folder, f"sub-{subj}")
            if os.path.isdir(subj_path):
                subjects.append(subj)

        if not subjects:
            print("No subjects found in BIDS folder.")

        return subjects

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(subjects={len(self.subject_list)}, "
            f"sessions={self.n_sessions}, extra_runs={self.extra_runs})>"
        )
