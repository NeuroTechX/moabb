"""Classifying three imaginary states of the same upper extremity.

Tavakolan, Frehlick, Yong, and Menon (2017), PLOS ONE.
DOI: 10.1371/journal.pone.0174161
"""

import logging
import os
import re
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
    PreprocessingMetadata,
    Tags,
)
from .utils import safe_extract_zip


log = logging.getLogger(__name__)

DRYAD_API_URL = "https://datadryad.org/api/v2/files/{file_id}/download"
DRYAD_TOKEN_URL = "https://datadryad.org/oauth/token"

# BCI2000 StimulusCode -> event name mapping.
# From the BCI2000 header: stimulus1=Rest, stimulus2=Wrist, stimulus3=Elbow,
# stimulus4=Reach-Hold the Glass.
_STIM_CODE_TO_EVENT = {
    1: "rest",
    2: "right_hand",
    3: "right_elbow_flexion",
}
# StimulusCode 4 (Reach-Hold the Glass) is excluded from the default event
# mapping because the paper analyses only three classes.

# Mapping of (subject, session) -> Dryad file_stream ID.
# Obtained from the Dryad API v2 for doi:10.5061/dryad.6qs86.
# fmt: off
_FILE_IDS = {
    (1, 1): 13762, (1, 2): 13764, (1, 3): 13766, (1, 4): 13768,
    (2, 1): 13770, (2, 2): 13792, (2, 3): 13814, (2, 4): 13836,
    (3, 1): 13772, (3, 2): 13794, (3, 3): 13816, (3, 4): 13838,
    (4, 1): 13774, (4, 2): 13796, (4, 3): 13818, (4, 4): 13840,
    (5, 1): 13776, (5, 2): 13798, (5, 3): 13820, (5, 4): 13842,
    (6, 1): 13778, (6, 2): 13800, (6, 3): 13822, (6, 4): 13844,
    (7, 1): 13780, (7, 2): 13802, (7, 3): 13824, (7, 4): 13846,
    (8, 1): 13782, (8, 2): 13804, (8, 3): 13826, (8, 4): 13848,
    (9, 1): 13784, (9, 2): 13806, (9, 3): 13828, (9, 4): 13850,
    (10, 1): 13786, (10, 2): 13808, (10, 3): 13830, (10, 4): 13852,
    (11, 1): 13788, (11, 2): 13810, (11, 3): 13832, (11, 4): 13854,
    (12, 1): 13790, (12, 2): 13812, (12, 3): 13834, (12, 4): 13856,
}
# fmt: on

# Number of EEG channels in the GSN-HydroCel-32 net (excluding Cz reference).
_N_EEG = 32

# Channel gain (µV per raw ADC unit) from the BCI2000 header.
_GAIN_UV = 0.0238419


class Tavakolan2017(BaseDataset):
    """Motor imagery dataset for three imaginary states of the same upper extremity.

    Dataset from [1]_.

    This dataset contains 32-channel EEG recordings from 12 healthy subjects
    performing motor imagery of the right upper extremity.  Subjects imagined
    three tasks: rest, grasping (opening/closing fingers to grab an object),
    and elbow flexion/extension (moving the forearm up and down).

    EEG was recorded at 1000 Hz using a 32-channel EGI Geodesic Sensor Net
    (GES 400 series amplifier) with Cz as the online reference.  Each subject
    completed 4 sessions on separate days, with 20 trials per class per session
    (80 trials total per session, 4 classes).

    Each trial consisted of a 3 s visual cue (during which the subject
    performed the imagery) followed by a 4-6 s rest interval.  The imagery
    interval [0, 3] s after cue onset is used for analysis.

    The data is stored on the Dryad Digital Repository [2]_ as ZIP archives
    (one per subject-session) containing BCI2000 ``.DAT`` files.

    .. note::
        Downloading requires Dryad API credentials.  Set the environment
        variables ``DRYAD_CLIENT_ID`` and ``DRYAD_CLIENT_SECRET`` before
        calling ``get_data()``.  You can obtain them by creating a free
        account at https://datadryad.org.

    .. note::
        Reading BCI2000 ``.DAT`` files requires the ``BCI2kReader`` package::

            pip install BCI2kReader

    Notes
    -----
    The original channel labels follow the EGI HydroCel Geodesic Sensor Net
    naming convention (E1-E32 plus Cz reference).  The ``GSN-HydroCel-32``
    montage from MNE is applied.

    The raw BCI2000 files contain 280 source channels; only the first 32 are
    EEG.  Channels are scaled from raw ADC units to volts using the gain
    from the BCI2000 header (0.0238419 µV per count).

    The BCI2000 files actually contain four stimulus classes (Rest, Wrist,
    Elbow, Reach-Hold the Glass) with StimulusCodes 1-4.  Following the
    paper's analysis of three classes, only codes 1-3 are mapped to events
    by default.

    References
    ----------
    .. [1] M. Tavakolan, Z. Frehlick, X. Yong, and C. Menon,
       "Classifying three imaginary states of the same upper extremity
       using time-domain features," PLoS ONE, vol. 12, no. 3, e0174161,
       2017. DOI: 10.1371/journal.pone.0174161

    .. [2] M. Tavakolan, Z. Frehlick, X. Yong, and C. Menon,
       "Data from: Classifying three imaginary states of the same upper
       extremity using time-domain features," Dryad, 2017.
       DOI: 10.5061/dryad.6qs86
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="GSN-HydroCel-32",
            hardware="EGI Geodesic Net Amps 400 series",
            sensor_type="Ag/AgCl sponge",
            reference="Cz",
            impedance_threshold_kohm=50,
            filters={"bandpass": [0.1, 100]},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={
                "rest": 1,
                "right_hand": 2,
                "right_elbow_flexion": 3,
            },
            paradigm="imagery",
            n_classes=3,
            class_labels=["rest", "right_hand", "right_elbow_flexion"],
            trial_duration=3.0,
            study_design=(
                "Three-class motor imagery of the same upper extremity: "
                "rest, grasping (MI-GRASP), and elbow flexion (MI-ELBOW). "
                "20 trials per class per session, 4 sessions per subject."
            ),
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "REST: relax without movement. "
                "MI-GRASP: imagine opening and closing all fingers to grab "
                "an object. MI-ELBOW: imagine moving the forearm up and down."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0174161",
            investigators=[
                "Mojgan Tavakolan",
                "Zack Frehlick",
                "Xinyi Yong",
                "Carlo Menon",
            ],
            senior_author="Carlo Menon",
            institution="Simon Fraser University",
            institution_department=(
                "MENRVA Research Group, School of Mechatronic Systems Engineering"
            ),
            country="CA",
            data_url="https://datadryad.org/stash/dataset/doi:10.5061/dryad.6qs86",
            repository="Dryad",
            license="CC0-1.0",
            publication_year=2017,
            ethics_approval=["Simon Fraser University Office of Research Ethics"],
            keywords=[
                "motor imagery",
                "EEG",
                "upper extremity",
                "same limb",
                "time-domain features",
                "SVM",
                "BCI",
            ],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="continuous",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["rest", "right_hand", "right_elbow_flexion"],
            cue_duration_s=3.0,
            imagery_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=2880,
            trials_context="12 subjects x 4 sessions x 60 trials (20 per class)",
            n_trials_per_class={
                "rest": 20,
                "right_hand": 20,
                "right_elbow_flexion": 20,
            },
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        sessions_per_subject=4,
        runs_per_session=1,
        file_format="BCI2000",
    )

    _events = {
        "rest": 1,
        "right_hand": 2,
        "right_elbow_flexion": 3,
    }

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=4,
            events=self._events,
            code="Tavakolan2017",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1371/journal.pone.0174161",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject across all sessions.

        Each session is stored as a separate ZIP file on Dryad containing a
        single BCI2000 ``.DAT`` file.  The first 32 channels are EEG
        (GSN-HydroCel-32 net).  Events are extracted from the ``StimulusCode``
        state variable.
        """
        sessions = {}
        for ses_idx in range(1, 5):
            dat_path = self.data_path(subject, session=ses_idx)
            raw = self._read_bci2000_dat(dat_path)
            sessions[str(ses_idx - 1)] = {"0": raw}
        return sessions

    def _read_bci2000_dat(self, dat_path):
        """Read a BCI2000 .DAT file and return an MNE Raw object."""
        try:
            from BCI2kReader.BCI2kReader import BCI2kReader
        except ImportError:
            raise ImportError(
                "BCI2kReader is required for Tavakolan2017.  "
                "Install it with: pip install BCI2kReader"
            )

        reader = BCI2kReader(dat_path)
        sfreq = reader.samplingrate

        # Extract first 32 EEG channels and scale to volts
        n_eeg = min(_N_EEG, reader.signals.shape[0])

        # Parse gain from the BCI2000 header
        gain_uv = _GAIN_UV
        if "SourceChGain" in reader.parameters:
            gain_str = reader.parameters["SourceChGain"][0]
            m = re.match(r"([0-9.eE+-]+)", gain_str)
            if m:
                gain_uv = float(m.group(1))

        data = reader.signals[:n_eeg].astype(np.float64) * gain_uv * 1e-6  # -> V

        # Channel names
        ch_names = [f"E{i}" for i in range(1, n_eeg + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)

        # Set montage
        montage = mne.channels.make_standard_montage("GSN-HydroCel-32")
        raw.set_montage(montage, match_case=False, on_missing="warn")

        # Extract events from StimulusCode
        stim = reader.states["StimulusCode"].flatten()
        transitions = np.where(np.diff(stim) != 0)[0] + 1
        for onset_idx in transitions:
            code = int(stim[onset_idx])
            if code in _STIM_CODE_TO_EVENT:
                onset_sec = onset_idx / sfreq
                raw.annotations.append(onset_sec, 3.0, _STIM_CODE_TO_EVENT[code])

        return raw

    @staticmethod
    def _get_dryad_token():
        """Obtain an OAuth Bearer token from Dryad.

        Requires ``DRYAD_CLIENT_ID`` and ``DRYAD_CLIENT_SECRET`` environment
        variables to be set (obtainable from https://datadryad.org).
        """
        import requests

        client_id = os.environ.get("DRYAD_CLIENT_ID", "")
        client_secret = os.environ.get("DRYAD_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise EnvironmentError(
                "Dryad API credentials are required.  Set the DRYAD_CLIENT_ID "
                "and DRYAD_CLIENT_SECRET environment variables.  You can obtain "
                "them from https://datadryad.org by creating an account."
            )
        resp = requests.post(
            DRYAD_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def _download_dryad_file(self, file_id, dest_dir, subject, session):
        """Download a file from Dryad using OAuth authentication.

        Returns the path to the downloaded ZIP file.
        """
        import requests

        zip_path = dest_dir / f"P{subject:02d}_Se{session:02d}.zip"
        if zip_path.exists():
            return str(zip_path)

        token = self._get_dryad_token()
        url = DRYAD_API_URL.format(file_id=file_id)
        log.info("Downloading %s from Dryad ...", url)

        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()

        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

        return str(zip_path)

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        session=None,
    ):
        """Return local path to the .DAT file for a given subject and session.

        Parameters
        ----------
        subject : int
            Subject number (1-12).
        path : str | None
            Custom download location.
        force_update : bool
            Force re-download.
        update_path : None
            Unused, kept for API compatibility.
        verbose : bool | None
            Verbosity level.
        session : int | None
            Session number (1-4).  If None, downloads session 1.

        Returns
        -------
        dat_path : str
            Path to the extracted BCI2000 .DAT file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        if session is None:
            session = 1
        if session not in (1, 2, 3, 4):
            raise ValueError(f"Invalid session number: {session}")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"P{subject:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Check if a .DAT file already exists for this session
        dat_files = list(subj_dir.glob(f"*Se{session:02d}*.DAT")) + list(
            subj_dir.glob(f"*Session{session:02d}*.DAT")
        )
        if dat_files and not force_update:
            return str(dat_files[0])

        # Download the ZIP from Dryad using OAuth API
        file_id = _FILE_IDS[(subject, session)]
        zip_path = self._download_dryad_file(file_id, subj_dir, subject, session)

        # Extract .DAT files from the ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            dat_members = [
                m
                for m in zf.infolist()
                if m.filename.upper().endswith(".DAT")
                and not m.filename.startswith("__MACOSX")
            ]
            safe_extract_zip(zf, subj_dir, members=dat_members)

        # Find the extracted .DAT file
        dat_files = list(subj_dir.glob("**/*.DAT"))
        if not dat_files:
            raise FileNotFoundError(
                f"No .DAT file found after extracting ZIP for subject "
                f"{subject}, session {session}"
            )

        return str(dat_files[0]) if len(dat_files) == 1 else str(sorted(dat_files)[0])
