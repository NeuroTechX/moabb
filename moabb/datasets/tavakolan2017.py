"""Classifying three imaginary states of the same upper extremity.

Tavakolan, Frehlick, Yong, and Menon (2017), PLOS ONE.
DOI: 10.1371/journal.pone.0174161
Data DOI (original): 10.5061/dryad.6qs86
"""

import logging
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
from .utils import download_and_extract_subject_zip, safe_extract_zip


log = logging.getLogger(__name__)

# Zenodo re-hosted data (originally from Dryad DOI: 10.5061/dryad.6qs86).
_ZENODO_RECORD = "18967205"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# BCI2000 StimulusCode -> event name mapping.
# From the BCI2000 header: stimulus1=Rest, stimulus2=Wrist, stimulus3=Elbow,
# stimulus4=Reach-Hold the Glass.
_STIM_CODE_TO_EVENT = {1: "rest", 2: "right_hand", 3: "right_elbow_flexion"}
# StimulusCode 4 (Reach-Hold the Glass) is excluded from the default event
# mapping because the paper analyses only three classes.

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

    The data was originally deposited on the Dryad Digital Repository [2]_
    and has been re-hosted on Zenodo for direct programmatic access.

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
            n_subjects=12, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events={"rest": 1, "right_hand": 2, "right_elbow_flexion": 3},
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
                "MENRVA Research Group, Schools of Mechatronic Systems Engineering"
                " and Engineering Science"
            ),
            country="CA",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
            repository="Zenodo",
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
        preprocessing=PreprocessingMetadata(data_state="continuous"),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["rest", "right_hand", "right_elbow_flexion"],
            cue_duration_s=3.0,
            imagery_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=2880,
            trials_context="12 subjects x 4 sessions x 60 trials (20 per class)",
            n_trials_per_class={"rest": 20, "right_hand": 20, "right_elbow_flexion": 20},
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SVM-RBF"],
            feature_extraction=[
                "autoregressive_coefficients",
                "waveform_length",
                "root_mean_square",
            ],
            frequency_bands={"bandpass": [6.0, 35.0]},
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10x10-fold", cv_folds=10, evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        sessions_per_subject=4,
        runs_per_session=1,
        file_format="BCI2000",
    )

    _events = {"rest": 1, "right_hand": 2, "right_elbow_flexion": 3}

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
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
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject across all sessions.

        Each session is stored as a session ZIP inside the per-subject ZIP
        on Zenodo, containing a single BCI2000 ``.DAT`` file.  The first 32
        channels are EEG (GSN-HydroCel-32 net).  Events are extracted from
        the ``StimulusCode`` state variable.
        """
        # Ensure all sessions are downloaded and extracted
        self.data_path(subject)

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, None)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"P{subject:02d}"

        sessions = {}
        for ses_idx in range(1, 5):
            dat_files = sorted(subj_dir.glob(f"*Se{ses_idx:02d}*.DAT"))
            if not dat_files:
                log.warning("Missing .DAT for subject %d session %d", subject, ses_idx)
                continue
            raw = self._read_bci2000_dat(str(dat_files[0]))
            sessions[str(ses_idx - 1)] = {"0": raw}
        return sessions

    def _read_bci2000_dat(self, dat_path):
        """Read a BCI2000 .DAT file and return an MNE Raw object."""
        try:
            from BCI2kReader.BCI2kReader import BCI2kReader
        except ImportError as err:
            raise ImportError(
                "BCI2kReader is required for Tavakolan2017.  "
                "Install it with: pip install BCI2kReader"
            ) from err

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

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return local path to the subject directory containing .DAT files.

        Downloads the per-subject ZIP from Zenodo if not already present,
        then extracts the nested session ZIPs to obtain the BCI2000 .DAT
        files.

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

        Returns
        -------
        data_dir : str
            Path to the dataset root directory.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"P{subject:02d}"

        # Check if .DAT files already exist for all 4 sessions
        dat_files = sorted(subj_dir.glob("*.DAT")) if subj_dir.is_dir() else []
        if len(dat_files) >= 4 and not force_update:
            return str(data_dir)

        # Download per-subject ZIP from Zenodo and extract session ZIPs.
        url = f"{_ZENODO_BASE}/P{subject:02d}.zip"
        download_and_extract_subject_zip(url, sign, subj_dir, path, force_update, verbose)

        # Each session ZIP (P{NN}_Se{NN}.zip) contains a BCI2000 .DAT file.
        # Extract the .DAT from each session ZIP.
        for ses_zip in sorted(subj_dir.glob("P*_Se*.zip")):
            with zipfile.ZipFile(ses_zip) as szf:
                dat_members = [
                    m
                    for m in szf.infolist()
                    if m.filename.upper().endswith(".DAT")
                    and not m.filename.startswith("__MACOSX")
                ]
                safe_extract_zip(szf, subj_dir, members=dat_members)

        # Verify extraction
        dat_files = sorted(subj_dir.glob("*.DAT"))
        if not dat_files:
            # .DAT files might be in subdirectories — flatten them
            for dat in subj_dir.rglob("*.DAT"):
                if dat.parent != subj_dir:
                    import shutil

                    shutil.move(str(dat), str(subj_dir / dat.name))
            dat_files = sorted(subj_dir.glob("*.DAT"))

        if not dat_files:
            raise FileNotFoundError(
                f"No .DAT files found for subject {subject} in {subj_dir}"
            )

        return str(data_dir)
