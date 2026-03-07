"""Upper-limb elbow-centered motor imagery dataset (10 classes, 12 subjects).

Zhang, Yong, and Menon (2017), PLoS ONE.
DOI: 10.1371/journal.pone.0188293
"""

import logging
import shutil
import subprocess
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


log = logging.getLogger(__name__)

_FIGSHARE_URL = "https://ndownloader.figshare.com/files/9700792"


def _find_rar_tool():
    """Return the command list for extracting RAR archives, or raise."""
    for tool, args in [
        ("unrar", ["unrar", "x", "-o+"]),
        ("unar", ["unar", "-f", "-o"]),
        ("7z", ["7z", "x", "-y", "-o"]),
    ]:
        if shutil.which(tool) is not None:
            return tool, args
    raise RuntimeError(
        "No RAR extraction tool found. Install one of: unrar, unar, or 7z. "
        "For example: 'brew install unar' (macOS) or 'apt install unrar' (Linux)."
    )


def _extract_rar(rar_path, dest_dir):
    """Extract a RAR archive into *dest_dir* using an available system tool."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rar_path = str(rar_path)

    tool, base_args = _find_rar_tool()

    if tool == "unrar":
        # unrar x -o+ archive.rar dest/
        cmd = base_args + [rar_path, str(dest_dir) + "/"]
    elif tool == "unar":
        # unar -f -o dest/ archive.rar
        cmd = base_args + [str(dest_dir), rar_path]
    elif tool == "7z":
        # 7z x -y -oDEST archive.rar
        cmd = base_args[:-1] + [base_args[-1] + str(dest_dir), rar_path]

    log.info("Extracting RAR with: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"RAR extraction failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def _read_bci2000_file(fpath):
    """Read a BCI2000 .dat file and return (signals, states, sfreq, parameters).

    Returns
    -------
    signals : ndarray, shape (n_channels, n_samples)
    states : dict
        State variable name -> 1D array of integer values.
    sfreq : float
        Sampling rate in Hz.
    parameters : dict-like
        BCI2000 parameter object.
    """
    try:
        from BCI2kReader.BCI2kReader import BCI2kReader
    except ImportError:
        raise ImportError(
            "The BCI2kReader package is required to read BCI2000 .dat files. "
            "Install it with: pip install BCI2kReader"
        )

    reader = BCI2kReader(str(fpath))
    signals = reader.signals  # (n_samples, n_channels)
    states = reader.states
    sfreq = reader.samplingrate
    parameters = reader.parameters
    reader.close()

    # BCI2kReader returns signals as (n_samples, n_channels) -> transpose
    signals = np.asarray(signals).T  # (n_channels, n_samples)

    # Convert states dict values to numpy arrays
    state_dict = {}
    for key in states.dtype.names:
        state_dict[key] = np.asarray(states[key]).ravel()

    return signals, state_dict, float(sfreq), parameters


def _bci2000_to_raw(fpath, event_mapping):
    """Convert a BCI2000 .dat file to an MNE Raw object with annotations.

    Parameters
    ----------
    fpath : str or Path
        Path to BCI2000 .dat file.
    event_mapping : dict
        Mapping from integer StimulusCode values to string event names.

    Returns
    -------
    raw : mne.io.RawArray
    """
    signals, states, sfreq, parameters = _read_bci2000_file(fpath)
    n_channels, n_samples = signals.shape

    # Build channel names - EGI uses numbered electrodes
    # BCI2000 with EGI typically names channels Ch1..ChN or numbered.
    # We use E1..E32 + Cz to match the GSN-HydroCel-32 montage.
    if n_channels == 32:
        ch_names = [f"E{i}" for i in range(1, 33)]
    elif n_channels == 33:
        # 32 EGI channels + Cz reference (if included)
        ch_names = [f"E{i}" for i in range(1, 33)] + ["Cz"]
    else:
        # Fallback: generic channel names
        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
        log.warning(
            "Unexpected number of channels (%d) for subject. "
            "Using generic channel names.",
            n_channels,
        )

    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Scale to Volts (BCI2000 data is typically in microvolts)
    raw = mne.io.RawArray(signals * 1e-6, info, verbose=False)

    # Set montage
    try:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-32")
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        log.warning("Could not set GSN-HydroCel-32 montage.")

    # Extract event annotations from StimulusCode state variable
    stim_key = None
    for candidate in ["StimulusCode", "TargetCode", "Stimulus"]:
        if candidate in states:
            stim_key = candidate
            break

    if stim_key is None:
        log.warning(
            "No StimulusCode or TargetCode state found in %s. "
            "Available states: %s. No events will be set.",
            fpath,
            list(states.keys()),
        )
        return raw

    stim = states[stim_key]

    # Find onsets: where stim code transitions from 0 to non-zero,
    # or changes to a different non-zero value
    onsets = []
    codes = []
    prev_val = 0
    for i, val in enumerate(stim):
        if val != prev_val and val != 0:
            if val in event_mapping:
                onsets.append(i / sfreq)
                codes.append(event_mapping[val])
        prev_val = val

    if onsets:
        # Use the minimum trial duration as a conservative estimate
        durations = [4.0] * len(onsets)
        annotations = mne.Annotations(onset=onsets, duration=durations, description=codes)
        raw.set_annotations(annotations)
    else:
        log.warning("No events found in StimulusCode for file %s", fpath)

    return raw


class Zhang2017(BaseDataset):
    """Upper-limb elbow-centered motor imagery dataset (10 classes).

    Dataset from [1]_.

    This dataset contains 32-channel EEG recordings from 12 healthy subjects
    (10 male, 2 female, ages 20-33, 11 right-handed) performing 10
    motor imagery tasks involving the dominant upper limb. Data was recorded
    using a 32-channel EGI Geodesic Sensor Net (N400 series) at 1000 Hz
    with Cz reference, using BCI2000 in Stimulus Presentation mode.

    The 10 tasks are:

    - **rest**: stay alert, look at center cross
    - **elbow_flexion**: simple elbow flexion/extension
    - **drawer**: opening/closing a drawer
    - **soup**: spoon-feeding (drinking soup with a spoon)
    - **weight_lifting**: lifting/lowering a dumbbell
    - **door**: opening/closing a door
    - **plate_cleaning**: plate-cleaning movements
    - **combing**: hair-combing
    - **pizza_cutting**: pizza-cutting motions
    - **pick_and_place**: picking a ball into a basket

    Each subject completed 15 runs (~3 minutes each). Each run contained
    24 trials: 4 rest + 4 elbow + 2 each of the 8 goal-directed tasks.
    Trial timing: 4-6 s cue display (randomized) with MI, then 4-6 s rest.
    Total: 60 rest trials + 30 trials per MI task = 330 trials per subject.

    The dataset is distributed as a single RAR archive on Figshare.
    Extraction requires ``unrar``, ``unar``, or ``7z`` to be installed
    on the system. The BCI2000 ``.dat`` files are read using the
    ``BCI2kReader`` package (``pip install BCI2kReader``).

    Notes
    -----
    Subject H5 is left-handed; all other subjects are right-handed.
    In the paper's analysis, H5's channels were flipped between hemispheres.
    This adapter does NOT apply any hemisphere flipping.

    Only 17 of the 32 channels were used in the paper's analysis (facial
    channels excluded). The raw data contains all 32 channels.

    References
    ----------
    .. [1] X. Zhang, X. Yong, and C. Menon, "Evaluating the versatility of
       EEG models generated from motor imagery tasks: An exploratory
       investigation on upper-limb elbow-centered motor imagery tasks,"
       PLoS ONE, vol. 12, no. 11, e0188293, 2017.
       DOI: 10.1371/journal.pone.0188293
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="GSN-HydroCel-32",
            hardware="EGI Geodesic Net Amps 400 series (N400)",
            sensor_type="Ag/AgCl sponge",
            reference="Cz",
            ground="COM",
            software="BCI2000 (Stimulus Presentation mode)",
            filters={"bandpass": [0.1, 40]},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"male": 10, "female": 2},
            age_min=20,
            age_max=33,
            handedness="right-handed (11/12)",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={
                "rest": 1,
                "elbow_flexion": 2,
                "drawer": 3,
                "soup": 4,
                "weight_lifting": 5,
                "door": 6,
                "plate_cleaning": 7,
                "combing": 8,
                "pizza_cutting": 9,
                "pick_and_place": 10,
            },
            paradigm="imagery",
            n_classes=10,
            class_labels=[
                "rest",
                "elbow_flexion",
                "drawer",
                "soup",
                "weight_lifting",
                "door",
                "plate_cleaning",
                "combing",
                "pizza_cutting",
                "pick_and_place",
            ],
            trial_duration=5.0,
            study_design=(
                "Upper-limb elbow-centered motor imagery with 9 goal-directed "
                "tasks plus rest. Each trial: 4-6 s cue (randomized) then "
                "4-6 s rest (randomized)."
            ),
            feedback_type="none",
            stimulus_type="picture cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "Participants were asked to repetitively perform the "
                "kinesthetic motor imagery task displayed on the screen "
                "without actually moving."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0188293",
            investigators=[
                "Xiaogang Zhang",
                "Xinyi Yong",
                "Carlo Menon",
            ],
            institution="Simon Fraser University",
            institution_department="School of Engineering Science",
            country="CA",
            data_url="https://doi.org/10.6084/m9.figshare.5579461.v1",
            publication_year=2017,
            senior_author="Carlo Menon",
            license="CC BY 4.0",
            repository="Figshare",
            keywords=[
                "motor imagery",
                "upper limb",
                "elbow",
                "BCI",
                "EEG",
                "kinesthetic imagery",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=15,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "elbow_flexion",
                "drawer",
                "soup",
                "weight_lifting",
                "door",
                "plate_cleaning",
                "combing",
                "pizza_cutting",
                "pick_and_place",
            ],
            cue_duration_s=5.0,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=330,
            trials_context=(
                "15 runs of 24 trials each (4 rest + 4 elbow + 2 each of "
                "8 goal tasks). Total: 60 rest + 30 per MI task = 330."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
        ),
        file_format="BCI2000",
    )

    # BCI2000 StimulusCode -> event name mapping
    # The exact integer codes depend on the BCI2000 configuration.
    # Common BCI2000 Stimulus Presentation mapping: codes 1-10 for 10 tasks.
    # This mapping will be auto-detected from the data if needed.
    _STIM_CODE_TO_EVENT = {
        1: "rest",
        2: "elbow_flexion",
        3: "drawer",
        4: "soup",
        5: "weight_lifting",
        6: "door",
        7: "plate_cleaning",
        8: "combing",
        9: "pizza_cutting",
        10: "pick_and_place",
    }

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events=dict(
                rest=1,
                elbow_flexion=2,
                drawer=3,
                soup=4,
                weight_lifting=5,
                door=6,
                plate_cleaning=7,
                combing=8,
                pizza_cutting=9,
                pick_and_place=10,
            ),
            code="Zhang2017",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1371/journal.pone.0188293",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject.

        Returns
        -------
        dict
            ``{session_key: {run_key: raw}}`` where each raw is an
            :class:`mne.io.RawArray`.
        """
        dat_files = self.data_path(subject)
        if not dat_files:
            raise FileNotFoundError(f"No BCI2000 .dat files found for subject {subject}.")

        runs = {}
        for run_idx, fpath in enumerate(sorted(dat_files)):
            raw = _bci2000_to_raw(fpath, self._STIM_CODE_TO_EVENT)
            runs[str(run_idx)] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return list of BCI2000 .dat file paths for a subject.

        Downloads and extracts the KI.rar archive from Figshare if needed.

        Parameters
        ----------
        subject : int
            Subject number (1-12).
        path : str or None
            Download destination. Defaults to MNE_DATA.
        force_update : bool
            Re-download even if local files exist.
        update_path : ignored
            Kept for API compatibility.
        verbose : ignored
            Kept for API compatibility.

        Returns
        -------
        list of str
            Paths to BCI2000 .dat files for this subject, sorted.
        """
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject number {subject}. " f"Must be in {self.subject_list}."
            )

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        extract_dir = data_dir / "extracted"

        # Subject identifier used in the data files
        subject_id = f"H{subject}"

        # Check if already extracted
        if not force_update and extract_dir.is_dir():
            dat_files = self._find_subject_files(extract_dir, subject_id)
            if dat_files:
                return dat_files

        # Download RAR archive
        rar_path = Path(dl.data_dl(_FIGSHARE_URL, sign, path, force_update, verbose))

        # Extract RAR
        _extract_rar(rar_path, extract_dir)

        # Find subject files
        dat_files = self._find_subject_files(extract_dir, subject_id)
        if not dat_files:
            # Log available files for debugging
            all_files = list(extract_dir.rglob("*"))
            log.error(
                "No .dat files found for subject %s in %s. " "Available files/dirs: %s",
                subject_id,
                extract_dir,
                [str(f.relative_to(extract_dir)) for f in all_files[:50]],
            )
            raise FileNotFoundError(
                f"No BCI2000 .dat files found for subject {subject_id} "
                f"after extracting {rar_path} to {extract_dir}. "
                f"Please verify the archive contents."
            )

        return dat_files

    @staticmethod
    def _find_subject_files(extract_dir, subject_id):
        """Find BCI2000 .dat files for a subject under extract_dir.

        Searches recursively for .dat files whose path contains the
        subject identifier (e.g. 'H1', 'H2', ..., 'H12').

        Parameters
        ----------
        extract_dir : Path
            Root directory of extracted archive.
        subject_id : str
            Subject identifier, e.g. 'H1'.

        Returns
        -------
        list of str
            Sorted list of .dat file paths.
        """
        # Strategy 1: Look for folder named after subject
        # (e.g., extract_dir/KI/H1/ or extract_dir/H1/)
        matches = []
        for dat_file in extract_dir.rglob("*.dat"):
            # Check if subject_id appears in the path (folder or filename)
            parts = dat_file.parts
            # Match folder names like "H1", "H01" or filenames containing "H1"
            for part in parts:
                if part == subject_id or part == subject_id.replace("H", "H0"):
                    matches.append(str(dat_file))
                    break
            else:
                # Also check if filename starts with subject_id
                stem = dat_file.stem
                if stem.startswith(subject_id) and (
                    len(stem) == len(subject_id) or not stem[len(subject_id)].isdigit()
                ):
                    matches.append(str(dat_file))

        return sorted(matches)
