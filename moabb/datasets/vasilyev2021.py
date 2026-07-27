"""Vasilyev2021 multiday motor-imagery EEG dataset (with and without feedback).

Vasilyev, A. (2021). "Multiday motor imagery with and without feedback."
figshare. Dataset. DOI: 10.6084/m9.figshare.14602872
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


log = logging.getLogger(__name__)

# Single figshare archive (rawData.zip, file id 28033899, ~4.2 GB) that holds
# every participant/session folder. The ndownloader endpoint serves the bytes
# directly; the local file is cached under MNE-Vasilyev2021-data.
FIGSHARE_ZIP_URL = "https://ndownloader.figshare.com/files/28033899"

# Two-letter participant codes, mapped positionally to MOABB subject numbers
# 1..7 in alphabetical order (folders are named <code><3-digit-session>).
_SUBJECT_CODES = ["av", "ks", "ly", "op", "py", "sl", "un"]

_N_SESSIONS = 6

# StimulusCode state values carried inside every BCI2000 .dat file:
#   1 -> motor-imagery cue (active condition)
#   2 -> visual attention to an abstract picture (non-motor reference)
#   0 -> inter-stimulus / blank screen (not an event)
_CODE_IMAGERY = 1
_CODE_NONMOTOR = 2

# BCI2000 DataFormat token -> numpy dtype (little-endian).
_DTYPE_MAP = {"int16": "<i2", "int32": "<i4", "float32": "<f4"}


class Vasilyev2021(BaseDataset):
    """Multiday motor-imagery EEG dataset with and without feedback [1]_.

    .. admonition:: Dataset summary

        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        ===========  =======  =======  ==========  =================  ============  ===============  ===========
        Vasilyev2021       7       30           2                            5s              500 Hz            6
        ===========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Seven participants completed six recording sessions (days) each, performing
    kinesthetic motor imagery under an active-vs-reference design recorded with
    BCI2000. Data were acquired with 30 channels at 500 Hz and are distributed in
    native BCI2000 ``.dat`` format. Each session folder is named by the
    participant's two-letter code and a three-digit session number
    (e.g. ``av001``); the files inside are continuous multichannel runs
    (``...R01``, ``...R02``, ...), 12 to 21 runs per session.

    Within every run the ``StimulusCode`` state channel marks the trial
    structure: value ``1`` = a motor-imagery cue, value ``2`` = visual attention
    to an abstract picture (a non-motor reference condition), and value ``0`` =
    the blank inter-stimulus interval. Each cue lasts about six seconds, with
    typically six imagery and six reference trials per run. This loader exposes
    the two data-borne, per-trial conditions read directly from ``StimulusCode``:

    ================  ============
    Event             Code
    ================  ============
    motor_imagery     1
    non_motor         2
    ================  ============

    A companion ``recData.mat`` table (not required by this loader) additionally
    records, for each run, which of six motor-imagery types was used
    (finger tapping left/right hand, arm circumduction left/right, and right
    thumb swipe without/with a video instruction) and whether online feedback
    was presented. That per-run imagery type is stored as a MATLAB string column
    inside the file's opaque object subsystem and is not exposed as separate
    classes here; every run therefore contributes its imagery trials to the
    single ``motor_imagery`` class contrasted against the ``non_motor``
    reference. The exposed interval spans the first five seconds after cue onset.

    No channel names or electrode montage are provided by the source, so the 30
    channels are exposed with generic names and without a montage; one channel
    (the last) carries a distinct amplifier gain in the file header and may be a
    reference or auxiliary channel.

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load (1..7). Defaults to all seven participants.
    sessions : list of int | None
        Sessions to load. Defaults to all recorded sessions.

    References
    ----------

    .. [1] Vasilyev, A. (2021). Multiday motor imagery with and without
       feedback. figshare. Dataset.
       DOI: https://doi.org/10.6084/m9.figshare.14602872

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=30,
            channel_types={"eeg": 30},
            montage="unknown",
            hardware="BCI2000 acquisition",
            software="BCI2000",
            reference=None,
            ground=None,
            sensors=[],
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=7, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["motor_imagery", "non_motor"],
            trial_duration=6.0,
            task_type="motor_imagery_vs_reference",
            study_design=(
                "Multiday kinesthetic motor imagery recorded with BCI2000 under "
                "an active-vs-reference design. Each run alternates motor-imagery "
                "cues (StimulusCode 1) with a non-motor visual-attention reference "
                "(StimulusCode 2). Runs additionally vary the specific imagery "
                "type and the presence of online feedback (see recData.mat)."
            ),
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="online",
            events={"motor_imagery": 1, "non_motor": 2},
            instructions=(
                "Perform kinesthetic motor imagery of the cued movement, or "
                "attend to the abstract reference picture during the non-motor "
                "condition."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.14602872",
            description=(
                "Multiday motor-imagery EEG from 7 participants across 6 sessions "
                "each, recorded with BCI2000 (30 channels, 500 Hz). Runs contrast "
                "motor imagery against a non-motor visual-attention reference, "
                "with and without online feedback."
            ),
            investigators=["Anatoly Vasilyev"],
            country="RU",
            data_url="https://doi.org/10.6084/m9.figshare.14602872",
            publication_year=2021,
            license="CC BY 4.0",
            repository="figshare",
            funding=["RFBR project number 19-315-60011"],
            keywords=[
                "motor imagery",
                "brain-computer interface",
                "EEG",
                "feedback",
                "multiday",
                "BCI2000",
            ],
        ),
        sessions_per_subject=6,
        runs_per_session=18,
        data_processed=False,
        file_format="BCI2000",
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(environment="lab", online_feedback=True),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "fingers_left",
                "fingers_right",
                "arm_left",
                "arm_right",
                "thumb_right",
                "thumb_right_video",
            ],
            imagery_duration_s=6.0,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=18,
            trials_context=(
                "Seven participants, six sessions each, 12-21 runs per session. "
                "Each run holds roughly six motor-imagery trials and six non-motor "
                "reference trials, about six seconds per cue."
            ),
        ),
        abstract=(
            "Multiday motor-imagery EEG recorded with BCI2000 from seven "
            "participants over six sessions each. Every run contrasts kinesthetic "
            "motor imagery against a non-motor visual-attention reference, and "
            "runs span six imagery types and both feedback and no-feedback "
            "conditions. Distributed as native BCI2000 .dat files plus a "
            "recData.mat description table."
        ),
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, len(_SUBJECT_CODES) + 1)),
            sessions_per_subject=_N_SESSIONS,
            events={"motor_imagery": 1, "non_motor": 2},
            code="Vasilyev2021",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.6084/m9.figshare.14602872",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    # ------------------------------------------------------------------ #
    # Download / extraction
    # ------------------------------------------------------------------ #
    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the path to the extracted directory for one subject.

        Downloads the shared figshare archive once (cached) and extracts only
        the requested subject's session folders.

        Parameters
        ----------
        subject : int
            Subject number (1..7).
        path : None | str
            Storage location override.
        force_update : bool
            Re-download even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of str
            Single-element list with the path to the extraction directory.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        code = _SUBJECT_CODES[subject - 1]

        zip_path = Path(
            dl.data_dl(FIGSHARE_ZIP_URL, self.code, path, force_update, verbose)
        )
        extract_dir = zip_path.parent

        wanted = {f"{code}{ses:03d}" for ses in range(1, _N_SESSIONS + 1)}
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                top = member.split("/")[0]
                if top not in wanted:
                    continue
                if not (extract_dir / member).exists():
                    zf.extract(member, extract_dir)

        return [str(extract_dir)]

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: Raw}}`` for one subject."""
        extract_dir = Path(self.data_path(subject)[0])
        code = _SUBJECT_CODES[subject - 1]

        sessions = {}
        for ses in range(1, _N_SESSIONS + 1):
            folder = extract_dir / f"{code}{ses:03d}"
            if not folder.is_dir():
                continue

            dat_files = sorted(
                folder.glob(f"{code}S{ses:03d}R*.dat"), key=self._run_number
            )
            runs = {}
            for run_idx, dat_path in enumerate(dat_files):
                try:
                    runs[str(run_idx)] = self._load_bci2000_run(dat_path)
                except Exception:  # noqa: BLE001
                    log.warning("Failed to load %s, skipping.", dat_path)

            if runs:
                sessions[str(ses - 1)] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No BCI2000 .dat runs found for subject {subject} "
                f"(code {code}) under {extract_dir}"
            )

        return sessions

    @staticmethod
    def _run_number(dat_path):
        """Sort key: the integer run index from a ``...R<NN>.dat`` filename."""
        match = re.search(r"R(\d+)\.dat$", dat_path.name)
        return int(match.group(1)) if match else 0

    @classmethod
    def _load_bci2000_run(cls, dat_path):
        """Read one BCI2000 ``.dat`` run and return an :class:`mne.io.RawArray`.

        Signals are calibrated to volts using the per-channel gain and offset
        stored in the file header, and a stim channel is built from single-sample
        markers at each ``StimulusCode`` cue onset (1 = motor imagery,
        2 = non-motor reference).
        """
        signals, stim, sfreq = cls._read_bci2000(str(dat_path))

        n_ch = signals.shape[0]
        ch_names = [f"EEG{i + 1}" for i in range(n_ch)] + ["STI"]
        ch_types = ["eeg"] * n_ch + ["stim"]

        data = np.vstack([signals, stim[np.newaxis, :]])
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw

    @staticmethod
    def _read_bci2000(filepath):
        """Parse a BCI2000 ``.dat`` file.

        Returns
        -------
        signals : ndarray, shape (n_channels, n_samples)
            Calibrated signal in volts.
        stim : ndarray, shape (n_samples,)
            Stim vector with the event code (1 or 2) at each cue onset, 0 else.
        sfreq : float
            Sampling frequency in Hz.
        """
        with open(filepath, "rb") as fh:
            raw_bytes = fh.read()

        newline = raw_bytes.find(b"\n")
        first_line = raw_bytes[:newline].decode("ascii", errors="replace")
        parts = first_line.split()
        header_len = int(parts[parts.index("HeaderLen=") + 1])
        source_ch = int(parts[parts.index("SourceCh=") + 1])
        state_vec_len = int(parts[parts.index("StatevectorLen=") + 1])
        data_format = parts[parts.index("DataFormat=") + 1]

        if data_format not in _DTYPE_MAP:
            raise ValueError(f"Unsupported BCI2000 DataFormat: {data_format}")
        sample_dtype = np.dtype(_DTYPE_MAP[data_format])
        sample_bytes = sample_dtype.itemsize

        header = raw_bytes[:header_len].decode("ascii", errors="replace")

        sfreq = float(_param_first_value(header, "SamplingRate", default="500"))
        gain = _param_float_list(header, "SourceChGain", source_ch)
        offset = _param_float_list(header, "SourceChOffset", source_ch)

        # StimulusCode state definition: "Name Length Value BytePos BitPos".
        sc_len, sc_byte, sc_bit = _stimulus_code_def(header)

        block_size = source_ch * sample_bytes + state_vec_len
        body = raw_bytes[header_len:]
        n_samples = len(body) // block_size
        block = np.frombuffer(body[: n_samples * block_size], dtype=np.uint8).reshape(
            n_samples, block_size
        )

        # Signal bytes -> (n_samples, source_ch) of the native sample dtype.
        sig_raw = (
            block[:, : source_ch * sample_bytes]
            .copy()
            .view(sample_dtype)
            .astype(np.float64)
        )
        # Calibrate to microvolts then to volts: (raw - offset) * gain.
        signals = (sig_raw - offset[np.newaxis, :]) * gain[np.newaxis, :] * 1e-6
        signals = signals.T  # (n_channels, n_samples)

        # StimulusCode: assemble the little-endian integer from its state bytes.
        state = block[:, source_ch * sample_bytes :]
        stim_code = _extract_state(state, sc_len, sc_byte, sc_bit)

        stim = np.zeros(n_samples, dtype=np.float64)
        for code in (_CODE_IMAGERY, _CODE_NONMOTOR):
            is_code = stim_code == code
            # Rising edge: sample is `code` while the previous sample was not.
            # Prepending False makes sample 0 an onset when it already holds
            # the code.
            prev = np.concatenate([[False], is_code[:-1]])
            onsets = np.where(is_code & ~prev)[0]
            stim[onsets] = code

        return signals, stim, sfreq


# ---------------------------------------------------------------------- #
# BCI2000 header helpers
# ---------------------------------------------------------------------- #
def _param_first_value(header, name, default=None):
    """Return the first value token of a BCI2000 parameter line."""
    match = re.search(rf"{name}=\s*(\S+)", header)
    return match.group(1) if match else default


def _param_float_list(header, name, count):
    """Return a length-``count`` float array from a BCI2000 list parameter.

    List parameters store ``<count> v1 v2 ... vN`` after the ``=``; the leading
    count token is dropped.
    """
    match = re.search(rf"{name}=\s*([^%\r\n]*)", header)
    if not match:
        return (
            np.zeros(count, dtype=np.float64)
            if name.endswith("Offset")
            else np.ones(count, dtype=np.float64)
        )
    tokens = match.group(1).split()
    values = [float(t) for t in tokens[1 : count + 1]]
    if len(values) != count:
        # Fall back to identity calibration if the header is malformed.
        return np.zeros(count) if name.endswith("Offset") else np.ones(count)
    return np.asarray(values, dtype=np.float64)


def _stimulus_code_def(header):
    """Return (length, byte_offset, start_bit) for the StimulusCode state."""
    in_states = False
    for line in header.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[ State Vector Definition ]"):
            in_states = True
            continue
        if stripped.startswith("[") and in_states:
            break
        if in_states and stripped.startswith("StimulusCode"):
            fields = stripped.split()
            # Name Length Value BytePosition BitPosition
            return int(fields[1]), int(fields[3]), int(fields[4])
    # Sensible default matching the observed files: 16-bit at byte 0, bit 0.
    return 16, 0, 0


def _extract_state(state_bytes, length, byte_offset, start_bit):
    """Extract an unsigned integer state channel from the state-byte matrix.

    Parameters
    ----------
    state_bytes : ndarray, shape (n_samples, state_vec_len), uint8
    length : int
        Number of bits in the state.
    byte_offset : int
        Starting byte of the state within the state vector.
    start_bit : int
        Starting bit (0-7) within the starting byte.

    Returns
    -------
    ndarray, shape (n_samples,), int64
    """
    n_samples = state_bytes.shape[0]
    values = np.zeros(n_samples, dtype=np.int64)
    for bit in range(length):
        abs_bit = start_bit + bit
        byte_idx = byte_offset + abs_bit // 8
        bit_idx = abs_bit % 8
        if byte_idx >= state_bytes.shape[1]:
            break
        bit_col = (state_bytes[:, byte_idx] >> bit_idx) & 1
        values |= bit_col.astype(np.int64) << bit
    return values
