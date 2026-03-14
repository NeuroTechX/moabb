"""P300 speller dataset with language model comparison.

Speier, Deshpande, and Pouratian (2017), PLoS ONE.
DOI: 10.1371/journal.pone.0175382
Data DOI: 10.7910/DVN/PHHHB6
"""

import logging
import struct
from pathlib import Path

import mne
import numpy as np

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_DOI = "10.1371/journal.pone.0175382"
_SIGN = "speier2017"

# fmt: off
_CH_NAMES = [
    "Fz", "FC1", "FCz", "FC2", "FC4", "FC6", "C4", "C6",
    "CP4", "CP6", "FC3", "FC5", "C3", "C5", "CP3", "CP5",
    "CP1", "P1", "Cz", "CPz", "Pz", "POz", "CP2", "P2",
    "PO7", "PO3", "O1", "Oz", "Iz", "O2", "PO4", "PO8",
]
# fmt: on

# Harvard Dataverse file IDs for each .dat file.
# Format: Subject_{N}_{condition}{run}.dat
# Conditions: FF (Famous Faces), Inv (Inverting)
# Runs: 1, 2, 3 (training, copy-mode), Online (free-mode, no labels)
_FILE_IDS = {
    "Subject_1_FF1.dat": 2863666,
    "Subject_1_FF2.dat": 2863665,
    "Subject_1_FF3.dat": 2863667,
    "Subject_1_FFOnline.dat": 2863668,
    "Subject_1_Inv1.dat": 2863133,
    "Subject_1_Inv2.dat": 2863134,
    "Subject_1_Inv3.dat": 2863135,
    "Subject_1_InvOnline.dat": 2863136,
    "Subject_2_FF1.dat": 2863671,
    "Subject_2_FF2.dat": 2863669,
    "Subject_2_FF3.dat": 2863670,
    "Subject_2_FFOnline.dat": 2863672,
    "Subject_2_Inv1.dat": 2863675,
    "Subject_2_Inv2.dat": 2863673,
    "Subject_2_Inv3.dat": 2863674,
    "Subject_2_InvOnline.dat": 2863676,
    "Subject_3_FF1.dat": 2863677,
    "Subject_3_FF2.dat": 2863679,
    "Subject_3_FF3.dat": 2863678,
    "Subject_3_FFOnline.dat": 2863680,
    "Subject_3_Inv1.dat": 2863683,
    "Subject_3_Inv2.dat": 2863681,
    "Subject_3_Inv3.dat": 2863682,
    "Subject_3_InvOnline.dat": 2863684,
    "Subject_4_FF1.dat": 2863690,
    "Subject_4_FF2.dat": 2863689,
    "Subject_4_FF3.dat": 2863691,
    "Subject_4_FFOnline.dat": 2863692,
    "Subject_4_Inv1.dat": 2863685,
    "Subject_4_Inv2.dat": 2863686,
    "Subject_4_Inv3.dat": 2863687,
    "Subject_4_InvOnline.dat": 2863688,
    "Subject_5_FF1.dat": 2863697,
    "Subject_5_FF2.dat": 2863698,
    "Subject_5_FF3.dat": 2863699,
    "Subject_5_FFOnline.dat": 2863700,
    "Subject_5_Inv1.dat": 2863695,
    "Subject_5_Inv2.dat": 2863693,
    "Subject_5_Inv3.dat": 2863694,
    "Subject_5_InvOnline.dat": 2863696,
    "Subject_6_FF1.dat": 2864035,
    "Subject_6_FF2.dat": 2864034,
    "Subject_6_FF3.dat": 2864036,
    "Subject_6_FFOnline.dat": 2864037,
    "Subject_6_Inv1.dat": 2863702,
    "Subject_6_Inv2.dat": 2863703,
    "Subject_6_Inv3.dat": 2863701,
    "Subject_6_InvOnline.dat": 2864033,
    "Subject_7_FF1.dat": 2864042,
    "Subject_7_FF2.dat": 2864043,
    "Subject_7_FF3.dat": 2864044,
    "Subject_7_FFOnline.dat": 2864046,
    "Subject_7_Inv1.dat": 2864039,
    "Subject_7_Inv2.dat": 2864040,
    "Subject_7_Inv3.dat": 2864038,
    "Subject_7_InvOnline.dat": 2864041,
    "Subject_8_FF1.dat": 2864051,
    "Subject_8_FF2.dat": 2864053,
    "Subject_8_FF3.dat": 2864052,
    "Subject_8_FFOnline.dat": 2864054,
    "Subject_8_Inv1.dat": 2864047,
    "Subject_8_Inv2.dat": 2864048,
    "Subject_8_Inv3.dat": 2864049,
    "Subject_8_InvOnline.dat": 2864050,
    "Subject_9_FF1.dat": 2864059,
    "Subject_9_FF2.dat": 2864061,
    "Subject_9_FF3.dat": 2864060,
    "Subject_9_FFOnline.dat": 2864062,
    "Subject_9_Inv1.dat": 2864055,
    "Subject_9_Inv2.dat": 2864057,
    "Subject_9_Inv3.dat": 2864056,
    "Subject_9_InvOnline.dat": 2864058,
    "Subject_10_FF1.dat": 2864063,
    "Subject_10_FF2.dat": 2864064,
    "Subject_10_FF3.dat": 2864065,
    "Subject_10_FFOnline.dat": 2864066,
    "Subject_10_Inv1.dat": 2864067,
    "Subject_10_Inv2.dat": 2864068,
    "Subject_10_Inv3.dat": 2864069,
    "Subject_10_InvOnline.dat": 2864070,
}


def _read_bci2000(filepath):
    """Minimal BCI2000 .dat file reader.

    Returns (signals, states) where signals is (n_channels, n_samples)
    float64 array in microvolts, and states is a dict of state vectors.
    """
    with open(filepath, "rb") as fh:
        # Read ASCII header.
        header_line = b""
        while True:
            byte = fh.read(1)
            if byte == b"\n" or byte == b"\r":
                break
            header_line += byte

        header_text = header_line.decode("ascii", errors="replace")
        parts = header_text.split()

        # Parse key header fields.
        header_len = int(parts[parts.index("HeaderLen=") + 1])
        source_ch = int(parts[parts.index("SourceCh=") + 1])
        state_vec_len = int(parts[parts.index("StatevectorLen=") + 1])
        data_format = parts[parts.index("DataFormat=") + 1]

        if data_format != "float32":
            raise ValueError(f"Unsupported BCI2000 DataFormat: {data_format}")

        # Read full header for state definitions.
        fh.seek(0)
        full_header = fh.read(header_len).decode("ascii", errors="replace")

        # Parse state definitions.
        state_defs = {}
        in_states = False
        for line in full_header.split("\n"):
            line = line.strip()
            if line.startswith("[ State Vector Definition ]"):
                in_states = True
                continue
            if line.startswith("[") and in_states:
                break
            if in_states and line:
                # Format: Name Length DefaultValue BytePosition BitPosition
                sparts = line.split()
                if len(sparts) >= 5:
                    name = sparts[0]
                    length = int(sparts[1])
                    byte_offset = int(sparts[3])
                    start_bit = int(sparts[4])
                    state_defs[name] = (length, start_bit, byte_offset)

        # Read binary data.
        fh.seek(header_len)
        raw_data = fh.read()

        # Each sample block: source_ch * 4 bytes (float32) + state_vec_len bytes.
        block_size = source_ch * 4 + state_vec_len
        n_samples = len(raw_data) // block_size

        signals = np.zeros((source_ch, n_samples), dtype=np.float64)
        state_bytes = np.zeros((n_samples, state_vec_len), dtype=np.uint8)

        for i in range(n_samples):
            offset = i * block_size
            # Read float32 signal values.
            for ch in range(source_ch):
                ch_offset = offset + ch * 4
                signals[ch, i] = struct.unpack_from("<f", raw_data, ch_offset)[0]
            # Read state vector.
            sv_offset = offset + source_ch * 4
            state_bytes[i] = np.frombuffer(
                raw_data, dtype=np.uint8, count=state_vec_len, offset=sv_offset
            )

        # Extract state variables.
        states = {}
        for name, (length, start_bit, byte_offset) in state_defs.items():
            if length > 16:
                continue  # Skip large states to avoid overflow.
            vals = np.zeros(n_samples, dtype=np.int32)
            for i in range(n_samples):
                # Extract bits from state vector.
                val = 0
                for b in range(length):
                    bit_pos = start_bit + b
                    byte_idx = byte_offset + bit_pos // 8
                    bit_idx = bit_pos % 8
                    if byte_idx < state_vec_len:
                        val |= ((int(state_bytes[i, byte_idx]) >> bit_idx) & 1) << b
                vals[i] = val
            states[name] = vals

    return signals, states


class Speier2017(BaseDataset):
    """P300 speller dataset from Speier et al 2017.

    Dataset from the paper [1]_.

    **Dataset Description**

    Ten subjects performed a P300 row-column speller task under two
    stimulus conditions: Famous Faces (FF, Einstein image overlay)
    and Inverting (Inv, color inversion). EEG was recorded at 256 Hz
    from 32 channels using g.tec amplifiers with BCI2000 software.

    Each condition has 3 training runs (copy-mode, 10 characters
    each) and 1 online run (free-mode, no ground truth). Only
    training runs are used by default (StimulusType labels available).

    Events: Target (flashed row/column contains target) = 2,
    NonTarget = 1.

    Parameters
    ----------
    include_online : bool
        If True, include online (free-mode) runs where
        StimulusType is always 0 (no ground truth). Default False.

    References
    ----------
    .. [1] Speier, W., Deshpande, A., & Pouratian, N. (2017). A
           comparison of stimulus types in online classification of
           the P300 speller using language models. PLoS ONE, 12(4),
           e0175382.
           https://doi.org/10.1371/journal.pone.0175382
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1005",
            hardware="g.tec amplifier",
            reference="left ear",
            ground="AFz",
            sensors=list(_CH_NAMES),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            age_min=20,
            age_max=35,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=1.0,
            study_design=(
                "P300 row-column speller; 2 stimulus conditions "
                "(Famous Faces, Inverting); 6x6 character matrix"
            ),
            feedback_type="visual",
            stimulus_type="flash / famous face overlay",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "William Speier",
                "Corey Arnold",
                "Aniket Deshpande",
                "Nader Pouratian",
            ],
            institution="University of California, Los Angeles",
            country="US",
            publication_year=2017,
            data_url="https://dataverse.harvard.edu/dataset.xhtml"
            "?persistentId=doi:10.7910/DVN/PHHHB6",
            license="CC0",
        ),
        sessions_per_subject=2,
        runs_per_session=3,
        tags=Tags(
            pathology=["Healthy"],
            modality=["ERP"],
            type=["P300"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            soa_ms=125.0,
            isi_ms=25.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="~1200 flashes per training run (10 chars x 10 seq x 12)",
            trials_context="per_run",
        ),
        data_processed=False,
        file_format="BCI2000",
    )

    def __init__(self, include_online=False, subjects=None, sessions=None):
        self._include_online = include_online

        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=2,
            events={"Target": 2, "NonTarget": 1},
            code="Speier2017",
            interval=[0, 0.8],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        self.data_path(subject)
        base = self._subject_dir(subject)

        sessions = {}
        conditions = [("FF", "0"), ("Inv", "1")]

        for cond, ses_key in conditions:
            runs = {}

            # Training runs (1, 2, 3).
            for run_idx in range(1, 4):
                fname = f"Subject_{subject}_{cond}{run_idx}.dat"
                fpath = base / fname
                if not fpath.exists():
                    continue
                try:
                    raw = self._load_bci2000(str(fpath))
                    if raw is not None:
                        runs[str(len(runs))] = raw
                except Exception:
                    log.warning("Failed to load %s, skipping.", fpath)

            # Online run (optional).
            if self._include_online:
                fname = f"Subject_{subject}_{cond}Online.dat"
                fpath = base / fname
                if fpath.exists():
                    try:
                        raw = self._load_bci2000(str(fpath), online=True)
                        if raw is not None:
                            runs[str(len(runs))] = raw
                    except Exception:
                        log.warning("Failed to load %s, skipping.", fpath)

            if runs:
                sessions[ses_key] = runs

        return sessions

    @staticmethod
    def _load_bci2000(filepath, online=False):
        """Load a BCI2000 .dat file and return Raw."""
        signals, states = _read_bci2000(filepath)

        # Scale uV -> V.
        signals = signals.astype(np.float64) * 1e-6

        n_ch, n_samples = signals.shape
        sfreq = 256.0

        # Build stim channel from StimulusBegin rising edges + StimulusType.
        # StimulusBegin marks the actual stimulus display onset (0→1 transition).
        # StimulusType indicates target (1) vs non-target (0).
        # NOTE: The "Res" variants (StimulusCodeRes, StimulusTypeRes) are delayed
        # result states (~208 samples / 812ms after the actual stimulus) and must
        # NOT be used for event timing.
        stim_begin = states.get("StimulusBegin", np.zeros(n_samples))
        stim_type = states.get("StimulusType", np.zeros(n_samples))

        stim = np.zeros(n_samples)

        if not online:
            # Detect flash onsets: StimulusBegin transitions from 0 to 1.
            prev = np.concatenate([[0], stim_begin[:-1]])
            onset_idx = np.where((stim_begin == 1) & (prev == 0))[0]

            for idx in onset_idx:
                if stim_type[idx] == 1:
                    stim[idx] = 2  # Target
                else:
                    stim[idx] = 1  # NonTarget

        ch_names = list(_CH_NAMES) + ["STI"]
        ch_types = ["eeg"] * n_ch + ["stim"]
        all_data = np.vstack([signals, stim[np.newaxis]])

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(all_data, info, verbose=False)
        raw.set_montage("standard_1005", on_missing="warn")

        return raw

    def _subject_dir(self, subject):
        path = dl.get_dataset_path(_SIGN, None)
        return Path(path) / f"MNE-{_SIGN}-data"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = self._subject_dir(subject)
        base.mkdir(parents=True, exist_ok=True)

        conditions = ["FF", "Inv"]
        suffixes = ["1", "2", "3"]
        if self._include_online:
            suffixes.append("Online")

        for cond in conditions:
            for suf in suffixes:
                fname = f"Subject_{subject}_{cond}{suf}.dat"
                local = base / fname
                if local.exists() and not force_update:
                    continue
                file_id = _FILE_IDS.get(fname)
                if file_id is None:
                    continue
                url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
                log.info("Downloading %s ...", fname)
                downloaded = dl.data_dl(url, _SIGN)
                downloaded = Path(downloaded)
                if downloaded != local:
                    downloaded.rename(local)

        return str(base)
