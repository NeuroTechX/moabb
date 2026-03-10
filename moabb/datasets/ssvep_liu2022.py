"""eldBETA: A Large Eldercare-oriented Benchmark Database of SSVEP-BCI.

Liu et al. (2022), Scientific Data.
DOI: 10.1038/s41597-022-01372-9
"""

import struct
import tarfile
import tempfile
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
    Tags,
)
from .utils import FIGSHARE_DL_URL, safe_extract_tar


# Figshare file IDs for per-subject tar.gz files (S1.tar.gz through S100.tar.gz)
# fmt: off
_FIGSHARE_FILE_IDS = {
    1: 34516952, 2: 34516955, 3: 34516958, 4: 34516961, 5: 34516964,
    6: 34516967, 7: 34516970, 8: 34516973, 9: 34516976, 10: 34516979,
    11: 34516982, 12: 34516985, 13: 34516988, 14: 34516994, 15: 34517015,
    16: 34517018, 17: 34517024, 18: 34517027, 19: 34517030, 20: 34517033,
    21: 34517036, 22: 34517039, 23: 34519832, 24: 34517045, 25: 34517048,
    26: 34517051, 27: 34517054, 28: 34519643, 29: 34517060, 30: 34517063,
    31: 34517066, 32: 34517069, 33: 34517072, 34: 34517075, 35: 34517078,
    36: 34517081, 37: 34517084, 38: 34517087, 39: 34517090, 40: 34517093,
    41: 34517096, 42: 34517099, 43: 34517102, 44: 34517105, 45: 34517108,
    46: 34517111, 47: 34517114, 48: 34517117, 49: 34517120, 50: 34517123,
    51: 34517126, 52: 34517129, 53: 34517132, 54: 34517135, 55: 34517138,
    56: 34517141, 57: 34517144, 58: 34517147, 59: 34517150, 60: 34517153,
    61: 34517156, 62: 34517159, 63: 34517162, 64: 34517165, 65: 34517168,
    66: 34517171, 67: 34517174, 68: 34517177, 69: 34517180, 70: 34517183,
    71: 34517186, 72: 34517192, 73: 34517195, 74: 34517198, 75: 34517201,
    76: 34517204, 77: 34517207, 78: 34517210, 79: 34517213, 80: 34517216,
    81: 34517219, 82: 34517222, 83: 34517225, 84: 34517228, 85: 34517231,
    86: 34517234, 87: 34517237, 88: 34517240, 89: 34517243, 90: 34517246,
    91: 34517249, 92: 34517252, 93: 34517255, 94: 34517258, 95: 34517261,
    96: 34517264, 97: 34517267, 98: 34517270, 99: 34517273, 100: 34517276,
}
# fmt: on

# GDF event annotations use target indices "1"-"9" (strings, as MNE annotations
# are always str). Map to frequency strings following the JFPM column-major order
# of the 3x3 stimulus grid.
_TARGET_TO_FREQ = {
    "1": "8",
    "2": "9.5",
    "3": "11",
    "4": "8.5",
    "5": "10",
    "6": "11.5",
    "7": "9",
    "8": "10.5",
    "9": "12",
}


def _read_patched_gdf(gdf_path):
    """Read a GDF file exported by biosig4octave, patching a known header bug.

    The eldBETA GDF 2.11 files contain an extra 256-byte metadata block
    (from ``BioSig/EEGLAB writeeeg.m``) that makes the declared header
    size (``header_nblocks``) one block larger than expected by MNE's GDF
    parser (which expects ``1 + nchan`` blocks). This function creates a
    temporary patched copy with the extra block removed and reads it with
    ``mne.io.read_raw_gdf``.
    """
    with open(gdf_path, "rb") as f:
        data = bytearray(f.read())

    # GDF 2.x fixed header: header_nblocks at byte 184 (uint16), nchan at byte 252
    nblocks = struct.unpack_from("<H", data, 184)[0]
    nchan = struct.unpack_from("<H", data, 252)[0]
    expected_nblocks = 1 + nchan

    if nblocks != expected_nblocks:
        # Patch header_nblocks to the expected value
        struct.pack_into("<H", data, 184, expected_nblocks)
        # Remove extra blocks between the variable header and data
        expected_end = expected_nblocks * 256
        actual_end = nblocks * 256
        data = bytes(data[:expected_end]) + bytes(data[actual_end:])

    tmp = tempfile.NamedTemporaryFile(suffix=".gdf", delete=False)
    tmp_path = tmp.name
    try:
        tmp.write(data)
        tmp.close()
        raw = mne.io.read_raw_gdf(tmp_path, preload=True, verbose=False)

        # GDF data records are 1-second blocks; the last block is
        # NaN-padded if the recording doesn't fill it completely.
        # Crop trailing NaN now, while the temp file still exists.
        first_chan = raw.get_data(picks=[0])
        if np.isnan(first_chan[0, -1]):
            last_valid = np.where(~np.isnan(first_chan[0]))[0][-1]
            raw.crop(tmax=raw.times[last_valid])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return raw


class Liu2022EldBETA(BaseDataset):
    """eldBETA SSVEP benchmark dataset for elderly population.

    Dataset from [1]_.

    The eldBETA database contains 64-channel EEG recordings from 100 elderly
    participants (33 males, 67 females, aged 51-81, mean age 63.17) performing
    a 9-target SSVEP-BCI task. Stimuli used joint frequency and phase
    modulation (JFPM) with 9 targets in a 3x3 matrix. Frequencies ranged
    from 8.0 to 12.0 Hz (0.5 Hz step).

    Each subject completed 7 blocks of 9 trials. Each trial consisted of a
    4 s target cue followed by 5 s of SSVEP stimulation and 1 s rest (10 s
    total per trial). EEG was recorded at 1000 Hz with a Synamps2 system
    (Neuroscan) using 64 channels.

    Data is loaded from the BIDS-formatted GDF files included in each
    subject's Figshare archive. The GDF files contain continuous recordings
    at 1000 Hz with event annotations marking each stimulus onset.

    Warnings
    --------
    The GDF files in the archive are mislabeled with ``.edf`` extension
    and contain an extra header block from the biosig4octave exporter.
    This adapter patches the header on-the-fly before reading.

    Like Wang2016 and BETA, this dataset uses the same 64-channel Tsinghua
    Neuroscan cap layout including 'CB1' and 'CB2' channels.

    References
    ----------
    .. [1] B. Liu, Y. Wang, X. Gao, and X. Chen, "eldBETA: A Large
       Eldercare-oriented Benchmark Database of SSVEP-BCI for the Aging
       Population," Scientific Data, vol. 9, p. 252, 2022.
       DOI: 10.1038/s41597-022-01372-9
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="standard_1005",
            hardware="Synamps2 (Neuroscan)",
            line_freq=50.0,
            reference="Cz",
            impedance_threshold_kohm=20,
        ),
        participants=ParticipantMetadata(
            n_subjects=100,
            health_status="healthy",
            gender={"male": 33, "female": 67},
            age_mean=63.17,
            age_min=51,
            age_max=81,
            age_std=6.05,
            # Per-subject demographics from BIDS participants.tsv
            ages=[
                58,
                51,
                58,
                54,
                57,
                51,
                54,
                59,
                57,
                60,
                56,
                51,
                61,
                56,
                60,
                75,
                55,
                68,
                65,
                57,
                56,
                62,
                68,
                81,
                63,
                61,
                75,
                58,
                71,
                71,
                61,
                68,
                69,
                51,
                67,
                62,
                65,
                57,
                67,
                59,
                63,
                71,
                60,
                69,
                70,
                70,
                70,
                68,
                66,
                58,
                61,
                70,
                59,
                54,
                69,
                68,
                68,
                68,
                65,
                62,
                54,
                59,
                62,
                67,
                71,
                66,
                71,
                64,
                61,
                63,
                70,
                58,
                69,
                68,
                67,
                65,
                66,
                62,
                74,
                61,
                58,
                57,
                61,
                58,
                57,
                63,
                71,
                60,
                57,
                65,
                66,
                60,
                61,
                59,
                67,
                69,
                66,
                66,
                67,
                67,
            ],
            sexes=[
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "female",
                "female",
                "female",
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "female",
                "male",
                "female",
                "female",
                "male",
                "female",
                "female",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "female",
                "male",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "male",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "male",
                "female",
                "female",
                "male",
                "female",
                "female",
                "female",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "female",
                "female",
                "female",
                "female",
                "female",
            ],
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=9,
            trial_duration=5.0,
            stimulus_type="JFPM visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            task_type="9-target SSVEP speller",
            feedback_type="visual",
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-022-01372-9",
            investigators=[
                "Bingchuan Liu",
                "Yijun Wang",
                "Xiaorong Gao",
                "Xiaogang Chen",
            ],
            senior_author="Xiaogang Chen",
            institution="Tsinghua University",
            country="CN",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.18032669",
            license="CC BY 4.0",
            publication_year=2022,
            institution_department="Department of Biomedical Engineering, School of Medicine, Tsinghua University",
            ethics_approval=[
                "Institutional Review Board of Tsinghua University, No. 20210032"
            ],
            funding=[
                "National Natural Science Foundation of China (No. 62171473)",
                "Doctoral Brain+X Seed Grant Program of Tsinghua University",
                "Strategic Priority Research Program of Chinese Academy of Sciences (No. XDB32040200)",
            ],
            keywords=["SSVEP", "BCI", "EEG", "elderly", "aging", "benchmark", "JFPM"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[
                8.0,
                8.5,
                9.0,
                9.5,
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
            ],
            frequency_resolution_hz=0.5,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=7,
            n_trials=63,
        ),
        bci_application=BCIApplicationMetadata(
            environment="lab",
            online_feedback=True,
            applications=["speller"],
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        sessions_per_subject=7,
        file_format="GDF (BIDS)",
    )

    # Derived from _TARGET_TO_FREQ: frequency string -> target index
    _events = {freq: int(idx) for idx, freq in _TARGET_TO_FREQ.items()}

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 101)),
            sessions_per_subject=7,
            events=self._events,
            code="Liu2022EldBETA",
            interval=[0, 6.0],
            paradigm="ssvep",
            doi="10.1038/s41597-022-01372-9",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across all 7 blocks from BIDS/GDF files."""
        gdf_paths = self.data_path(subject)

        montage = mne.channels.make_standard_montage("standard_1005")

        sessions = {}
        for block_idx, gdf_path in enumerate(gdf_paths):
            raw = _read_patched_gdf(gdf_path)

            # Rename annotations from target indices ("1"-"9") to
            # frequency strings ("8", "9.5", ...) matching _events
            raw.annotations.rename(_TARGET_TO_FREQ)
            raw.set_montage(montage, on_missing="ignore")

            sessions[str(block_idx)] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return list of 7 GDF file paths (one per session/block).

        Downloads and extracts the subject's Figshare tar.gz archive if needed.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        extract_dir = data_dir / "eldBETA"

        sub_label = f"sub-{subject:03d}"
        gdf_paths = []
        for ses in range(1, self.n_sessions + 1):
            ses_label = f"ses-{ses:02d}"
            gdf_file = (
                extract_dir
                / sub_label
                / ses_label
                / "eeg"
                / f"{sub_label}_{ses_label}_task-ssvep_eeg.edf"
            )
            gdf_paths.append(gdf_file)

        if all(p.exists() for p in gdf_paths) and not force_update:
            return [str(p) for p in gdf_paths]

        # Download and extract the subject archive
        file_id = _FIGSHARE_FILE_IDS[subject]
        url = f"{FIGSHARE_DL_URL}{file_id}"
        tar_path = dl.data_dl(url, sign, path, force_update, verbose)

        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            safe_extract_tar(tf, extract_dir)

        # Verify files exist after extraction
        missing = [p for p in gdf_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Could not find {len(missing)} GDF files for subject {subject} "
                f"after extraction. First missing: {missing[0]}"
            )

        return [str(p) for p in gdf_paths]
