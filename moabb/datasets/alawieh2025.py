"""Alawieh2025 longitudinal motor-imagery BCI dataset (TESS neuromodulation)."""

import warnings
from pathlib import Path

import mne

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.perdikis2018 import _keep_class_cues
from moabb.datasets.utils import download_and_extract_subject_zip


# Zenodo record 15454355 (concept DOI 10.5281/zenodo.15454354)
ALAWIEH2025_BASE = "https://zenodo.org/records/15454355/files/{name}?download=1"

# Per-cohort archives with the 27 participants. The d2 "SlowLearners" archive
# (6-month follow-up of four d1 subjects) is intentionally excluded.
_ARCHIVES = {
    "d1": ("d1_Main_Group_n20.zip", "d1_Main_Group_n20"),
    "d3": ("d3_SinglePulse_n5.zip", "d3_SinglePulse_n5"),
    "d4": ("d4_SCI_patients.zip", "d4_SCI_patients"),
}

# subject index (1..27) -> (archive key, zero-padded id token used in the
# on-disk folder name "Subject_<token>_..._Offline")
_D1_REST = (4, 8, 9, 10, 13, 17, 18, 19, 20, 22)  # REST_n10 cohort
_D1_TESS = (2, 3, 5, 6, 7, 11, 14, 15, 16, 21)  # TESS_n10 cohort
_SUBJECT_MAP = dict(
    enumerate(
        [("d1", f"{n:03d}") for n in _D1_REST + _D1_TESS]
        + [("d3", str(n)) for n in range(501, 506)]  # SinglePulse cohort
        + [("d4", "0001"), ("d4", "0002")],
        start=1,
    )
)

# The 32 EEG electrodes recorded (in file order). M1/M2 are mastoids.
_EEG_CHANNELS = (
    "FP1 FPZ FP2 F7 F3 FZ F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 CZ "
    "C4 T8 M2 CP5 CP1 CP2 CP6 P7 P3 PZ P4 P8 POZ O1 OZ O2"
).split()

# Auxiliary sensor channels present in the GDF header.
_AUX_CHANNELS = ["sens7", "sens8", "sens9"]

# GDF event type codes for the two motor-imagery classes (standard Graz/CNBI
# encoding: 0x0301 = left-hand cue, 0x0302 = right-hand cue).
_CLASS_CODES = {"769": "left_hand", "770": "right_hand"}

# MNE does not interpret the GDF unit metadata, so the payload stays in
# microvolts; the same scale applies to EEG and auxiliary channels.
ALAWIEH2025_EEG_SCALE_TO_VOLTS = 1e-6


class Alawieh2025(BaseDataset):
    """Motor-imagery BCI dataset with transcutaneous spinal stimulation [1]_.

    **Dataset description**

    Longitudinal two-class (left vs right hand) motor-imagery BCI training of 27
    participants (25 able-bodied, 2 with spinal cord injury) studying
    transcutaneous electrical spinal stimulation (TESS) [1]_; 32 EEG + 3
    auxiliary channels at 512 Hz, GDF (CNBI/Graz convention). Zenodo cohorts:
    ``d1_Main_Group_n20`` (REST and TESS groups, n=10 each), ``d3_SinglePulse_n5``
    and ``d4_SCI_patients`` (n=2); the ``d2`` follow-up of four d1 subjects is
    not loaded.

    Only the offline cue-based recordings are exposed, one session per
    participant with one run per GDF; the online closed-loop recordings have no
    discrete class cues. Events are the class cues and the interval is the 4 s
    after the cue. The microvolt payload (unscaled by MNE) is converted to volts,
    and the O2/OZ order swapped in d3/d4 recordings is restored.

    References
    ----------

    .. [1] Alawieh, H., Deland, L., Madera, J., Kumar, S., Racz, F. S.,
       Majewicz Fey, A., & Millán, J. del R. (2025). A Multi-Session EEG Dataset
       of Longitudinal Motor Imagery BCI Training with Transcutaneous Spinal
       Stimulation in Able-Bodied and Spinal Cord Injury Participants. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.15454354

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            channel_types={"eeg": 32, "eog": 3},
            montage="10-20",
            reference="mastoids (M1, M2)",
            sensors=[*_EEG_CHANNELS, *_AUX_CHANNELS],
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=True, eog_channels=3),
        ),
        participants=ParticipantMetadata(
            n_subjects=27,
            health_status="able-bodied and spinal cord injury",
            clinical_population="25 able-bodied, 2 spinal cord injury (SCI)",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            feedback_type="kinesthetic",
            synchronicity="cue-based",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.15454354",
            description=(
                "Longitudinal two-class (left/right hand) motor-imagery BCI "
                "training dataset with transcutaneous electrical spinal "
                "stimulation, in able-bodied and spinal cord injury participants."
            ),
            investigators=[
                "Hussein Alawieh",
                "Liu Deland",
                "Jonathan Madera",
                "Satyam Kumar",
                "Frigyes Samuel Racz",
                "Ann Majewicz Fey",
                "José del R. Millán",
            ],
            country="US",
            data_url="https://doi.org/10.5281/zenodo.15454354",
            publication_year=2025,
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="GDF",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 27 + 1)),
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="Alawieh2025",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.5281/zenodo.15454354",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the sorted paths of the subject's offline GDF recordings."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        archive_key, token = _SUBJECT_MAP[subject]
        zip_name, root_name = _ARCHIVES[archive_key]
        url = ALAWIEH2025_BASE.format(name=zip_name)

        data_dir = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        archive_root = data_dir / root_name
        if force_update or not archive_root.exists():
            download_and_extract_subject_zip(
                url, self.code, data_dir, path, force_update, verbose
            )

        # Resolve the exact subject-level offline folders first. Some d3
        # inner session directories carry the preceding subject's token, so a
        # substring match over the full file path would assign eight runs to
        # two subjects. d4 legitimately has two same-depth roots (REST/TESS).
        prefix = f"Subject_{token}_"
        subject_roots = [
            candidate
            for candidate in archive_root.rglob(f"{prefix}*_Offline")
            if candidate.is_dir()
        ]
        if not subject_roots:
            raise FileNotFoundError(
                f"No offline roots for subject {subject} in {archive_root}"
            )
        subject_paths = sorted(
            {
                str(file_path)
                for subject_root in subject_roots
                for file_path in subject_root.rglob("*.gdf")
            }
        )
        if not subject_paths:
            raise FileNotFoundError(f"No offline GDF recordings for subject {subject}")
        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {run: Raw}}``, one run per offline GDF recording."""
        paths = self.data_path(subject)
        return {"0": {str(i): self._read_run(p) for i, p in enumerate(paths)}}

    @staticmethod
    def _read_run(file_path):
        """Read one GDF recording into a clean, annotated ``mne.io.Raw``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")

        # Drop the trigger channel so events are read from GDF annotations, and
        # drop any channel not part of the recorded EEG/aux set.
        keep = set(_EEG_CHANNELS) | set(_AUX_CHANNELS)
        raw.drop_channels([ch for ch in raw.ch_names if ch not in keep])
        aux_present = [ch for ch in _AUX_CHANNELS if ch in raw.ch_names]
        raw.set_channel_types(dict.fromkeys(aux_present, "eog"))

        missing_eeg = [ch for ch in _EEG_CHANNELS if ch not in raw.ch_names]
        if missing_eeg:
            raise ValueError(
                "Alawieh2025 recording is missing required EEG channels: "
                f"{missing_eeg}; available channels: {raw.ch_names}"
            )
        raw.apply_function(
            lambda data: data * ALAWIEH2025_EEG_SCALE_TO_VOLTS,
            picks=[*_EEG_CHANNELS, *aux_present],
            channel_wise=False,
        )
        # d3/d4 archive recordings swap O2 and OZ in their GDF channel order.
        # The channel set is unchanged, so restore the documented acquisition
        # order before Braindecode caches are combined across cohorts.
        raw.reorder_channels([*_EEG_CHANNELS, *aux_present])

        _keep_class_cues(raw, _CLASS_CODES)

        montage = mne.channels.make_standard_montage("colin27_1005")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = raw.set_montage(
                montage, match_case=False, on_missing="ignore", verbose=False
            )
        return raw
