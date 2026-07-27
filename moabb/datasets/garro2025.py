"""Garro2025 (NeBULA) standardized reaching motor-execution EEG dataset."""

import warnings
import zipfile
from pathlib import Path

from mne.channels import make_standard_montage
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


# Figshare article hosting the BIDS BrainVision dataset.
FIGSHARE_ARTICLE_ID = "27301629"
FIGSHARE_FILE_URL = "https://ndownloader.figshare.com/files/{file_id}"

# Top-level BIDS metadata files (needed by mne-bids to read the recordings).
_ROOT_FILES = ("dataset_description.json", "participants.tsv", "README.txt")
_ROOT_FILE_IDS = {
    "dataset_description.json": 49987098,
    "participants.tsv": 49987440,
    "README.txt": 49987452,
}
_SUBJECT_FILE_IDS = {
    1: 49987455,
    2: 49987458,
    3: 49987461,
    4: 49987464,
    5: 49987467,
    6: 49987473,
    7: 49987476,
    8: 49987482,
    9: 49987485,
    10: 49987488,
    11: 49987491,
    12: 49987494,
    13: 49987497,
    14: 49987500,
    15: 49987503,
    16: 49987506,
    17: 49987509,
    18: 49987512,
    19: 49987515,
    20: 49987518,
    21: 49987521,
    22: 49987536,
    23: 49987539,
    24: 49987545,
    25: 49987554,
    26: 49987560,
    27: 49987563,
    29: 49987575,
    30: 49987578,
    31: 49987581,
    32: 49987584,
    33: 49987587,
    34: 49987599,
    35: 49987605,
    36: 49987608,
    37: 49987611,
    38: 49987614,
    39: 49987617,
    40: 49987632,
}

# 127 recorded EEG channels (actiCHamp 128-cap, FCz used as online reference and
# therefore not stored). Order matches the BrainVision header.
# fmt: off
_CH_NAMES = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
    "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1",
    "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "Cz",
    "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2",
    "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "C1",
    "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz",
    "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6",
    "C2", "FC4", "FT8", "F6", "AF8", "AF4", "F2", "F9",
    "AFF1h", "FFC1h", "FFC5h", "FTT7h", "FCC3h", "CCP1h", "CCP5h", "TPP7h",
    "P9", "PPO9h", "PO9", "O9", "OI1h", "PPO1h", "CPP3h", "CPP4h",
    "PPO2h", "OI2h", "O10", "PO10", "PPO10h", "P10", "TPP8h", "CCP6h",
    "CCP2h", "FCC4h", "FTT8h", "FFC6h", "FFC2h", "AFF2h", "F10", "AFp1",
    "AFF5h", "FFT9h", "FFT7h", "FFC3h", "FCC1h", "FCC5h", "FTT9h", "TTP7h",
    "CCP3h", "CPP1h", "CPP5h", "TPP9h", "POO9h", "PPO5h", "POO1", "POO2",
    "PPO6h", "POO10h", "TPP10h", "CPP6h", "CPP2h", "CCP4h", "TTP8h", "FTT10h",
    "FCC6h", "FCC2h", "FFC4h", "FFT8h", "FFT10h", "AFF6h", "AFp2",
]
# fmt: on

# The 3 reaching movement types (targets) are the decoding classes. The
# eeg/ folder ships no events.tsv (only sub-01/emg/ has one), so
# read_raw_bids does not rebuild annotations from events.tsv -- it keeps the
# BrainVision .vmrk marker descriptions as-is: "Stimulus/R 1" / "Stimulus/R 2"
# / "Stimulus/R 3" (with the "Stimulus/" prefix and an internal space).
_EVENTS = {"reach_1": 1, "reach_2": 2, "reach_3": 3}
_ANNOT_RENAME = {
    "Stimulus/R 1": "reach_1",
    "Stimulus/R 2": "reach_2",
    "Stimulus/R 3": "reach_3",
}

# The three assistance-level recordings become runs (in acquisition order).
# The released BIDS task entity labels are lowercase
# (get_entity_vals(root, "task") == ["free", "high", "low"]), unlike Table 1
# of the data descriptor.
_TASK_TO_RUN = {"free": "0free", "low": "1low", "high": "2high"}

# Subjects with a demographics entry but no released recording (39 usable of 40).
_MISSING_SUBJECTS = (28,)


class Garro2025(BaseDataset):
    """Standardized reaching motor-execution EEG dataset (NeBULA) [1]_.

    **Dataset description**

    The NeBULA (Neuromechanical Biomarkers for Upper Limb Assessment) dataset
    contains high-density EEG (and synchronized surface EMG) recorded while
    participants performed a standardized upper-limb reaching task with the
    right arm. Seated participants reached to one of three illuminated targets
    when a target light turned on, at a comfortable pace, and returned the hand
    to a resting position on the right leg.

    Each participant performed the task under three assistance levels, provided
    here as three runs:

    - ``free``: no robot, free reaching movement.
    - ``low``: movement assisted by an exoskeleton at assistance level 1.
    - ``high``: movement assisted by an exoskeleton at assistance level 2.

    Within every run, trials cover three reaching movement types (three target
    positions), which define the three decoding classes. The BrainVision markers
    encode a trial as ``StartTrial`` -> ``G n`` (go cue / target illumination)
    -> ``R n`` (reach onset) -> ``EndTrial``, where ``n`` in {1, 2, 3} is the
    movement type. Epoching is locked to the reach onset ``R n``.

    EEG was acquired with a 128-channel Brain Products actiCHamp system at
    1000 Hz using FCz as the online reference (127 channels are stored). Surface
    EMG from 11 selected upper-limb muscles (Cometa Wave Plus) is available as a
    non-EEG modality and is not returned by the imagery paradigm.

    40 participants have demographic entries; subject 28 has no released
    recording, so 39 subjects provide usable data.

    References
    ----------
    .. [1] Garro, F., Fenoglio, E., Ceroni, I., Forsiuk, I., Canepa, M.,
           Mozzon, M., Bruschi, A., Zippo, F., Laffranchi, M., De Michieli, L.,
           Buccelli, S., Chiappalone, M., & Semprini, M. (2025). An EEG-EMG
           dataset from a standardized reaching task for biomarker research in
           upper limb assessment. Scientific Data.
           DOI: https://doi.org/10.1038/s41597-025-05042-4

    Notes
    -----
    .. versionadded:: 1.2.1
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=127,
            channel_types={"eeg": 127},
            montage="standard_1005",
            hardware="Brain Products actiCHamp (128-channel actiCAP)",
            cap_manufacturer="Brain Products",
            sensor_type="active electrodes",
            reference="FCz",
            line_freq=50.0,
            sensors=list(_CH_NAMES),
            auxiliary_channels=AuxiliaryChannelsMetadata(has_emg=True, emg_channels=11),
        ),
        participants=ParticipantMetadata(
            n_subjects=39,
            health_status="healthy",
            gender={"female": 19, "male": 21},
            age_min=25.0,
            age_max=71.0,
            handedness="right",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=3,
            class_labels=list(_EVENTS.keys()),
            study_design=(
                "Standardized right-arm reaching to one of three illuminated "
                "targets (three movement types), performed under three "
                "assistance levels (free, low, high)."
            ),
            feedback_type="none",
            stimulus_type="target light",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
            instructions=(
                "Seated participant reached a target when a light turned on with "
                "the right arm, at a normal pace, then returned to resting "
                "position with the hand on the right leg."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-05042-4",
            description=(
                "NeBULA: high-density EEG and surface EMG recorded during a "
                "standardized upper-limb reaching task under three robotic "
                "assistance levels, for neuromechanical biomarker research."
            ),
            investigators=[
                "Federica Garro",
                "Elisa Fenoglio",
                "Ilaria Ceroni",
                "Iryna Forsiuk",
                "Michela Canepa",
                "Marta Mozzon",
                "Alessandro Bruschi",
                "Fabio Zippo",
                "Matteo Laffranchi",
                "Lorenzo De Michieli",
                "Stefano Buccelli",
                "Michela Chiappalone",
                "Marianna Semprini",
            ],
            institution="Istituto Italiano di Tecnologia",
            institution_address="Via Morego 30, Genova, Italy",
            institution_department="Rehab Technologies",
            country="IT",
            repository="Figshare",
            data_url="https://doi.org/10.1038/s41597-025-05042-4",
            publication_year=2025,
            funding=[
                "Istituto Nazionale Assicurazione Infortuni sul Lavoro (INAIL), "
                "project grant PR19-RR-P2"
            ],
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=3,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=list(_EVENTS.keys())
        ),
        data_structure=DataStructureMetadata(
            n_trials=30,
            trials_context=(
                "~10 trials per movement type per assistance-level run "
                "(3 classes x 3 runs per subject)."
            ),
        ),
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        all_subjects = [s for s in range(1, 41) if s not in _MISSING_SUBJECTS]
        super().__init__(
            subjects=all_subjects,
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Garro2025",
            interval=[0, 2],
            paradigm="imagery",
            doi="10.1038/s41597-025-05042-4",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _download_root(self, subject, path=None, force_update=False):
        """Download and extract one subject's BIDS folder, return the BIDS root."""
        root = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        root.mkdir(parents=True, exist_ok=True)

        def local_or_download(file_id):
            cached = root / "files" / str(file_id)
            if cached.is_file() and not force_update:
                return cached
            url = FIGSHARE_FILE_URL.format(file_id=file_id)
            return Path(dl.data_dl(url, self.code, path=path, force_update=force_update))

        # Ensure the top-level BIDS metadata files are present.
        for fname in _ROOT_FILES:
            target = root / fname
            if force_update or not target.exists():
                downloaded = local_or_download(_ROOT_FILE_IDS[fname])
                target.write_bytes(downloaded.read_bytes())

        # Download and extract this subject's zip if not already extracted.
        subj_dir = root / f"sub-{subject:02d}"
        if force_update or not any(subj_dir.rglob("*.vhdr")):
            try:
                file_id = _SUBJECT_FILE_IDS[subject]
            except KeyError:
                raise ValueError(f"No data released for subject {subject}") from None
            zip_path = local_or_download(file_id)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(root)

        return root

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the BIDS paths (one per assistance-level run) for one subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn't exist, the "~/mne_data" directory is used. If the
            dataset is not found under the given path, the data will be
            automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for signature compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list of :class:`mne_bids.BIDSPath` objects, one per run.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        root = self._download_root(subject, path=path, force_update=force_update)

        tasks = get_entity_vals(root, "task")
        # Keep a deterministic run order: free, low, high.
        ordered = [t for t in ("free", "low", "high") if t in tasks]

        bids_paths = []
        for task in ordered:
            bids_paths.append(
                BIDSPath(
                    subject=f"{subject:02d}",
                    task=task,
                    suffix="eeg",
                    datatype="eeg",
                    extension=".vhdr",
                    root=root,
                    check=True,
                )
            )
        return bids_paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: raw}}."""
        bids_paths = self.data_path(subject)
        montage = make_standard_montage("standard_1005")

        runs = {}
        for bids_path in bids_paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = read_raw_bids(bids_path=bids_path, verbose=False)
                raw.load_data(verbose=False)

                # Reach markers carry the class label; keep only the ones present.
                present = set(raw.annotations.description)
                rename = {k: v for k, v in _ANNOT_RENAME.items() if k in present}
                if rename:
                    raw.annotations.rename(rename)

                raw.set_montage(montage, on_missing="ignore", verbose=False)

            run_key = _TASK_TO_RUN.get(bids_path.task, f"0{bids_path.task}")
            runs[run_key] = raw

        return {"0": runs}
