"""BCI Competition 2020 Track 3 - Imagined Speech dataset.

Lee et al., 2020 International BCI Competition.
Data: https://osf.io/pq7vb/
"""

import logging
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


log = logging.getLogger(__name__)

_SIGN = "bcicomp2020is"
_SFREQ = 256.0

# fmt: off
_CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
    "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2",
    "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz",
    "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1", "C2", "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7",
    "PO3", "POz", "PO4", "PO8",
]
# fmt: on

_CLASS_NAMES = ["Hello", "Helpme", "Stop", "Thankyou", "Yes"]

# For download, we use the OSF storage API.
_OSF_API = "https://api.osf.io/v2/nodes/pq7vb/files/osfstorage/"


class BCIComp2020IS(BaseDataset):
    """BCI Competition 2020 Track 3 - Imagined Speech Classification.

    Dataset from the 2020 International BCI Competition [1]_.

    **Dataset Description**

    Fifteen subjects (aged 20-30) performed imagined speech of five
    phrases: "Hello", "Help me", "Stop", "Thank you", "Yes". EEG was
    recorded at 1000 Hz using 64 channels in a 10-20 configuration with
    a BrainAmp amplifier (BrainProducts GmbH), FCz reference, Fpz ground.
    Data is stored at the native epoch sampling rate of 256 Hz.

    Each trial begins with an auditory cue (one of the five words),
    followed by 4 repetitions of: fixation cross (0.8-1.2 s jittered)
    then 2 s imagined speech. A 3 s relaxation phase separates blocks.
    Epochs span -500 ms to 2600 ms relative to cue onset.

    Each subject has 300 training trials (60 per class) and 50
    validation trials (10 per class). Test trials (50 per subject)
    have no labels (competition holdout) and are not loaded.
    Best competition result was 82.6% accuracy.

    References
    ----------
    .. [1] Jeong, J.-H. et al. (2022). 2020 International brain-computer
           interface competition: A review. Frontiers in Human Neuroscience,
           16, 898300. https://doi.org/10.3389/fnhum.2022.898300
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="standard_1005",
            hardware="BrainAmp (BrainProducts GmbH)",
            software="BrainVision with MATLAB 2019a",
            reference="FCz",
            ground="Fpz",
            sensors=list(_CH_NAMES),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            age_min=20,
            age_max=30,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Hello": 1, "Helpme": 2, "Stop": 3, "Thankyou": 4, "Yes": 5},
            paradigm="imagery",
            n_classes=5,
            class_labels=_CLASS_NAMES,
            trial_duration=3.1,
            study_design=(
                "Auditory cue followed by 4 repetitions of fixation cross "
                "(0.8-1.2 s jittered) + 2 s imagined speech, with 3 s "
                "relaxation between blocks. Black screen during imagery."
            ),
            stimulus_type="auditory cue",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "Imagine silent pronunciation as if performing real speech. "
                "No articulator movement, no sound, no blinking."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2022.898300",
            investigators=[
                "Ji-Hoon Jeong",
                "Jeong-Hyun Cho",
                "Young-Eun Lee",
                "Seo-Hyun Lee",
                "Gi-Hwan Shin",
                "Young-Seok Kweon",
                "Jose del R. Millan",
                "Klaus-Robert Mueller",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            institution_department="Department of Brain and Cognitive Engineering",
            institution_address="Seoul, South Korea",
            country="KR",
            data_url="https://osf.io/pq7vb/",
            publication_year=2022,
            license="CC-BY-4.0",
            repository="OSF",
            senior_author="Seong-Whan Lee",
            contact_info=["bcicompetition2020@gmail.com"],
            associated_paper_doi="10.3389/fnhum.2022.898300",
            keywords=[
                "brain-computer interface",
                "electroencephalogram",
                "imagined speech",
                "competition",
                "open datasets",
                "neural decoding",
            ],
            description=(
                "BCI Competition 2020 Track 3: Imagined speech classification "
                "with 5 phrases using 64-channel EEG. Best competition accuracy "
                "82.6%. IRB: KUIRB-2019-0143-01."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Competition"]),
        preprocessing=PreprocessingMetadata(
            data_state="epoched", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=_CLASS_NAMES,
            cue_duration_s=2.0,
            imagery_duration_s=2.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=5250,
            n_trials_per_class={
                "Hello": 1050,
                "Helpme": 1050,
                "Stop": 1050,
                "Thankyou": 1050,
                "Yes": 1050,
            },
            trials_context=("15 subjects x 350 trials (70 per class: 60 train + 10 val)"),
        ),
        data_processed=False,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events={"Hello": 1, "Helpme": 2, "Stop": 3, "Thankyou": 4, "Yes": 5},
            code="BCIComp2020IS",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.3389/fnhum.2022.898300",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _load_epoch_mat(self, fpath, epo_key):
        """Load a MATLAB epoch file and return (data, labels, ch_names).

        Parameters
        ----------
        fpath : str
            Path to the .mat file.
        epo_key : str
            Key for the epoch struct (e.g. 'epo_train').

        Returns
        -------
        data : ndarray, shape (n_trials, n_channels, n_times)
        labels : ndarray of int, shape (n_trials,)
        ch_names : list of str

        Returns None, None, None if labels are not available.
        """
        try:
            mat = loadmat(fpath, squeeze_me=False)
            epo = mat[epo_key]

            x = epo["x"][0, 0]  # (n_times, n_channels, n_trials)
            y = epo["y"][0, 0]  # (n_classes, n_trials) one-hot

            # Transpose to (n_trials, n_channels, n_times)
            data = np.transpose(x, (2, 1, 0))

            # Convert one-hot to integer labels (1-indexed)
            labels = np.argmax(y, axis=0) + 1

            # Channel names from the file
            clab = epo["clab"][0, 0][0]
            ch_names = [str(c[0]) for c in clab]

        except NotImplementedError:
            # HDF5 / MATLAB v7.3 format (e.g. test files).
            import h5py

            with h5py.File(fpath, "r") as f:
                epo = f[epo_key]

                # HDF5 transposes: (n_trials, n_channels, n_times)
                x = epo["x"][:]
                data = x

                # Check if labels are available.
                y = epo["y"][:]
                if y.ndim < 2 or y.shape[0] < 2 or np.all(y == 0):
                    return None, None, None

                labels = np.argmax(y, axis=0) + 1

                # Channel names from object references.
                clab = epo["clab"]
                ch_names = []
                for i in range(clab.shape[0]):
                    ref = clab[i, 0]
                    ch = f[ref][:]
                    name = "".join(chr(c) for c in ch.flat)
                    ch_names.append(name)

        return data, labels, ch_names

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        self.data_path(subject)
        base = self._subject_dir()
        runs = {}

        for run_idx, (split, epo_key) in enumerate(
            [
                ("training", "epo_train"),
                ("validation", "epo_validation"),
                ("test", "epo_test"),
            ]
        ):
            fpath = base / split / f"Data_Sample{subject:02d}.mat"
            if not fpath.exists():
                log.warning("File not found: %s", fpath)
                continue

            data, labels, ch_names = self._load_epoch_mat(str(fpath), epo_key)
            if data is None:
                log.info("Skipping %s (no labels available).", split)
                continue
            raw = build_raw_from_epochs(
                data, ch_names, _SFREQ, labels, montage_name="standard_1005"
            )
            runs[str(run_idx)] = raw

        return {"0": runs}

    def _subject_dir(self):
        path = dl.get_dataset_path(_SIGN, None)
        return Path(path) / f"MNE-{_SIGN}-data"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = self._subject_dir()

        # Check if files already exist.
        train_file = base / "training" / f"Data_Sample{subject:02d}.mat"
        if train_file.exists() and not force_update:
            return str(base)

        # Try to find files in alternate location (manual download).
        mne_data = Path(dl.get_dataset_path(_SIGN, path))
        alt_paths = [mne_data / "bci_comp_2020_imagined_speech"]

        for alt in alt_paths:
            alt_train = alt / "training" / f"Data_Sample{subject:02d}.mat"
            if alt_train.exists():
                for split in ["training", "validation", "test"]:
                    src = alt / split / f"Data_Sample{subject:02d}.mat"
                    dst = base / split / f"Data_Sample{subject:02d}.mat"
                    if src.exists() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src), str(dst))
                return str(base)

        # Download from OSF.
        log.info("Downloading subject %d from OSF...", subject)
        self._download_from_osf(subject, base, path, force_update, verbose)

        return str(base)

    def _download_from_osf(self, subject, base, path, force_update, verbose):
        """Download subject files from the OSF API."""
        import requests

        fname = f"Data_Sample{subject:02d}.mat"
        resp = requests.get(_OSF_API, timeout=30)
        if not resp.ok:
            raise ConnectionError(
                f"Failed to list OSF files (HTTP {resp.status_code}). "
                "Download manually from https://osf.io/pq7vb/"
            )

        folder_map = {}
        for item in resp.json().get("data", []):
            folder_map[item["attributes"]["name"]] = item["relationships"]["files"][
                "links"
            ]["related"]["href"]

        for split in ["training", "validation"]:
            dest_dir = base / split
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / fname

            if dest.exists() and not force_update:
                continue

            folder_url = folder_map.get(split)
            if folder_url is None:
                continue

            freq = requests.get(folder_url, timeout=30)
            if not freq.ok:
                continue

            for fitem in freq.json().get("data", []):
                if fitem["attributes"]["name"] == fname:
                    download_url = fitem["links"]["download"]
                    downloaded = dl.data_dl(
                        download_url,
                        _SIGN,
                        path=path,
                        force_update=force_update,
                        verbose=verbose,
                    )
                    downloaded = Path(downloaded)
                    if downloaded != dest:
                        shutil.move(str(downloaded), str(dest))
