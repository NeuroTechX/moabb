"""BETA: A Large Benchmark Database Toward SSVEP-BCI Application.

Liu et al. (2020), Frontiers in Neuroscience.
DOI: 10.3389/fnins.2020.00627
"""

import tarfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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
from .utils import TSINGHUA_64CH_NAMES, build_raw_from_epochs, safe_extract_tar


BETA_URL = "http://bci.med.tsinghua.edu.cn/upload/liubingchuan/"

# fmt: off
# Frequencies follow the BETA keyboard layout (row-major reading of 5x8 grid),
# starting from 8.6 Hz and wrapping around: 8.6, 8.8, ..., 15.8, 8.0, 8.2, 8.4
_EVENTS = {
    "8.6": 1, "8.8": 2, "9": 3, "9.2": 4, "9.4": 5, "9.6": 6, "9.8": 7, "10": 8,
    "10.2": 9, "10.4": 10, "10.6": 11, "10.8": 12, "11": 13, "11.2": 14, "11.4": 15, "11.6": 16,
    "11.8": 17, "12": 18, "12.2": 19, "12.4": 20, "12.6": 21, "12.8": 22, "13": 23, "13.2": 24,
    "13.4": 25, "13.6": 26, "13.8": 27, "14": 28, "14.2": 29, "14.4": 30, "14.6": 31, "14.8": 32,
    "15": 33, "15.2": 34, "15.4": 35, "15.6": 36, "15.8": 37, "8": 38, "8.2": 39, "8.4": 40,
}
# fmt: on


class Liu2020BETA(BaseDataset):
    """BETA SSVEP benchmark dataset.

    Dataset from [1]_.

    The BETA database contains 64-channel EEG recordings from 70 healthy
    subjects (42 males, 28 females, aged 9-64 years, mean age 25.14) performing
    a 40-target cued-spelling SSVEP-BCI task. Unlike Wang2016 which was collected
    in a shielded room, BETA was recorded in a normal classroom, providing a more
    realistic BCI benchmark.

    Stimuli used joint frequency and phase modulation (JFPM) with 40 targets
    arranged in a 5x8 QWERTY virtual keyboard. Frequencies ranged from 8.0 to
    15.8 Hz (0.2 Hz step) with initial phases from 0 to 19.5*pi (0.5*pi step).

    Each subject completed 4 blocks of 40 trials. Stimulation duration was 2 s
    for subjects S1-S15 (experienced) and 3 s for subjects S16-S70 (naive), plus
    a 0.5 s visual cue. EEG was recorded at 1000 Hz with a Synamps2 system
    (Neuroscan) using 64 channels in the international 10-10 system, then
    downsampled to 250 Hz.

    Data are stored as 4D matrices [64, 750, 4, 40] corresponding to
    [channels, time points, block index, target index]. Each epoch is 3 s
    (750 samples at 250 Hz).

    Warnings
    --------
    Like Wang2016, this dataset includes channels 'CB1' and 'CB2' which are
    not part of the standard 10-20 montage. They are treated as standard EEG
    channels with ``on_missing="ignore"`` for montage setting.

    The data is downloaded from the Tsinghua BCI Lab server in tar.gz archives
    grouped by 10 subjects each. The download may be slow depending on server
    availability.

    References
    ----------
    .. [1] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "BETA: A Large
       Benchmark Database Toward SSVEP-BCI Application," Frontiers in
       Neuroscience, vol. 14, p. 627, 2020.
       DOI: 10.3389/fnins.2020.00627
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="standard_1005",
            hardware="Synamps2 (Neuroscan)",
            sensors=TSINGHUA_64CH_NAMES,
            line_freq=50.0,
            reference="Cz",
            impedance_threshold_kohm=10,
        ),
        participants=ParticipantMetadata(
            n_subjects=70,
            health_status="healthy",
            gender={"male": 42, "female": 28},
            age_mean=25.14,
            age_min=9,
            age_max=64,
            age_std=7.97,
            bci_experience="mixed",
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            events=dict(_EVENTS),
            n_classes=40,
            trial_duration=3.0,
            stimulus_type="JFPM visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            task_type="cued-spelling",
            feedback_type="visual",
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2020.00627",
            investigators=[
                "Bingchuan Liu",
                "Xiaoshan Huang",
                "Yijun Wang",
                "Xiaogang Chen",
                "Xiaorong Gao",
            ],
            senior_author="Xiaorong Gao",
            institution="Tsinghua University",
            country="CN",
            repository="Tsinghua BCI Lab",
            data_url="http://bci.med.tsinghua.edu.cn/upload/liubingchuan/",
            license="Non-commercial research use",
            publication_year=2020,
            institution_department="Department of Biomedical Engineering, Tsinghua University",
            ethics_approval=["Ethics Committee of Tsinghua University, No. 20190002"],
            funding=[
                "National Key Research and Development Program of China (No. 2017YFB1002505)",
                "Strategic Priority Research Program of Chinese Academy of Sciences (No. XDB32040200)",
                "Key Research and Development Program of Guangdong Province (No. 2018B030339001)",
                "National Natural Science Foundation of China (Grant No. 61431007)",
            ],
            keywords=["SSVEP", "BCI", "EEG", "benchmark", "JFPM"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            downsampled_to_hz=250.0,
            notch_hz=50,
            filter_type="zero-phase FIR",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[8.0 + i * 0.2 for i in range(40)],
            frequency_resolution_hz=0.2,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=4,
            n_trials=160,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["TRCA", "msTRCA", "FBCCA", "CCA"],
            feature_extraction=["CCA", "TRCA", "FBCCA"],
            frequency_bands={
                "bandpass": [3.0, 100.0],
            },
            spatial_filters=["CCA", "TRCA"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-block-out",
            cv_folds=4,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            environment="classroom",
            online_feedback=True,
            applications=["speller"],
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        file_format="MAT",
    )

    _events = _EVENTS

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 71)),
            sessions_per_subject=1,
            events=self._events,
            code="Liu2020BETA",
            interval=[0, 3.0],
            paradigm="ssvep",
            doi="10.3389/fnins.2020.00627",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        n_channels, n_blocks = 64, 4
        n_classes = len(self.event_id)

        fname = self.data_path(subject)
        mat = loadmat(fname)  # no squeeze for reliable struct access

        # Struct: data.EEG shape [64, 750, 4, 40] (ch, time, blocks, targets)
        raw_data = mat["data"]
        eeg = raw_data["EEG"][0, 0]

        # Extract per-subject demographics from suppl_info
        suppl = raw_data["suppl_info"][0, 0]
        age = int(round(float(suppl["age"][0, 0].flat[0])))
        gender = str(suppl["gender"][0, 0].flat[0])

        # Transpose to [targets, blocks, channels, time] then reshape
        data = np.transpose(eeg, axes=(3, 2, 0, 1))
        data = np.reshape(data, (-1, n_channels, eeg.shape[1]))

        event_ids = np.repeat(np.arange(1, n_classes + 1), n_blocks)
        raw = build_raw_from_epochs(
            data, TSINGHUA_64CH_NAMES, 250, event_ids, "standard_1005"
        )

        # Set subject_info for BIDS export (sex, his_id are MNE-supported keys)
        _sex_map = {"male": 1, "female": 2}
        raw.info["subject_info"] = {
            "sex": _sex_map.get(gender.lower(), 0),
            "his_id": str(subject),
        }
        # Store age separately (MNE SubjectInfo doesn't support custom keys)
        raw._moabb_subject_age = age

        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if the extracted .mat file already exists
        mat_file = data_dir / f"S{subject}.mat"
        if mat_file.exists() and not force_update:
            return str(mat_file)

        # Download the tar.gz archive containing this subject
        start = ((subject - 1) // 10) * 10 + 1
        tar_name = f"S{start}-S{start + 9}.tar.gz"
        url = BETA_URL + tar_name
        tar_path = dl.data_dl(url, sign, path, force_update, verbose)

        # Extract only the needed subject's .mat file
        target_name = f"S{subject}.mat"
        data_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [
                member for member in tf.getmembers() if member.name.endswith(target_name)
            ]
            safe_extract_tar(tf, data_dir, members=members)

        if mat_file.exists():
            return str(mat_file)

        # Search subdirectories in case tar has nested structure
        found = next(data_dir.rglob(target_name), None)
        if found:
            return str(found)

        raise FileNotFoundError(
            f"Could not find S{subject}.mat after extracting {tar_name}"
        )
