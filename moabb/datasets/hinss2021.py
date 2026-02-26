import os
import os.path as osp
import zipfile as z

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FilterDetails,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)


URL = "https://zenodo.org/record/5055046/files/"
# private dictionary to map events to integers
_HINNS_EVENTS = dict(rs=1, easy=2, medium=3, diff=4)


class Hinss2021(BaseDataset):
    """Neuroergonomic 2021 dataset.

    We describe the experimental procedures for a dataset that is publicly available
    at https://zenodo.org/records/5055046.
    This dataset contains electroencephalographic recordings of 15 subjects (6 female, with an
    average age of 25 years). A total of 62 active Ag–AgCl
    electrodes were available in the dataset.

    The participants engaged in 3 (2 available here) distinct experimental sessions, each of which
    was separated by 1 week.

    At the beginning of each session, the resting state of the participant
    (measured as 1 minute with eyes open) was recorded.

    Subsequently, participants undertook 3 tasks of varying difficulty levels
    (i.e., easy, medium, and difficult). The task assignments
    were randomized. For more details, please check [Hinss2021]_.

    Notes
    -----

    .. versionadded:: 1.0.1

    References
    ----------

    .. [Hinss2021] M. Hinss, B. Somon, F. Dehais & R. N. Roy (2021)
            Open EEG Datasets for Passive Brain-Computer
            Interface Applications: Lacks and Perspectives.
            IEEE Neural Engineering Conference.

    .. [Hinss2023] M. F. Hinss, et al. (2023)
            An EEG dataset for cross-session mental workload estimation:
            Passive BCI competition of the Neuroergonomics Conference 2021.
            Scientific Data, 10, 85.
            https://doi.org/10.1038/s41597-022-01898-y
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-20",
            hardware="Brain Products",
            sensor_type="active Ag-AgCl",
            reference="Car",
            software="EEGlab",
            filters="none",
            impedance_threshold_kohm=25,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["ecg", "gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=29,
            gender={"female": 6, "male": 9},
            age_mean=23.9,
            handedness={"left": 2, "right": 27},
        ),
        experiment=ExperimentMetadata(
            paradigm="rstate",
            n_classes=1,
            class_labels=["rest"],
            study_design="Four cognitive tasks: N-Back (working memory/mental workload), MATB-II (multi-tasking/workload), PVT (vigilance), Flanker (decision-making/conflict)",
            feedback_type="trial-based feedback (Flanker task provides correct/incorrect/miss feedback)",
            stimulus_type="avatar",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-022-01898-y",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.6874128",
            license="CC-BY-SA-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            filter_details=FilterDetails(
                highpass_hz=1.0,
                filter_type="FIR",
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=250,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["MDM", "Riemannian"],
            feature_extraction=["Bandpower", "Covariance/Riemannian", "ICA"],
            frequency_bands=FrequencyBands(
                alpha=[8.0, 13.0],
                theta=[4.0, 8.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold",
            cv_folds=5,
            evaluation_type=["cross_subject", "cross_session", "transfer_learning"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=70.67,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar", "communication"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="rstate",
        ),
        data_structure=DataStructureMetadata(
            n_trials=90,
            n_blocks=2,
            trials_context="total",
        ),
        data_processed=False,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 16)),  # 15 participants
            sessions_per_subject=2,  # 2 sessions per subject
            events=_HINNS_EVENTS,
            code="Hinss2021",
            interval=[0, 2],  # Epochs are 2-second long
            paradigm="rstate",
            doi="10.1038/s41597-022-01898-y",
        )

    def _get_stim_channel(self, rs_epochs, easy_epochs, med_epochs, n_epochs, n_samples):
        n_epochs_rs = rs_epochs.get_data().shape[0]
        n_epochs_easy = easy_epochs.get_data().shape[0]
        n_epochs_med = med_epochs.get_data().shape[0]
        stim = np.zeros((1, n_epochs * n_samples))
        for i in range(n_epochs):
            stim[0, n_samples * i + 1] = (
                _HINNS_EVENTS["rs"]
                if i < n_epochs_rs
                else (
                    _HINNS_EVENTS["easy"]
                    if i < n_epochs_rs + n_epochs_easy
                    else (
                        _HINNS_EVENTS["medium"]
                        if i < n_epochs_rs + n_epochs_easy + n_epochs_med
                        else _HINNS_EVENTS["diff"]
                    )
                )
            )
        return stim

    def _get_epochs(self, session_path, subject, session, event_file):
        raw = os.path.join(
            session_path,
            f"alldata_sbj{str(subject).zfill(2)}_sess{session}_{event_file}.set",
        )
        epochs = mne.io.read_epochs_eeglab(raw)
        return epochs

    def _get_single_subject_data(self, subject):
        """Load data for a single subject."""
        data = {}

        subject_path = self.data_path(subject)[0]

        for session in range(1, self.n_sessions + 1):
            session_path = os.path.join(subject_path, f"S{session}/eeg/")

            # get 'resting state'
            rs_epochs = self._get_epochs(session_path, subject, session, "RS")

            # get task 'easy'
            easy_epochs = self._get_epochs(session_path, subject, session, "MATBeasy")

            # get task 'med'
            med_epochs = self._get_epochs(session_path, subject, session, "MATBmed")

            # get task 'diff'
            diff_epochs = self._get_epochs(session_path, subject, session, "MATBdiff")

            # concatenate raw data
            raw_data = np.concatenate(
                (
                    rs_epochs.get_data(),
                    easy_epochs.get_data(),
                    med_epochs.get_data(),
                    diff_epochs.get_data(),
                )
            )

            # reshape data in the form n_channel x n_sample
            raw_data = raw_data.transpose((1, 0, 2))
            n_channel, n_epochs, n_samples = raw_data.shape
            raw_data = raw_data.reshape((n_channel, n_epochs * n_samples))

            # add stim channel
            stim = self._get_stim_channel(
                rs_epochs, easy_epochs, med_epochs, n_epochs, n_samples
            )
            raw_data = np.concatenate((raw_data, stim))

            # create info
            self._chnames = rs_epochs.ch_names + ["stim"]
            self._chtypes = ["eeg"] * (raw_data.shape[0] - 1) + ["stim"]

            info = mne.create_info(
                ch_names=self._chnames, sfreq=500, ch_types=self._chtypes, verbose=False
            )
            raw = mne.io.RawArray(raw_data, info)

            # Only one run => "0"
            data[str(session)] = {"0": raw}

        return data

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # check if has the .zip
        url = f"{URL}P{subject:02}.zip"

        path_zip = dl.data_dl(url, "Neuroergonomics2021")
        path_folder = path_zip.strip(f"P{subject:02}.zip")

        # check if has to unzip
        if not (osp.isdir(path_folder + f"P{subject:02}")) and not (
            osp.isdir(path_folder + f"P{subject:02}")
        ):
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        final_path = f"{path_folder}P{subject:02}"
        return [final_path]
