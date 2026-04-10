import json
import os
import string
import warnings
from pathlib import Path

import mne
from tqdm import tqdm

from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
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

from .base import BaseDataset


_manifest_link = "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/MQOVEY"
_api_base_url = "https://dataverse.harvard.edu/api/access/datafile/"


class Kojima2024A(BaseDataset):
    """Class for Kojima2024A dataset management. P300 dataset.

    **Dataset description**

    This dataset [1]_ originates from a study investigating a three-class auditory BCI
    based on auditory stream segregation (ASME-BCI) [2]_.

    In the experiment, participants focused on one of three auditory streams, leveraging
    auditory stream segregation to selectively attend to stimuli in the target stream.
    Each stream contained a two-stimulus oddball sequence composed of one deviant
    stimulus and one standard stimulus.

    The sequence below illustrates an example trial. For instance, when D2 is the target
    stimulus, the participant attended to Stream2 and selectively listened for D2.
    In this case, D2 is the target, and D1 and D3 are considered non-target stimuli.

    .. code-block:: text

        Stream3  ----- S3 -------- S3 -------- S3 -------- D3 -------- S3 -----
        Stream2  -- S2 -------- S2 -------- D2 -------- S2 -------- S2 --------
        Stream1  S1 -------- D1 -------- S1 -------- S1 -------- S1 -----------

    Each participant completed 1 session consisting of 6 runs.
    Each run lasted approximately 5 minutes.
    In each run, all deviant stimuli (D1--D4) were presented approximately 60 times.

    Recording Details:
        - EEG signals were recorded using a BrainAmp system (Brain Products, Germany)
          at a sampling rate of 1000 Hz.

        - Data were collected in Tokyo, Japan, where the power line frequency is 50 Hz.

        - EEG was recorded from 64 scalp electrodes according to the international 10--20 system:
          Fp1, Fp2, AF7, AF3, AFz, AF4, AF8, F7, F5, F3, F1, Fz, F2, F4, F6, F8,
          FT9, FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8, FT10, T7, C5, C3, C1,
          Cz, C2, C4, C6, T8, TP9, TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8,
          TP10, P7, P5, P3, P1, Pz, P2, P4, P6, P8, PO7, PO3, POz, PO4, PO8,
          O1, Oz, O2

          EEG signals were referenced to the right mastoid and grounded to the left mastoid.

        - EOG was recorded using 2 electrodes (vEOG and hEOG), placed above/below and
          lateral to one eye.

    References
    ----------

    .. [1] Kojima, S. (2024).
        Replication Data for: An auditory brain-computer interface based on selective attention to multiple tone streams.
        Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/MQOVEY
    .. [2] Kojima, S. & Kanoh, S. (2024).
        An auditory brain-computer interface based on selective attention to multiple tone streams.
        PLoS ONE 19(5): e0303565. DOI: https://doi.org/10.1371/journal.pone.0303565
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64, "eog": 2},
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "AFz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT10",
                "FT7",
                "FT8",
                "FT9",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
                "hEOG",
                "vEOG",
            ],
            sensor_type="eeg",
            reference="right earlobe",
            ground="left earlobe",
            hardware="Brain Amp DC (Brain Products GmbH, Germany) and MR plus (Brain Products GmbH, Germany)",
            software=None,
            filters={"bandpass": "0.1 Hz to 100 Hz"},
            line_freq=50.0,
            montage="standard_1020",
            impedance_threshold_kohm=None,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["vertical", "horizontal"],
                has_emg=False,
                emg_channels=0,
                other_physiological=None,
            ),
            cap_manufacturer="EASYCAP GmbH",
            cap_model=None,
            electrode_type=None,
            electrode_material="Ag-AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=11,
            health_status="healthy",
            gender={"male": 10, "female": 1},
            age_mean=22.5,
            age_std=None,
            age_min=22.0,
            age_max=23.0,
            ages=None,
            handedness=None,
            clinical_population=None,
            bci_experience=None,
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
                "male",
                "female",
            ],
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="auditory selective attention",
            events={"S1": 1, "D1": 2, "S2": 3, "D2": 4, "S3": 5, "D3": 6},
            n_classes=3,
            class_labels=["Stream 1", "Stream 2", "Stream 3"],
            trials_per_class=None,
            trial_duration=None,
            tasks=["attend to Stream 1", "attend to Stream 2", "attend to Stream 3"],
            study_design="within-subject",
            study_domain="auditory BCI",
            feedback_type="none",
            stimulus_type="auditory musical tones",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Subjects were requested to attend to one of three streams and to count the number of target stimuli in the attended stream",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "method": "Digital signal processor (System3, Tucker-Davis Technologies, USA) and headphones (HDA200, Sennheiser)",
                "ear": "right ear only",
                "tone_generator": "Software synthesizer (Piano tones Grand Piano 1 SE from SampleTank3, IK multimedia Production, Italy)",
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0303565",
            description="A 3-class auditory BCI using three tone sequences based on auditory stream segregation. Musical tones were presented to subjects' right ear, and subjects attended to one of three streams while counting target stimuli. P300 activity was elicited by target stimuli in the attended stream.",
            investigators=["Simon Kojima", "Shin'ichiro Kanoh"],
            institution="Shibaura Institute of Technology",
            country="JP",
            repository="Harvard Dataverse",
            data_url="https://doi.org/10.7910/DVN/MQOVEY",
            license="Creative Commons Attribution License",
            publication_year=2024,
            senior_author="Shin'ichiro Kanoh",
            contact_info=["nb21106@shibaura-it.ac.jp"],
            associated_paper_doi="10.1371/journal.pone.0303565",
            funding=["JSPS KAKENHI Grant Number JP23K11811"],
            institution_address="Koto-ku, Tokyo, Japan",
            institution_department="Graduate School of Engineering and Science; College of Engineering",
            ethics_approval=[
                "Review Board on Bioengineering Research Ethics of Shibaura Institute of Technology",
                "Declaration of Helsinki",
            ],
            acknowledgements=None,
            how_to_acknowledge=None,
            keywords=[
                "auditory BCI",
                "P300",
                "auditory stream segregation",
                "selective attention",
                "oddball paradigm",
                "Riemannian geometry",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        sessions=None,
        contributing_labs=None,
        n_contributing_labs=1,
        data_processed=False,
        file_format="BrainVision",
        external_links={
            "source": "https://doi.org/10.7910/DVN/MQOVEY",
            "paper": "https://doi.org/10.1371/journal.pone.0303565",
        },
        tags=Tags(
            pathology=["Healthy"], modality=["auditory"], type=["EEG", "P300", "BCI"]
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            preprocessing_steps=None,
            highpass_hz=None,
            lowpass_hz=None,
            bandpass=None,
            notch_hz=None,
            filter_type=None,
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Logistic Regression", "Minimum Distance to Mean (MDM)"],
            feature_extraction=[
                "xDAWN spatial filtering",
                "Riemannian geometry covariance matrices",
            ],
            frequency_bands={"analyzed_range": [1.0, 40.0]},
            spatial_filters=["xDAWN"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold cross validation",
            cv_folds=10,
            evaluation_type=["within-subject"],
        ),
        performance={
            "description": "Classification accuracy over 80% for 5 subjects, over 75% for 9 subjects",
            "metric": "MCC (Matthews correlation coefficient)",
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=3,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=180.0,
            imagery_tasks=None,
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=None,
            n_trials_per_class=None,
            n_blocks=6,
            block_duration_s=300.0,
            trials_context="Each task block had 3 runs (5 minutes each). Subjects counted target stimuli in Streams 1, 2, and 3 on the 1st, 2nd, and 3rd measurements respectively. Task block was repeated twice.",
        ),
        abstract="In this study, we attempted to improve brain-computer interface (BCI) systems by means of auditory stream segregation in which alternately presented tones are perceived as sequences of various different tones (streams). A 3-class BCI using three tone sequences, which were perceived as three different tone streams, was investigated and evaluated. Each presented musical tone was generated by a software synthesizer. Eleven subjects took part in the experiment. Stimuli were presented to each user's right ear. Subjects were requested to attend to one of three streams and to count the number of target stimuli in the attended stream. In addition, 64-channel electroencephalogram (EEG) and two-channel electrooculogram (EOG) signals were recorded from participants with a sampling frequency of 1000 Hz. The measured EEG data were classified based on Riemannian geometry to detect the object of the subject's selective attention. P300 activity was elicited by the target stimuli in the segregated tone streams. In five out of eleven subjects, P300 activity was elicited only by the target stimuli included in the attended stream. In a 10-fold cross validation test, a classification accuracy over 80% for five subjects and over 75% for nine subjects was achieved. For subjects whose accuracy was lower than 75%, either the P300 was also elicited for nonattended streams or the amplitude of P300 was small. It was concluded that the number of selected BCI systems based on auditory stream segregation can be increased to three classes, and these classes can be detected by a single ear without the aid of any visual modality.",
        methodology="Musical tones generated by a digital auditory workstation were used as auditory stimuli. Piano tones from a MIDI sound source were presented using a digital signal processor and headphones to participants' right ear only. Three tone streams were created using auditory stream segregation, each consisting of standard (90% probability) and deviant (10% probability) tones. The duration of each tone was 150 ms with stimulus onset asynchrony of 180 ms. The 64-channel EEG and 2-channel EOG signals were recorded at 1000 Hz. Each experiment consisted of two task blocks with three runs each (5 minutes per run). Subjects counted target stimuli in different streams across runs. Data analysis involved bandpass filtering (0.1-40 Hz for ERP analysis, 1-40 Hz for classification), baseline correction, artifact rejection (±100μV for EEG, ±500μV for EOG), xDAWN spatial filtering, and classification using Riemannian geometry with covariance matrices and logistic regression. Performance was evaluated using 10-fold cross validation with accuracy and Matthews correlation coefficient (MCC) metrics.",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):

        self.n_channels = 64

        super().__init__(
            list(range(1, 12)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 0},
            code="Kojima2024A",
            interval=[-0.5, 1.2],
            paradigm="p300",
            doi="10.7910/DVN/MQOVEY",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_files_list(self, subject, manifest):

        subject_id = self.convert_subject_to_subject_id(subject)

        manifest_files = manifest["datasetVersion"]["files"]

        files_to_load = []

        for file in manifest_files:
            if (f"sub-{subject_id}" not in file["label"]) or (
                "_eeg" not in file["label"]
            ):
                continue

            fname = file["label"]
            directory = "/".join(file["directoryLabel"].split("/")[1:])
            file_id = file["dataFile"]["id"]

            files_to_load.append(
                {"fname": fname, "directory": directory, "file_id": file_id}
            )

        return files_to_load

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            A dictionary containing the raw data for the subject.
        """

        # Get the file path for the subject's data
        files_path = self.data_path(subject)
        runs = {}
        for file in files_path:
            fname = file.name

            run_id = fname.split("_")[2].split("-")[1]
            task = fname.split("_")[1].split("-")[1]

            annotations_map = {
                "Stimulus/S  2": "NonTarget",
                "Stimulus/S  8": "NonTarget",
                "Stimulus/S 32": "NonTarget",
            }

            if task == "low":
                annotations_map["Stimulus/S  2"] = "Target"
            elif task == "mid":
                annotations_map["Stimulus/S  8"] = "Target"
            elif task == "high":
                annotations_map["Stimulus/S 32"] = "Target"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                raw = mne.io.read_raw_brainvision(file, eog=["vEOG", "hEOG"])
                raw = raw.load_data()

                raw = raw.set_montage("standard_1020")

                raw.annotations.rename(annotations_map)

            runs.update({f"{run_id}{task}": raw})

        sessions = {"0": runs}

        return sessions

    def convert_subject_to_subject_id(self, subjects):
        """
        Convert subject number(s) to subject ID(s).
        (In this dataset, subject IDs are encoded using alphabet letters.)

        Parameters
        ----------
        subjects : int or list of int
            Subject number(s) to convert.

        Returns
        -------
        subject_id : str or list of str
            Converted subject ID(s).
        """

        if isinstance(subjects, int):
            subject_id = list(string.ascii_uppercase)[subjects - 1]
        elif isinstance(subjects, list):
            subject_id = []
            for subject in subjects:
                subject_id.append(list(string.ascii_uppercase)[subject - 1])
        else:
            raise TypeError("Type of subejcts must be either int or list.")

        return subject_id

    def data_path(self, subject, path=None):
        """
        Return the data paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn't exist, the “~/mne_data” directory is used. If the
            dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.

        Returns
        -------
        list
            A list containing the ``pathlib.Path`` object for the subject's data file.
        """

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_by_subject(subject=subject, path=path)

        subject_id = self.convert_subject_to_subject_id(subject)

        files = os.listdir(dataset_path / f"sub-{subject_id}" / "eeg")

        paths = []
        for file in files:
            if file.endswith(".vhdr"):
                paths.append(dataset_path / f"sub-{subject_id}" / "eeg" / file)

        return paths

    def download_by_subject(self, subject, path=None):
        """
        Download and extract the dataset.

        Parameters
        ----------
        subject : int
            The subject number to download the dataset for.

        path : str | None
            The path to the directory where the dataset should be downloaded.
            If None, the default directory is used.


        Returns
        -------
        path : str
            The dataset path.
        """

        path = Path(dl.get_dataset_path(self.code, path)) / (f"MNE-{self.code}-data")

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "kojima2024_manifest.json", _manifest_link)

        with open(path / "kojima2024_manifest.json", "r") as f:
            manifest = json.load(f)

        files = self._get_files_list(subject, manifest)

        for file in tqdm(files):
            download_url = _api_base_url + str(file["file_id"])
            dl.download_if_missing(
                path / file["directory"] / file["fname"], download_url, warn_missing=False
            )

        return path
