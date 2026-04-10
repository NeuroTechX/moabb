"""Liu2024 Motor imagery dataset."""

import os
import shutil
import warnings
import zipfile as z
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import mne
import numpy as np
import pandas as pd
from mne.channels import make_dig_montage

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
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.utils import _handle_deprecated_kwargs


# Link to the raw data
LIU2024_URL = "https://ndownloader.figshare.com/files/38516654"

# Links to the electrodes and events information files
LIU2024_ELECTRODES = "https://ndownloader.figshare.com/files/38516078"
LIU2024_EVENTS = "https://ndownloader.figshare.com/files/38516084"


class Liu2024(BaseDataset):
    """Dataset [1]_ from the study on motor imagery [2]_.

    **Dataset description**
    This dataset includes data from 50 acute stroke patients (the time after stroke ranges from 1 day to 30 days)
    admitted to the stroke unit of Xuanwu Hospital of Capital Medical University. The patients included 39 males (78%)
    and 11 females (22%), aged between 31 and 77 years, with an average age of 56.70 years (SD = 10.57)
    Before the start of the experiment, the subject sat in a chair in a position as comfortable as possible with an
    EEG cap placed on their head; subjects were positioned approximately 80 cm away from a computer screen in front of them.
    The computer played audio instructions to the patient about the procedure. Each experiment lasted approximately 20 minutes,
    including preparation time and approximately 10 minutes of signal recording. Before the start of the MI experiment,
    the patients opened their eyes and closed their eyes for 1 minute each. The MI experiment was divided into 40 trials, and
    each trial took 8 seconds, which consisted of three stages (instruction, MI and break). In the instruction stage, patients
    were prompted to imagine grasping a spherical object with the left- or right-hand. In the MI stage, participants imagined
    performing this action, a video of gripping motion is played on the computer, which leads the patient imagine grabbing the
    ball. This video stays playing for 4 s. Patients only imagine one hand movement.In the break stage, participants were allowed
    to relax and rest. The MI experiments alternated between the left- and right-hand, and the patients moved onto the next stage
    of the experiment according to the instructions.

    The EEG data were collected through a wireless multichannel EEG acquisition system (ZhenTec NT1, Xi’an ZhenTec Intelligence
    Technology Co., Ltd., China). The system includes an EEG cap, an EEG acquisition amplifier, a data receiver and host computer
    software. The EEG cap had electrodes placed according to the international 10-10 system, including 29 EEG recording electrodes
    and 2 electrooculography (EOG) electrodes. The reference electrode located at CPz position and the grounding electrode located
    at FPz position. All the EEG electrodes and grounding electrode are Ag/AgCl semi-dry EEG electrodes based on highly absorbable
    porous sponges that are dampened with 3% NaCl solution. The EOG electrodes are composed by Ag/AgCl electrodes and conductive
    adhesive hydrogel. The common-mode rejection ratio was 120 dB, the input impedance was 1 GΩ, the input noise was less than
    0.4 μVrms, and the resolution was 24 bits. The acquisition impedance was less than or equal to 20 kΩ. The sampling frequency
    was 500 Hz.

    References
    ----------

    .. [1] Liu, Haijie; Lv, Xiaodong (2022). EEG datasets of stroke patients.
        figshare. Dataset. DOI: https://doi.org/10.6084/m9.figshare.21679035.v5

    .. [2] Liu, Haijie, Wei, P., Wang, H. et al. An EEG motor imagery dataset
       for brain computer interface in acute stroke patients. Sci Data 11, 131
       (2024). DOI: https://doi.org/10.1038/s41597-023-02787-8

    Notes
    -----
    To add the break and instruction events, set the `break_events` and
    `instr_events` parameters to True while instantiating the class.

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=29,
            channel_types={"eeg": 29, "eog": 2},
            montage="10-10",
            hardware="ZhenTec NT1 wireless multichannel EEG acquisition system",
            cap_manufacturer="Xi'an ZhenTec Intelligence Technology Co., Ltd.",
            cap_model="ZhenTec NT1",
            sensor_type="semi-dry Ag/AgCl",
            electrode_type="semi-dry",
            reference="CPz",
            ground="FPz",
            software=None,
            impedance_threshold_kohm=20,
            sensors=[
                "C3",
                "C4",
                "CP3",
                "CP4",
                "Cz",
                "F3",
                "F4",
                "F7",
                "F8",
                "FC3",
                "FC4",
                "FCz",
                "FP1",
                "FP2",
                "FT7",
                "FT8",
                "Fz",
                "HEOL",
                "O1",
                "O2",
                "Oz",
                "P3",
                "P4",
                "Pz",
                "T3",
                "T4",
                "T5",
                "T6",
                "TP7",
                "TP8",
                "VEOR",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True, eog_channels=2, eog_type=["horizontal", "vertical"]
            ),
            electrode_material="Ag/AgCl semi-dry electrodes based on highly absorbable porous sponges dampened with 3% NaCl solution",
        ),
        participants=ParticipantMetadata(
            n_subjects=50,
            health_status="acute stroke patients",
            clinical_population="acute stroke patients (1-30 days post-stroke)",
            gender={"male": 39, "female": 11},
            age_mean=56.7,
            age_std=10.57,
            age_min=31.0,
            age_max=77.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 20, "right_hand": 20},
            trial_duration=8.0,
            study_design="Imagining grasping a spherical object with left or right hand while watching a video of gripping motion. Each trial: instruction stage (prompt), MI stage (4s video-guided imagery), break stage (rest).",
            feedback_type="none",
            stimulus_type="video and audio",
            stimulus_modalities=["visual", "audio"],
            synchronicity="cue-based",
            mode="offline",
            has_training_test_split=True,
            events={"left_hand": 1, "right_hand": 2},
            instructions="Subject sat approximately 80 cm from computer screen. Computer played audio instructions. Patients imagined grasping spherical object with prompted hand during 4s video playback.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-023-02787-8",
            description="EEG motor imagery dataset from 50 acute stroke patients performing left- and right-handed hand-grip imagination tasks. First open dataset addressing left- and right-handed motor imagery in acute stroke patients.",
            investigators=[
                "Haijie Liu",
                "Penghu Wei",
                "Haochong Wang",
                "Xiaodong Lv",
                "Wei Duan",
                "Meijie Li",
                "Yan Zhao",
                "Qingmei Wang",
                "Xinyuan Chen",
                "Gaige Shi",
                "Bo Han",
                "Junwei Hao",
            ],
            senior_author="Junwei Hao",
            contact_info=["haojunwei@vip.163.com"],
            institution="Xuanwu Hospital Capital Medical University",
            institution_address="Beijing, 100053, China",
            institution_department="Department of Neurology",
            country="CN",
            data_url="https://doi.org/10.6084/m9.figshare.21679035.v5",
            publication_year=2024,
            funding=[
                "National Natural Science Foundation of China (grant nos. 82090043 and 81825008)"
            ],
            ethics_approval=[
                "Ethics Committee of Xuanwu Hospital of Capital Medical University (No. 2021-236)"
            ],
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "stroke patients",
                "EEG",
                "rehabilitation",
                "acute stroke",
                "hand-grip imagery",
                "databases",
                "scientific data",
            ],
            license="CC-BY-4.0",
            repository="Figshare",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "baseline removal (mean removal method)",
                "FIR filtering (0.5-40 Hz)",
            ],
            highpass_hz=0.5,
            lowpass_hz=40.0,
            bandpass=[0.5, 40.0],
            filter_type="FIR",
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            epoch_window=[0.0, 8.0],
            notes="Preprocessed with EEGLAB toolbox in MATLAB R2019b. Filtered data split into trials x channels x time-samples format by marker '1'. Some motion artifacts present in subjects 4, 5, 13, 14, 18, 24, 28, 33, 42, 43, 47, 48, 49.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+LDA", "FBCSP+SVM", "TSLDA+DGFMDRM", "TWFB+DGFMDM"],
            feature_extraction=[
                "CSP",
                "FBCSP",
                "ERD/ERS",
                "Riemannian geometry (SCMs on SPD manifolds)",
                "Tangent Space",
                "Time-Frequency (Morlet wavelet)",
                "TWFB (Time Window Filter Bank)",
            ],
            frequency_bands={
                "alpha": [8.0, 15.0],
                "beta": [15.0, 30.0],
                "analyzed_range": [8.0, 30.0],
            },
            spatial_filters=["CSP", "FBCSP", "Discriminant Geodesic Filtering"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold cross-validation",
            cv_folds=10,
            evaluation_type=["within_subject"],
        ),
        performance={
            "CSP+LDA_accuracy": 55.57,
            "FBCSP+SVM_accuracy": 57.57,
            "TSLDA+DGFMDRM_accuracy": 61.20,
            "TWFB+DGFMDM_accuracy": 72.21,
            "TWFB+DGFMDM_kappa": 0.4442,
            "TWFB+DGFMDM_precision": 0.7543,
            "TWFB+DGFMDM_sensitivity": 0.7845,
        },
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"], environment="hospital", online_feedback=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=40,
            n_trials_per_class={"left_hand": 20, "right_hand": 20},
            trials_context="40 trials per subject total (20 left-hand, 20 right-hand), alternating. Each trial: 8s total (instruction + 4s MI + break). Training/test split: 60%/40%.",
        ),
        file_format="MAT and EDF",
        data_processed=True,
        contributing_labs=["Xuanwu Hospital Capital Medical University"],
        n_contributing_labs=1,
        abstract="The brain-computer interface (BCI) is a technology that involves direct communication with parts of the brain and has evolved rapidly in recent years; it has begun to be used in clinical practice, such as for patient rehabilitation. Patient electroencephalography (EEG) datasets are critical for algorithm optimization and clinical applications of BCIs but are rare at present. We collected data from 50 acute stroke patients with wireless portable saline EEG devices during the performance of two tasks: 1) imagining right-handed movements and 2) imagining left-handed movements. The dataset consists of four types of data: 1) the motor imagery instructions, 2) raw recording data, 3) pre-processed data after removing artefacts and other manipulations, and 4) patient characteristics. This is the first open dataset to address left- and right-handed motor imagery in acute stroke patients.",
        methodology="50 acute stroke patients (1-30 days post-stroke) performed 40 trials of hand-grip motor imagery (20 left, 20 right). Each 8s trial included instruction, 4s video-guided imagery, and rest phases. EEG recorded with ZhenTec NT1 wireless system (29 EEG + 2 EOG channels) at 500 Hz. Data organized in EEG-BIDS format with raw (.mat) and preprocessed (.edf) versions. Clinical assessments: NIHSS (mean=4.16, SD=2.85), MBI (mean=70.94, SD=18.22), mRS (mean=2.66, SD=1.44). 23 patients right hemiplegia, 27 left hemiplegia.",
    )

    def __init__(
        self,
        break_events=False,
        instr_events=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {
            "BreakEvents": "break_events",
            "InstrEvents": "instr_events",
            "Subjects": "subjects",
            "Sessions": "sessions",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "Liu2024")
        break_events = resolved.get("break_events", break_events)
        instr_events = resolved.get("instr_events", instr_events)
        subjects = resolved.get("subjects", subjects)
        sessions = resolved.get("sessions", sessions)

        self.break_events = break_events
        self.instr_events = instr_events
        self.events = {"left_hand": 1, "right_hand": 2}
        if instr_events:
            self.events["instr"] = 3
        if break_events:
            self.events["break"] = 4
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events=self.events,
            code="Liu2024",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.1038/s41597-023-02787-8",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn’t exist, the “~/mne_data” directory is used. If the
            dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config
            to the given path.
            If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list containing the path to the subject's data file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download the zip file containing the data
        path_zip = dl.data_dl(LIU2024_URL, self.code)
        path_zip = Path(path_zip)
        path_folder = path_zip.parent

        # Extract the zip file if it hasn't been extracted yet
        if not (path_folder / "edffile").is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(dl.data_dl(LIU2024_URL, self.code, force_update=True))
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        subject_paths = []
        sub = f"sub-{subject:02d}"

        # Construct the path to the subject's data file
        subject_path = (
            path_folder / "edffile" / sub / "eeg" / f"{sub}_task-motor-imagery_eeg.edf"
        )
        subject_paths.append(str(subject_path))

        return subject_paths

    def encoding(self, events_df):
        """Encode the columns 'value' and 'trial_type' into a single event type.

        Parameters
        ----------
        events_df : :class:`pandas.DataFrame`
            DataFrame containing the events information.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of encoded event types.
        dict
            Mapping from event codes to event names.
        set
            Set of STI channel values included in the encoding.

        Notes
        -----
        The 'trial_type' variable can take the following values:
         - 1 : Left hand
         - 2 : Right hand

        The 'value' variable can take the following values:
         - 1 : instructions
         - 2 : MI
         - 3 : break

        """
        # Define the mapping dictionary
        encoding_mapping = {
            (1, 2): 1,  # Left hand MI (trial_type=1 per paper = left)
            (2, 2): 2,  # Right hand MI (trial_type=2 per paper = right)
        }

        mapping = {1: "left_hand", 2: "right_hand"}

        if self.instr_events:
            encoding_mapping.update(
                {
                    (1, 1): 3,  # Left hand, instructions
                    (2, 1): 3,  # Right hand, instructions
                }
            )
            mapping[3] = "instr"

        if self.break_events:
            encoding_mapping.update(
                {
                    (1, 3): 4,  # Left hand, break
                    (2, 3): 4,  # Right hand, break
                }
            )
            mapping[4] = "break"

        # Collect the set of STI channel values included in the encoding
        stim_values = {v for (_, v) in encoding_mapping.keys()}

        # Filter out rows that won't be encoded
        valid_tuples = encoding_mapping.keys()
        events_df = events_df[
            events_df.apply(
                lambda row: (row["trial_type"], row["value"]) in valid_tuples, axis=1
            )
        ]

        # Apply the mapping to the DataFrame
        event_category = events_df.apply(
            lambda row: encoding_mapping[(row["trial_type"], row["value"])], axis=1
        )

        return event_category, mapping, stim_values

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

        file_path_list = self.data_path(subject)[0]
        path_electrodes, path_events = self.data_infos()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Read the subject's raw data
            raw = mne.io.read_raw_edf(
                file_path_list, verbose=False, infer_types=True, stim_channel=""
            )

        # Always drop reference channel (constant zeros, not a useful modality)
        raw = raw.drop_channels(["CPz"])

        # Renaming channels accurately
        raw.rename_channels({"HEOR": "VEOR", "": "STI"})

        # Create a dictionary with the channel names and their new types
        mapping = {"STI": "stim", "VEOR": "eog", "HEOL": "eog"}

        # Set the new channel types
        raw.set_channel_types(mapping)

        # Read electrode positions from TSV file and create montage
        electrodes_df = pd.read_csv(path_electrodes, sep="\t")
        ch_pos = {
            row["name"]: np.array([row["X"], row["Y"], row["Z"]])
            for _, row in electrodes_df.iterrows()
        }
        montage = make_dig_montage(ch_pos=ch_pos, coord_frame="head")

        events_df = pd.read_csv(path_events, sep="\t")

        # Encode the events
        event_category, mapping, stim_values = self.encoding(events_df=events_df)

        events = self.create_event_array(
            raw=raw, event_category=event_category, stim_values=stim_values
        )

        # Creating and setting annotations from the events
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=mapping
        )

        raw = raw.set_annotations(annotations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Removing the stimulus channels
            if not self.return_all_modalities:
                raw = raw.pick(["eeg", "eog"])
            else:
                # Drop original STI; stim_channels_with_selected_ids adds a clean one
                raw = raw.drop_channels(["STI"])
            # Setting the montage
            raw = raw.set_montage(montage, verbose=False)
        # Loading dataset
        raw = raw.load_data(verbose=False)
        # There is only one session
        sessions = {"0": {"0": stim_channels_with_selected_ids(raw, self.event_id)}}

        return sessions

    def data_infos(self):
        """Returns the data paths of the electrodes and events information

        This function downloads the necessary data files for electrodes
        and events from their respective URLs and returns their local file paths.

        Returns
        -------
        tuple
            A tuple containing the local file paths to the channels, electrodes,
            and events information files.
        """

        path_electrodes = dl.data_dl(LIU2024_ELECTRODES, self.code)

        path_events = dl.data_dl(LIU2024_EVENTS, self.code)

        return path_electrodes, path_events

    @staticmethod
    def _normalize_extension(file_name: str) -> str:
        # Renaming the .tsv file to make sure it's recognized as .tsv
        # Check if the file already has the ".tsv" extension

        file_electrodes_tsv = file_name + ".tsv"

        if not os.path.exists(file_electrodes_tsv):
            # Perform the rename operation only if the target file
            # doesn't exist
            shutil.copy(file_name, file_electrodes_tsv)

        return file_electrodes_tsv

    @staticmethod
    def create_event_array(
        raw: Any, event_category: np.ndarray, stim_values: set
    ) -> np.ndarray:
        """
        This method creates an event array based on the stimulus channel.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw data.
        event_category : :class:`numpy.ndarray`
            The event categories.
        stim_values : set
            Set of STI channel values to select triggers for.

        Returns
        -------
        events : :class:`numpy.ndarray`
            The created events array.
        """
        sti_data = raw.copy().pick("STI").get_data().flatten()
        # Only select triggers matching the requested stage values
        idx_trigger = np.where(np.isin(sti_data, list(stim_values)))[0]
        n_label_stim = len(event_category)
        # Create the events array based on the stimulus channel
        events = np.column_stack(
            (idx_trigger[:n_label_stim], np.zeros_like(event_category), event_category)
        )
        return events
