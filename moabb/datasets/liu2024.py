"""Liu2024 Motor imagery dataset."""

import os
import shutil
import warnings
import zipfile as z
from pathlib import Path
from typing import Any, Dict, Tuple

import mne
import numpy as np
import pandas as pd
from mne.channels import read_custom_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


# Link to the raw data
LIU2024_URL = "https://figshare.com/ndownloader/files/38516654"

# Links to the electrodes and events information files
LIU2024_ELECTRODES = "https://figshare.com/ndownloader/files/38516078"
LIU2024_EVENTS = "https://figshare.com/ndownloader/files/38516084"


class Liu2024(BaseDataset):
    """

    Dataset [1]_ from the study on motor imagery [2]_.

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

    def __init__(self, break_events=False, instr_events=False):
        self.break_events = break_events
        self.instr_events = instr_events
        events = {"left_hand": 1, "right_hand": 2}
        if break_events:
            events["instr"] = 3
        if instr_events:
            events["break"] = 4
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events=events,
            code="Liu2024",
            interval=(2, 6),
            paradigm="imagery",
            doi="10.1038/s41597-023-02787-8",
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
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths = []
        sub = f"sub-{subject:02d}"

        # Construct the path to the subject's data file
        subject_path = (
            path_folder / "edffile" / sub / "eeg" / f"{sub}_task-motor-imagery_eeg.edf"
        )
        subject_paths.append(str(subject_path))

        return subject_paths

    def encoding(self, events_df: pd.DataFrame) -> Tuple[np.array, Dict[int, str]]:
        """Encode the columns 'value' and 'trial_type' into a single event type.

        Parameters
        ----------
        events_df : pd.DataFrame
            DataFrame containing the events information.

        Returns
        -------
        np.ndarray
            Array of encoded event types.

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
            (2, 2): 1,  # Left hand, MI
            (1, 2): 2,  # Right hand, MI
        }

        mapping = {
            1: "left_hand",
            2: "right_hand",
        }

        if self.instr_events:
            encoding_mapping.update(
                {
                    (1, 1): 3,  # Right hand, instructions
                    (2, 1): 3,  # Left hand, instructions
                }
            )
            mapping[3] = "instr"

        if self.break_events:
            encoding_mapping.update(
                {
                    (1, 3): 4,  # Right hand, break
                    (2, 3): 4,  # Left hand, break
                }
            )
            mapping[4] = "break"

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

        return event_category, mapping

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

        # Dropping reference channels with constant values
        raw = raw.drop_channels(["CPz"])

        # Renaming channels accurately
        raw.rename_channels({"HEOR": "VEOR", "": "STI"})

        # Create a dictionary with the channel names and their new types
        mapping = {"STI": "stim", "VEOR": "eog", "HEOL": "eog"}

        # Set the new channel types
        raw.set_channel_types(mapping)

        # Normalize and Read the montage
        path_electrodes = self._normalize_extension(path_electrodes)
        # Read and set the montage
        montage = read_custom_montage(path_electrodes)

        events_df = pd.read_csv(path_events, sep="\t")

        # Encode the events
        event_category, mapping = self.encoding(events_df=events_df)

        events = self.create_event_array(raw=raw, event_category=event_category)

        # Creating and setting annotations from the events
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=mapping
        )

        raw = raw.set_annotations(annotations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Removing the stimulus channels
            raw = raw.pick(["eeg", "eog"])
            # Setting the montage
            raw = raw.set_montage(montage, verbose=False)
        # Loading dataset
        raw = raw.load_data(verbose=False)
        # There is only one session
        sessions = {"0": {"0": raw}}

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
    def create_event_array(raw: Any, event_category: np.ndarray) -> np.ndarray:
        """
        This method creates an event array based on the stimulus channel.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw data.
        event_category : np.ndarray
            The event categories.

        Returns
        -------
        events : np.ndarray
            The created events array.
        """
        _, idx_trigger = np.nonzero(raw.copy().pick("STI").get_data())
        n_label_stim = len(event_category)
        # Create the events array based on the stimulus channel
        events = np.column_stack(
            (idx_trigger[:n_label_stim], np.zeros_like(event_category), event_category)
        )
        return events
