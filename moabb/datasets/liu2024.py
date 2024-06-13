import os
import zipfile as z
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.channels import read_custom_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


# Link to the raw data
LIU2024_URL = "https://figshare.com/ndownloader/files/38516654"

# Links to the channels, electrodes and events information files
url_channels = "https://figshare.com/ndownloader/files/38516069"
url_electrodes = "https://figshare.com/ndownloader/files/38516078"
url_events = "https://figshare.com/ndownloader/files/38516084"


class Liu2024(BaseDataset):
    """

    Dataset [1]_ from the study on motor imagery [2]_.

    .. admonition:: Dataset summary


        ============  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ============  =======  =======  ==========  =================  ============  ===============  ===========
        Liu2024        50       29         2                40              4s           500Hz           1
        ============  =======  =======  ==========  =================  ============  ===============  ===========


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

    .. [1] Liu, Haijie; Lv, Xiaodong (2022). EEG datasets of stroke patients. figshare. Dataset.
           DOI: https://doi.org/10.6084/m9.figshare.21679035.v5

    .. [2] Liu, Haijie, Wei, P., Wang, H. et al. An EEG motor imagery dataset for brain computer interface in acute stroke
           patients. Sci Data 11, 131 (2024).
           DOI: https://doi.org/10.1038/s41597-023-02787-8

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events={"left_hand": 0, "right_hand": 1},
            code="Liu2024",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.1038/s41597-023-02787-8",
        )

    def data_infos(self):
        """Returns the data paths of the electrodes and events information

        This function downloads the necessary data files for channels, electrodes,
        and events from their respective URLs and returns their local file paths.

        Returns
        -------
        tuple
            A tuple containing the local file paths to the channels, electrodes, and events information files.
        """

        path_channels = dl.data_dl(url_channels, self.code)
        path_electrodes = dl.data_dl(url_electrodes, self.code)
        path_events = dl.data_dl(url_events, self.code)

        return path_channels, path_electrodes, path_events

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject, in our case only one path is returned.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None, the environment
            variable or config parameter MNE_DATASETS_(dataset)_PATH is used. If it doesn’t exist,
            the “~/mne_data” directory is used. If the dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config to the given path.
            If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list containing the path to the subject's data file. The list contains only one path.
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

    def encoding(self, events_df: pd.DataFrame):
        """Encode the columns 'value' and 'trial_type' in the events file into a single event type.

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
        'trial_type' can take values  { 1 : Left hand, 2 : Right hand }, but for convenience we use 0 and 1.
        'value' can take values { 1 : instructions, 2 : MI, 3 : break}.
        For example, if trial_type = 1 and value = 2, the event type will be 12.
        If trial_type = 0 and value = 2, the event type will be 2.
        """

        event_type = events_df["value"].values + (events_df["trial_type"].values - 1) * 10

        return event_type

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

        file_path_list = self.data_path(subject)
        path_channels, path_electrodes, path_events = self.data_infos()

        # Read the subject's raw data
        raw = mne.io.read_raw_edf(file_path_list[0], preload=False)

        # Selecting the EEG channels and the STIM channel excluding the CPz
        # reference channel and the EOG channels
        selected_channels = raw.info["ch_names"][:-3] + [""]
        selected_channels.remove("CPz")
        raw = raw.pick(selected_channels)

        # Updating the types of the channels after extracting them from the channels file
        channels_info = pd.read_csv(path_channels, sep="\t")
        channel_types = [type.lower() for type in channels_info["type"].tolist()[:-2]] + [
            "stim"
        ]
        channel_dict = dict(zip(selected_channels, channel_types))
        raw.info.set_channel_types(channel_dict)

        # Renaming the .tsv file to make sure it's recognized as .tsv
        # Check if the file already has the ".tsv" extension
        if not path_electrodes.endswith(".tsv"):
            # Create the new path
            new_path_electrodes = path_electrodes + ".tsv"
            # Check if the target filename already exists
            if not os.path.exists(new_path_electrodes):
                # Perform the rename operation only if the target file doesn't exist
                os.rename(path_electrodes, new_path_electrodes)
                path_electrodes = new_path_electrodes
            else:
                # If the file already exists, simply keep the original path
                path_electrodes = new_path_electrodes

        # Read and set the montage
        montage = read_custom_montage(path_electrodes)
        raw.set_montage(montage, on_missing="ignore")

        events_df = pd.read_csv(path_events, sep="\t")

        # Convert onset from milliseconds to seconds
        onset_seconds = events_df["onset"].values / 1000

        # Encode the events
        event_type = self.encoding(events_df)

        # Convert onset from seconds to samples for the events array
        sfreq = raw.info["sfreq"]
        onset_samples = (onset_seconds * sfreq).astype(int)

        # Create the events array
        events = np.column_stack(
            (onset_samples, np.zeros_like(onset_samples), event_type)
        )

        # Creating and setting annotations from the events
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=event_type
        )
        raw.set_annotations(annotations)

        # There is only one session
        sessions = {"0": {"run_1": raw}}

        return sessions
