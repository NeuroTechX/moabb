import zipfile as z
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.utils import add_stim_channel_epoch, add_stim_channel_trial
from mne.channels import read_custom_montage


_LIU2024_URL = (
    "https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5"
)


class Liu2024(BaseDataset):
    """

    Dataset [1]_ from the study on motor imagery [2]_.

    .. admonition:: Dataset summary


        ============  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ============  =======  =======  ==========  =================  ============  ===============  ===========
        BNCI2014_001       50       33           2                40   4s            500Hz                      1
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

    .. [2] Liu, H., Wei, P., Wang, H. et al. An EEG motor imagery dataset for brain computer interface in acute stroke
           patients. Sci Data 11, 131 (2024).
           DOI: https://doi.org/10.1038/s41597-023-02787-8

    Notes
    -----

    .. versionadded:: 1.1.0

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events={ "left_hand": 0, "right_hand": 1},
            code="Liu2024",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.6084/m9.figshare.21679035.v5",
        )
    
    def data_infos(self):
        """Returns the data paths of the channels, electrodes and events informations"""
        
        url_electrodes = "https://figshare.com/ndownloader/files/38516078"
        url_channels = "https://figshare.com/ndownloader/files/38516069"
        url_events = "https://figshare.com/ndownloader/files/38516084"
        
        path_channels = dl.data_dl(url_channels, self.code + "channels" )  
        path_electrodes = dl.data_dl(url_electrodes, self.code + "electrodes" )
        path_events = dl.data_dl(url_events, self.code + "events" )

        return path_channels, path_electrodes, path_events


    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []

        url = "https://figshare.com/ndownloader/files/38516654"
        path_zip = dl.data_dl(url, self.code)
        path_zip = Path(path_zip)
        path_folder = path_zip.parent

        if not (path_folder / "edffile").is_dir():
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)
        sub = f"sub-{subject:02d}"

        subject_path = (
            path_folder / "edffile" / sub / "eeg" / f"{sub}_task-motor-imagery_eeg.edf"
        )
        subject_paths.append(str(subject_path))

        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)
        path_channels, path_electrodes, path_events = self.data_infos()

        # Read the subject's data
        raw = mne.io.read_raw_edf(file_path_list[0], preload=False)
        
        # Selecting the EEG channels and the STIM channel
        selected_channels = raw.info['ch_names'][:-3] + ['']
        selected_channels.remove("CPz")

        raw = raw.pick(selected_channels)

        # Updating the types of the channels 
        channel_types = channel_types[:-2] + ['stim']
        channel_dict = dict(zip(selected_channels, channel_types))
        raw.info.set_channel_types(channel_dict)

        montage = read_custom_montage(path_electrodes)
        raw.set_montage(montage, on_missing='ignore')

        # There is only one session
        sessions = {"0": {}}
        sessions["0"] = raw

        return sessions