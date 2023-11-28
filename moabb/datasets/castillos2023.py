import os.path as osp
import zipfile as z

import mne

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


Castillos2023_URL = "https://zenodo.org/records/8255618"


# Each trial contained 12 cycles of a 2.2 second code
NR_CYCLES_PER_TRIAL = 15

# Codes were presented at a 60 Hz monitor refresh rate
PRESENTATION_RATE = 60


class BaseCastillos2023(BaseDataset):
    """c-VEP and Burst-VEP dataset from Castillos et al. (2023)

    Dataset [1]_ from the study on burst-VEP [2]_.

    .. admonition:: Dataset summary

        =============  =======  =======  ==============================  ===============  ===============  ===========
        Name             #Subj    #Chan     #Trials / class              Trials length    Sampling rate      #Sessions
        =============  =======  =======  ==============================  ===============  ===============  ===========
        Castillos2023       12       32   15 "1"/15 "2"/ 15 "3"/ 15 "4"  0.25s             500Hz                     1
        =============  =======  =======  ==============================  ===============  ===============  ===========

    **Dataset description**

    Participants were comfortably seated and instructed to read and sign the informed consent. EEG data were recorded
    using a BrainProduct LiveAmp 32 active electrodes wet-EEG setup with a sample rate of 500 Hz to record the surface
    brain activity. The 32 electrodes were placed following the 10–20 international system on a BrainProduct Acticap. The
    ground electrode was placed at the FPz electrode location and all electrodes were referenced to the FCz electrode. The
    impedance of all electrodes was brought below 25kΩ prior to recording onset. Once equipped with the EEG system,
    volunteers were asked to focus on four targets that were cued sequentially in a random order for 0.5 s, followed by a
    2.2 s stimulation phase, before a 0.7 s inter-trial period. The cue sequence for each trial was pseudo-random and
    different for each block. After each block, a pause was observed and subjects had to press the space bar to continue.
    The participants were presented with fifteen blocks of four trials for each of the four conditions (burst or msequence ×
    40% or 100%), see Fig. 2 - left. The task was implemented in Python using the Psychopy toolbox.1 The four discs were all
    150 pixels, without borders, and were presented on the following LCD monitor: Dell P2419HC, 1920 × 1080 pixels, 265
    cd/m2, and 60 Hz refresh rate. After completing the experiment and removing the EEG equipment, the participants were
    asked to provide subjective ratings for the different stimuli conditions. These stimuli included burst c-VEP with 100%
    amplitude, burst c-VEP with 40% amplitude, m-sequences with 100% amplitude, and m-sequences with 40% amplitude. Each
    stimulus was presented three times in a pseudo-random order. Following the presentation of each stimulus, participants
    were presented with three 11-points scales and were asked to rate the visual comfort, visual tiredness, and
    intrusiveness using a mouse. In total, participants completed 12 ratings (3 repetitions 𝑥 4 types of stimuli) for
    each of the three scales.

    References
    ----------

    .. [1] Kalou Cabrera Castillos. (2023). 4-class code-VEP EEG data [Data set]. Zenodo.(dataset).
           DOI: https://doi.org/10.5281/zenodo.8255618

    .. [2] Kalou Cabrera Castillos, Simon Ladouce, Ludovic Darmet, Frédéric Dehais. Burst c-VEP Based BCI: Optimizing stimulus
           design for enhanced classification with minimal calibration data and improved user experience,NeuroImage,Volume 284,
           2023,120446,ISSN 1053-8119
           DOI: https://doi.org/10.1016/j.neuroimage.2023.120446

    Notes
    -----

    .. versionadded:: 0.6.0

    """

    def __init__(self, events, sessions_per_subject, code, paradigm, paradigm_type):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=sessions_per_subject,
            events=events,
            code=code,
            interval=(0, 0.25),
            paradigm=paradigm,
            doi="https://doi.org/10.1016/j.neuroimage.2023.120446",
        )
        self.paradigm_type = paradigm_type

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject, self.paradigm_type)

        # Codes
        #################### METTRE LES CODE#########################

        raw = mne.io.read_raw_eeglab(file_path_list[0], preload=True, verbose=False)

        # There is only one session, one trial of 60 subtrials
        sessions = {"0": {}}
        sessions["0"]["0"] = raw

        return sessions

    def data_path(
        self,
        subject,
        paradigm_type,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []

        url = "https://zenodo.org/records/8255618/files/4Class-CVEP.zip"
        path_zip = dl.data_dl(
            url, "4Class-VEP", path="C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD"
        )
        path_folder = "C" + path_zip.strip("4Class-VEP.zip")

        # check if has to unzip
        if not (osp.isdir(path_folder + "4Class-VEP")):
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths.append(
            path_folder
            + "4Class-CVEP/P{:d}/P{:d}_{:s}.set".format(subject, subject, paradigm_type)
        )

        return subject_paths


class CasitllosBurstVEP100(BaseCastillos2023):
    """BurstVEP CasitllosBurstVEP100 dataset.

    .. admonition:: Dataset summary


        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        Name                      #Subj    #Chan    #Classes    #Trials / class    Trials length    Sampling rate      #Sessions
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        CasitllosBurstVEP100       12      32            4      15                 2.2s               500Hz                    1
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={
                "011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000_1": 1,
                "000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000_2": 2,
                "000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000_3": 3,
                "000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111_4": 4,
            },
            sessions_per_subject=1,
            code="CasitllosBurstVEP100",
            paradigm="burstVEP",
            paradigm_type="burst100",
        )


class CasitllosBurstVEP40(BaseCastillos2023):
    """BurstVEP CasitllosBurstVEP40 dataset.

    .. admonition:: Dataset summary


        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        Name                      #Subj    #Chan    #Classes    #Trials / class    Trials length    Sampling rate      #Sessions
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        CasitllosBurstVEP40        12      32            4      15                 2.2s               500Hz                    1
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={
                "011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000011110000000000000000000000000111100000000000000000000000011110000000000000000000001111000000000000000000000011110000000000000000011110000000000000000000111100000000000000111100000_1": 1,
                "000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000000000000000000000000001111000000000000000000011110000000000000011110000000000000111100000000000000001111000000000000000000011110000000000000000000000111100000000000000000011110000_2": 2,
                "000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000000000000000000000111100000000000000000000111100000000000001111000000000000000000111100000000000000000000000011110000000000000000000000001111000000000000000000000000111100000000000_3": 3,
                "000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111000000000000000111100000000000000000000001111000000000000000000000000111100000000000000111100000000000000000000001111000000000001111000000000000000000001111000000000000000000000111_4": 4,
            },
            sessions_per_subject=1,
            code="CasitllosBurstVEP40",
            paradigm="burstVEP",
            paradigm_type="burst40",
        )


class CasitllosCVEP100(BaseCastillos2023):
    """CVEP CasitllosCVEP100 dataset.

    .. admonition:: Dataset summary


        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        Name                      #Subj    #Chan    #Classes    #Trials / class    Trials length    Sampling rate      #Sessions
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        CasitllosCVEP100          12       32            4      15                 2.2s               500Hz                    1
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={
                "111111111111110000111100001111000011001111001100001100111111000000110011111100001100110000001111001100000011001100001100000000001100_1": 1,
                "000011110000000011111111110000000000001111000011001100110011110011000000110000001111110011001111111111000011000000111100001100111111_2": 2,
                "111100000000110000111100000000000011001111001100110011000000111111001111110011001111000000111100111111000000000011111100001100110011_3": 3,
                "111100111100111100111100000011111100000011111111110000110011000011110000000011000000001111111111110011001100001111000011000000110011_4": 4,
            },
            sessions_per_subject=1,
            code="CasitllosBurstVEP100",
            paradigm="cvep",
            paradigm_type="mseq100",
        )


class CasitllosCVEP40(BaseCastillos2023):
    """CVEP CasitllosCVEP40 dataset.

    .. admonition:: Dataset summary


        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        Name                      #Subj    #Chan    #Classes    #Trials / class    Trials length    Sampling rate      #Sessions
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========
        CasitllosCVEP40           12       32            4      15                 2.2s             500Hz                    1
        ====================      =======  =======  ==========  =================  ===============  ===============  ===========

    """

    def __init__(self):
        super().__init__(
            events={
                "111111111111110000111100001111000011001111001100001100111111000000110011111100001100110000001111001100000011001100001100000000001100_1": 1,
                "000011110000000011111111110000000000001111000011001100110011110011000000110000001111110011001111111111000011000000111100001100111111_2": 2,
                "111100000000110000111100000000000011001111001100110011000000111111001111110011001111000000111100111111000000000011111100001100110011_3": 3,
                "111100111100111100111100000011111100000011111111110000110011000011110000000011000000001111111111110011001100001111000011000000110011_4": 4,
            },
            sessions_per_subject=1,
            code="CasitllosBurstVEP40",
            paradigm="cvep",
            paradigm_type="mseq40",
        )
