"""
Erpcore2021 dataset
"""

import json
import os
import warnings
from abc import abstractmethod
from functools import partialmethod
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pooch
from mne.datasets import fetch_dataset
from mne_bids import BIDSPath, read_raw_bids

from moabb.datasets.base import BaseDataset


OSF_BASE_URL = "https://files.osf.io/v1/resources/"

# Ids for the buckets on OSF and the folder OSF hash.
OSF_IDS = {
    "ERN": ["q6gwp", "600df65e75226b017d517f6d"],
    "LRP": ["28e6c", "600dffbf327cbe019d7b6a0c"],
    "MMN": ["5q4xs", "6007896286541a091d14b102"],
    "N170": ["pfde9", "60060f8ae80d370812a5b15d"],
    "N2pc": ["yefrq", "60077f09ba010908a4892b3a"],
    "N400": ["29xpq", "6007857286541a092614c5d3"],
    "P3": ["etdkz", "60077b04ba010908a78927e9"],
}


DATASET_PARAMS = {
    task: dict(
        archive_name=f"ERPCORE2021_{task}.zip",
        url=OSF_BASE_URL + f"{osf[0]}/providers/osfstorage/{osf[1]}/?zip=",
        folder_name=f"MNE-erpcore{task.lower()}2021-data",
        dataset_name=f"MNE-erpcore{task.lower()}2021",
        hash=None,
        config_key=f"MNE_ERPCORE_{task.upper()}_PATH",  # I need to check this
    )
    for task, osf in OSF_IDS.items()
}


class Erpcore2021(BaseDataset):
    """Abstract base dataset class for Erpcore2021.

    Datasets [1]_ from the study [2]_.

    **Dataset Description**

    The ERP CORE dataset includes data from 40 neurotypical young adults
    (25 female, 15 male; Mean years of age = 21.5, SD = 2.87, Range 18–30; 38 right handed)
    from the University of California. Each participant had native English competence and normal
    color perception, normal or corrected-to-normal vision, and no history of neurological injury
    or disease (as indicated by self-report). They participated in six 10-minutes optimized
    experiments designed to measure seven widely used ERP components: N170, Mismatch Negativity
    (MMN), N2pc, N400, P3, Lateralized Readiness Potential (LRP), and Error-Related Negativity
    (ERN). These experiments were conducted to standardize ERP paradigms and protocols across
    studies.

    **Experimental procedures**:
    - **N170**: Subjects viewed faces and objects to elicit the N170 component. In this task,
    an image of a face, car, scrambled face, or scrambled car was presented on each trial in
    the center of the screen, and participants responded whether the stimulus was an “object”
    (face or car) or a “texture” (scrambled face or scrambled car).
    - **MMN**: Subjects were exposed to a sequence of auditory stimuli to evoke the mismatch
    negativity response, indicating automatic detection of deviant sounds.  Standard tones
    (presented at 80 dB, with p = .8) and deviant tones (presented at 70 dB, with p = .2)
    were presented over speakers while participants watched a silent video and ignored the tones.
    - **N2pc**: Participants were given a target color of pink or blue at the beginning of a
    trial block, and responded on each trial whether the gap in the target color square was
    on the top or bottom.
    - **N400**: On each trial, a red prime word was followed by a green target word.
    Participants responded whether the target word was semantically related or unrelated
    to the prime word.
    - **P3**: The letters A, B, C, D, and E were presented in random order (p = .2 for each
    letter). One letter was designated the target for a given block of trials, and the other
    4 letters were non-targets. Thus, the probability of the target category was .2, but the
    same physical stimulus served as a target in some blocks and a nontarget in others.
    Participants responded whether the letter presented on each trial was the target or a
    non-target for that block.
    - **LRP & ERN**: A central arrowhead pointing to the left or right was flanked on both
    sides by arrowheads that pointed in the same direction (congruent trials) or the opposite
    direction (incongruent trials). Participants indicated the direction of the central
    arrowhead on each trial with a left- or right-hand buttonpress.


    The continuous EEG was recorded using a Biosemi ActiveTwo recording system with active
    electrodes (Biosemi B.V., Amsterdam, the Netherlands). Recording from 30 scalp electrodes,
    mounted in an elastic cap and placed according to the International 10/20 System (FP1, F3,
    F7, FC3, C3, C5, P3, P7, P9, PO7, PO3, O1, Oz, Pz, CPz, FP2, Fz, F4, F8, FC4, FCz, Cz, C4,
    C6, P4, P8, P10, PO8, PO4, O2; see Supplementary Fig. S1). The common mode sense (CMS)
    electrode was located at PO1, and the driven right leg (DRL) electrode was located at PO2.
    The horizontal electrooculogram (HEOG) was recorded from electrodes placed lateral to the
    external canthus of each eye. The vertical electrooculogram (VEOG) was recorded from an
    electrode placed below the right eye. Signals were incidentally also recorded from 37 other
    sites, but these sites were not monitored during the recording and are not included in
    the ERP CORE data set. All signals were low-pass filtered using a fifth order sinc filter
    with a half-power cutoff at 204.8 Hz and then digitized at 1024 Hz with 24 bits of resolution.
    The signals were recorded in single-ended mode (i.e., measuring the voltage between the active
    and ground electrodes without the use of a reference), and referencing was performed offline.

    References
    ----------
    .. [1] Emily S. Kappenman, Jaclyn L. Farrens, Wendy Zhang, Andrew X. Stewart, Steven J. Luck.
        (2020). ERP CORE: An open resource for human event-related potential research. NeuroImage.
        DOI: https://doi.org/10.1016/j.neuroimage.2020.117465

    .. [2] Emily S. Kappenman, Jaclyn L. Farrens, Wendy Zhang, Andrew X. Stewart, Steven J. Luck.
        ERP CORE: An open resource for human event-related potential research.
        DOI: https://doi.org/10.1016/j.neuroimage.2020.117465

    """

    def __init__(self, task):
        if task == "N170":
            interval = (-0.2, 0.8)
            events = {
                "Stimulus - car - normal": 1,
                "Stimulus - car - scrambled": 2,
                "Stimulus - face - normal": 3,
                "Stimulus - face - scrambled": 4,
                "Response - correct": 5,
                "Response - error": 6,
            }
        elif task == "MMN":
            interval = (-0.2, 0.8)
            events = {"Stimulus - deviant:70": 1, "Stimulus - standard:80": 2}
        elif task == "N2pc":
            interval = (-0.2, 0.8)
            events = {
                "Target left": 0,
                "Target right": 1,
                "response_correct": 2,
                "response_error": 3,
            }
        elif task == "P3":
            interval = (-0.2, 0.8)
            events = dict(match=1, no_match=2, response_correct=3, response_error=4)

        elif task == "N400":
            interval = (-0.2, 0.8)
            events = dict(related=1, unrelated=2, response_correct=3, response_error=4)

        elif task == "ERN":
            interval = (-0.8, 0.2)
            events = dict(right=1, left=2)
        elif task == "LRP":
            interval = (-0.6, 0.4)
            events = dict(right=1, left=2)
        else:
            raise ValueError(f"Unknown task {task}")

        self.task = task
        # self.meta_info = None

        super().__init__(
            subjects=list(range(1, 40 + 1)),
            sessions_per_subject=1,
            events=events,
            code=f"Erpcore2021-{task}",
            interval=interval,
            paradigm="p300",
            doi="10.1016/j.neuroimage.2020.117465",
        )

    # def load_meta_info(self):
    #    """
    #    Load original value mapping from a JSON file.
    #    """
    #    file_path = self.data_path(1)[0]
    #    json_file = pd.read_json(file_path)
    #    self.meta_info = json_file["value"]["Levels"]

    def get_meta_data(self, subject):
        """
        Retrieve original events mapping and original event data for a given subject.

        Parameters
        ----------
        subject : int
            The subject number for which to retrieve data.

        Returns ------- tuple A tuple containing the original events mapping
        and the original events DataFrame.
        """
        # Get the path to the events file for the subject
        events_path = self.events_path(subject)
        # Read the events data
        original_events = pd.read_csv(events_path, sep="\t")

        file_path = self.data_path(1)[0]

        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

        # Extract the value mapping
        meta_info = data["value"]["Levels"]

        # if self.meta_info is not None:
        #    return self.meta_info, original_events

        return meta_info, original_events

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
        file_path = self.data_path(subject)[0]

        # Load the meta_data
        # self.load_meta_info()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Read the subject's raw data and set the montage
            raw = read_raw_bids(bids_path=file_path, verbose=False)
            raw = raw.set_montage("standard_1020", match_case=False)

        # Shift the stimulus event codes forward in time
        # to account for the LCD monitor delay
        # (26 ms on our monitor, as measured with a photosensor).
        if self.task != "MMN":
            raw.annotations.onset = raw.annotations.onset + 0.026

        events_path = self.events_path(subject)
        raw = self.handle_events_reading(events_path, raw)

        # There is only one session
        sessions = {"0": {"0": raw}}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """
        Return the data BIDS paths of a single subject.

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
            A list containing the BIDSPath object for the subject's data file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_and_extract(
            path=path, force_update=force_update, update_path=update_path
        )

        # Create a BIDSPath object for the subject
        bids_path = BIDSPath(
            subject=f"{subject:03d}",
            task=self.task,
            suffix="eeg",
            datatype="eeg",
            root=dataset_path,
        )

        subject_paths = [bids_path]

        return subject_paths

    def download_and_extract(self, path=None, force_update=False, update_path=None):
        """
        Download and extract the dataset.

        Parameters
        ----------
        path : str | None
            The path to the directory where the dataset should be downloaded.
            If None, the default directory is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path: bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config
            to the given path.

        Returns
        -------
        path : str
            The dataset path.
        """
        if path is not None:
            path = Path(path) / DATASET_PARAMS[self.task]["folder_name"]
        else:
            path = Path.home() / "mne_data" / DATASET_PARAMS[self.task]["folder_name"]

        # Check if the dataset already exists
        if not force_update and path.exists():
            print(f"Dataset already exists at {path}. Skipping download.")
            return path

        fetch_dataset(
            DATASET_PARAMS[self.task],
            path=path,
            force_update=force_update,
            update_path=update_path,
            processor=pooch.Unzip(extract_dir=path),
        )
        return path

    def events_path(self, subject):
        """
        Get the path to the events file for a given subject.

        Parameters
        ----------
        subject : int
            The subject number for which to get the events file path.

        Returns
        -------
        str
            The path to the events file.
        """
        # Get the BIDSPath object for the subject
        bids_path = self.data_path(subject)[0]
        # Construct the path to the events file
        events_path = os.path.join(
            bids_path.directory,
            bids_path.update(suffix="events", extension=".tsv").basename,
        )
        return events_path

    def handle_events_reading(self, events_path, raw):
        """Read associated events.tsv and populate raw with annotations.

        Parameters
        ----------
        events_path : str
            The path to the events file.
        raw : mne.io.Raw
            The raw EEG data object.

        Returns
        -------
        mne.io.Raw
            The updated raw EEG data object with annotations.
        """

        events_df = pd.read_csv(events_path, sep="\t")

        # Encode the events
        event_category, mapping = self.encoding(events_df=events_df)

        # Create the event array using the sample column and the encoded
        # event categories
        events = np.column_stack(
            (events_df["sample"].values, np.zeros(len(event_category)), event_category)
        )

        # Create and set annotations from the events
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=mapping
        )
        raw.set_annotations(annotations)

        return raw

    @staticmethod
    @abstractmethod
    def encode_event(row: str):
        """
        Encode a single event values based on the task-specific criteria.

        Parameters
        ----------
        row : pd.Series
            A row of the events DataFrame.

        Returns
        -------
        str
            Encoded event value.
        """

    @abstractmethod
    def encoding(self, events_df: pd.DataFrame):
        """
        Encode the column value in the events DataFrame.

        Parameters
        ----------
        events_df : DataFrame
            DataFrame containing the events information.

        Returns
        -------
        tuple
            A tuple containing the encoded event values and the mapping dictionary.
        """


class Erpcore2021_N170(Erpcore2021):
    """
    .. admonition:: Dataset summary


        ================  =======  =======  =================  ===============  ===============  ===========
        Name                #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
        ================  =======  =======  =================  ===============  ===============  ===========
        Erpcore2021_N170       40       30                80               1s           1024Hz            1
        ================  =======  =======  =================  ===============  ===============  ===========

    """

    __init__ = partialmethod(Erpcore2021.__init__, "N170")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # Stimulus - faces and cars  :  Stimulus - faces : 1 - 40 and Stimulus - cars : 41 - 80
        if 1 <= value <= 80:
            return f"00{value:02d}"
        # Stimulus - scrambled faces and cars : Stimulus - scrambled faces : 101 - 140 and Stimulus - scrambled cars : 141 - 180
        if 101 <= value <= 180:
            return f"01{value - 100:02d}"
        # Response - correct
        if value == 201:
            return "11"
        # Response - error
        if value == 202:
            return "10"

        return value

    def encoding(self, events_df):
        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        # Stimulus - faces : 1 - 40 and Stimulus - cars : 41 - 80
        mapping = {
            f"00{val:02d}": f"Stimulus - {desc}"
            for val, desc in zip(range(1, 81), ["faces"] * 40 + ["cars"] * 40)
        }
        # Stimulus - scrambled faces : 101 - 140 and Stimulus - scrambled cars : 141 - 180
        mapping.update(
            {
                f"01{val:02d}": f"Stimulus - scrambled {desc}"
                for val, desc in zip(range(1, 81), ["faces"] * 40 + ["cars"] * 40)
            }
        )
        mapping.update({"11": "Response - correct", "10": "Response - error"})

        return encoded_column.values, mapping


class Erpcore2021_MMN(Erpcore2021):
    """
    ================  =======  =======  =========================  ===============  ===============  ===========
    Name                #Subj    #Chan  #Trials / class            Trials length    Sampling rate     #Sessions
    ================  =======  =======  =========================  ===============  ===============  ===========
    Erpcore2021_MMN       40       30   800 Deviant/200 Standard              1s           1024Hz            1
    ================  =======  =======  =========================  ===============  ===============  ===========


    """

    __init__ = partialmethod(Erpcore2021.__init__, "MMN")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # Standard stimulus
        if value in {80, 180}:
            return "01"
        # Deviant stimulus
        if value == 70:
            return "02"
        return value

    def encoding(self, events_df):
        # Remove first and last rows, which correspond to trial_type STATUS
        events_df.drop([0, len(events_df) - 1], inplace=True)
        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "01": "Stimulus - standard",
            "02": "Stimulus - deviant",
        }

        return encoded_column.values, mapping


class Erpcore2021_N2pc(Erpcore2021):
    """
    ================  =======  =======  =================  ===============  ===============  ===========
    Name                #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
    ================  =======  =======  =================  ===============  ===============  ===========
    Erpcore2021_N2pc       40       30               160               1s           1024Hz            1
    ================  =======  =======  =================  ===============  ===============  ===========

    """

    __init__ = partialmethod(Erpcore2021.__init__, "N2pc")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        if value in {111, 112, 211, 212}:
            return "1"
        elif value in {121, 122, 221, 222}:
            return "2"
        return value

    def encoding(self, events_df):

        # Drop rows corresponding to the responses
        events_df.drop(events_df[events_df["value"].isin([201, 202])].index, inplace=True)

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Stimulus - target left",
            "2": "Stimulus - target right",
        }

        return encoded_column.values, mapping


class Erpcore2021_P3(Erpcore2021):
    """
    ================  =======  =======  =================  ===============  ===============  ===========
    Name                #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
    ================  =======  =======  =================  ===============  ===============  ===========
    Erpcore2021_P3       40       30      160 NT / 40 T                1s           1024Hz            1
    ================  =======  =======  =================  ===============  ===============  ===========

    """

    __init__ = partialmethod(Erpcore2021.__init__, "P3")

    @staticmethod
    # keeping only the stimulus without the response
    def encode_event(row):
        value = row["value"]
        if value in {11, 22, 33, 44, 55}:
            return "1"
        if value in {
            21,
            31,
            41,
            51,
            12,
            32,
            42,
            52,
            13,
            23,
            43,
            53,
            14,
            24,
            34,
            54,
            15,
            25,
            35,
            45,
        }:
            return "2"
        return value

    def encoding(self, events_df):

        # Drop rows corresponding to the responses
        events_df.drop(events_df[events_df["value"].isin([201, 202])].index, inplace=True)

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary

        mapping = {
            "1": "Target",
            "2": "NonTarget",
        }

        return encoded_column.values, mapping


class Erpcore2021_N400(Erpcore2021):
    """
    ================  =======  =======  =================  ===============  ===============  ===========
    Name                #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
    ================  =======  =======  =================  ===============  ===============  ===========
    Erpcore2021_N400       40       30                60               1s           1024Hz            1
    ================  =======  =======  =================  ===============  ===============  ===========

    """

    __init__ = partialmethod(Erpcore2021.__init__, "N400")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # Related word pair
        if value in {211, 212}:
            return "1"
        # Unrelated word pair
        if value in {221, 222}:
            return "2"
        return value

    def encoding(self, events_df):

        # Drop rows corresponding to the responses
        events_df.drop(
            events_df[events_df["value"].isin([111, 112, 121, 122, 201, 202])].index,
            inplace=True,
        )

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Stimulus - related word pair",
            "2": "Stimulus - unrelated word pair",
        }

        return encoded_column.values, mapping


class Erpcore2021_ERN(Erpcore2021):
    """
    ================  =======  =======  =================  ===============  ===============  ===========
    Name                #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
    ================  =======  =======  =================  ===============  ===============  ===========
    Erpcore2021_ERN       40       30            Depends               1s           1024Hz            1
    ================  =======  =======  =================  ===============  ===============  ===========

    """

    __init__ = partialmethod(Erpcore2021.__init__, "ERN")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # correct
        if value in {111, 121, 212, 222}:
            return "1"
        # incorrect
        if value in {112, 122, 211, 221}:
            return "2"

        return value

    def encoding(self, events_df):

        # Keep rows corresponding to the responses
        events_df.drop(
            events_df[
                ~events_df["value"].isin([111, 112, 121, 122, 211, 212, 221, 222])
            ].index,
            inplace=True,
        )

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Correct response",
            "2": "Incorrect response",
        }

        return encoded_column.values, mapping


class Erpcore2021_LRP(Erpcore2021):
    """
    ================  =======  =======  =================  ===============  ===============  ===========
    Name                #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
    ================  =======  =======  =================  ===============  ===============  ===========
    Erpcore2021_LRP       40       30            Depends               1s           1024Hz            1
    ================  =======  =======  =================  ===============  ===============  ===========

    """

    __init__ = partialmethod(Erpcore2021.__init__, "LRP")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # left
        if value in {111, 112, 121, 122}:
            return "1"
        # right
        if value in {211, 212, 221, 222}:
            return "2"
        return value

    def encoding(self, events_df):

        # Keep rows corresponding to the responses
        events_df.drop(
            events_df[
                ~events_df["value"].isin([111, 112, 121, 122, 211, 212, 221, 222])
            ].index,
            inplace=True,
        )

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Response - left",
            "2": "Response - right",
        }

        return encoded_column.values, mapping
