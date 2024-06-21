import os
import shutil
import urllib.request
import warnings
import zipfile
from abc import abstractmethod
from functools import partialmethod
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from moabb.datasets.base import BaseDataset

# Links to the raw BIDS format data for each dataset
N170_URL = "https://files.osf.io/v1/resources/pfde9/providers/osfstorage/60060f8ae80d370812a5b15d/?zip="
MMN_URL = "https://files.osf.io/v1/resources/5q4xs/providers/osfstorage/6007896286541a091d14b102/?zip="
N2pc_URL = "https://files.osf.io/v1/resources/yefrq/providers/osfstorage/60077f09ba010908a4892b3a/?zip="
P3_URL = "https://files.osf.io/v1/resources/etdkz/providers/osfstorage/60077b04ba010908a78927e9/?zip="
N_400_URL = "https://files.osf.io/v1/resources/29xpq/providers/osfstorage/6007857286541a092614c5d3/?zip="
ERN_URL = "https://files.osf.io/v1/resources/q6gwp/providers/osfstorage/600df65e75226b017d517f6d/?zip="
LRP_URL = "https://files.osf.io/v1/resources/28e6c/providers/osfstorage/600dffbf327cbe019d7b6a0c/?zip="


class ERPCore2021(BaseDataset):
    """Base dataset class for ERPCore2021."""

    TASK_URLS = {
        "N170": N170_URL,
        "MMN": MMN_URL,
        "N2pc": N2pc_URL,
        "P3": P3_URL,
        "N400": N_400_URL,
        "ERN": ERN_URL,
        "LRP": LRP_URL,
    }

    def __init__(self, task):

        if task == "N170":
            interval = (-0.2, 0.8)
            events = {
                "Stimulus - car - normal": 0,
                "Stimulus - car - scrambled": 1,
                "Stimulus - face - normal": 2,
                "Stimulus - face - scrambled": 3,
                "Response - correct": 4,
                "Response - error": 5,
            }
            original_mapping = {
                "1-40": "Stimulus - faces",
                "41-80": "Stimulus - cars",
                "101-140": "Stimulus - scrambled faces",
                "141-180": "Stimulus - scrambled cars",
                "201": "Response - correct",
                "202": "Response - error",
            }
        elif task == "MMN":
            interval = (-0.2, 0.8)
            events = {"Stimulus - deviant:70": 0, "Stimulus - standard:80": 1}
            original_mapping = {
                "80": "Stimulus - standard",
                "70": "Stimulus - deviant",
                "180": "Stimulus - first stream of standards",
            }
        elif task == "N2pc":
            interval = (-0.2, 0.8)
            events = {"Target left": 0, "Target right": 1, "response_correct":2, "response_error":3}
            original_mapping = {
                "111": "Stimulus - target blue, target left, gap at top",
                "112": "Stimulus - target blue, target left, gap at bottom",
                "121": "Stimulus - target blue, target right, gap at top",
                "122": "Stimulus - target blue, target right, gap at bottom",
                "211": "Stimulus - target pink, target left, gap at top",
                "212": "Stimulus - target pink, target left, gap at bottom",
                "221": "Stimulus - target pink, target right, gap at top",
                "222": "Stimulus - target pink, target right, gap at bottom",
                "201": "Response - correct",
                "202": "Response - error",
            }
        elif task == "P3":
            interval = (-0.2, 0.8)
            events = dict(match=0, no_match=1, response_correct=2, response_error=3)
            original_mapping = {
                "11": "Stimulus - block target A, trial stimulus A",
                "21": "Stimulus - block target B, trial stimulus A",
                "31": "Stimulus - block target C, trial stimulus A",
                "41": "Stimulus - block target D, trial stimulus A",
                "51": "Stimulus - block target E, trial stimulus A",
                "12": "Stimulus - block target A, trial stimulus B",
                "22": "Stimulus - block target B, trial stimulus B",
                "32": "Stimulus - block target C, trial stimulus B",
                "42": "Stimulus - block target D, trial stimulus B",
                "52": "Stimulus - block target E, trial stimulus B",
                "13": "Stimulus - block target A, trial stimulus C",
                "23": "Stimulus - block target B, trial stimulus C",
                "33": "Stimulus - block target C, trial stimulus C",
                "43": "Stimulus - block target D, trial stimulus C",
                "53": "Stimulus - block target E, trial stimulus C",
                "14": "Stimulus - block target A, trial stimulus D",
                "24": "Stimulus - block target B, trial stimulus D",
                "34": "Stimulus - block target C, trial stimulus D",
                "44": "Stimulus - block target D, trial stimulus D",
                "54": "Stimulus - block target E, trial stimulus D",
                "15": "Stimulus - block target A, trial stimulus E",
                "25": "Stimulus - block target B, trial stimulus E",
                "35": "Stimulus - block target C, trial stimulus E",
                "45": "Stimulus - block target D, trial stimulus E",
                "55": "Stimulus - block target E, trial stimulus E",
                "201": "Response - correct",
                "202": "Response - error"
            }
        elif task == "N400":
            interval = (-0.2, 0.8)
            events = dict(related=0, unrelated=1, response_correct=2, response_error=3)
            original_mapping = {
                "111": "Stimulus - prime word, related word pair, list 1",
                "112": "Stimulus - prime word, related word pair, list 2",
                "121": "Stimulus - prime word, unrelated word pair, list 1",
                "122": "Stimulus - prime word, unrelated word pair, list 2",
                "211": "Stimulus - target word, related word pair, list 1",
                "212": "Stimulus - target word, related word pair, list 2",
                "221": "Stimulus - target word, unrelated word pair, list 1",
                "222": "Stimulus - target word, unrelated word pair, list 2",
                "201": "Response - correct",
                "202": "Response - error"
            }
        elif task == "ERN":
            interval = (-0.8, 0.2)
            events = dict(right=1, left=2) # right-response /correct : 1, right-response /error : 2, left-response /correct : 3, left-response /error : 4
            original_mapping = {
                "11": "Stimulus - compatible flankers, target left",
                "12": "Stimulus - compatible flankers, target right",
                "21": "Stimulus - incompatible flankers, target left",
                "22": "Stimulus - incompatible flankers, target right",
                "111": "Response - left, compatible flankers, target left",
                "112": "Response - left, compatible flankers, target right",
                "121": "Response - left, incompatible flankers, target left",
                "122": "Response - left, incompatible flankers, target right",
                "211": "Response - right, compatible flankers, target left",
                "212": "Response - right, compatible flankers, target right",
                "221": "Response - right, incompatible flankers, target left",
                "222": "Response - right, incompatible flankers, target right"
            }
        elif task == "LRP":
            interval = (-0.6, 0.4)
            events = dict(right=1, left=2)
            original_mapping = {
                "11": "Stimulus - compatible flankers, target left",
                "12": "Stimulus - compatible flankers, target right",
                "21": "Stimulus - incompatible flankers, target left",
                "22": "Stimulus - incompatible flankers, target right",
                "111": "Response - left, compatible flankers, target left",
                "112": "Response - left, compatible flankers, target right",
                "121": "Response - left, incompatible flankers, target left",
                "122": "Response - left, incompatible flankers, target right",
                "211": "Response - right, compatible flankers, target left",
                "212": "Response - right, compatible flankers, target right",
                "221": "Response - right, incompatible flankers, target left",
                "222": "Response - right, incompatible flankers, target right"
            }                                                                                  
        else:
            raise ValueError('unknown task "{}"'.format(task))
        self.task = task
        self.meta_info = original_mapping
        super().__init__(
            subjects=list(range(1, 40 + 1)),
            sessions_per_subject=1,
            events=events,
            code="ERPCore-" + task,
            interval=interval,
            paradigm="p300",
            doi="10.1016/j.neuroimage.2020.117465",
        )

    def get_meta_data(self, subject):
        """
        Retrieve original events mapping and original event data for a given subject.

        Parameters
        ----------
        subject : int
            The subject number for which to retrieve data.

        Returns
        -------
        tuple
            A tuple containing the original events mapping and the original events DataFrame.
        """
        # Get the path to the events file for the subject
        events_path = self.events_path(subject)
        # Read the events data
        original_events = pd.read_csv(events_path, sep="\t")

        return self.meta_info, original_events

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

    # returns bids path is it okay ?
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
            raise (ValueError("Invalid subject number"))

        # Get the URL for the task data
        url = self.TASK_URLS.get(self.task)
        # Download and extract the dataset
        bids_root = self.download_and_extract(url, self.task.lower())  # self.code

        # Create a BIDSPath object for the subject
        bids_path = BIDSPath(
            subject=f"{subject:03d}",
            task=self.task,
            suffix="eeg",
            datatype="eeg",
            root=bids_root,
        )

        subject_paths = [bids_path]

        return subject_paths

    @staticmethod
    def download_and_extract(url, sign, path=None, force_update=False, verbose=None):
        """Download and extract a zip file dataset from the given url.

        Parameters
        ----------
        url : str
            Path to the remote location of data.
        sign : str
            Signifier of dataset.
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see :func:`mne.verbose`).

        Returns
        -------
        str
            Local path to the extracted data directory.
        """
        # Set the default path if none is provided
        if path is None:
            path = Path(
                os.getenv(f"MNE_DATASETS_{sign.upper()}_PATH", Path.home() / "mne_data")
            )
        else:
            path = Path(path)

        # Construct the destination paths
        key_dest = f"MNE-{sign.lower()}-data"
        destination = path / key_dest / f"{sign}-raw-data-BIDS"
        local_zip_path = str(destination)
        extract_path = path / key_dest / "extracted"

        # Check if the zip file and extracted directory already exist
        if destination.is_file() and extract_path.exists() and not force_update:
            return str(extract_path)

        # Download the file using urllib if it doesn't already exist or force_update is True
        if not destination.is_file() or force_update:
            if destination.is_file():
                destination.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading {url} to {local_zip_path}")
            with urllib.request.urlopen(url) as response, open(
                local_zip_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
        print(f"Extracting {local_zip_path} to {extract_path}")
        # Extract the contents of the zip file
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        return str(extract_path)

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

        # Create the event array using the sample column and the encoded event categories
        events = np.column_stack(
            (events_df["sample"].values, np.zeros(len(event_category)), event_category)
        )

        # Create and set annotations from the events
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=mapping
        )
        raw.set_annotations(annotations)

        return raw

    @abstractmethod
    def encode_event(row):
        """
        Encode a single event values based on the task specific criteria.

        Parameters
        ----------
        row : pd.Series
            A row of the events DataFrame.

        Returns
        -------
        str
            Encoded event value.
        """
        pass

    @abstractmethod
    def encoding(self, events_df):
        """
        Encode the columns value in the events DataFrame.

        Parameters
        ----------
        events_df : pd.DataFrame
            DataFrame containing the events information.

        Returns
        -------
        tuple
            A tuple containing the encoded event values and the mapping dictionary.
        """
        pass


class ERPCore2021_N170(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "N170")

    @staticmethod
    def encode_event(row):
        """
        Encode a single event values based on the task specific criteria.

        Parameters
        ----------
        row : pd.Series
            A row of the events DataFrame.

        Returns
        -------
        str
            Encoded event value.
        """

        value = row["value"]
        if 1 <= value <= 80:
            return f"00{value:02d}"
        elif 101 <= value <= 180:
            return f"01{value-100:02d}"
        elif value == 201:
            return "11"
        elif value == 202:
            return "10"
        else:
            return "Unknown"

    def encoding(self, events_df):
        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            f"00{val:02d}": f"Stimulus - {desc}"
            for val, desc in zip(range(1, 81), ["faces"] * 40 + ["cars"] * 40)
        }
        mapping.update(
            {
                f"01{val:02d}": f"Stimulus - scrambled {desc}"
                for val, desc in zip(range(1, 81), ["faces"] * 40 + ["cars"] * 40)
            }
        )
        mapping.update({"11": "Response - correct", "10": "Response - error"})

        return encoded_column.values, mapping


class ERPCore2021_MMN(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "MMN")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # Standard stimulus
        if value in {80, 180}:
            return "01"
        # Deviant stimulus
        elif value == 70:
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


class ERPCore2021_N2pc(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "N2pc")

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
            "2": "Stimulus - target left",
        }

        return encoded_column.values, mapping


class ERPCore2021_P3(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "P3")

    @staticmethod
    # keeping only the stimulus without the response
    def encode_event(row):
        value = row["value"]
        if value in {11, 22, 33, 44, 55}:
            return "1"
        elif value in {21, 31, 41, 51, 12, 32, 42, 52, 13, 23, 43, 53, 14, 24, 34, 54, 15, 25, 35, 45}:
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


class ERPCore2021_N400(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "N400")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # Related word pair
        if value in {211, 212}:
            return "1"
        # Unrelated word pair
        elif value in {221, 222}:
            return "2"
        return value

    def encoding(self, events_df):

        # Drop rows corresponding to the responses
        events_df.drop(events_df[events_df["value"].isin([111, 112, 121, 122, 201, 202])].index, inplace=True)

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Stimulus - related word pair",
            "2": "Stimulus - unrelated word pair",
        }

        return encoded_column.values, mapping


class ERPCore2021_ERN(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "ERN")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # correct
        if value in {111, 121, 212, 222}:
            return "1"
        # incorrect
        elif value in {112, 122, 211, 221}:
            return "2"
        return value

    def encoding(self, events_df):

        # Keep rows corresponding to the responses
        events_df.drop(events_df[~events_df["value"].isin([111, 112, 121, 122, 211, 212, 221, 222])].index, inplace=True)

        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Correct response",
            "2": "Incorrect response",
        }

        return encoded_column.values, mapping

class ERPCore2021_LRP(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "LRP")

    @staticmethod
    def encode_event(row):
        value = row["value"]
        # left
        if value in {111, 112, 121, 122}:
            return "1"
        # right
        elif value in {211, 212, 221, 222}:
            return "2"
        return value

    def encoding(self, events_df):

        # Keep rows corresponding to the responses
        events_df.drop(events_df[~events_df["value"].isin([111, 112, 121, 122, 211, 212, 221, 222])].index, inplace=True)


        # Apply the encoding function to each row
        encoded_column = events_df.apply(self.encode_event, axis=1)

        # Create the mapping dictionary
        mapping = {
            "1": "Response - left",
            "2": "Response - right",
        }

        return encoded_column.values, mapping