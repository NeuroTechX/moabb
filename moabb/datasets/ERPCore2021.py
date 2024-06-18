import os
from pathlib import Path
import warnings
from functools import partialmethod
import urllib.request
import zipfile
import shutil

import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne_bids.read import _drop, _from_tsv
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


N170_URL = "https://files.osf.io/v1/resources/pfde9/providers/osfstorage/60060f8ae80d370812a5b15d/?zip="
MMN_URL = "https://files.osf.io/v1/resources/5q4xs/providers/osfstorage/6007896286541a091d14b102/?zip="
N2pc_URL = "https://files.osf.io/v1/resources/yefrq/providers/osfstorage/60077f09ba010908a4892b3a/?zip="
P3_URL = "https://files.osf.io/v1/resources/etdkz/providers/osfstorage/60077b04ba010908a78927e9/?zip="
N_400_URL = "https://files.osf.io/v1/resources/29xpq/providers/osfstorage/6007857286541a092614c5d3/?zip="
ERN_URL = "https://files.osf.io/v1/resources/q6gwp/providers/osfstorage/600df65e75226b017d517f6d/?zip="
LRP_URL = "https://files.osf.io/v1/resources/28e6c/providers/osfstorage/600dffbf327cbe019d7b6a0c/?zip="

class ERPCore2021(BaseDataset):
    """Base dataset class for Lee2019."""

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
            #object=0, texture=1
            events = dict(("stimulus:" + str(i), 1) for i in range(1, 81))
            events.update(dict(("stimulus:" + str(i), 0) for i in range(101, 181)))
            events.update({"response:201": 201, "response:202": 202})
        elif task == "MMN":
            interval = (-0.2, 0.8)
            # deviant: 1, standard: 0
            events = {"stimulus:70": 1, "stimulus:80": 0}
        elif task == "N2pc":
            interval = (-0.2, 0.8)
            events = dict(top=1, bottom=2)
        elif task == "N400":
            interval = (-0.2, 0.8)
            events = dict(related=1, unrelated=2)
        elif task == "P3":
            interval = (-0.2, 0.8)
            events = dict(match=1, no_match=2)
        elif task == "ERN":
            interval = (-0.8, 0.2)
            events = dict(right=1, left=2)
        elif task == "LRP":
            interval = (-0.6, 0.4)
            events = {
                    "response:111": 0,
                    "response:112": 0,
                    "response:121": 0,
                    "response:122": 0,
                    "response:211": 1,
                    "response:212": 1,
                    "response:221": 1,
                    "response:222": 1,
                    }
        else:
            raise ValueError('unknown task "{}"'.format(task))
        self.task = task

        super().__init__(
            subjects=list(range(1, 40 + 1)),
            sessions_per_subject=1,
            events=events,
            code="ERPCore-" + task,
            interval=interval,
            paradigm="p300",
            doi=" ",
        )
        self.task = task

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

        file_path = self.data_path(subject)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Read the subject's raw data
            raw = read_raw_bids(bids_path=file_path, verbose=False)
            raw = raw.set_montage("standard_1020", match_case=False)

            # Shift the stimulus event codes forward in time
            # to account for the LCD monitor delay
            # (26 ms on our monitor, as measured with a photosensor).
            if self.task != "MMN":
                raw.annotations.onset = raw.annotations.onset + 0.026

        raw = read_annotations_core(file_path, raw)
        # There is only one session
        sessions = {"0": {"0": raw}}

        return sessions

    # returns bids path is it okay ?
    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = self.TASK_URLS.get(self.task)
        bids_root = download_and_extract(url, self.code)

        bids_path = BIDSPath(
            subject=str(subject),
            task=self.task,
            suffix="eeg",
            session="1",
            datatype="eeg",
            root=bids_root,
        )

        # layout = BIDSLayout(bids_root)
        # sub = f"{subject:03d}"
        # subject_paths = [eeg_file.path for eeg_file in layout.get(subject=f"{subject:03d}", extension='set')]

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
        extracted_path : str
            Local path to the extracted data directory.
        """
        # Set the default path if none is provided
        if path is None:
            path = Path(os.getenv(f'MNE_DATASETS_{sign.upper()}_PATH', Path.home() / 'mne_data'))
        else:
            path = Path(path)
            
        # Construct the destination paths
        key_dest = f"MNE-{sign.lower()}-data"
        destination = path / key_dest / Path(url).name
        local_zip_path = str(destination)
        extract_path = path / key_dest / 'extracted'

        # Download the file using urllib
        if not destination.is_file() or force_update:
            if destination.is_file():
                destination.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)

            with urllib.request.urlopen(url) as response, open(local_zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        # Extract the contents of the zip file
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        return str(extract_path)

    def encoding_to_events(self):

        # Cars: 0, Faces: 1
        custom_mapping = dict(("stimulus:" + str(i), 1) for i in range(0, 41))     # 0 -> 40 ? bcz we start from 0 
        custom_mapping.update(dict(("stimulus:" + str(i), 0) for i in range(41, 81)))
        custom_mapping.update({"response:201": 201, "response:202": 202})

        #object=0, texture=1
        custom_mapping = dict(("stimulus:" + str(i), 1) for i in range(1, 81))
        custom_mapping.update(dict(("stimulus:" + str(i), 0) for i in range(101, 181)))
        custom_mapping.update({"response:201": 201, "response:202": 202})


        

def read_annotations_core(bids_path, raw):
    events_path = os.path.join(
        bids_path.directory,
        bids_path.update(suffix="events", extension=".tsv").basename,
    )
    return _handle_events_reading_core(events_path, raw)

def _handle_events_reading_core(events_path, raw):
    """Read associated events.tsv and populate raw.
    Handle onset, duration, and description of each event.
    """
    events_dict = _from_tsv(events_path)
    
    # Get the descriptions of the events
    descriptions = np.asarray(
        [
            a + ":" + b
            for a, b in zip(events_dict["trial_type"], events_dict["value"])
        ],
        dtype=str,
    )

        events_df = pd.read_csv(events_path, sep="\t")

        # Encode the events
        event_category, mapping = self.encoding(events_df=events_df)

        events = self.create_event_array(raw=raw, event_category=event_category)

        # Creating and setting annotations from the events
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=mapping
        )
    

    # 
    ons = [on for on in events_dict["onset"]]
    dus = [du for du in events_dict["duration"]]
    onsets = np.asarray(ons, dtype=float)
    durations = np.asarray(dus, dtype=float)

    # Add Events to raw as annotations
    annot_from_events = mne.Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=None
    )
    raw.set_annotations(annot_from_events)

    return raw


def _handle_events_reading_core(events_fname, raw):
    """Read associated events.tsv and populate raw.
    Handle onset, duration, and description of each event.
    """
    events_dict = _from_tsv(events_fname)

    if ("value" in events_dict) and ("trial_type" in events_dict):
        events_dict = _drop(events_dict, "n/a", "trial_type")
        events_dict = _drop(events_dict, "n/a", "value")

        descriptions = np.asarray(
            [
                a + ":" + b
                for a, b in zip(events_dict["trial_type"], events_dict["value"])
            ],
            dtype=str,
        )

        # Get the descriptions of the events
    elif "trial_type" in events_dict:

        # Drop events unrelated to a trial type
        events_dict = _drop(events_dict, "n/a", "trial_type")
        descriptions = np.asarray(events_dict["trial_type"], dtype=str)

    # If we don't have a proper description of the events, perhaps we have
    # at least an event value?
    elif "value" in events_dict:
        # Drop events unrelated to value
        events_dict = _drop(events_dict, "n/a", "value")
        descriptions = np.asarray(events_dict["value"], dtype=str)
    # Worst case, we go with 'n/a' for all events
    else:
        descriptions = "n/a"

    # Deal with "n/a" strings before converting to float
    ons = [np.nan if on == "n/a" else on for on in events_dict["onset"]]
    dus = [0 if du == "n/a" else du for du in events_dict["duration"]]
    onsets = np.asarray(ons, dtype=float)
    durations = np.asarray(dus, dtype=float)
    # Keep only events where onset is known
    good_events_idx = ~np.isnan(onsets)
    onsets = onsets[good_events_idx]
    durations = durations[good_events_idx]
    descriptions = descriptions[good_events_idx]
    del good_events_idx
    # Add Events to raw as annotations
    annot_from_events = mne.Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=None
    )
    raw.set_annotations(annot_from_events)

    return raw



class ERPCore2021_N170(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "N170")


class ERPCore2021_MMN(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "MMN")


class ERPCore2021_N2pc(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "N2pc")


class ERPCore2021_P3(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "P3")

class ERPCore2021_N400(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "N400")


class ERPCore2021_ERN(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "ERN")


class ERPCore2021_LRP(ERPCore2021):
    """ """

    __init__ = partialmethod(ERPCore2021.__init__, "LRP")
