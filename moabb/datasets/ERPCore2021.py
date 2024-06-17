from functools import partialmethod

import os
import warnings
import mne
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne_bids.read import _drop, _from_tsv

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

from mne_bids import BIDSPath, read_raw_bids

from bids import BIDSLayout

ERPCore_URL = ""

N170_URL = "https://files.osf.io/v1/resources/pfde9/providers/osfstorage/60060f8ae80d370812a5b15d/?zip="
MMN_URL = "https://files.osf.io/v1/resources/5q4xs/providers/osfstorage/6007896286541a091d14b102/?zip="

class ERPCore2021(BaseDataset):
    """Base dataset class for Lee2019."""
    
    TASK_URLS = {
        "N170": N170_URL,
        "MMN": MMN_URL,
    }

    def __init__(
        self,
        task
    ):

        if task == "N170":
            interval = (-0.2, 0.8)
            events = dict(object=1, texture=2)
        elif task == "MMN":
            interval = (-0.2, 0.8)
            events = dict(standard_tone =1, deviant_tone=2)
        elif task == "N2pc":
            interval = (-0.2, 0.8)
            events = dict( top = 1, bottom= 2)
        elif task == "N400":
            interval = (-0.2, 0.8)
            events = dict( related = 1, unrelated= 2)
        elif task == "P3":
            interval = (-0.2, 0.8)
            events = dict( match = 1, no_match= 2)
        elif task == "ERN":
            interval = (-0.8, 0.2)
            events = dict( right = 1, left= 2)
        elif task == "LRP":
            interval = (-0.6, 0.4)
            events = dict( right = 1, left= 2)     
        else:
            raise ValueError('unknown task "{}"'.format(task))
        self.task = task

        super().__init__(
            subjects=list(range(1, 40 + 1)),
            sessions_per_subject=1,
            events=events,
            code="ERPCore-" + task,
            interval=interval,
            paradigm= "p300",
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
        bids_root = dl.data_dl(url, self.code, path, force_update, verbose)

        bids_path = BIDSPath(
            subject=str(subject),
            task=self.task,
            suffix='eeg',
            session='1',
            datatype='eeg',
            root=bids_root,
        ) 
         
        #layout = BIDSLayout(bids_root)
        #sub = f"{subject:03d}"
        #subject_paths = [eeg_file.path for eeg_file in layout.get(subject=f"{subject:03d}", extension='set')]

        subject_paths = [bids_path]

        return subject_paths



def read_annotations_core(bids_path, raw):
    tsv = os.path.join(
        bids_path.directory,
        bids_path.update(suffix="events", extension=".tsv").basename,
    )
    return _handle_events_reading_core(tsv, raw)

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
    """

    """

    __init__ = partialmethod(ERPCore2021.__init__, "N170")


class ERPCore2021_MMN(ERPCore2021):
    """
    """

    __init__ = partialmethod(ERPCore2021.__init__, "MMN")


class ERPCore2021_N2pc(ERPCore2021):
    """

    """

    __init__ = partialmethod(ERPCore2021.__init__, "N2pc")

class ERPCore2021_P3(ERPCore2021):
    """

    """

    __init__ = partialmethod(ERPCore2021.__init__, "P3")

class ERPCore2021_ERN(ERPCore2021):
    """

    """

    __init__ = partialmethod(ERPCore2021.__init__, "ERN")

class ERPCore2021_LRP(ERPCore2021):
    """

    """

    __init__ = partialmethod(ERPCore2021.__init__, "LRP")
