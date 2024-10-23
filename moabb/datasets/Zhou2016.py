"""Simple and compound motor imagery.

https://doi.org/10.1371/journal.pone.0114853
"""

import os
import shutil
import zipfile as z

import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_cnt
from pooch import retrieve

from .base import BaseDataset
from .download import get_dataset_path


DATA_PATH = "https://ndownloader.figshare.com/files/3662952"


def local_data_path(base_path, subject):
    if not os.path.isdir(os.path.join(base_path, "subject_{}".format(subject))):
        if not os.path.isdir(os.path.join(base_path, "data")):
            retrieve(DATA_PATH, None, fname="data.zip", path=base_path, progressbar=True)
            with z.ZipFile(os.path.join(base_path, "data.zip"), "r") as f:
                f.extractall(base_path)
            os.remove(os.path.join(base_path, "data.zip"))
        datapath = os.path.join(base_path, "data")
        for i in range(1, 5):
            os.makedirs(os.path.join(base_path, "subject_{}".format(i)))
            for session in range(1, 4):
                for run in ["A", "B"]:
                    os.rename(
                        os.path.join(datapath, "S{}_{}{}.cnt".format(i, session, run)),
                        os.path.join(
                            base_path,
                            "subject_{}".format(i),
                            "{}{}.cnt".format(session, run),
                        ),
                    )
        shutil.rmtree(os.path.join(base_path, "data"))
    subjpath = os.path.join(base_path, "subject_{}".format(subject))
    return [
        [os.path.join(subjpath, "{}{}.cnt".format(y, x)) for x in ["A", "B"]]
        for y in ["1", "2", "3"]
    ]


class Zhou2016(BaseDataset):
    """Motor Imagery dataset from Zhou et al 2016.

    Dataset from the article *A Fully Automated Trial Selection Method for
    Optimization of Motor Imagery Based Brain-Computer Interface* [1]_.
    This dataset contains data recorded on 4 subjects performing 3 type of
    motor imagery: left hand, right hand and feet.

    Every subject went through three sessions, each of which contained two
    consecutive runs with several minutes inter-run breaks, and each run
    comprised 75 trials (25 trials per class). The intervals between two
    sessions varied from several days to several months.

    A trial started by a short beep indicating 1 s preparation time,
    and followed by a red arrow pointing randomly to three directions (left,
    right, or bottom) lasting for 5 s and then presented a black screen for
    4 s. The subject was instructed to immediately perform the imagination
    tasks of the left hand, right hand or foot movement respectively according
    to the cue direction, and try to relax during the black screen.

    References
    ----------

    .. [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated
           Trial Selection Method for Optimization of Motor Imagery Based
           Brain-Computer Interface. PLoS ONE 11(9).
           https://doi.org/10.1371/journal.pone.0162657
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 5)),
            sessions_per_subject=3,
            events=dict(left_hand=1, right_hand=2, feet=3),
            code="Zhou2016",
            # MI 1-6s, prepare 0-1, break 6-10
            # boundary effects
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1371/journal.pone.0162657",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        files = self.data_path(subject)

        out = {}
        for sess_ind, runlist in enumerate(files):
            sess_key = str(sess_ind)
            out[sess_key] = {}
            for run_ind, fname in enumerate(runlist):
                run_key = str(run_ind)
                raw = read_raw_cnt(fname, preload=True, eog=["VEOU", "VEOL"])
                stim = raw.annotations.description.astype(np.dtype("<10U"))
                stim[stim == "1"] = "left_hand"
                stim[stim == "2"] = "right_hand"
                stim[stim == "3"] = "feet"
                raw.annotations.description = stim
                out[sess_key][run_key] = self._create_stim_channels(raw)
                out[sess_key][run_key].set_montage(make_standard_montage("standard_1005"))
        return out

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("ZHOU", path)
        basepath = os.path.join(path, "MNE-zhou-2016")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)
        return local_data_path(basepath, subject)

    @staticmethod
    def _create_stim_channels(raw):
        # Define a consistent mapping from event descriptions to integer IDs
        desired_event_id = dict(left_hand=1, right_hand=2, feet=3)

        # Get events using the consistent event_id mapping
        events, _ = mne.events_from_annotations(raw, event_id=desired_event_id)

        # Filter the events array to include only desired events
        desired_event_ids = list(desired_event_id.values())
        filtered_events = events[np.isin(events[:, 2], desired_event_ids)]

        # Create annotations from filtered events using the inverted mapping
        event_desc = {v: k for k, v in desired_event_id.items()}
        annot_from_events = mne.annotations_from_events(
            events=filtered_events,
            event_desc=event_desc,
            sfreq=raw.info["sfreq"],
            orig_time=raw.info["meas_date"],
        )
        raw.set_annotations(annot_from_events)

        # Create the stim channel data array
        stim_channs = np.zeros((1, raw.n_times))
        for event in filtered_events:
            sample_index = event[0]
            event_code = event[2]  # Consistent event IDs
            stim_channs[0, sample_index] = event_code

        # Create the stim channel and add it to raw
        stim_channel_name = "STIM"
        stim_info = mne.create_info(
            [stim_channel_name], sfreq=raw.info["sfreq"], ch_types=["stim"]
        )
        stim_raw = mne.io.RawArray(stim_channs, stim_info, verbose=False)
        raw_with_stim = raw.copy().add_channels([stim_raw], force_update_info=True)

        return raw_with_stim
