from functools import partialmethod

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

from bids import BIDSLayout

ERPCore_URL = ""

N170_URL = "https://files.osf.io/v1/resources/pfde9/providers/osfstorage/60060f8ae80d370812a5b15d/?zip="
MMN_URL = "https://files.osf.io/v1/resources/5q4xs/providers/osfstorage/6007896286541a091d14b102/?zip="

class ERPCore2021(BaseDataset):
    """Base dataset class for Lee2019."""

    def __init__(
        self,
        task
    ):

        if task == "N170":
            interval = (0, 0.3)
            events = dict(object=1, texture=2)
        elif task == "MMN":
            interval = (0, 0.1)
            events = dict(standard_tone =1, deviant_tone=2)
        elif task == "N2pc":
            interval = (0, 0.5)
            events = dict( top = 1, bottom= 2)
        elif task == "N400":
            interval = (0, 0.2)
            events = dict( related = 1, unrelated= 2)
        elif task == "P3":
            interval = (0, 0.2)
            events = dict( match = 1, no_match= 2)
        elif task in ["ERN", "LRP"]:
            interval = (0, 0.2)
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


    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = self.task + "_URL"
        bids_root = dl.data_dl(url, self.code, path, force_update, verbose)

        layout = BIDSLayout(bids_root)
        sub = f"{subject:03d}"
        subject_paths = [eeg_file.path for eeg_file in layout.get(subject=f"{subject:03d}", extension='set')]

        return subject_paths

    

class ERPCore2021_N170(ERPCore):
    """

    """

    __init__ = partialmethod(ERPCore.__init__, "N170")


class ERPCore2021_MMN(ERPCore):
    """
    """

    __init__ = partialmethod(ERPCore.__init__, "MMN")


class ERPCore2021_N2pc(ERPCore):
    """

    """

    __init__ = partialmethod(ERPCore.__init__, "N2pc")

class ERPCore2021_P3(ERPCore):
    """

    """

    __init__ = partialmethod(ERPCore.__init__, "P3")

class ERPCore2021_ERN(ERPCore):
    """

    """

    __init__ = partialmethod(ERPCore.__init__, "ERN")

class ERPCore2021_LRP(ERPCore):
    """

    """

    __init__ = partialmethod(ERPCore.__init__, "LRP")
