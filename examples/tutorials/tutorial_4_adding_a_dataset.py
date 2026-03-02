"""
====================================
Tutorial 4: Creating custom datasets
====================================

MOABB provides several ways to integrate a custom dataset, depending on the
format of your data:

1. :class:`~moabb.datasets.base.BaseDataset` — for datasets with arbitrary
   file formats. Requires implementing data downloading and file reading.
2. :class:`~moabb.datasets.base.BaseBIDSDataset` /
   :class:`~moabb.datasets.base.LocalBIDSDataset` — for datasets already
   provided in `BIDS format <https://bids.neuroimaging.io/>`_.
   ``BaseBIDSDataset`` is used for online datasets (only the download step
   needs to be implemented); ``LocalBIDSDataset`` is used for local or private
   datasets with no subclassing required at all.

**BIDS is the preferred format for new datasets in MOABB.**

This tutorial illustrates both approaches.
"""

# Authors: Pedro L. C. Rodrigues, Sylvain Chevallier
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

import mne
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from scipy.io import loadmat, savemat
from sklearn.pipeline import make_pipeline

from moabb.datasets import download as dl
from moabb.datasets.base import BaseBIDSDataset, BaseDataset
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


##############################################################################
# 1. Creating a dataset class from scratch (BaseDataset)
# ======================================================
#
# Creating some Data
# ------------------
#
# To illustrate the creation of a dataset class in MOABB, we first create an
# example dataset saved in .mat file. It contains a single fake recording on
# 8 channels lasting for 150 seconds (sampling frequency 256 Hz). We have
# included the script that creates this dataset and have uploaded it online.
# The fake dataset is available on the
# `Zenodo website <https://zenodo.org/records/14973598>`_


def create_example_dataset():
    """Create a fake example for a dataset."""
    sfreq = 256
    t_recording = 150
    t_trial = 1  # duration of a trial
    intertrial = 2  # time between end of a trial and the next one
    n_chan = 8

    x = np.zeros((n_chan + 1, t_recording * sfreq))  # electrodes + stimulus
    stim = np.zeros(t_recording * sfreq)
    t_offset = 1.0  # offset where the trials start
    n_trials = 40

    rep = np.linspace(0, 4 * t_trial, t_trial * sfreq)
    signal = np.sin(2 * np.pi / t_trial * rep)
    for n in range(n_trials):
        label = n % 2 + 1  # alternate between class 0 and class 1
        tn = int(t_offset * sfreq + n * (t_trial + intertrial) * sfreq)
        stim[tn] = label
        noise = 0.1 * np.random.randn(n_chan, len(signal))
        x[:-1, tn : (tn + t_trial * sfreq)] = label * signal + noise
    x[-1, :] = stim
    return x, sfreq


# Create the fake data
for subject in [1, 2, 3]:
    x, fs = create_example_dataset()
    filename = "subject_" + str(subject).zfill(2) + ".mat"
    mdict = {}
    mdict["x"] = x
    mdict["fs"] = fs
    savemat(filename, mdict)

##############################################################################
# Creating a Dataset Class
# ------------------------
#
# We will create now a dataset class using the fake data simulated with the
# code from above. For this, we first need to import the right classes from
# MOABB:
#
# - ``dl`` is a very useful script that downloads automatically a dataset online
#   if it is not yet available in the user's computer. The script knows where
#   to download the files because we create a global variable telling the URL
#   where to fetch the data.
# - ``BaseDataset`` is the basic class that we overload to create our dataset.
#
# The global variable with the dataset's URL should specify an online
# repository where all the files are stored.

ExampleDataset_URL = "https://zenodo.org/records/14973598"


##############################################################################
# The ``ExampleDataset`` needs to implement only 3 functions:
#
# - ``__init__`` for indicating the parameter of the dataset
# - ``_get_single_subject_data`` to define how to process the data once they
#   have been downloaded
# - ``data_path`` to define how the data are downloaded.


class ExampleDataset(BaseDataset):
    """Dataset used to exemplify the creation of a dataset class in MOABB.

    The data samples have been simulated and has no physiological
    meaning whatsoever.
    """

    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3],
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="ExampleDataset",
            interval=[0, 0.75],
            paradigm="imagery",
            doi="",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_path_list = self.data_path(subject)

        data = loadmat(file_path_list[0])
        x = data["x"]
        fs = data["fs"]
        ch_names = ["ch" + str(i) for i in range(8)] + ["stim"]
        ch_types = ["eeg" for i in range(8)] + ["stim"]
        info = mne.create_info(ch_names, float(np.squeeze(fs)), ch_types)
        raw = mne.io.RawArray(x, info)

        sessions = {}
        sessions["0"] = {}
        sessions["0"]["0"] = raw
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from one subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}/files/subject_0{:d}.mat".format(ExampleDataset_URL, subject)
        path = dl.data_dl(url, "ExampleDataset")
        return [path]  # it has to return a list


##############################################################################
# Using the ExampleDataset
# ------------------------
#
# Now that the `ExampleDataset` is defined, it could be instantiated directly.
# The rest of the code follows the steps described in the previous tutorials.

dataset = ExampleDataset()

paradigm = LeftRightImagery()
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=dataset, overwrite=False, suffix="newdataset"
)
pipelines = {}
pipelines["MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann"))
scores = evaluation.process(pipelines)

print(scores)

##############################################################################
# Pushing on MOABB Github
# -----------------------
#
# If you want to make your dataset available to everyone, you could upload
# your data on public server (like Zenodo or Figshare) and signal that you
# want to add your dataset to MOABB in the  `dedicated issue <https://github.com/NeuroTechX/moabb/issues/1>`_.  # noqa: E501
# You could then follow the instructions on `how to contribute <https://github.com/NeuroTechX/moabb/blob/master/CONTRIBUTING.md>`_  # noqa: E501

##############################################################################
# 2. Creating a BIDS dataset class (BaseBIDSDataset)
# ==================================================
#
# If your dataset is already provided in the
# `BIDS format <https://bids.neuroimaging.io/>`_, you can subclass
# :class:`~moabb.datasets.base.BaseBIDSDataset` instead of
# :class:`~moabb.datasets.base.BaseDataset`.
# The ``BaseBIDSDataset`` base class handles reading the BIDS files
# automatically via ``mne-bids``; you only need to implement the
# ``_download_subject`` method that downloads the data for a single subject
# and returns the local path to the **root** of the BIDS dataset.
#
# Several MOABB datasets already use this approach — for example
# :class:`moabb.datasets.Zhou2016`.
#
# The skeleton below shows the minimal implementation required.
# **Note:** ``ExampleBIDSDataset`` is intentionally non-functional — its
# ``_download_subject`` raises ``NotImplementedError``. Replace it with your
# own download logic before instantiating the class.


class ExampleBIDSDataset(BaseBIDSDataset):
    """Skeleton showing how to wrap an online BIDS dataset.

    Replace ``_download_subject`` with the actual download logic for your
    dataset (e.g. fetching a zip archive from Zenodo and extracting it).
    """

    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3],
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="ExampleBIDSDataset",
            interval=[0, 0.75],
            paradigm="imagery",
            doi="",
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        """Download the BIDS dataset for *subject* and return the BIDS root."""
        # Example (not executed here — replace with your actual download URL):
        #
        #   url = f"https://zenodo.org/records/XXXXX/files/bids_dataset.zip"
        #   bids_root = dl.data_dl(url, "ExampleBIDSDataset")
        #   return bids_root
        raise NotImplementedError("Replace this with your actual download logic.")


##############################################################################
# Using LocalBIDSDataset for local/private BIDS datasets
# -------------------------------------------------------
#
# If you already have a BIDS dataset on your local machine and you do **not**
# want to write a dedicated class, you can use
# :class:`~moabb.datasets.base.LocalBIDSDataset` directly.
# It auto-discovers subjects and sessions from the BIDS directory structure:
#
# .. code-block:: python
#
#    from moabb.datasets.base import LocalBIDSDataset
#
#    dataset = LocalBIDSDataset(
#        bids_root="/path/to/bids/dataset",
#        events={"left_hand": 1, "right_hand": 2},
#        interval=[0, 0.75],
#        paradigm="imagery",
#    )
#
#    paradigm = LeftRightImagery()
#    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])
