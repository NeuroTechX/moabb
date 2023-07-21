"""
===============================
Convert a MOABB dataset to BIDS
===============================

The Brain Imaging Data Structure (BIDS) format
is a standard format for storing neuroimaging data.
It follows fixed principles to facilitate the
sharing of neuroimaging data between researchers.

The MOABB library allows to convert any dataset to
BIDS. This example shows how this can be done.

We will use the AlexMI dataset as example
because it is relatively small and can be
downloaded quickly.

"""
# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import shutil
import tempfile
from pathlib import Path

import mne

from moabb import set_log_level
from moabb.datasets import AlexMI


set_log_level("info")

###############################################################################
# Basic usage
# -----------
#
# The conversion of any MOABB dataset to a BIDS-compliant structure can be done
# by simply calling its ``get_data`` method. Here we will save the BIDS version
# of the dataset in a temporary folder:
temp_dir = Path(tempfile.mkdtemp())
dataset = AlexMI()
_ = dataset.get_data(cache_config=dict(path=temp_dir, save_raw=True))


###############################################################################
# Before / after the folder structure
# -----------------------------
#
# To investigate what was saved, we will first define a function to print
# the folder structure of a given path:
def print_tree(p: Path, last=True, header=""):
    elbow = "└──"
    pipe = "│  "
    tee = "├──"
    blank = "   "
    print(header + (elbow if last else tee) + p.name)
    if p.is_dir():
        children = list(p.iterdir())
        for i, c in enumerate(children):
            print_tree(
                c, header=header + (blank if last else pipe), last=i == len(children) - 1
            )


###############################################################################
# Now, we will retrieve the location of the original dataset. It is stored
# in the MNE data directory, which can be found with the ``"MNE_DATA"`` key:
mne_data = Path(mne.get_config("MNE_DATA"))
print(f"MNE data directory: {mne_data}")

###############################################################################
# Now, we can print the folder structure of the original dataset:
print("Before conversion:")
print_tree(mne_data / "MNE-alexeeg-data")

###############################################################################
# As we can see, before conversion all the data (i.e. from all subjects,
# sessions and runs) is stored in a single folder. This follows no particular
# standard and can vary from one dataset to another.
#
# After conversion, the data is stored in a BIDS-compliant structure:
print("After conversion:")
print_tree(temp_dir / "MNE-alexandre motor imagery-bids-cache")

###############################################################################
# In the BIDS version of our dataset, the raw files are saved in EDF.
# The data is organized in a hierarchy of folders,
# starting with the subjects, then the sessions, then the runs. Metadata files
# are stored to describe the data. For more details on the BIDS structure,
# please refer to the `BIDS website <https://bids.neuroimaging.io>`_ and the
# `BIDS specification <https://bids-specification.readthedocs.io/en/stable/>`_.
#
# Under the hood, saving datasets to BIDS is done through the caching system
# of MOABB. Only raw EEG files are officially supported by the BIDS specification.
# However, MOABB's caching mechanism also offers the possibility to save the data
# in a pseudo-BIDS after different preprocessing steps. In particular, we can save
# :func:`mne.Epochs` and ``np.ndarray`` objects.  For more details on the caching system,
# please refer to the `disk cache tutorial :doc:`example_disk_cache`.
#
# Cleanup
# -------
#
# Finally, we can delete the temporary folder:
shutil.rmtree(temp_dir)
