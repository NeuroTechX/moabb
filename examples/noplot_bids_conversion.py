"""
===============================
Convert a MOABB dataset to BIDS
===============================

The Brain Imaging Data Structure (BIDS) format
is standard for storing neuroimaging data.
It follows fixed principles to facilitate the
sharing of neuroimaging data between researchers.

The MOABB library allows to convert any MOABB dataset to
BIDS [1]_ and [2]_.

In this example, we will convert the AlexMI dataset to BIDS using the
option ``cache_config=dict(path=temp_dir, save_raw=True)`` of the ``get_data``
method from the dataset object.

This will automatically save the raw data in the BIDS format and allow to use
a cache for the next time the dataset is used.

We will use the AlexMI dataset [3]_, one of the smallest in
people and one that can be downloaded quickly.
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
# Here, we will save the BIDS version of the dataset in a temporary folder
temp_dir = Path(tempfile.mkdtemp())
# The conversion of any MOABB dataset to a BIDS-compliant structure can be done
# by simply calling its ``get_data`` method and using the ``cache_config``
# parameter. This parameter is a dictionary.
dataset = AlexMI()
_ = dataset.get_data(cache_config=dict(path=temp_dir, save_raw=True))


###############################################################################
# Before / after folder structure
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
# As we can see, before conversion, all the data (i.e. from all subjects,
# sessions and runs) is stored in a single folder. This follows no particular
# standard and can vary from one dataset to another.
#
# After conversion, the data is stored in a BIDS-compliant way:
print("After conversion:")
print_tree(temp_dir / "MNE-BIDS-alexandre-motor-imagery")

###############################################################################
# In the BIDS version of our dataset, the raw files are saved in EDF.
# The data is organized in a hierarchy of folders,
# starting with the subjects, then the sessions, and then the runs. Metadata
# files are stored to describe the data. For more details on the BIDS
# structure, please refer to the `BIDS website <https://bids.neuroimaging.io>`_
# and the `BIDS spec <https://bids-specification.readthedocs.io/en/stable/>`_.
#
# Under the hood, saving datasets to BIDS is done through the caching system
# of MOABB. Only raw EEG files are officially supported by the BIDS
# specification.
# However, MOABB's caching mechanism also offers the possibility to save
# the data in a pseudo-BIDS after different preprocessing steps.
# In particular, we can save :class:`mne.Epochs` and ``np.ndarray`` objects.
# For more details on the caching system,
# please refer to the tutorial :doc:`./plot_disk_cache`.
#
# Cleanup
# -------
#
# Finally, we can delete the temporary folder:
shutil.rmtree(temp_dir)

###############################################################################
# References
# -----------
#
# .. [1] Pernet, C.R., Appelhoff, S., Gorgolewski, K.J. et al. EEG-BIDS,
#        An extension to the brain imaging data structure for
#        electroencephalography. Sci Data 6, 103 (2019).
#        https://doi.org/10.1038/s41597-019-0104-8
#
# .. [2] Appelhoff et al., (2019). MNE-BIDS: Organizing electrophysiological
#        data into the BIDS format and facilitating their analysis.
#        Journal of Open Source Software, 4(44), 1896,
#        https://doi.org/10.21105/joss.01896
#
# .. [3] Barachant, A., 2012. Commande robuste d'un effecteur par une
#        interface cerveau machine EEG asynchrone (Doctoral dissertation,
#        Université de Grenoble).
#        https://tel.archives-ouvertes.fr/tel-01196752
