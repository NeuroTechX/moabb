"""
=================================================
Cache on disk intermediate data processing states
=================================================

This example shows how intermediate data processing
states can be cached on disk to speed up the loading
of this data in subsequent calls.

When a MOABB paradigm processes a dataset, it will
first apply processing steps to the raw data, this is
called the ``raw_pipeline``. Then, it will convert the
raw data into epochs and apply processing steps on the
epochs, this is called the ``epochs_pipeline``.
Finally, it will eventually convert the epochs into arrays,
this is called the ``array_pipeline``. In summary:

``raw_pipeline`` --> ``epochs_pipeline`` --> ``array_pipeline``

After each step, MOABB offers the possibility to save on disk
the result of the step. This is done by setting the ``cache_config``
parameter of the paradigm's ``get_data`` method.
The ``cache_config`` parameter is a dictionary that can take all
the parameters of ``moabb.datasets.base.CacheConfig`` as keys,
they are the following: ``use``, ``save_raw``, ``save_epochs``,
``save_array``, ``overwrite_raw``, ``overwrite_epochs``,
``overwrite_array``, and ``path``.  You can also directly pass a
``CacheConfig`` object as ``cache_config``.

If ``use=False``, the ``save_*`` and ``overwrite_*``
parameters are ignored.

When trying to use the cache (i.e. ``use=True``), MOABB will
first check if there exist a cache of the result of the full
pipeline (i.e. ``raw_pipeline`` --> ``epochs_pipeline`` ->
``array_pipeline``).
If there is none, we remove the last step of the pipeline and
look for its cached result. We keep removing steps and looking
for a cached result until we find one or until we reach an
empty pipeline.
Every time, if the ``overwrite_*`` parameter
of the corresponding step is true, we first try to erase the
cache of this step.
Once a cache has been found or the empty pipeline has been reached,
depending on the case we either load the cache or the original dataset.
Then, apply the missing steps one by one and save their result
if their corresponding ``save_*`` parameter is true.

By default, only the result of the ``raw_pipeline`` is saved.
This is usually a good compromise between speed and disk space
because, when using cached raw data, the epochs can be obtained
without preloading the whole raw signals, only the necessary
intervals. Yet, because only the raw data is cached, the epoching
parameters can be changed without creating a new cache each time.
However, if your epoching parameters are fixed, you can directly
cache the epochs or the arrays to speed up the loading and
reduce the disk space used.

.. note::
    The ``cache_config`` parameter is also available for the ``get_data``
    method of the datasets. It works the same way as for a
    paradigm except that it will save un-processed raw recordings.
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import shutil
import tempfile

###############################################################################
import time
from pathlib import Path

from moabb import set_log_level
from moabb.datasets import Zhou2016
from moabb.paradigms import LeftRightImagery


set_log_level("info")

###############################################################################
# Basic usage
# -----------
#
# The ``cache_config`` parameter is a dictionary that has the
# following default values:
default_cache_config = dict(
    save_raw=False,
    save_epochs=False,
    save_array=False,
    use=False,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
    path=None,
)

###############################################################################
# You don not need to specify all the keys of ``cache_config``, only the ones
# you want to change.
#
# By default, the cache is saved at the MNE data directory (i.e. when
# ``path=None``).  The MNE data directory can be found with
# ``mne.get_config('MNE_DATA')``. For this example, we will save it  in a
# temporary directory instead:
temp_dir = Path(tempfile.mkdtemp())

###############################################################################
# We will use the Zhou2016 dataset and the LeftRightImagery paradigm in this
# example, but this works for any dataset and paradigm pair.:
dataset = Zhou2016()
paradigm = LeftRightImagery()

###############################################################################
# And we will only use the first subject for this example:
subjects = [1]

###############################################################################
# Then, saving a cache can simply be done by setting the desired parameters
# in the ``cache_config`` dictionary:
cache_config = dict(
    use=True,
    save_raw=True,
    save_epochs=True,
    save_array=True,
    path=temp_dir,
)
_ = paradigm.get_data(dataset, subjects, cache_config=cache_config)

###############################################################################
# Time comparison
# ---------------
#
# Now, we will compare the time it takes to load the with different levels of
# cache. For this, we will use the cache saved in the previous block and
# overwrite the steps results one by one so that we can compare the time it
# takes to load the data and compute the missing steps with an increasing
# number of missing steps.
#
# Using array cache:
cache_config = dict(
    use=True,
    path=temp_dir,
    save_raw=False,
    save_epochs=False,
    save_array=False,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)
t0 = time.time()
_ = paradigm.get_data(dataset, subjects, cache_config=cache_config)
t_array = time.time() - t0

###############################################################################
# Using epochs cache:
cache_config = dict(
    use=True,
    path=temp_dir,
    save_raw=False,
    save_epochs=False,
    save_array=False,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=True,
)
t0 = time.time()
_ = paradigm.get_data(dataset, subjects, cache_config=cache_config)
t_epochs = time.time() - t0

###############################################################################
# Using raw cache:
cache_config = dict(
    use=True,
    path=temp_dir,
    save_raw=False,
    save_epochs=False,
    save_array=False,
    overwrite_raw=False,
    overwrite_epochs=True,
    overwrite_array=True,
)
t0 = time.time()
_ = paradigm.get_data(dataset, subjects, cache_config=cache_config)
t_raw = time.time() - t0

###############################################################################
# Using no cache:
cache_config = dict(
    use=False,
    path=temp_dir,
    save_raw=False,
    save_epochs=False,
    save_array=False,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)
t0 = time.time()
_ = paradigm.get_data(dataset, subjects, cache_config=cache_config)
t_nocache = time.time() - t0

###############################################################################
# Time needed to load the data with different levels of cache:
print(f"Using array cache: {t_array:.2f} seconds")
print(f"Using epochs cache: {t_epochs:.2f} seconds")
print(f"Using raw cache: {t_raw:.2f} seconds")
print(f"Without cache: {t_nocache:.2f} seconds")

###############################################################################
# As you can see, using a raw cache is more than 5 times faster than
# without cache.
# This is because when using the raw cache, the data is not preloaded, only
# the desired epochs are loaded in memory.
#
# Using the epochs cache is a little faster than the raw cache. This is because
# there are several preprocessing steps done after the epoching by the
# ``epochs_pipeline``. This difference would be greater if the ``resample``
# argument was different that the sampling frequency of the dataset. Indeed,
# the data loading time is directly proportional to its sampling frequency
# and the resampling is done by the ``epochs_pipeline``.
#
# Finally, we observe very little difference between array and epochs cache.
# The main interest of the array cache is when the user passes a
# computationally heavy but fixed additional preprocessing (for example
# computing the covariance matrices of the epochs). This can be done by using
# the ``postprocess_pipeline`` argument. The output of this additional pipeline
# (necessary a numpy array) will be saved to avoid re-computing it each time.
#
#
# Technical details
# -----------------
#
# Under the hood, the cache is saved on disk in a Brain Imaging Data Structure
# (BIDS) compliant format. More details on this structure can be found in the
# tutorial :doc:`./plot_bids_conversion`.
#
# However, there are two particular aspects of the way MOABB saves the data
# that are not specific to BIDS:
#
# * For each file, we set a
#   `description key <https://bids-specification.readthedocs.io/en/stable/appendices/entities.html#desc>`_.
#   This key is a code that corresponds to a hash of the
#   pipeline that was used to generate the data (i.e. from raw to the state
#   of the cache). This code is unique for each different pipeline and allows
#   to identify all the files that were generated by the same pipeline.
# * Once we finish saving all the files for a given combination of dataset,
#   subject, and pipeline, we write a file ending in ``"_lockfile.json"`` at
#   the root directory of this subject. This file serves two purposes:
#
#   * It indicates that the cache is complete for this subject and pipeline.
#     If it is not present, it means that something went wrong during the
#     saving process and the cache is incomplete.
#   * The file contains the un-hashed string representation of the pipeline.
#     Therefore, it can be used to identify the pipeline used without having
#     to decode the description key.
#
# Cleanup
# -------
#
# Finally, we can delete the temporary folder:
shutil.rmtree(temp_dir)
