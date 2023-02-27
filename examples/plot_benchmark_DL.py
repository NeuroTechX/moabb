"""
====================================================================
Benchmarking with MOABB with DeepLearning architecture in Tensorflow
====================================================================
This example shows how to use MOABB to benchmark a set of DeepLearning pipeline (Tensorflow)
on all available datasets.
For this example, we will use only one dataset to keep the computation time low, but this benchmark is designed
to easily scale to many datasets.
"""
# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

import os

import matplotlib.pyplot as plt
import tensorflow as tf
from absl.logging import ERROR, set_verbosity
from tensorflow import keras

from moabb import benchmark, set_log_level
from moabb.analysis.plotting import score_plot
from moabb.datasets import BNCI2014001
from moabb.utils import setup_seed


set_log_level("info")
# Avoid output Warning
set_verbosity(ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Print Information Tensorflow
print(f"Tensorflow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

CPU = len(tf.config.list_physical_devices("CPU")) > 0
print("CPU is", "AVAILABLE" if CPU else "NOT AVAILABLE")

GPU = len(tf.config.list_physical_devices("GPU")) > 0
print("GPU is", "AVAILABLE" if GPU else "NOT AVAILABLE")

###############################################################################
###############################################################################
# In this example, we will use only the dataset 'BNCI2014001'.
#
# Running the benchmark
# ---------------------
#
# The benchmark is run using the ``benchmark`` function. You need to specify the
# folder containing the pipelines to use, the kind of evaluation and the paradigm
# to use. By default, the benchmark will use all available datasets for all
# paradigms listed in the pipelines. You could restrict to specific evaluation and
# paradigm using the ``evaluations`` and ``paradigms`` arguments.
#
# To save computation time, the results are cached. If you want to re-run the
# benchmark, you can set the ``overwrite`` argument to ``True``.
#
# It is possible to indicate the folder to cache the results and the one to save
# the analysis & figures. By default, the results are saved in the ``results``
# folder, and the analysis & figures are saved in the ``benchmark`` folder.
#
# This code is implemented to run on CPU. If you're using a GPU, do not use multithreading
# (i.e. set n_jobs=1)

# Set up reproducibility of Tensorflow
setup_seed(42)

# Restrict this example only on the first two subject of BNCI2014001
dataset = BNCI2014001()
dataset.subject_list = dataset.subject_list[:2]
datasets = [dataset]

results = benchmark(
    pipelines="./pipelines_DL",
    evaluations=["WithinSession"],
    paradigms=["LeftRightImagery"],
    include_datasets=datasets,
    results="./results/",
    overwrite=False,
    plot=False,
    output="./benchmark/",
    n_jobs=-1,
)

###############################################################################
# Benchmark prints a summary of the results. Detailed results are saved in a
# pandas dataframe, and can be used to generate figures. The analysis & figures
# are saved in the ``benchmark`` folder.

score_plot(results)
plt.show()
