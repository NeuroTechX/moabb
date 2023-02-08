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

import matplotlib.pyplot as plt

from moabb import benchmark, set_log_level
from moabb.analysis.plotting import score_plot
import random
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


set_log_level("info")

import absl.logging
# Avoid output Warning
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Print Information Tensorflow
print(f"Tensorflow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

CPU = len(tf.config.list_physical_devices("CPU")) > 0
print("CPU is", "AVAILABLE" if CPU else "NOT AVAILABLE")

GPU = len(tf.config.list_physical_devices("GPU")) > 0
print("GPU is", "AVAILABLE" if GPU else "NOT AVAILABLE")

# Allow parallelization on all physical devices available
mirrored_strategy = tf.distribute.MirroredStrategy()

# Set up reproducibility of Tensorflow
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


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

results = benchmark(
    pipelines="./pipelines_DL",
    evaluations=["WithinSession"],
    paradigms=["LeftRightImagery"],
    include_datasets=["001-2014"],
    results="./results/",
    overwrite=True,
    plot=False,
    output="./benchmark/",
)

###############################################################################
# Benchmark prints a summary of the results. Detailed results are saved in a
# pandas dataframe, and can be used to generate figures. The analysis & figures
# are saved in the ``benchmark`` folder.

score_plot(results)
plt.show()