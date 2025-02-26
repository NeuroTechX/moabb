"""
====================================================================
Benchmarking on MOABB with Tensorflow deep net architectures
====================================================================
This example shows how to use MOABB to benchmark a set of Deep Learning pipeline (Tensorflow)
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

from moabb import benchmark, set_log_level
from moabb.analysis.plotting import score_plot
from moabb.datasets import BNCI2014_001
from moabb.utils import setup_seed


set_log_level("info")
# Avoid output Warning
set_verbosity(ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


CPU = len(tf.config.list_physical_devices("CPU")) > 0
print("CPU is", "AVAILABLE" if CPU else "NOT AVAILABLE")

GPU = len(tf.config.list_physical_devices("GPU")) > 0
print("GPU is", "AVAILABLE" if GPU else "NOT AVAILABLE")

###############################################################################
# In this example, we will use only the dataset ``BNCI2014_001``.
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

# Restrict this example only on the first two subject of BNCI2014_001
dataset = BNCI2014_001()
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
# The deep learning architectures implemented in MOABB are:
# - Shallow Convolutional Network [1]_
# - Deep Convolutional Network [1]_
# - EEGNet [2]_
# - EEGTCNet [3]_
# - EEGNex [4]_
# - EEGITNet [5]_
#
# Benchmark prints a summary of the results. Detailed results are saved in a
# pandas dataframe, and can be used to generate figures. The analysis & figures
# are saved in the ``benchmark`` folder.

score_plot(results)
plt.show()

##############################################################################
# References
# ----------
# .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
#    Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
#    `Deep learning with convolutional neural networks for EEG decoding and
#    visualization <https://doi.org/10.1002/hbm.23730>`_.
#    Human brain mapping, 38(11), 5391-5420.
# .. [2] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
#    Hung, C. P., & Lance, B. J. (2018). `EEGNet: a compact convolutional neural
#    network for EEG-based brain-computer interfaces.
#    <https://doi.org/10.1088/1741-2552/aace8c>`_
#    Journal of neural engineering, 15(5), 056013.
# .. [3] Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N., Cavigelli, L., &
#    Benini, L. (2020, October). `EEG-TCNet: An accurate temporal convolutional
#    network for embedded motor-imagery brain-machine interfaces.
#    <https://doi.org/10.1109/SMC42975.2020.9283028>`_
#    In 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
#    (pp. 2958-2965). IEEE.
# .. [4] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2022). `Toward reliable
#    signals decoding for electroencephalogram: A benchmark study to EEGNeX.
#    <https://doi.org/10.48550/arXiv.2207.12369>`_
#    arXiv preprint arXiv:2207.12369.
# .. [5] Salami, A., Andreu-Perez, J., & Gillmeister, H. (2022). `EEG-ITNet: An
#    explainable inception temporal convolutional network for motor imagery
#    classification
#    <https://doi.org/10.1109/ACCESS.2022.3161489>`_.
#    IEEE Access, 10, 36672-36685.
