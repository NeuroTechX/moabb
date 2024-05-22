"""Examples of how to use MOABB to benchmark pipelines.
=======================
Benchmarking with MOABB
=======================

This example shows how to use MOABB to benchmark a set of pipelines
on all available datasets. For this example, we will use only one
dataset to keep the computation time low, but this benchmark is designed
to easily scale to many datasets.
"""

# Authors: Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

from moabb import benchmark, set_log_level
from moabb.analysis.plotting import score_plot
from moabb.paradigms import LeftRightImagery


set_log_level("info")

###############################################################################
# Loading the pipelines
# ---------------------
#
# The ML pipelines used in benchmark are defined in YAML files, following a
# simple format. It simplifies sharing and reusing pipelines across benchmarks,
# reproducing state-of-the-art results.
#
# MOABB comes with complete list of pipelines that cover most of the successful
# approaches in the literature. You can find them in the
# `pipelines folder <https://github.com/NeuroTechX/moabb/tree/develop/pipelines>`_.
# For this example, we will use a folder with only 2 pipelines, to keep the
# computation time low.
#
# This is an example of a pipeline defined in YAML, defining on which paradigms it
# can be used, the original publication, and the steps to perform using a
# scikit-learn API. In this case, a CSP + SVM pipeline, the covariance are estimated
# to compute a CSP filter and then a linear SVM is trained on the CSP filtered
# signals.

with open("sample_pipelines/CSP_SVM.yml", "r") as f:
    lines = f.readlines()
    for line in lines:
        print(line, end="")

###############################################################################
# The ``sample_pipelines`` folder contains a second pipeline, a logistic regression
# performed in the tangent space using Riemannian geometry.
#
# Selecting the datasets (optional)
# ---------------------------------
#
# If you want to limit your benchmark on a subset of datasets, you can use the
# ``include_datasets`` and ``exclude_datasets`` arguments. You will need either
# to provide the dataset's object, or a the dataset's code. To get the list of
# available dataset's code for a given paradigm, you can use the following command:

paradigm = LeftRightImagery()
for d in paradigm.datasets:
    print(d.code)

###############################################################################
# In this example, we will use only the last dataset, 'Zhou 2016'.
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
    pipelines="./sample_pipelines/",
    evaluations=["WithinSession"],
    paradigms=["LeftRightImagery"],
    include_datasets=["Zhou2016"],
    results="./results/",
    overwrite=False,
    plot=False,
    output="./benchmark/",
)

###############################################################################
# Benchmark prints a summary of the results. Detailed results are saved in a
# pandas dataframe, and can be used to generate figures. The analysis & figures
# are saved in the ``benchmark`` folder.

score_plot(results)
plt.show()
