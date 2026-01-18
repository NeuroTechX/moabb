"""
========================================
Benchmarking with MOABB with Grid Search
========================================

This example shows how to use MOABB to benchmark a set of pipelines
on all available datasets. In particular we run the Gridsearch to select the best hyperparameter of some pipelines
and save the gridsearch.
For this example, we will use only one dataset to keep the computation time low, but this benchmark is designed
to easily scale to many datasets.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

from moabb import benchmark, set_log_level
from moabb.analysis.plotting import score_plot


set_log_level("info")

###############################################################################
# In this example, we will use only the dataset 'Zhou 2016'.
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

# In the results folder we will save the gridsearch evaluation
# When write the pipeline in ylm file we need to specify the parameter that we want to test, in format
# pipeline-name__estimator-name_parameter. Note that pipeline and estimator names MUST
# be in lower case (no capital letters allowed).
# If the grid search is already implemented it will load the previous results

results = benchmark(
    pipelines="./pipelines_grid/",
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
