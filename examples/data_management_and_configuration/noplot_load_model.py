"""
==============================================
Load Model (Scikit) with MOABB
==============================================

This example shows how to use load the pretrained pipeline in MOABB.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

from pickle import load

from moabb import set_log_level
from moabb.utils import setup_seed


set_log_level("info")

###############################################################################
# In this example, we will use the results computed by the following examples
#
# - plot_benchmark
#
# ---------------------

# Set up reproducibility of Tensorflow and PyTorch
setup_seed(42)

###############################################################################
# Loading the Scikit-learn pipelines

with open(
    "../how_to_benchmark/results/Models_WithinSession/Zhou2016/1/0/csp+svm/fitted_model_best.pkl",
    "rb",
) as pickle_file:
    CSP_SVM_Trained = load(pickle_file)
