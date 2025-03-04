"""
==============================================
Load Model (Scikit, Keras) with MOABB
==============================================

This example shows how to use load the pretrained pipeline in MOABB.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

from pickle import load

import keras
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline

from moabb import set_log_level
from moabb.pipelines.features import StandardScaler_Epoch
from moabb.utils import setup_seed


set_log_level("info")

###############################################################################
# In this example, we will use the results computed by the following examples
#
# - plot_benchmark_
# - plot_benchmark_braindecode_
# - plot_benchmark_DL_
# ---------------------

# Set up reproducibility of Tensorflow and PyTorch
setup_seed(42)

###############################################################################
# Loading the Scikit-learn pipelines

with open(
    "./results/Models_WithinSession/Zhou2016/1/0/CSP + SVM/fitted_model_best.pkl",
    "rb",
) as pickle_file:
    CSP_SVM_Trained = load(pickle_file)

###############################################################################
# Loading the Keras model
# We load the single Keras model, if we want we can set in the exact same pipeline.

model_Keras = keras.models.load_model(
    "./results/Models_WithinSession/BNCI2014-001/1/1E/Keras_DeepConvNet/kerasdeepconvnet_fitted_model_best.h5"
)
# Now we need to instantiate a new SciKeras object since we only saved the Keras model
Keras_DeepConvNet_Trained = KerasClassifier(model_Keras)
# Create the pipelines


pipes_keras = Pipeline(
    [
        ("StandardScaler_Epoch", StandardScaler_Epoch),
        ("Keras_DeepConvNet_Trained", Keras_DeepConvNet_Trained),
    ]
)
