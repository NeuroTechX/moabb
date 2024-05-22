"""
==============================================
Load Model (Scikit, Pytorch, Keras) with MOABB
==============================================

This example shows how to use load the pretrained pipeline in MOABB.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#
# License: BSD (3-clause)

from pickle import load

import keras
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGInception
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

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

###############################################################################
# Loading the PyTorch model

# Hyperparameter
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
BATCH_SIZE = 64
SEED = 42
VERBOSE = 1
EPOCH = 2
PATIENCE = 3

# Define a Skorch classifier
clf = EEGClassifier(
    module=EEGInception,
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=SEED),
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
    ],
    verbose=VERBOSE,  # Not printing the results for each epoch
)

clf.initialize()

f_params = "./results/Models_CrossSession/BNCI2014-001/1/braindecode_EEGInception/EEGInception_fitted_best_model.pkl"
f_optimizer = "./results/Models_CrossSession/BNCI2014-001/1/braindecode_EEGInception/EEGInception_fitted_best_optim.pkl"
f_history = "./results/Models_CrossSession/BNCI2014-001/1/braindecode_EEGInception/EEGInception_fitted_best_history.json"

clf.load_params(f_params=f_params, f_optimizer=f_optimizer, f_history=f_history)

# Create the pipelines
pipes_pytorch = make_pipeline(clf)
