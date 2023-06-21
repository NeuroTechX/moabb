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
from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from moabb import set_log_level
from moabb.pipelines.features import StandardScaler_Epoch
from moabb.pipelines.utils_pytorch import BraindecodeDatasetLoader, InputShapeSetterEEG
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
    "./results/Models_WithinSession/Zhou 2016/1/session_0/CSP + SVM/best_model.pkl", "rb"
) as pickle_file:
    CSP_SVM_Trained = load(pickle_file)

###############################################################################
# Loading the Keras model
# We load the single Keras model, if we want we can set in the exact same pipeline.

model_Keras = keras.models.load_model(
    "./results/Models_WithinSession/001-2014/1/session_E/Keras_DeepConvNet/best_model"
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

# Set EEG Inception model
model = EEGInception(in_channels=22, n_classes=2)

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
    module=model,
    criterion=torch.nn.CrossEntropyLoss,
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
        InputShapeSetterEEG(
            params_list=["in_channels", "input_window_samples", "n_classes"],
        ),
    ],
    verbose=VERBOSE,  # Not printing the results for each epoch
)

clf.initialize()

clf.load_params(
    f_params="./results/Models_CrossSession/001-2014/1/braindecode_EEGInception/best_model.pkl",
    f_optimizer="./results/Models_CrossSession/001-2014/1/braindecode_EEGInception/best_opt.pkl",
    f_history="./results/Models_CrossSession/001-2014/1/braindecode_EEGInception/best_history.json",
)

# Create the dataset
create_dataset = BraindecodeDatasetLoader(drop_last_window=False)

# Create the pipelines
pipes_pytorch = Pipeline([("Braindecode_dataset", create_dataset), ("EEGInception", clf)])
