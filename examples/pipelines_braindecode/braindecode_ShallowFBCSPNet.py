import torch
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from moabb.pipelines.features import Resampler_Epoch


# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

# Hyperparameter
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
BATCH_SIZE = 64
SEED = 42
VERBOSE = 1
EPOCH = 5
PATIENCE = 3

# Define a Skorch classifier
clf = EEGClassifier(
    module=ShallowFBCSPNet,
    module__final_conv_length="auto",
    optimizer=torch.optim.Adam,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=SEED, stratified=True),
    device=device,
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

# Create the pipelines
pipes = Pipeline(
    [
        ("resample", Resampler_Epoch(250)),
        ("ShallowFBCSPNet", clf),
    ]
)

# this is what will be loaded
PIPELINE = {
    "name": "braindecode_ShallowFBCSPNet",
    "paradigms": ["LeftRightImagery", "MotorImagery"],
    "pipeline": pipes,
}
