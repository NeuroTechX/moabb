import os.path as osp
# Set up the Directory for made it run on a server.
import sys
import os

sys.path.append(r'/home/icarrara/Documents/Project/HolographicEEG')  # Server

import matplotlib.pyplot as plt
import mne
import seaborn as sns
import torch
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode.util import set_random_seeds
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery
from moabb.utils import set_download_dir
from sklearn.pipeline import Pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from braindecode.datasets import create_from_X_y
from Feature_Extraction.Transformer import Transformer

mne.set_log_level(False)

set_download_dir(osp.join(osp.expanduser("~"), "mne_data"))

# Set up GPU if it is there
cuda = torch.cuda.is_available()
print(cuda)
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True


# Class to load the data
class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, kw_args=None):
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        dataset = create_from_X_y(
            X.get_data(),
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=False,
            sfreq=X.info["sfreq"],
        )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


# Set random seed to be able to reproduce results
seed = 42
set_random_seeds(seed=seed, cuda=cuda)

# Hyperparameter ShallowEEG
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 10
patience = 5
fmin = 4
fmax = 100
tmin = 0
tmax = None
sub_numb = 1

# Load the dataset
dataset = BNCI2014001()
events = ["right_hand", "left_hand"]
# events = ["right_hand", "left_hand", "tongue", "feet"]
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
subjects = [sub_numb]
X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
# Define Transformer of Dataset compatible with Brain Decode
create_dataset = Transformer()

# ========================================================================================================
# Define Model
# ========================================================================================================
model = EEGNetv4(
    in_chans=X.shape[1],
    n_classes=len(events),
    input_window_samples=X.shape[2],
    final_conv_length="auto",
    drop_prob=0.5
)

# Send model to GPU
if cuda:
    model.cuda()

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=lr,
    batch_size=batch_size,
    max_epochs=n_epochs,
    train_split=ValidSplit(0.2),
    device=device,
    callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
               EpochScoring(scoring='accuracy', on_train=True, name='train_acc', lower_is_better=False),
               EpochScoring(scoring='accuracy', on_train=False, name='valid_acc', lower_is_better=False)],
    verbose=1  # Not printing the results foe each epoch
)

# ========================================================================================================
# Define the pipeline
# ========================================================================================================
pipes = {}
pipes["ShallowFBCSPNet"] = Pipeline([
    ("Braindecode_dataset", create_dataset),
    ("Net", clf)])

# ========================================================================================================
# Evaluation For MOABB
# ========================================================================================================
dataset.subject_list = dataset.subject_list[int(sub_numb) - 1:int(sub_numb)]

evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    suffix="braindecode_example",
    overwrite=True,
    return_epochs=True,
    n_jobs=1
)

results = evaluation.process(pipes)

print(results.head())

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results. We the first plot is a pointplot with the average
# performance of each pipeline across session and subjects.
# The second plot is a paired scatter plot. Each point representing the score
# of a single session. An algorithm will outperforms another is most of the
# points are in its quadrant.

fig, axes = plt.subplots(1, 1, figsize=[8, 4], sharey=True)

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=axes,
    jitter=True,
    alpha=0.5,
    palette="Set1",
)
sns.pointplot(
    data=results, y="score", x="pipeline", ax=axes, palette="Set1"
)

axes.set_ylabel("ROC AUC")
axes.set_ylim(0.4, 1.0)

plt.show()
