"""
===================
Cross Session SSVEP
===================

This Example show how to perform a cross-session SSVEP analysis on the
MAMEM dataset 1, using a CCA pipeline.

The cross session evaluation context will evaluate performance using a leave
one session out cross-validation. For each session in the dataset, a model
is trained on every other session and performance are evaluated on the current
session.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
#
# License: BSD (3-clause)

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import MAMEM1, MAMEM2, MAMEM3
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import SSVEP
from moabb.pipelines import SSVEP_CCA


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
moabb.set_log_level('info')

###############################################################################
# Loading dataset
# ---------------
#
# Load 2 subjects of MAMEM1 2 and 3 datasets, with 5 session each

subj = [2, 5]
for s in subj:
    for d in [MAMEM1(), MAMEM2(), MAMEM3()]:
        d._get_single_subject_data(s)
datasets = [MAMEM1(), MAMEM2(), MAMEM3()]
for d in datasets:
    d.subject_list = subj

###############################################################################
# Choose paradigm
# ---------------
#
# We select the paradigm SSVEP, applying a bandpass filter (3-15 Hz) on
# the data and we keep all the 5 classes, that is stimulation
# frequency of 6.66, 7.50 and 8.57, 10 and 12 Hz.

paradigm = SSVEP(fmin=3, fmax=15, n_classes=None)

##############################################################################
# Create pipelines
# ----------------
#
# Use a Canonical Correlation Analysis classifier

interval = datasets[0].interval
freqs = paradigm.used_events(datasets[0])

pipeline = {}
pipeline["CCA"] = make_pipeline(SSVEP_CCA(interval=interval, freqs=freqs, n_harmonics=3))

##############################################################################
# Get data (optional)
# -------------------
#
# To get access to the EEG signals downloaded from the dataset, you could
# use `dataset._get_single_subject_data(subject_id) to obtain the EEG under
# an MNE format, stored in a dictionary of sessions and runs.
# Otherwise, `paradigm.get_data(dataset=dataset, subjects=[subject_id])`
# allows to obtain the EEG data in scikit format, the labels and the meta
# information.

X_all, labels_all, meta_all = [], [], []
for d in datasets:
    # sessions = d._get_single_subject_data(2)
    X, labels, meta = paradigm.get_data(dataset=d, subjects=[2])
    X_all.append(X)
    labels_all.append(labels)
    meta_all.append(meta)

##############################################################################
# Evaluation
# ----------
#
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.

overwrite = True  # set to True if we want to overwrite cached results

evaluation = CrossSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix='examples', overwrite=overwrite
)
results = evaluation.process(pipeline)

print(results.head())

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results, indicating the score for each session and subject

sns.catplot(
    data=results,
    x='session',
    y='score',
    hue='subject',
    col='dataset',
    kind='bar',
    palette='viridis',
)
plt.show()
