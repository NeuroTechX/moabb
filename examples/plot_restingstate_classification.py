"""
================================
Resting state classification
================================

This example compares the performance of different
pipelines with resting state datasets.

"""

# License: BSD (3-clause)

import warnings

from moabb.evaluations.evaluations import WithinSessionEvaluation
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.base import TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb import set_log_level
from moabb.datasets import Cattan2019_PHMD, Hinss2021, Rodrigues2017
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter


# Suppressing future and runtime warnings for cleaner output
set_log_level("info")


##############################################################################
# Initialization Process
# ----------------------
#
# 1) Load the datasets
# 2) Define the paradigms
# 3) Select a subset of subjects and specific events for analysis

# We define a list with the dataset to use
datasets = [Cattan2019_PHMD(), Hinss2021(), Rodrigues2017()]

# To reduce the computation time in the example, we will only use the
# first two subjects.
start_subject = 1
stop_subject = 2
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[start_subject:stop_subject]

# Here we define the mne events for the RestingState paradigm.
events_cattan = dict(on=1, off=2)
events_hinss = dict(easy=2, diff=3)
events_rodrigues = dict(closed=1, open=2)

# Create a paradigm by dataset as the events and lenght of epochs are different.
paradigm_cattan = RestingStateToP300Adapter(events=events_cattan, tmin=10, tmax=50)
paradigm_hinss = RestingStateToP300Adapter(events=events_hinss, tmin=0, tmax=0.5)
paradigm_rodrigues = RestingStateToP300Adapter(events=events_rodrigues, tmin=0, tmax=10)
paradigms = [paradigm_cattan, paradigm_hinss, paradigm_rodrigues]

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of scikit-learning pipeline transformer.
# For the sake of the example, we will just limit ourselves to two pipelines
# but more can be added.

pipelines = {}

pipelines["Xdawn+Cov+TS+LDA"] = make_pipeline(
    Xdawn(nfilter=4), Covariances(estimator="lwf"), TangentSpace(), LDA()
)

pipelines["Cov+MDM"] = make_pipeline(
    Covariances(estimator="lwf"), MDM()
)

##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipeline using a cross session evaluation.

evaluation_cattan = WithinSessionEvaluation(
        paradigm=paradigm_cattan,
        datasets=[datasets[0]],
        overwrite=False,
    )
results_cattan = evaluation_cattan.process(pipelines)

evaluation_hinss = CrossSessionEvaluation(
        paradigm=paradigm_hinss,
        datasets=[datasets[1]],
        overwrite=False,
    )
results_hinss = evaluation_hinss.process(pipelines)

evaluation_rodrigues = WithinSessionEvaluation(
        paradigm=paradigm_rodrigues,
        datasets=[datasets[2]],
        overwrite=False,
    )
results_rodrigues= evaluation_rodrigues.process(pipelines)

results = pd.concat([results_cattan, results_hinss, results_rodrigues], ignore_index=True)

###############################################################################
# Here, with the ElSel+Cov+TS+LDA pipeline, we reduce the computation time
# in approximately 8 times to the Cov+ElSel+TS+LDA pipeline.

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

###############################################################################
# Plot Results
# -------------
#
# Here, we plot the results to compare two pipelines


fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1").set(
    title=title
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()

###############################################################################
# Key Observations:
# -----------------
# - `Xdawn` is not ideal for the resting state paradigm. This is due to its specific design for Event-Related Potential (ERP).
# - Electrode selection strategy based on covariance matrices demonstrates less variability and typically yields better performance.
# - However, this strategy is more time-consuming compared to the simpler electrode selection based on time epoch data.
