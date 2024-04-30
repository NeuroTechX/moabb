"""
================================
Hinss2021 classification example
================================

This example show how to use the Hinss2021 dataset
with the resting state paradigm.

For the sake of the example, we will try to answer which 
channel selection strategy is the best for the Hinss2021 dataset:
Xdawn, electrode selection on time epochs data, or
electrode selection on covariance matrices. 

"""

# License: BSD (3-clause)

import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.base import TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb import set_log_level
from moabb.datasets import Hinss2021
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter


##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

set_log_level("info")

##############################################################################
# Create util transformer
# ----------------------
#
# Let's create a simple transformer, that will
# select electrodes based on the covariance information


class EpochSelectChannel(TransformerMixin):
    """Select channels based on covariance information,"""

    def __init__(self, n_chan, est):
        self.n_chan = n_chan
        self.est = est

    def fit(self, X, _y=None):
        # Get the covariances of the channels for each epoch.
        covs = Covariances(estimator=self.est).fit_transform(X)
        # Get the average covariance between the channels
        m = np.mean(covs, axis=0)
        n_feats, _ = m.shape
        # Select the `n_chan` channels having the maximum covariances.
        all_max = []
        for i in range(n_feats):
            for j in range(n_feats):
                if len(all_max) <= self.n_chan:
                    all_max.append(m[i, j])
                else:
                    if m[i, j] > max(all_max):
                        all_max[np.argmin(all_max)] = m[i, j]
        indices = []
        for v in all_max:
            indices.extend(np.argwhere(m == v).flatten())
        # We will keep only these channels for the transform step.
        indices = np.unique(indices)
        self._elec = indices
        return self

    def transform(self, X):
        return X[:, self._elec, :]


##############################################################################
# Initialization
# ----------------
#
# 1) Create paradigm
# 2) Load datasets
# 3) Select a few subjects and events


events = dict(easy=2, diff=3)

paradigm = RestingStateToP300Adapter(events=events, tmin=0, tmax=0.5)

datasets = [Hinss2021()]

# reduce the number of subjects.
start_subject = 1
stop_subject = 3
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[start_subject:stop_subject]

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

pipelines = {}

pipelines["Xdawn+Cov+TS+LDA"] = make_pipeline(
    Xdawn(nfilter=4), Covariances(estimator="lwf"), TangentSpace(), LDA()  # 8 components
)

pipelines["Cov+ElSel+TS+LDA"] = make_pipeline(
    Covariances(estimator="lwf"), ElectrodeSelection(nelec=8), TangentSpace(), LDA()
)

# Pay attention here that the channel selection took place before computing the covariances:
# It is done on time epochs.
pipelines["ElSel+Cov+TS+LDA"] = make_pipeline(
    EpochSelectChannel(8, "lwf"), Covariances(estimator="lwf"), TangentSpace(), LDA()
)

##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipeline using a cross session evaluation.

# Here should be cross session
evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

###############################################################################
# Plot Results
# -------------
#
# Here we plot the results to compare two pipelines


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

# A few observation that you may make:
#      - Xdawn is not appropriate for resting state paradigm.
#        This is kind of specific are the filter was designed for ERP
#      - Electrode selection on covariance matrices as less variability, and 
#        in general perform best
#      - however, it takes also more time than the simple electrode selection on 
#        time epoch