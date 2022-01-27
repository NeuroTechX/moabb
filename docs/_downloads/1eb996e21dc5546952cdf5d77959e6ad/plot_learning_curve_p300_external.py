"""
=======================================
Within Session P300 with Learning Curve
=======================================

This example shows how to perform a within session analysis while also
creating learning curves for a P300 dataset.
Additionally, we will evaluate external code. Make sure to have tdlda installed, which
can be found in requirements_external.txt

We will compare three pipelines :

- Riemannian geometry
- Jumping Means-based Linear Discriminant Analysis
- Time-Decoupled Linear Discriminant Analysis

We will use the P300 paradigm, which uses the AUC as metric.
"""
# Authors: Jan Sosulski
#
# License: BSD (3-clause)

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from tdlda import TimeDecoupledLda
from tdlda import Vectorizer as JumpingMeansVectorizer

import moabb
from moabb.datasets import BNCI2014009
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300


# getting rid of the warnings about the future (on s'en fout !)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


moabb.set_log_level("info")

##############################################################################
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
processing_sampling_rate = 128
pipelines = {}

# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

# Riemannian geometry based classification
pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(nfilter=5, estimator="lwf", xdawn_estimator="scm"),
    TangentSpace(),
    LDA(solver="lsqr", shrinkage="auto"),
)

# Simple LDA pipeline using averaged feature values in certain time intervals
jumping_mean_ivals = [
    [0.10, 0.139],
    [0.14, 0.169],
    [0.17, 0.199],
    [0.20, 0.229],
    [0.23, 0.269],
    [0.27, 0.299],
    [0.30, 0.349],
    [0.35, 0.409],
    [0.41, 0.449],
    [0.45, 0.499],
]
jmv = JumpingMeansVectorizer(
    fs=processing_sampling_rate, jumping_mean_ivals=jumping_mean_ivals
)

pipelines["JM+LDA"] = make_pipeline(jmv, LDA(solver="lsqr", shrinkage="auto"))

# Time-decoupled Covariance classifier, needs information about number of
# channels and time intervals
c = TimeDecoupledLda(N_channels=16, N_times=10)
# TD-LDA needs to know about the used jumping means intervals
c.preproc = jmv
pipelines["JM+TD-LDA"] = make_pipeline(jmv, c)


##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (P300) and use the BNCI 2014-009 dataset for it.
# The evaluation will return a dataframe containing AUCs for each permutation
# and dataset size.

paradigm = P300(resample=processing_sampling_rate)
dataset = BNCI2014009()
# Remove the slicing of the subject list to evaluate multiple subjects
dataset.subject_list = dataset.subject_list[0:1]
datasets = [dataset]
overwrite = True  # set to True if we want to overwrite cached results
data_size = dict(policy="ratio", value=np.geomspace(0.02, 1, 6))
# When the training data is sparse, peform more permutations than when we have a lot of data
n_perms = np.floor(np.geomspace(20, 2, len(data_size["value"]))).astype(int)
print(n_perms)
# Guarantee reproducibility
np.random.seed(7536298)
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    data_size=data_size,
    n_perms=n_perms,
    suffix="examples_lr",
    overwrite=overwrite,
)


results = evaluation.process(pipelines)

##############################################################################
# Plot Results
# ------------
#
# Here we plot the results.

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

n_subs = len(dataset.subject_list)

if n_subs > 1:
    r = results.groupby(["pipeline", "subject", "data_size"]).mean().reset_index()
else:
    r = results

sns.pointplot(data=r, x="data_size", y="score", hue="pipeline", ax=ax, palette="Set1")

errbar_meaning = "subjects" if n_subs > 1 else "permutations"
title_str = f"Errorbar shows Mean-CI across {errbar_meaning}"
ax.set_xlabel("Amount of training samples")
ax.set_ylabel("ROC AUC")
ax.set_title(title_str)
fig.tight_layout()
plt.show()
