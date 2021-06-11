"""
================================================
Within Session Motor Imagery with learning curve
================================================

This Example show how to perform a within session motor imagery analysis on the
very popular dataset 2a from the BCI competition IV.

We will compare two pipelines :

- CSP + LDA
- Riemannian Geometry + Logistic Regression

We will use the LeftRightImagery paradigm. this will restrict the analysis
to two classes (left hand versus righ hand) and use AUC as metric.
"""
# Original author: Alexandre Barachant <alexandre.barachant@gmail.com>
# Learning curve modification: Jan Sosulski
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


moabb.set_log_level("info")

##############################################################################
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
#
# The csp implementation from MNE is used. We selected 8 CSP components, as
# usually done in the litterature.
#
# The riemannian geometry pipeline consists in covariance estimation, tangent
# space mapping and finaly a logistic regression for the classification.

pipelines = {}

pipelines["CSP+LDA"] = make_pipeline(
    CSP(n_components=8), LDA(solver="lsqr", shrinkage="auto")
)

pipelines["RG+LR"] = make_pipeline(
    Covariances(), TangentSpace(), LogisticRegression(solver="lbfgs")
)

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (LeftRightImagery) and the dataset (BNCI2014001).
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwrited if necessary.

paradigm = LeftRightImagery()
dataset = BNCI2014001()
dataset.subject_list = dataset.subject_list[:1]
datasets = [dataset]
overwrite = True  # set to True if we want to overwrite cached results
# Evaluate for a specific number of training samples per class
data_size = dict(policy="per_class", value=np.array([5, 10, 30, 50]))
# When the training data is sparse, peform more permutations than when we have a lot of data
n_perms = np.floor(np.geomspace(20, 2, len(data_size["value"]))).astype(int)
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    suffix="examples",
    overwrite=overwrite,
    data_size=data_size,
    n_perms=n_perms,
)

results = evaluation.process(pipelines)

print(results.head())

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
