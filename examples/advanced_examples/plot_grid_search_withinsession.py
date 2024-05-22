"""
============================
GridSearch within a session
============================

This example demonstrates how to make a model selection in pipelines
for finding the best model parameter, using grid search. Two models
are compared, one "vanilla" model with model tuned via grid search.
"""

import os
from pickle import load

import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmax = None

# Select the Subject
subjects = [1]
# Load the dataset
dataset = BNCI2014_001()

events = ["right_hand", "left_hand"]

paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax
)

# Create a path and folder for every subject
path = os.path.join(str("Results"))
os.makedirs(path, exist_ok=True)

##############################################################################
# Create the Pipelines
# --------------------
# Two pipelines implementing elastic net classifiers, one using a fixed
# l1_ratio ("VanillaEN") and the other using a range of values to select
# l1_ratio ("GridSearchEN")

pipelines = {}
pipelines["VanillaEN"] = Pipeline(
    steps=[
        ("Covariances", Covariances("cov")),
        ("Tangent_Space", TangentSpace(metric="riemann")),
        (
            "LogistReg",
            LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.75,
                intercept_scaling=1000.0,
                solver="saga",
                max_iter=1000,
            ),
        ),
    ]
)

pipelines["GridSearchEN"] = Pipeline(
    steps=[
        ("Covariances", Covariances("cov")),
        ("Tangent_Space", TangentSpace(metric="riemann")),
        (
            "LogistReg",
            LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.70,
                intercept_scaling=1000.0,
                solver="saga",
                max_iter=1000,
            ),
        ),
    ]
)

##############################################################################
# The search space for parameters is defined as a dictionary, specifying the
# name of the estimator and the parameter name as a key.

param_grid = {}
param_grid["GridSearchEN"] = {
    "LogistReg__l1_ratio": [0.15, 0.30, 0.45, 0.60, 0.75],
}

##############################################################################
# Running the Evaluation
# ----------------------
# If a param_grid is specified during process, the specified pipelines will
# automatically be run with a grid search.

dataset.subject_list = dataset.subject_list[:1]
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    overwrite=True,
    random_state=42,
    hdf5_path=path,
    n_jobs=-1,
    save_model=True,
)
result = evaluation.process(pipelines, param_grid)

#####################################################################
# Plot Results
# ------------
# The grid search allows to find better parameter during the
# evaluation, leading to better accuracy results.

fig, axes = plt.subplots(1, 1, figsize=[8, 5], sharey=True)

sns.stripplot(
    data=result,
    y="score",
    x="pipeline",
    ax=axes,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=result, y="score", x="pipeline", ax=axes, palette="Set1")
axes.set_ylabel("ROC AUC")

##########################################################
# Load Best Model Parameter
# -------------------------
# The best model are automatically saved in a pickle file, in the
# results directory. It is possible to load those model for each
# dataset, subject and session. Here, we could see that the grid
# search found a l1_ratio that is different from the baseline
# value.

with open(
    "./Results/Models_WithinSession/BNCI2014-001/1/1test/GridSearchEN/fitted_model_best.pkl",
    "rb",
) as pickle_file:
    GridSearchEN_Session_E = load(pickle_file)

print(
    "Best Parameter l1_ratio Session_E GridSearchEN ",
    GridSearchEN_Session_E.best_params_["LogistReg__l1_ratio"],
)

print(
    "Best Parameter l1_ratio Session_E VanillaEN: ",
    pipelines["VanillaEN"].steps[2][1].l1_ratio,
)

with open(
    "./Results/Models_WithinSession/BNCI2014-001/1/0train/GridSearchEN/fitted_model_best.pkl",
    "rb",
) as pickle_file:
    GridSearchEN_Session_T = load(pickle_file)

print(
    "Best Parameter l1_ratio Session_T GridSearchEN ",
    GridSearchEN_Session_T.best_params_["LogistReg__l1_ratio"],
)

print(
    "Best Parameter l1_ratio Session_T VanillaEN: ",
    pipelines["VanillaEN"].steps[2][1].l1_ratio,
)
