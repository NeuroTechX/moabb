"""
============================
GridSearch within a session
============================

This example shows how to use GridSearchCV within a session.

"""
import os

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmin = 0
tmax = None

# Select the Subject
subjects = [1]
# Load the dataset
dataset = BNCI2014001()

events = ["right_hand", "left_hand"]

paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax
)

# Create a path and folder for every subject
path = os.path.join(str("Results"))
os.makedirs(path, exist_ok=True)

# Pipelines
pipelines = {}
# Define the different algorithm to test and assign a name in the dictionary
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
                l1_ratio=0.15,
                intercept_scaling=1000.0,
                solver="saga",
                max_iter=1000,
            ),
        ),
    ]
)

##############################################################################
# GridSearch Parameter
# -------------
param_grid = {}
param_grid["GridSearchEN"] = {
    "LogistReg__l1_ratio": [0.15, 0.30, 0.45, 0.60, 0.75],
}

##############################################################################
# Evaluation For MOABB
# -------------
dataset.subject_list = dataset.subject_list[:1]
# Select an evaluation Within Session
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    overwrite=True,
    random_state=42,
    hdf5_path=path,
    n_jobs=-1,
)

# Print the results
# result = evaluation.process(pipelines)
result = evaluation.process(pipelines, param_grid)

#####################################################################
# Plot Results
# ----------------------------------
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
# Load best model Parameter
# -----------------------------------------------
search_session_E = joblib.load(
    os.path.join(
        path,
        "GridSearch_WithinSession",
        "001-2014",
        "subject1",
        "session_E",
        "GridSearchEN",
        "Grid_Search_WithinSession.pkl",
    )
)
print(
    "Best Parameter l1_ratio Session_E GridSearchEN ",
    search_session_E.best_params_["LogistReg__l1_ratio"],
)
print(
    "Best Parameter l1_ratio Session_E VanillaEN: ",
    pipelines["VanillaEN"].steps[2][1].l1_ratio,
)

search_session_T = joblib.load(
    os.path.join(
        path,
        "GridSearch_WithinSession",
        "001-2014",
        "subject1",
        "session_T",
        "GridSearchEN",
        "Grid_Search_WithinSession.pkl",
    )
)
print(
    "Best Parameter l1_ratio Session_T GridSearchEN ",
    search_session_T.best_params_["LogistReg__l1_ratio"],
)
print(
    "Best Parameter l1_ratio Session_T VanillaEN: ",
    pipelines["VanillaEN"].steps[2][1].l1_ratio,
)
