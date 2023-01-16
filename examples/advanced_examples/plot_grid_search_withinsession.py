"""
============================
GridSearch within a session
============================

This example shows how to use GridSearchCV within a session.

"""
import os

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from moabb.datasets import Zhou2016
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


sub_numb = 1

# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmin = 0
tmax = None

# Select the Subject
subjects = [int(sub_numb)]
# Load the dataset, right now you have added Nothing events to DATA using new stim channel STI
dataset = Zhou2016()

events = ["right_hand", "feet"]

paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax
)

# Create a path and folder for every subject
path = os.path.join(str("try_Subject_" + str(sub_numb)))
os.makedirs(path, exist_ok=True)

# Pipelines
pipelines = {}
# Define the different algorithm to test and assign a name in the dictionary
pipelines["CSP+LDA"] = Pipeline(
    steps=[
        ("Covariances", Covariances("cov")),
        ("csp", CSP(nfilter=6)),
        ("lda", LDA(solver="lsqr", shrinkage="auto")),
    ]
)

pipelines["Cov+EN"] = Pipeline(
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

# ====================================================================================================================
# GridSearch
# ====================================================================================================================
param_grid = {}
param_grid["Cov+EN"] = {
    "LogistReg__l1_ratio": [0.15, 0.30, 0.45, 0.60, 0.75],
}

# Evaluation For MOABB
# ========================================================================================================
dataset.subject_list = dataset.subject_list[int(sub_numb) - 1 : int(sub_numb)]
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
