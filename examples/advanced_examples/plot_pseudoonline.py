"""
==================================================
Pseudo-Online Motor Imagery with Sliding Window
==================================================

This example shows how to perform a pseudo-online motor imagery evaluation
using sliding window overlap. The ``overlap`` parameter in the paradigm
generates overlapping epochs from the original trials, simulating an online
BCI scenario.

We use the BNCI2014-001 dataset with two Riemannian pipelines (MDM and FgMDM)
and a within-session evaluation.
"""

import numpy as np
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from sklearn.pipeline import Pipeline

from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


sub = 1

# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 30
tmax = 3

# Load dataset and configure overlap in the paradigm pipeline
dataset = BNCI2014_001()

events = list(dataset.event_id.keys())

paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax, overlap=50
)

X, y, meta = paradigm.get_data(dataset=dataset, subjects=[sub])
unique, counts = np.unique(y, return_counts=True)
print("Number of trials per class:", dict(zip(unique, counts)))


pipelines = {}
pipelines["MDM"] = Pipeline(
    steps=[
        ("Covariances", Covariances("cov")),
        ("MDM", MDM(metric=dict(mean="riemann", distance="riemann"))),
    ]
)

pipelines["FgMDM"] = Pipeline(
    steps=[("Covariances", Covariances("cov")), ("FgMDM", FgMDM())]
)

dataset.subject_list = dataset.subject_list[int(sub) - 1 : int(sub)]
# Select an evaluation Within Session
evaluation_online = WithinSessionEvaluation(
    paradigm=paradigm, datasets=dataset, overwrite=True, random_state=42, n_jobs=1
)

# Print the results
results_ALL = evaluation_online.process(pipelines)
results_pipeline = results_ALL.groupby(["pipeline"], as_index=False)["score"].mean()
results_pipeline_std = results_ALL.groupby(["pipeline"], as_index=False)["score"].std()
results_pipeline["std"] = results_pipeline_std["score"]
print(results_pipeline)
