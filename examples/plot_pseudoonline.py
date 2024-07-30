# Set up the Directory for made it run on a server.

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

# Load Dataset and switch to Pseudoonline mode
dataset = BNCI2014_001()
dataset.pseudoonline = True

# events = ["right_hand", "left_hand"]
events = list(dataset.event_id.keys())

paradigm = MotorImagery(
    events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax, overlap=50
)

X, y, meta = paradigm.get_data(dataset=dataset, subjects=[sub])
print("Print Events_id:", y)
unique, counts = np.unique(y, return_counts=True)
print("Number of events per class:", dict(zip(unique, counts)))


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
    paradigm=paradigm, datasets=dataset, overwrite=True, random_state=42, n_jobs=-1
)

# Print the results
results_ALL = evaluation_online.process(pipelines)
results_pipeline = results_ALL.groupby(["pipeline"], as_index=False)["score"].mean()
results_pipeline_std = results_ALL.groupby(["pipeline"], as_index=False)["score"].std()
results_pipeline["std"] = results_pipeline_std["score"]
print(results_pipeline)
