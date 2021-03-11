import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters, cv=3)
pipe = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), clf)

# this is what will be loaded
PIPELINE = {"name": "TS + optSVM", "paradigms": ["MotorImagery"], "pipeline": pipe}
