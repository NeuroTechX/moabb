import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from moabb.pipelines.features import LogVariance


parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters)
pipe = make_pipeline(LogVariance(), clf)

# this is what will be loaded
PIPELINE = {"name": "AM + optSVM", "paradigms": ["MotorImagery"], "pipeline": pipe}
