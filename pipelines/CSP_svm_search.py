from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


parameters = {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}
clf = GridSearchCV(SVC(), parameters, cv=3)
pipe = make_pipeline(Covariances("oas"), CSP(6), clf)

# this is what will be loaded
PIPELINE = {"name": "CSP + optSVM", "paradigms": ["LeftRightImagery"], "pipeline": pipe}
