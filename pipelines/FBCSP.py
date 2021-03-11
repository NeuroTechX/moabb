import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from moabb.pipelines.utils import FilterBank


parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters)
fb = FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4)))
pipe = make_pipeline(fb, SelectKBest(score_func=mutual_info_classif, k=10), clf)

# this is what will be loaded
PIPELINE = {
    "name": "FBCSP + optSVM",
    "paradigms": ["FilterBankMotorImagery"],
    "pipeline": pipe,
}
