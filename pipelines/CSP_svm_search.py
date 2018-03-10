from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}
clf = GridSearchCV(SVC(), parameters)
pipe = make_pipeline(Covariances('oas'), CSP(6), clf)

# this is what will be loaded
PIPELINE = {'name': 'CSP + SVM with optim',
            'paradigms': ['LeftRightImagery'],
            'pipeline': pipe}
