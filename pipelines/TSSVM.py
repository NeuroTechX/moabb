from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
import numpy as np

parameters = {'C': np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel='linear'), parameters)
pipe = make_pipeline(Covariances('oas'), TangentSpace(metric='riemann'), clf)

# this is what will be loaded
PIPELINE = {'name': 'TS + optSVM',
            'paradigms': ['MotorImagery'],
            'pipeline': pipe}
