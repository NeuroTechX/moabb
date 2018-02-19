from moabb.contexts.evaluations import * 
from moabb.contexts.motor_imagery import LeftRightImagery
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier, MDM
from sklearn.pipeline import make_pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from collections import OrderedDict
from moabb import utils

datasets = utils.dataset_search('imagery',events=['right_hand','left_hand'],
                                exact_events=True)

pipelines = OrderedDict()
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM())
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
pipelines['CSP+SVM'] = make_pipeline(Covariances('oas'), CSP(8), SVC())  # 

context = LeftRightImagery(pipelines, WithinSessionEvaluation(), datasets)

context.process()

for v in context.results.data.values():
    print(v)
