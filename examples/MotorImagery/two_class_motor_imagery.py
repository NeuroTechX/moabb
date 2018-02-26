from moabb.contexts.evaluations import * 
from moabb.contexts.motor_imagery import LeftRightImagery
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier, MDM
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from collections import OrderedDict
from moabb import utils
from moabb.viz import meta_analysis as ma

datasets = utils.dataset_search('imagery',events=['right_hand','left_hand'],
                                exact_events=True, min_subjects=2, multi_session=False)

pipelines = OrderedDict()
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
# pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
# pipelines['CSP+SVM'] = make_pipeline(Covariances('oas'), CSP(8), SVC())  # 

context = LeftRightImagery(pipelines, WithinSessionEvaluation(), datasets)

results = context.process(results='/is/ei/vjayaram/code/git/moabb/examples/MotorImagery/2class.hdf5')

df = results.to_dataframe()
print(df[df['pipeline'] == 'TS'])


