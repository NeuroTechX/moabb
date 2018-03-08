from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms.motor_imagery import LeftRightImagery
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from collections import OrderedDict
from moabb.datasets import utils
from moabb.analysis import analyze

import mne
mne.set_log_level(False)

import logging
import coloredlogs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
coloredlogs.install(level=logging.INFO)

datasets = utils.dataset_search('imagery', events=['right_hand', 'left_hand'],
                                has_all_events=True, min_subjects=2,
                                multi_session=False)

paradigm = LeftRightImagery()

context = WithinSessionEvaluation(paradigm=paradigm,
                                  datasets=datasets,
                                  random_state=42)

pipelines = OrderedDict()
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
pipelines['CSP+SVM'] = make_pipeline(Covariances('oas'), CSP(8), SVC())  #

results = context.process(pipelines)

analyze(results, './')
