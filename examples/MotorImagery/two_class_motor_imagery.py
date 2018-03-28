from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms.motor_imagery import ImageryNClass
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances

from collections import OrderedDict
from moabb.datasets import utils
from moabb.analysis import analyze

import mne
mne.set_log_level(False)

import logging
import coloredlogs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level=logging.DEBUG)

datasets = utils.dataset_search('imagery', events=['supination', 'hand_close'],
                                has_all_events=False, min_subjects=2,
                                multi_session=False)

for d in datasets:
    d.subject_list = d.subject_list[:10]

paradigm = ImageryNClass(2)
context = WithinSessionEvaluation(paradigm=paradigm,
                                  datasets=datasets,
                                  random_state=42)

pipelines = OrderedDict()
pipelines['av+TS'] = make_pipeline(Covariances(estimator='oas'), TSclassifier())
pipelines['av+CSP+LDA'] = make_pipeline(Covariances(estimator='oas'), CSP(8), LDA())

results = context.process(pipelines, overwrite=True)

analyze(results, './')
