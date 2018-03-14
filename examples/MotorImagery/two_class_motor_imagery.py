from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms.motor_imagery import LeftRightImageryMultiPass
from moabb.pipelines import multi_pass as mp
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from collections import OrderedDict
from moabb.datasets import utils
from moabb.analysis import analyze

import mne
import numpy as np
mne.set_log_level(False)

import logging
import coloredlogs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level=logging.DEBUG)

datasets = utils.dataset_search('imagery', events=['right_hand', 'left_hand'],
                                has_all_events=True, min_subjects=2,
                                multi_session=False)

for d in datasets:
    d.subject_list = d.subject_list[:10]

paradigm = LeftRightImageryMultiPass(fbands=np.array([[8, 12],
                                                      [12, 16],
                                                      [16, 20],
                                                      [20, 24],
                                                      [24, 28],
                                                      [28, 32]]))

context = WithinSessionEvaluation(paradigm=paradigm,
                                  datasets=datasets,
                                  random_state=42)

pipelines = OrderedDict()
pipelines['av+TS'] = make_pipeline(
    mp.AverageCovariance(estimator='oas'), TSclassifier())
pipelines['av+CSP+LDA'] = make_pipeline(
    mp.AverageCovariance(estimator='oas'), CSP(8), LDA())
pipelines['FBCSP+LDA'] = make_pipeline(
    mp.MultibandCovariances(estimator='oas'), mp.FBCSP(), LDA())  #

results = context.process(pipelines, overwrite=True)

analyze(results, './')
