"""
================================
Select Electrodes and Resampling
================================

Within paradigm, it is possible to restrict analysis only to a subset of
electrodes and to resample to a specific sampling rate. There is also a
utility function to select common electrodes shared between datasets.
This tutorial demonstrates how to use this functionality.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
#
# License: BSD (3-clause)
import matplotlib.pyplot as plt
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import make_pipeline

import moabb.analysis.plotting as moabb_plt
from moabb.datasets import BNCI2014001, Zhou2016
from moabb.datasets.utils import find_intersecting_channels
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


##############################################################################
# Datasets
# --------
#
# Load 2 subjects of BNCI 2014-004 and Zhou2016 datasets, with 2 sessions each

subj = [1, 2]
datasets = [Zhou2016(), BNCI2014001()]
for d in datasets:
    d.subject_list = subj

##############################################################################
# Paradigm
# --------
#
# Restrict further analysis to specified channels, here C3, C4, and Cz.
# Also, use a specific resampling. In this example, all datasets are
# set to 200 Hz.

paradigm = LeftRightImagery(channels=["C3", "C4", "Cz"], resample=200.0)

##############################################################################
# Evaluation
# ----------
#
# The evaluation is conducted on with CSP+LDA, only on the 3 electrodes, with
# a sampling rate of 200 Hz.

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets)
csp_lda = make_pipeline(CSP(n_components=2), LDA())
ts_lr = make_pipeline(
    Covariances(estimator="oas"), TangentSpace(metric="riemann"), LR(C=1.0)
)
results = evaluation.process({"csp+lda": csp_lda, "ts+lr": ts_lr})
print(results.head())

##############################################################################
# Electrode Selection
# -------------------
#
# It is possible to select the electrodes that are shared by all datasets
# using the `find_intersecting_channels` function. Datasets that have 0
# overlap with others are discarded. It returns the set of common channels,
# as well as the list of datasets with valid channels.

electrodes, datasets = find_intersecting_channels(datasets)
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, overwrite=True, suffix="resample"
)
results = evaluation.process({"csp+lda": csp_lda, "ts+lr": ts_lr})
print(results.head())

##############################################################################
# Plot Results
# ------------
#
# Compare the obtained results with the two pipelines, CSP+LDA and logistic
# regression computed in the tangent space of the covariance matrices.

fig = moabb_plt.paired_plot(results, "csp+lda", "ts+lr")
plt.show()
