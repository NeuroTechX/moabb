"""
===========================
Cross Subject SSVEP
===========================
This example shows how to perform a cross subject analysis on a SSVEP dataset.
We will compare two pipelines :
- Riemannian Geometry
- CCA
We will use the SSVEP paradigm, which uses the AUC as metric.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
#
# License: BSD (3-clause)

from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import SSVEP, FilterBankSSVEP
from moabb.datasets import SSVEPExo
import moabb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
moabb.set_log_level('info')

###############################################################################
# Auxiliary functions
# -------------------
#
# We will need an auxiliary transformer for filtering the signal.
# This auxiliary transformer allows to reshape the signal for building
# extended covariances matrices

class FilteredSignal(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        out = X.transpose((0, 3, 1, 2))
        n_trials, n_freqs, n_channels, n_times = out.shape
        out = out.reshape((n_trials, n_channels * n_freqs, n_times))
        return out

###############################################################################
# The following class defines a SSVEP CCA classifier, where the CCA is computed
# from the set of training signals and some pure sinusoids to act as reference.
# Classification is made by taking the frequency with the max correlation, as
# proposed in Bin et al. (2009) https://doi.org/10.1088/1741-2560/6/4/046002

class SSVEP_CCA(BaseEstimator, ClassifierMixin):

    def __init__(self, interval, freqs, n_harmonics=3):
        self.Yf = dict()
        self.cca = CCA(n_components=1)
        self.slen = interval[1] - interval[0]
        self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.one_hot = {}
        for i, k in enumerate(freqs.keys()):
            self.one_hot[k] = i

    def fit(self, X, y, sample_weight=None):
        """fit."""
        n_times = X.shape[2]

        for f in self.freqs:
            if f.replace('.', '', 1).isnumeric():
                freq = float(f)
                yf = []
                for h in range(1, self.n_harmonics + 1):
                    yf.append(np.sin(2 * np.pi * freq * h *
                                     np.linspace(0, self.slen, n_times)))
                    yf.append(np.cos(2 * np.pi * freq * h *
                                     np.linspace(0, self.slen, n_times)))
                self.Yf[f] = np.array(yf)
        return self

    def predict(self, X):
        """predict"""
        y = []
        for i, x in enumerate(X):
            corr_f = {}
            for f in self.freqs:
                if f.replace('.', '', 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    corr_f[f] = np.corrcoef(S_x.T, S_y.T)[0, 1]
            y.append(self.one_hot[max(corr_f, key=lambda k: corr_f[k])])
        return y

    def predict_proba(self, X):
        """predict proba"""
        P = np.zeros(shape=(len(X), len(self.freqs)))
        for i, x in enumerate(X):
            for j, f in enumerate(self.freqs):
                if f.replace('.', '', 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    P[i, j] = np.corrcoef(S_x.T, S_y.T)[0, 1]
        return P / np.resize(P.sum(axis=1), P.T.shape).T


# Loading the SSVEP paradigm and the SSVEP Exo dataset, restricting to
# the first three classes (here 13, 17 and 21 Hz) and the first 2 subjects.
# Two paradigms are created, one for the CCA classification and another one
# based on filter-bank decomposition for designing extended covariance
# matrices.


n_subject = 2
for i in range(n_subject):
    SSVEPExo()._get_single_subject_data(i + 1)
dataset = SSVEPExo()
dataset.subject_list = dataset.subject_list[:n_subject]
interval = dataset.interval
paradigm = SSVEP(fmin=10, fmax=25, n_classes=3)
paradigm_fb = FilterBankSSVEP(filters=None, n_classes=3)

# Classes are defined by the frequency of the stimulation, here we use
# the first two frequencies of the dataset, 13 and 17 Hz.
# The evaluation function uses a LabelEncoder, transforming them
# to 0 and 1

freqs = paradigm.used_events(dataset)

##############################################################################
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
# The first pipeline uses Riemannian geometry, by building an extended
# covariance matrices from the signal filtered around the considered
# frequency and applying a logistic regression in the tangent plane.
# The second pipeline relies on the above defined CCA classifier.

pipelines_fb = {}
pipelines_fb['RG + LogReg'] = make_pipeline(
    FilteredSignal(),
    Covariances(estimator='lwf'),
    TangentSpace(),
    LogisticRegression(solver='lbfgs', multi_class='auto'))

pipelines = {}
pipelines['CCA'] = make_pipeline(
    SSVEP_CCA(interval=interval, freqs=freqs, n_harmonics=3))
#    BroadbandSignal(),

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (SSVEP) and use the dataset available for it.
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

overwrite = False  # set to True if we want to overwrite cached results

evaluation = CrossSubjectEvaluation(paradigm=paradigm,
                                    datasets=dataset, overwrite=overwrite)
results = evaluation.process(pipelines)

# Filter bank processing, determine automatically the filter from the
# stimulation frequency values of events.
evaluation_fb = CrossSubjectEvaluation(paradigm=paradigm_fb,
                                    datasets=dataset, overwrite=overwrite)
results_fb = evaluation_fb.process(pipelines_fb)

###############################################################################
# After processing the two, we simply concatenate the results.

results = pd.concat([results, results_fb])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results.

fig, ax = plt.subplots(facecolor='white', figsize=[8, 4])
sns.stripplot(data=results, y='score', x='pipeline', ax=ax, jitter=True,
              alpha=.5, zorder=1, palette="Set1")
sns.pointplot(data=results, y='score', x='pipeline', ax=ax,
              zorder=1, palette="Set1")
ax.set_ylabel('Accuracy')
ax.set_ylim(0.1, 0.6)
plt.savefig('ssvep.png')
fig.show()
