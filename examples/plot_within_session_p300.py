"""
===========================
Within Session P300
===========================

This Example shows how to perform a within session analysis on three different
P300 datasets.

We will compare two pipelines :

- Riemannian Geometry
- xDawn with Linear Discriminant Analysis

We will use the P300 paradigm, which uses the AUC as metric.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

# getting rid of the warnings about the future (on s'en fout !)
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances, Xdawn
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from moabb.datasets import EPFLP300
import moabb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


moabb.set_log_level('info')

# This is an auxiliary transformer that allows one to vectorize data
# structures in a pipeline For instance, in the case of a X with dimensions
# Nt x Nc x Ns, one might be interested in a new data structure with
# dimensions Nt x (Nc.Ns)


class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (X.shape[0], -1))

##############################################################################
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.


pipelines = {}

# we have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {'Target': 1, 'NonTarget': 0}

pipelines['RG + LDA'] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        classes=[
            labels_dict['Target']],
        estimator='lwf',
        xdawn_estimator='lwf'),
    TangentSpace(),
    LDA(solver='lsqr', shrinkage='auto'))

pipelines['Xdw + LDA'] = make_pipeline(Xdawn(nfilter=2, estimator='lwf'),
                                       Vectorizer(), LDA(solver='lsqr',
                                                         shrinkage='auto'))

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (P300) and use all three datasets available for it.
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

paradigm = P300(resample=128)
dataset = EPFLP300()
dataset.subject_list = dataset.subject_list[:2]
datasets = [dataset]
overwrite = True  # set to True if we want to overwrite cached results
evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                     datasets=datasets,
                                     suffix='examples', overwrite=overwrite)
results = evaluation.process(pipelines)

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

ax.set_ylabel('ROC AUC')
ax.set_ylim(0.5, 1)

fig.show()
