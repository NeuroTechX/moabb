"""
===========================
Within Session P300
===========================

This Example shows how to perform a within session analysis while also
creating learning curves for a P300 dataset.

We will compare three pipelines :

- Riemannian Geometry
- Jumping Means based Linear Discriminant Analysis
- Time-Decoupled Linear Discriminant Analysis

We will use the P300 paradigm, which uses the AUC as metric.
"""
# Authors: Jan Sosulski
#
# License: BSD (3-clause)

# getting rid of the warnings about the future (on s'en fout !)
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances, Xdawn
from moabb.evaluations import WithinSessionEvaluationIncreasingData
from moabb.paradigms import P300
from moabb.datasets import BNCI2014009
import moabb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
from tdlda import Vectorizer as JumpingMeansVectorizer
from tdlda import TimeDecoupledLda as TDLDA

start_time = time.time()
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


processing_sampling_rate = 128
pipelines = {}

# we have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {'Target': 1, 'NonTarget': 0}

# Riemannian geometry based classification
pipelines['RG + LDA'] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        estimator='lwf',
        xdawn_estimator='scm'),
    TangentSpace(),
    LDA(solver='lsqr', shrinkage='auto'))

jumping_mean_ivals = [[0.10, 0.139], [0.14, 0.169], [0.17, 0.199], [0.20, 0.229],
                      [0.23, 0.269], [0.27, 0.299], [0.30, 0.349], [0.35, 0.409],
                      [0.41, 0.449], [0.45, 0.499]]
jmv = JumpingMeansVectorizer(fs=processing_sampling_rate, jumping_mean_ivals=jumping_mean_ivals)

pipelines['JM + LDA'] = make_pipeline(jmv, LDA(solver='lsqr', shrinkage='auto'))

c = TDLDA(N_channels=16, N_times=10)
# TD-LDA needs to know about the used jumping means preprocessing
c.preproc = jmv
pipelines['JM + TD-LDA'] = make_pipeline(jmv, c)
#
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

paradigm = P300(resample=processing_sampling_rate)
dataset = BNCI2014009()
dataset.subject_list = dataset.subject_list[0:1]
datasets = [dataset]
overwrite = True  # set to True if we want to overwrite cached results
data_size = dict(policy='ratio', value=np.geomspace(0.05, 1, 5))
n_perms = np.floor(np.geomspace(50, 3, 5)).astype(np.int)
print(n_perms)
# Guarantee reproducibility
np.random.seed(7536298)
evaluation = WithinSessionEvaluationIncreasingData(paradigm=paradigm, datasets=datasets,
                                                   data_size=data_size, n_perms=n_perms,
                                                   suffix='examples_lr', overwrite=overwrite)


results = evaluation.process(pipelines)
# %%
##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results.

fig, ax = plt.subplots(facecolor='white', figsize=[8, 4])

n_subs = len(dataset.subject_list)

if n_subs > 1:
    r = results.groupby(['pipeline', 'subject', 'data_size']).mean().reset_index()
else:
    r = results

sns.pointplot(data=r, x='data_size', y='score', hue='pipeline', ax=ax, palette="Set1")

errbar_meaning = "subjects" if n_subs > 1 else "permutations"
title_str = f"Errorbar shows Mean-CI across {errbar_meaning}"
ax.set_xlabel('Amount of training samples')
ax.set_ylabel('ROC AUC')
ax.set_title(title_str)
fig.tight_layout()
plt.show()
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time/60} minutes.")
