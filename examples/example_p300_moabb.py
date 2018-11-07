"""
===========================
Within Session P300
===========================

This Example shows how to perform a within session analysis on two different
P300 datasets.

We will compare two pipelines :

- Riemannian Geometry
- ?

We will use the P300 paradigm, which uses the AUC as metric.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import moabb

from moabb.datasets import BNCI2014008, BNCI2014009
from moabb.paradigms import P300
from moabb.evaluations import WithinSessionEvaluation

from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.classification import MDM

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

moabb.set_log_level('info')

##############################################################################
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
#
# The riemannian geometry pipeline consists in covariance estimation, tangent
# space mapping and finaly a logistic regression for the classification.

pipelines = {}

labels_dict = {'Target':1, 'NonTarget':0}
pipelines['ERPCovs + MDM'] = make_pipeline(ERPCovariances(estimator='lwf', classes=[labels_dict['Target']]), MDM())
pipelines['XdwCovs + MDM'] = make_pipeline(XdawnCovariances(estimator='lwf', xdawn_estimator='lwf', classes=[labels_dict['Target']]), MDM())

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (P300) and the datasets (BNCI2014008 and BNCI2014009).
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

paradigm = P300()
datasets = [BNCI2014008(), BNCI2014009()]
overwrite = False  # set to True if we want to overwrite cached results
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets,
                                    suffix='examples', overwrite=overwrite)

results = evaluation.process(pipelines)

print(results.head())

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results. We the first plot is a pointplot with the average
# performance of each pipeline across session and subjects.
# The second plot is a paired scatter plot. Each point representing the score
# of a single session. An algorithm outperforms another if most of the
# points are in its quadrant.

fig, axes = plt.subplots(1, 2, figsize=[8, 4], sharey=True)

sns.stripplot(data=results, y='score', x='pipeline', ax=axes[0], jitter=True,
              alpha=.5, zorder=1, palette="Set1")
sns.pointplot(data=results, y='score', x='pipeline', ax=axes[0],
              zorder=1, palette="Set1")

axes[0].set_ylabel('ROC AUC')
axes[0].set_ylim(0.5, 1)

# paired plot
paired = results.pivot_table(values='score', columns='pipeline',
                             index=['subject', 'session'])
paired = paired.reset_index()

sns.regplot(data=paired, y='ERPCovs + MDM', x='XdwCovs + MDM', ax=axes[1],
            fit_reg=False)
axes[1].plot([0, 1], [0, 1], ls='--', c='k')
axes[1].set_xlim(0.5, 1)

plt.show()
