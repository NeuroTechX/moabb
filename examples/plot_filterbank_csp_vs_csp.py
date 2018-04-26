"""
=========================
FilterBank CSP versus CSP
=========================

This Example show a comparison of CSP versus FilterBank CSP on the
very popular dataset 2a from the BCI competition IV.
"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import moabb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP

from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery, FilterBankLeftRightImagery
from moabb.evaluations import CrossSessionEvaluation
from moabb.pipelines.utils import FilterBank

moabb.set_log_level('info')

##############################################################################
# Create pipelines
# ----------------
#
# The CSP implementation from MNE is used. We selected 8 CSP components, as
# usually done in the litterature.
#
# The second pipeline is the filter bank CSP. We use the FilterBank object
# with a CSP estimator. We set up the CSP to 4 components, to compensate for
# the higher dimensionality.
#
# The two pipelines will be applied on two different paradigm, so they have
# their own dict.

pipelines = {}
pipelines['CSP + LDA'] = make_pipeline(CSP(n_components=8),
                                       LDA())

pipelines_fb = {}
pipelines_fb['FBCSP + LDA'] = make_pipeline(FilterBank(CSP(n_components=4)),
                                            LDA())

##############################################################################
# Evaluation
# ----------
#
# Since two different preprocessing will be applied, we have two different
# paradigm objects. We have to make sure their filter matchs so the comparison
# will be fair.
#
# The first one is a standard `LeftRightImagery` with a 8 to 35 Hz broadband
# filter.
#
# The second is a `FilterBankLeftRightImagery` with a bank of 6 filter, ranging
# from 8 to 35 Hz.

datasets = [BNCI2014001()]
overwrite = False  # set to True if we want to overwrite cached results

# broadband filters
fmin=8
fmax=35
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=datasets,
                                    suffix='examples', overwrite=overwrite)
results = evaluation.process(pipelines)

# cashed results might return other pipelines
results = results[results.pipeline == 'CSP + LDA']

# bank of 6 filter, by 4 Hz increment
filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
paradigm = FilterBankLeftRightImagery(filters=filters)
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=datasets,
                                    suffix='examples', overwrite=overwrite)
results_fb = evaluation.process(pipelines_fb)

###############################################################################
# After processing the two, we simply concatenate the results.

results = pd.concat([results, results_fb])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results. We the first plot is a pointplot with the average
# performance of each pipeline across session and subjects.
# The second plot is a paired scatter plot. Each point representing the score
# of a single session. An algorithm will outperforms another is most of the
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

sns.regplot(data=paired, y='FBCSP + LDA', x='CSP + LDA', ax=axes[1],
            fit_reg=False)
axes[1].plot([0, 1], [0, 1], ls='--', c='k')
axes[1].set_xlim(0.5, 1)

plt.show()
