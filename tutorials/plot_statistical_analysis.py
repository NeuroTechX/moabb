"""=======================
Statistical Analysis
=======================

The MOABB codebase comes with convenience plotting utilities and some
statistical testing. This tutorial focuses on what those exactly are and how
they can be used.

"""
# Authors: Vinay Jayaram <vinayjayaram13@gmail.com>
#
# License: BSD (3-clause)

import moabb
import matplotlib.pyplot as plt
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import find_significant_differences, compute_dataset_statistics  # flake8: noqa

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import CrossSessionEvaluation

moabb.set_log_level('info')

print(__doc__)

###############################################################################
# Results Generation
# ---------------------
#
# First we need to set up a paradigm, dataset list, and some pipelines to
# test. This is explored more in the examples -- we choose a left vs right
# imagery paradigm with a single bandpass. There is only one dataset here but
# any number can be added without changing this workflow.
#
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
#
# The csp implementation from MNE is used. We selected 8 CSP components, as
# usually done in the litterature.
#
# The riemannian geometry pipeline consists in covariance estimation, tangent
# space mapping and finaly a logistic regression for the classification.

pipelines = {}

pipelines['CSP + LDA'] = make_pipeline(CSP(n_components=8),
                                       LDA())

pipelines['RG + LR'] = make_pipeline(Covariances(),
                                     TangentSpace(),
                                     LogisticRegression())

pipelines['CSP + LR'] = make_pipeline(CSP(n_components=8),
                                      LogisticRegression())

pipelines['RG + LDA'] = make_pipeline(Covariances(),
                                      TangentSpace(),
                                      LDA())

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (LeftRightImagery) and the dataset (BNCI2014001).
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

paradigm = LeftRightImagery()
datasets = [BNCI2014001()]
overwrite = False  # set to True if we want to overwrite cached results
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=datasets,
                                    suffix='examples', overwrite=overwrite)

results = evaluation.process(pipelines)


##############################################################################
# MOABB plotting
# ----------------
#
# Here we plot the results using some of the convenience methods within the
# toolkit.  The score_plot visualizes all the data with one score per subject
# for every dataset and pipeline.

fig = moabb_plt.score_plot(results)
plt.show()

###############################################################################
# For a comparison of two algorithms, there is the paired_plot, which plots
# performance in one versus the performance in the other over all chosen
# datasets. Note that there is only one score per subject, regardless of the
# number of sessions.

fig = moabb_plt.paired_plot(results, 'CSP + LDA', 'RG + LDA')
plt.show()


###############################################################################
# Statistical testing and further plots
# ----------------------------------------
#
# If the statistical significance of results is of interest, the method
# compute_dataset_statistics allows one to show a meta-analysis style plot as
# well. For an overview of how all algorithms perform in comparison with each
# other, the method find_significant_differences and the summary_plot are
# possible.


stats = compute_dataset_statistics(results)
P, T = find_significant_differences(stats)

################################################################################
# The meta-analysis style plot shows the standardized mean difference within
# each tested dataset for the two algorithms in question, in addition to a
# meta-effect and significances both per-dataset and overall.
fig = moabb_plt.meta_analysis_plot(stats, 'CSP + LDA', 'RG + LDA')
plt.show()


################################################################################
# The summary plot shows the effect and significance related to the hypothesis
# that the algorithm on the y-axis significantly out-performed the algorithm on
# the x-axis over all datasets
moabb_plt.summary_plot(P, T)
plt.show()
