"""
===========================
Cross-Subject SSVEP
===========================
This example shows how to perform a cross-subject analysis on an SSVEP dataset.
We will compare two pipelines :

- Riemannian Geometry
- CCA

We will use the SSVEP paradigm, which uses the AUC as metric.
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
#
# License: BSD (3-clause)

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import SSVEPExo
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import SSVEP, FilterBankSSVEP
from moabb.pipelines import SSVEP_CCA, SSVEP_TRCA, ExtendedSSVEPSignal


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
moabb.set_log_level("info")

###############################################################################
# Loading Dataset
# ---------------
#
# We will load the data from the first 2 subjects of the ``SSVEP_Exo`` dataset
# and compare two algorithms on this set. One of the algorithms could only
# process class associated with a stimulation frequency, we will thus drop
# the resting class. As the resting class is the last defined class, picking
# the first three classes (out of four) allows to focus only on the stimulation
# frequency.

n_subject = 2
dataset = SSVEPExo()
dataset.subject_list = dataset.subject_list[:n_subject]
interval = dataset.interval

###############################################################################
# Choose Paradigm
# ---------------
#
# We define the paradigms (SSVEP, SSSVEP_TRCA and FilterBankSSVEP) and use the dataset
# SSVEPExo. The SSVEP paradigm applied a bandpass filter (10-25 Hz) on
# the data, SSVEP_TRCA applied a bandpass filter (1-110 Hz) which correspond to almost
# no filtering, while the FilterBankSSVEP paradigm uses as many bandpass filters as
# there are stimulation frequencies (here 2). For each stimulation frequency
# the EEG is filtered with a 1 Hz-wide bandpass filter centered on the
# frequency. This results in ``n_classes`` copies of the signal, filtered for each
# class, as used in the filterbank motor imagery paradigms.

paradigm = SSVEP(fmin=10, fmax=25, n_classes=3)
paradigm_TRCA = SSVEP(fmin=1, fmax=110, n_classes=3)
paradigm_fb = FilterBankSSVEP(filters=None, n_classes=3)

###############################################################################
# Classes are defined by the frequency of the stimulation, here we use
# the first two frequencies of the dataset, 13 and 17 Hz.
# The evaluation function uses a LabelEncoder, transforming them
# to 0 and 1

freqs = paradigm.used_events(dataset)

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
# The first pipeline uses Riemannian geometry, by building an extended
# covariance matrices from the signal filtered around the considered
# frequency and applying a logistic regression in the tangent plane.
# The second pipeline relies on the above defined CCA classifier.
# The third pipeline relies on the TRCA algorithm.

pipelines_fb = {}
pipelines_fb["RG+LogReg"] = make_pipeline(
    ExtendedSSVEPSignal(),
    Covariances(estimator="lwf"),
    TangentSpace(),
    LogisticRegression(solver="lbfgs", multi_class="auto"),
)

pipelines = {}
pipelines["CCA"] = make_pipeline(SSVEP_CCA(interval=interval, freqs=freqs, n_harmonics=3))

pipelines_TRCA = {}
pipelines_TRCA["TRCA"] = make_pipeline(
    SSVEP_TRCA(interval=interval, freqs=freqs, n_fbands=5)
)

##############################################################################
# Evaluation
# ----------
#
# The evaluation will return a DataFrame containing an accuracy score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

overwrite = False  # set to True if we want to overwrite cached results

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm, datasets=dataset, overwrite=overwrite
)
results = evaluation.process(pipelines)

###############################################################################
# Filter bank processing, determine the filter automatically from the
# stimulation frequency values of events.

evaluation_fb = CrossSubjectEvaluation(
    paradigm=paradigm_fb, datasets=dataset, overwrite=overwrite
)
results_fb = evaluation_fb.process(pipelines_fb)

###############################################################################
# TRCA processing also relies on filter bank that is automatically designed.

evaluation_TRCA = CrossSubjectEvaluation(
    paradigm=paradigm_TRCA, datasets=dataset, overwrite=overwrite
)
results_TRCA = evaluation_TRCA.process(pipelines_TRCA)

###############################################################################
# After processing the three, we simply concatenate the results.

results = pd.concat([results, results_fb, results_TRCA])

##############################################################################
# Plot Results
# ----------------
#
# Here we display the results as stripplot, with a pointplot for error bar.

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])
sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, zorder=1, palette="Set1")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.1, 0.6)
plt.show()
