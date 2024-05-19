"""
================================
Resting state classification
================================

This example compares the performance of different
pipelines with resting state datasets.

Different evaluation methods are used for the different
datasets, then the results are aggregated and statistics
are plot by pipeline and dataset.

"""

# License: BSD (3-clause)


import pandas as pd
from matplotlib import pyplot as plt
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb.analysis.plotting as moabb_plt
from moabb import set_log_level
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
)
from moabb.datasets import Cattan2019_PHMD, Hinss2021, Rodrigues2017
from moabb.evaluations import CrossSessionEvaluation
from moabb.evaluations.evaluations import WithinSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter


# Suppressing future and runtime warnings for cleaner output
set_log_level("info")


##############################################################################
# Initialization Process
# ----------------------
#
# 1) Load the datasets
# 2) Define the paradigms
# 3) Select a subset of subjects and specific events for analysis

# We define a list with the dataset to use
datasets = [Cattan2019_PHMD(), Hinss2021(), Rodrigues2017()]

# To reduce the computation time in the example, we will only use the
# first two subjects.
start_subject = 1
stop_subject = 4
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[start_subject:stop_subject]

# Here we define the mne events for the RestingState paradigms.
events_cattan = dict(on=1, off=2)
events_hinss = dict(easy=2, diff=3)
events_rodrigues = dict(closed=1, open=2)

# Create a paradigm by dataset as the events and length of epochs are different.
paradigm_cattan = RestingStateToP300Adapter(events=events_cattan, tmin=10, tmax=50)
paradigm_hinss = RestingStateToP300Adapter(events=events_hinss, tmin=0, tmax=0.5)
paradigm_rodrigues = RestingStateToP300Adapter(events=events_rodrigues, tmin=0, tmax=10)
paradigms = [paradigm_cattan, paradigm_hinss, paradigm_rodrigues]

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of scikit-learn pipeline transformer.
# For the sake of the example, we will just limit ourselves to two pipelines
# but more can be added.

pipelines = {}

pipelines["Xdawn+Cov+TS+LDA"] = make_pipeline(
    Xdawn(nfilter=4), Covariances(estimator="lwf"), TangentSpace(), LDA()
)

pipelines["Cov+MDM"] = make_pipeline(Covariances(estimator="lwf"), MDM())

##############################################################################
# Run evaluation
# ----------------
#
# Compare pipelines using a within and cross-sessionevaluation.
# (the evaluation method is different depending on the dataset)

evaluation_cattan = WithinSessionEvaluation(
    paradigm=paradigm_cattan,
    datasets=[datasets[0]],
    n_jobs=-1,
    overwrite=False,
)
results_cattan = evaluation_cattan.process(pipelines)

evaluation_hinss = CrossSessionEvaluation(
    paradigm=paradigm_hinss,
    datasets=[datasets[1]],
    n_jobs=-1,
    overwrite=False,
)
results_hinss = evaluation_hinss.process(pipelines)

evaluation_rodrigues = WithinSessionEvaluation(
    paradigm=paradigm_rodrigues,
    datasets=[datasets[2]],
    n_jobs=-1,
    overwrite=False,
)
results_rodrigues = evaluation_rodrigues.process(pipelines)

# Display results by pipeline and dataset
results = pd.concat([results_cattan, results_hinss, results_rodrigues], ignore_index=True)

print("Averaging the session performance:")
print(results)
print(results.groupby(["pipeline", "dataset"]).mean("score")[["score", "time"]])


###############################################################################
# Plot Results
# -------------
#
# Here, we plot the results to compare the performance of the two pipelines
# with the different datasets.

stats = compute_dataset_statistics(results)
stats.fillna(0, inplace=True)
print(stats)
moabb_plt.meta_analysis_plot(stats, "Xdawn+Cov+TS+LDA", "Cov+MDM")
plt.show()
