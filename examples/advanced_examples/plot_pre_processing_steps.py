"""
======================================
Playing with the pre-processing steps
======================================
By default, MOABB uses **fundamental** and **robust** pre-processing steps defined in
each paradigm. We present some discussion on our largest benchmark paper [1]_ for some
used steps. Pre-processing steps are known to shape the rank and
metric results of the EEG Decoding [2]_, [3]_, [4]_.

Here, we follow the philosophical principles of Findability, Accessibility, Interoperability,
and Reusability - FAIR, that is, the data and the benchmark code are findable, accessible,
interoperable and reusable.

In other words, we try as much as possible to use raw data, which we apply pre-processing steps
to construct the epoch object and later convert it to an array to compute the evaluation.

In the pre-processing, all the datasets receive the same pre-processing steps associated with
the paradigm object is to avoid the bias of the pre-processing steps, which are extremely common
in the field.

Behind the curtains of the evaluation and paradigm, we have scikit-learn Pipeline steps that
are applied to construct the final array object with the class labels.

In this example, we will show how to use the `make_process_pipelines` method to create a
custom pre-processing pipeline. We will use the MinMaxScaler from `sklearn` to scale the
data channels to the range [0, 1].
"""

# Authors: Bruno Aristimunha Pinto <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

##############################################################################
# What is applied precisely to each paradigm?
# -----------------------------------------------
#
# Each paradigm has a set of pre-processing steps that are applied to the raw data that
# allow the construction of the dataset as numpy arrays.
#
# At the moabb, you can apply the pre-processing steps to the `raw` object to the `epoch`
# object, or to the `array` object outputted after the pre-processing steps.
# By default, the pre-processing steps are applied as listed above.
#
# First things, let's define one dataset and one paradigm.
# Here, we will use the BNCI2014_001 dataset and the LeftRightImagery paradigm.
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler

from moabb.datasets import BNCI2014_001
from moabb.datasets.bids_interface import StepType
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import FilterBankLeftRightImagery, LeftRightImagery


dataset = BNCI2014_001()
# Select one subject for the example. You can use the dataset for all subjects
dataset.subject_list = dataset.subject_list[:1]

paradigm = LeftRightImagery()

##############################################################################
# Exposing the pre-processing steps
# ----------------------------------
#
# The most efficient way to expose the pre-processing steps is to use the
# `make_process_pipelines` method. This method will return a list of pipelines that
# are applied to the raw data. The pipelines are defined in the paradigm object.

process_pipeline = paradigm.make_process_pipelines(dataset)

# On the not filterbank paradigm, we have only one branch of possible steps steps:
process_pipeline[0]

##############################################################################
# Filter Bank Paradigm
# ---------------------
#
# On the filterbank paradigm, we have n branches in the case of multiple filters:


paradigm_filterbank = FilterBankLeftRightImagery()
pre_procesing_filter_bank_steps = paradigm_filterbank.make_process_pipelines(dataset)

# By default, we have six filter banks, and each filter bank has the same steps.
for i, step in enumerate(pre_procesing_filter_bank_steps):
    print(f"Filter bank {i}: {step}")

##############################################################################
# How to include extra steps?
# -------------------------------
#
# By default, the paradigm object accepts parameters to configure common
# pre-processing and epoching steps applied to the raw data. These include:
#
# - Bandpass filtering (`filters`)
# - Event selection for epoching (`events`)
# - Epoch time window definition (`tmin`, `tmax`)
# - Baseline correction (`baseline`)
# - Channel selection (`channels`)
# - Resampling (`resample`)
#
# The following example demonstrates how to add custom processing steps
# beyond these built-in options.
# we want to add a min-max function step to the raw data to do this.
# We need to do pipeline surgery and use the evaluation function.
##############################################################################


process_pipeline = paradigm.make_process_pipelines(dataset)[0]

process_pipeline.steps.insert(2, (StepType.RAW, MinMaxScaler()))


##############################################################################
# Now that you have defined some special pre-processing, you will need to run with
# `evaluation` function to get the results.
# Here, we will use the `DummyClassifier` from sklearn to run the evaluation.

classifier_pipeline = {}
classifier_pipeline["dummy"] = DummyClassifier()

evaluation = CrossSessionEvaluation(paradigm=paradigm)

generator_results = evaluation.evaluate(
    dataset=dataset,
    pipelines=classifier_pipeline,
    param_grid=None,
    process_pipeline=process_pipeline,
)
# The evaluation function will return a generator object that contains the results
# of the evaluation. You can use the `list` function to convert it to a list.
results = list(generator_results)

##############################################################################
# Plot Results
# ------------
#
# Compare the obtained results with the two pipelines, CSP+LDA and logistic
# regression computed in the tangent space of the covariance matrices.

df_results = pd.DataFrame(results)

df_results.plot(
    x="pipeline",
    y="score",
    kind="bar",
    title="Results of the evaluation with custom pre-processing steps",
    xlabel="Pipeline",
    ylabel="Score",
)
