"""
======================================
Playing with the pre-processing steps
======================================
By default, MOABB uses **fundamental** and **robust** pre-processing steps defined in
each paradigm.

Behind the curtains, these steps are defined in a scikit-learn Pipeline.
This pipeline receives raw signals and applies various signal processing steps
to construct the final array object and class labels, which will be used
to train and evaluate the classifiers.

Pre-processing steps are known to shape the rank and
metric results of the EEG Decoding [2]_, [3]_, [4]_,
and we present some discussion in our largest benchmark paper [1]_
on why we used those specific steps.
Using the same pre-processing steps for all datasets also avoids biases
and makes results more comparable.

However, there might be cases where these steps are not adequate.
MOABB allows you to modify the pre-processing pipeline.
In this example, we will show how to use the `make_process_pipelines` method to create a
custom pre-processing pipeline. We will use the MinMaxScaler from `sklearn` to scale the
data channels to the range [0, 1].

References
----------
.. [1] Chevallier, S., Carrara, I., Aristimunha, B., Guetschel, P., Sedlar, S., Lopes, B., ... & Moreau, T. (2024). The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. arXiv preprint arXiv:2404.15319.

.. [2] Kessler, R., Enge, A., & Skeide, M. A. (2024). How EEG preprocessing shapes decoding performance. arXiv preprint arXiv:2410.14453.

.. [3] Delorme, A. (2023). EEG is better left alone. Scientific reports, 13(1), 2372.

.. [4] Clayson, P. E. (2024). Beyond single paradigms, pipelines, and outcomes: Embracing multiverse analyses in psychophysiology. International Journal of Psychophysiology, 197, 112311.
"""

# Authors: Bruno Aristimunha Pinto <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

##############################################################################
# What is applied precisely to each paradigm?
# -----------------------------------------------
#
# Each paradigm defines a set of pre-processing steps that are applied to the raw data
# in order to construct the numpy arrays and class labels used for classification.
# In MOABB, the pre-processing steps are divided into three groups:
# the steps which are applied over the `raw` objects, those applied to the `epoch` objects,
# and those for  the `array` objects.
#
# First things, let's define one dataset and one paradigm.
# Here, we will use the BNCI2014_001 dataset and the LeftRightImagery paradigm.
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer, minmax_scale

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
# The paradigm object accepts parameters to configure common
# pre-processing and epoching steps applied to the raw data. These include:
#
# - Bandpass filtering (`filters`)
# - Event selection for epoching (`events`)
# - Epoch time window definition (`tmin`, `tmax`)
# - Baseline correction (`baseline`)
# - Channel selection (`channels`)
# - Resampling (`resample`)
#
# The following example demonstrates how you can surgically add custom processing steps
# beyond these built-in options.
#
# In this example, we want to add a min-max function step to the raw data to do this.
# We need to do pipeline surgery and use the evaluation function.
# We will use the `FunctionTransformer` instead of the `MinMaxScaler` to avoid
# the need to fit the raw data. The `FunctionTransformer` will apply the function
# to the data without fitting it.


def minmax_raw(raw):
    """Apply min-max scaling to the raw data."""
    return raw.apply_function(
        minmax_scale, picks="eeg", n_jobs=1, verbose=True, channel_wise=True
    )


process_pipeline = paradigm.make_process_pipelines(dataset)[0]

process_pipeline.steps.insert(2, (StepType.RAW, FunctionTransformer(minmax_raw)))


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
# Then you can follow the common procedure for analyzing the results.

df_results = pd.DataFrame(results)

df_results.plot(
    x="pipeline",
    y="score",
    kind="bar",
    title="Results of the evaluation with custom pre-processing steps",
    xlabel="Pipeline",
    ylabel="Score",
)
