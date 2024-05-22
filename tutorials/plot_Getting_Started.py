"""
============================
Tutorial 0: Getting Started
============================

This tutorial takes you through a basic working example of how to use this
codebase, including all the different components, up to the results
generation. If you'd like to know about the statistics and plotting, see the
next tutorial.

"""

# Authors: Vinay Jayaram <vinayjayaram13@gmail.com>
#
# License: BSD (3-clause)


##########################################################################
# Introduction
# --------------------
# To use the codebase you need an evaluation and a paradigm, some algorithms,
# and a list of datasets to run it all on. You can find those in the following
# submodules; detailed tutorials are given for each of them.

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

##########################################################################
# If you would like to specify the logging level when it is running, you can
# use the standard python logging commands through the top-level moabb module
import moabb
from moabb.datasets import BNCI2014_001, utils
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.pipelines.features import LogVariance


##########################################################################
# In order to create pipelines within a script, you will likely need at least
# the make_pipeline function. They can also be specified via a .yml file. Here
# we will make a couple pipelines just for convenience


moabb.set_log_level("info")

##############################################################################
# Create pipelines
# ----------------
#
# We create two pipelines: channel-wise log variance followed by LDA, and
# channel-wise log variance followed by a cross-validated SVM (note that a
# cross-validation via scikit-learn cannot be described in a .yml file). For
# later in the process, the pipelines need to be in a dictionary where the key
# is the name of the pipeline and the value is the Pipeline object

pipelines = {}
pipelines["AM+LDA"] = make_pipeline(LogVariance(), LDA())
parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters)
pipe = make_pipeline(LogVariance(), clf)

pipelines["AM+SVM"] = pipe

##############################################################################
# Datasets
# -----------------
#
# Datasets can be specified in many ways: Each paradigm has a property
# 'datasets' which returns the datasets that are appropriate for that paradigm

print(LeftRightImagery().datasets)

##########################################################################
# Or you can run a search through the available datasets:
print(utils.dataset_search(paradigm="imagery", min_subjects=6))

##########################################################################
# Or you can simply make your own list (which we do here due to computational
# constraints)

dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:2]
datasets = [dataset]

##########################################################################
# Paradigm
# --------------------
#
# Paradigms define the events, epoch time, bandpass, and other preprocessing
# parameters. They have defaults that you can read in the documentation, or you
# can simply set them as we do here. A single paradigm defines a method for
# going from continuous data to trial data of a fixed size. To learn more look
# at the tutorial Exploring Paradigms

fmin = 8
fmax = 35
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)

##########################################################################
# Evaluation
# --------------------
#
# An evaluation defines how the training and test sets are chosen. This could
# be cross-validated within a single recording, or across days, or sessions, or
# subjects. This also is the correct place to specify multiple threads.

evaluation = CrossSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=False
)
results = evaluation.process(pipelines)

##########################################################################
# Results are returned as a pandas DataFrame, and from here you can do as you
# want with them

print(results.head())
