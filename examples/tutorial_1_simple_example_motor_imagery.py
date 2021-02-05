"""
===========================
Motor Imagery CSP + LDA Classification
===========================
In this example, we will go through all the steps to make a simple BCI classification task, 
downloading a dataset and using a standard classifier. We choose the dataset 2a from BCI 
Competition IV, a motor imagery task. We will use a CSP to enhance the signal-to-noise
ratio of the EEG epochs and a LDA to classify these signals.
"""
# Authors: Pedro L. C. Rodrigues, Marco Congedo
# Sylvain Chevallier
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

import os

import moabb
from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

moabb.set_log_level('info')
import warnings
warnings.filterwarnings("ignore")

##############################################################################
# Instantiating dataset
# ----------------
#
# The first thing to do is instantiating the dataset that we want to analyse. 
# MOABB has a list of many different datasets, each one #containing all the 
# necessary information for describing them, such as number of subjects, size
# of trials, names of classes, etc.
#
# The dataset class has methods for:
#
# - downloading its files from some online source (e.g. Zenodo)
# - importing the data from the files in whatever extension they might be 
# (like .mat, .gdf, etc.) and instantiate a Raw object from the MNE package

dataset = BNCI2014001()

##############################################################################
# Acessing EEG recording
# ----------------
#
# As an example, we may access the EEG recording from a given session and a given run as follows:
#
# This returns a MNE Raw object that can be manipulated. For some people, this
# might be enough, since the pre-processing and epoching steps can be easily
# done via MNE. However, when assessing the score of a given set of classifiers
# on a loop of several subjects, MOABB ends up being a more appropriate option.

sessions = dataset._get_single_subject_data(subject=1)
session_name = 'session_T'
run_name = 'run_1'
raw = sessions[session_name][run_name]


##############################################################################
# Choosing a paradigm
# ----------------
#
# Once we instantiate a dataset, we have to choose a paradigm. This object is 
# responsable for filtering the data, epoching it, and extracting the labels for
# each epoch. Note that each dataset comes with the names of the paradigms to
# which it might be associated -- It wouldn't make sense to process a P300 
# dataset with a MI paradigm object.
#
# For the example that follows, we will consider the paradigm associated to
# left-hand/right-hand tasks, but there are other options in MOABB as well.
print(dataset.paradigm)
paradigm = LeftRightImagery()

X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

##############################################################################
# Create Pipeline
# ----------
#
# Our goal is to evaluate the performance of a given classification pipeline 
# (or several of them) when it is applied to the epochs from the dataset chosen
# previously. We will consider a very simple classification pipeline in which the
# epochs are dimension reduced via a CSP step and then classified via linear
# discriminant analysis be overwrited if necessary.

pipeline = make_pipeline(CSP(n_components=8), LDA())

##############################################################################
# Evaluation
# ----------
#
# To evaluate the score of this pipeline, we use the 'evaluation' class. When 
# instantiating it, we say which paradigm we want to consider, a list with the 
# datasets to analyse, and whether the scores should be recalculated each time 
# we run the evaluation or if MOABB should create a cache file.
#
# Note that there are different ways of evaluating a classifier; in this example,
# we choose 'WithinSessionEvaluation', which consists of doing a cross-validation 
# procedure where the training and testing partitions are from the same recording 
# session of the dataset. We could have used 'BetweenSessionEvaluation', which takes 
# one session as training partition and another one as testing partition.

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset], overwrite=True)

# We obtain the results in the form of a pandas dataframe
results = evaluation.process({'csp+lda':pipeline}) 

if not os.path.exists('./results'):
    os.mkdir('./results')
results.to_csv('./results/results_part2-1.csv')

##############################################################################
# Plotting results
# ----------
# 
# We create a figure with the seaborn package comparing the classification score
# for each subject on each session. Note that the 'subject' field from the results
# 'object' is given in terms of integers, but seaborn accepts only strings for its
# labeling. This is why we create the field 'subj'.


results = pd.read_csv('./results/results_part2-1.csv')
fig, ax = plt.subplots(figsize=(8,7))
results["subj"] = results["subject"].apply(str)
sns.barplot(x="score", y="subj", hue='session', data=results, orient='h', palette='viridis', ax=ax)
#sns.catplot(kind='bar', x="score", y="subj", hue='session', data=results, orient='h', palette='viridis')
fig.show()
plt.show()