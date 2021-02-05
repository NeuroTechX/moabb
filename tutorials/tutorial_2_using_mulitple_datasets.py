"""
===========================
Motor Imagery CSP + LDA Classification (Part 2)
===========================
We extend the previous example to a case where we want to analyse the score of a classifier with three different MI datasets instead of just one. As before, we begin by importing all relevant libraries.
"""
# Authors: Pedro L. C. Rodrigues, Marco Congedo
# Sylvain Chevallier
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

import os

import moabb
from moabb.datasets import BNCI2014001, Weibo2014, Zhou2016
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

import mne
mne.set_log_level("CRITICAL")

##############################################################################
# Initializing datasets
# ----------------------
#
# Then, we instantiate the three diferent datasets that interest us; they all 
# follow the MI paradigm (with left-hand/right-hand classes) but were recorded 
# with different number of electrodes, different number of trials, etc.

datasets = [Zhou2016(), BNCI2014001()]

# The following lines go exactly as in the previous example, where we end up obtaining a pandas dataframe containing the results of the evaluation.

paradigm = LeftRightImagery()
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False)
pipeline = make_pipeline(CSP(n_components=8), LDA())
results = evaluation.process({'csp+lda':pipeline}) 
if not os.path.exists('./results'):
    os.mkdir('./results')
results.to_csv('./results/results_part2-2.csv')

##############################################################################
# Plotting results
# ----------
# 
# Once again, we plot the results using the seaborn library. Note how easy it
# is to plot the results from the three datasets with just one line.

results = pd.read_csv('./results/results_part2-2.csv')
results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
g = sns.catplot(kind='bar', x="score", y="subj", col="dataset", data=results, orient='h', palette='viridis')
plt.show()