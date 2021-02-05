"""
===========================
Motor Imagery CSP + LDA Classification (Part 3)
===========================
In this last part, we extend the previous example by assessing the classification score of not one but three classification pipelines. Once again, we begin by importing all the required packages to make the script work.
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
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mne
mne.set_log_level("CRITICAL")

moabb.set_log_level('info')
import warnings
warnings.filterwarnings("ignore")

##############################################################################
# Creating Pipelines
# ----------------------
# 
# Then, we instantiate the three different classiciation pipelines to be considered
# in the analysis. The object that gathers each pipeline is a dictionary.
pipelines = {}
pipelines['csp+lda'] = make_pipeline(CSP(n_components=8), LDA())
pipelines['tgsp+svm'] = make_pipeline(Covariances('oas'), TangentSpace(metric='riemann'), SVC(kernel='linear'))
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM(metric='riemann'))

# The following lines go exactly as in the previous example, where we end up obtaining a pandas dataframe containing the results of the evaluation.
datasets = [BNCI2014001(), Zhou2016()]
paradigm = LeftRightImagery()
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)
results = evaluation.process(pipelines) 
if not os.path.exists('./results'):
    os.mkdir('./results')
results.to_csv('./results/results_part2-3.csv')

##############################################################################
# Plotting results
# ----------

results = pd.read_csv('./results/results_part2-3.csv')
results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
g = sns.catplot(kind='bar', x="score", y="subj", hue="pipeline", col="dataset", height=12, aspect=0.5, data=results, orient='h', palette='viridis')
plt.show()