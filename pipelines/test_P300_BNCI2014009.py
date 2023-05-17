"""
Comparison of the classification accuracy of the ERPCovariances and XdawnCovariances using
Minimum Distance to Mean (MDM) and Tangent Space classifiers

Based on: https://github.com/NeuroTechX/moabb/blob/develop/examples/plot_within_session_p300.py
"""

import matplotlib.pyplot as plt
import seaborn as sb

import moabb
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from moabb.datasets import BNCI2014009

from sklearn.pipeline import make_pipeline

from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.classification import MDM, TSclassifier
from sklearn.linear_model import LogisticRegression as LR

# Defining paradigm and database
######################################################################################################################
p300_paradigm = P300(resample=128)
p300_dataset = BNCI2014009()
moabb.set_log_level("info")
labels_dict = {"Target": 1, "NonTarget": 0}

# ERPCovariances features
######################################################################################################################
pipelines = {}
pipelines["ERPCov+MDM"] = make_pipeline(
    ERPCovariances(classes=[labels_dict["Target"]], estimator='scm'),
    MDM())
pipelines["ERPCov+TS+LR_L1"] = make_pipeline(
    ERPCovariances(classes=[labels_dict["Target"]], estimator='scm'), 
    TSclassifier(metric='riemann', clf=LR(penalty='l1', max_iter=1000, solver='liblinear')))
pipelines["ERPCov+TS+LR_L2"] = make_pipeline(
    ERPCovariances(classes=[labels_dict["Target"]], estimator='scm'), 
    TSclassifier(metric='riemann', clf=LR(penalty='l2', max_iter=1000, solver='saga')))
pipelines["ERPCov+TS+LR_EN"] = make_pipeline(
    ERPCovariances(classes=[labels_dict["Target"]], estimator='scm'), 
    TSclassifier(metric='riemann', clf=LR(penalty='elasticnet', max_iter=1000, l1_ratio=0.5, solver='saga')))

# XdawnCovariances features
######################################################################################################################
pipelines["XdawnCov+MDM"] = make_pipeline(
    XdawnCovariances(nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"),
    MDM())
pipelines["XdawnCov+TS+LR_L1"] = make_pipeline(
    XdawnCovariances(nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"), 
    TSclassifier(metric='riemann', clf=LR(penalty='l1', max_iter=1000, solver='liblinear')))
pipelines["XdawnCov+TS+LR_L2"] = make_pipeline(
    XdawnCovariances(nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"), 
    TSclassifier(metric='riemann', clf=LR(penalty='l2', max_iter=1000, solver='saga')))
pipelines["XdawnCov+TS+LR_EN"] = make_pipeline(
    XdawnCovariances(nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"), 
    TSclassifier(metric='riemann', clf=LR(penalty='elasticnet', max_iter=1000, l1_ratio=0.5, solver='saga')))

# Run within session evaluations 
######################################################################################################################
p300_dataset.subject_list = p300_dataset.subject_list[:1]
datasets = [p300_dataset]
evaluation = WithinSessionEvaluation(paradigm=p300_paradigm, datasets=p300_dataset, suffix="examples", overwrite=True)
results = evaluation.process(pipelines)

# Generating plot and saving results
######################################################################################################################
fig, ax = plt.subplots(facecolor="white", figsize=[20, 5])
sb.stripplot(data=results, y="score", x="pipeline", ax=ax, jitter=True, alpha=0.5, zorder=1, palette="Set1")
sb.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1")
ax.set_ylabel("ROC AUC")
ax.set_ylim(0.5, 1)

# Saving results per session and subjects
results.to_csv("./results_P300.csv")
fig.savefig("./results_P300.png")


