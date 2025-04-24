"""
==========================
Pipelines using the mne-features library
==========================

This example shows how to evaluate a pipeline constructed using the
mne-features library

"""

# Authors: Alexander de Ranitz <alexanderderanitz@gmail.com>
#          Luuk Neervens <luuk.neervens@ru.nl>
#          Charlynn van Osch <charlynn.vanosch@ru.nl>
#
# License: BSD (3-clause)

import warnings

import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne_features.feature_extraction import FeatureExtractor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001, Zhou2016
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

##############################################################################
# Creating Pipelines using mne-features
# ------------------
#
# Here, we closely follow Tutorial 3, but we create pipelines using features
# extracted using the mne-features library. We instantiate the three different
# classiciation pipelines to be considered in the analysis.
# See the mne-features docs to learn more about the available features:
# https://mne.tools/mne-features/api.html#api-documentation


# mne-feature's FeatureExtractor can be used directly in our pipelines
# as it implements the fit() and transform() methods
# (note that fit does not have any effect, but it is implemented for compatibility).

# We can specify which features we want to extract as a list of strings, see
# https://mne.tools/mne-features/generated/mne_features.feature_extraction.FeatureExtractor.html#mne_features.feature_extraction.FeatureExtractor
variance = FeatureExtractor(250.0, ["variance"])
ptp_amp = FeatureExtractor(250.0, ["ptp_amp"])

# We can also extract several features by passing more than one feature.
both = FeatureExtractor(250.0, ["ptp_amp", "variance"])

pipelines = {}
pipelines["var+LDA"] = make_pipeline(variance, LDA())
pipelines["ptp_amp+LDA"] = make_pipeline(ptp_amp, LDA())
pipelines["var+ptp_amp+LDA"] = make_pipeline(both, LDA())

##############################################################################
# The rest is the same as in previous tutorials!

datasets = [BNCI2014_001(), Zhou2016()]
subj = [1, 2, 3]
for d in datasets:
    d.subject_list = subj
paradigm = LeftRightImagery()
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, overwrite=False
)
results = evaluation.process(pipelines)

##############################################################################
# Plotting Results
# ----------------
#
# The following plot shows a comparison of the three classification pipelines
# for each subject of each dataset.

results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
g = sns.catplot(
    kind="bar",
    x="score",
    y="subj",
    hue="pipeline",
    col="dataset",
    height=12,
    aspect=0.5,
    data=results,
    orient="h",
    palette="viridis",
)
plt.show()
