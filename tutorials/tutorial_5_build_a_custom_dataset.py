"""
====================================
Tutorial 5: Creating a dataset class
====================================
"""
# Author: Gregoire Cattan
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from sklearn.pipeline import make_pipeline

from moabb.datasets import VirtualReality
from moabb.datasets.braininvaders import bi2014a
from moabb.datasets.shopping import GoShoppingDataset
from moabb.datasets.utils import blocks_reps
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms.p300 import P300


##############################################################################
# Initialization
# ------------------
#
# This tutorial illustrates how to use the GoShoppingDataset to:
# 1) Select a few subjects/sessions/runs in an existing dataset
# 2) Merge two GoShoppingDataset into a new one
# 3) ... and finally use this new dataset on a pipeline
# (this steps is not specific to GoShoppingDataset)
#
# Let's define a paradigm and a pipeline for evaluation first.

paradigm = P300()
pipelines = {}
pipelines["MDM"] = make_pipeline(ERPCovariances(estimator="lwf"), MDM(metric="riemann"))

##############################################################################
# Creation a selection of subject
# ------------------
#
# We are going to great two GoShoppingDataset, namely CustomDataset1 &  2.
# A GoShoppingDataset accepts a shopping_list of subjects.
# It is a list of tuple. A tuple contains 4 values:
# - the original dataset
# - the subject number to select
# - the sessions. It can be:
#   - a session name ('session_0')
#   - a list of sessions (['session_0', 'session_1'])
#   - `None` to select all the sessions attributed to a subjet
# - the runs. As for sessions, it can be a single run name, a list or `None`` (to select all runs).


class CustomDataset1(GoShoppingDataset):
    def __init__(self):
        biVR = VirtualReality(virtual_reality=True, screen_display=True)
        runs = blocks_reps([1, 3], [1, 2, 3, 4, 5])
        shopping_list = [
            (biVR, 1, "VR", runs),
            (biVR, 2, "VR", runs),
        ]
        GoShoppingDataset.__init__(
            self,
            shopping_list=shopping_list,
            events=dict(Target=2, NonTarget=1),
            code="D1",
            interval=[0, 1.0],
            paradigm="p300",
        )


class CustomDataset2(GoShoppingDataset):
    def __init__(self):
        bi2014 = bi2014a()
        shopping_list = [
            (bi2014, 4, None, None),
            (bi2014, 7, None, None),
        ]
        GoShoppingDataset.__init__(
            self,
            shopping_list=shopping_list,
            events=dict(Target=2, NonTarget=1),
            code="D2",
            interval=[0, 1.0],
            paradigm="p300",
        )


##############################################################################
# Merging the datasets
# ------------------
#
# We are now going to merge the two GoShoppingDataset into a single one.
# The implementation is straigh forward. Instead of providing a list of subjects,
# you should provide a list of GoShoppingDataset.
# shopping_list = [CustomDataset1(), CustomDataset2()]


class CustomDataset3(GoShoppingDataset):
    def __init__(self):
        shopping_list = [CustomDataset1(), CustomDataset2()]
        GoShoppingDataset.__init__(
            self,
            shopping_list=shopping_list,
            events=dict(Target=2, NonTarget=1),
            code="D3",
            interval=[0, 1.0],
            paradigm="p300",
        )


##############################################################################
# Evaluate and display
# ------------------
#
# Let's use a WithinSessionEvaluation to evaluate our new dataset.
# If you already new how to do this, nothing changed:
# The GoShoppingDataset can be used as a `normal` dataset.

datasets = [CustomDataset3()]
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, overwrite=False, suffix="newdataset"
)
scores = evaluation.process(pipelines)

print(scores)
