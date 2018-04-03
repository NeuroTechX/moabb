"""
=======================
Explore Paradigm Object
=======================

A paradigm defines how the raw data will be converted to trials ready
to be processed by a decoding algorithm. This is a function of the paradigm
used, i.e. in motor imagery one can have two-class, multi-class,
or continuous paradigms; similarly, different preprocessing is necessary
for ERP vs ERD paradigms.

A paradigm also defines the appropriate evaluation metric, for example AUC
for binary classification problem, accuracy for multiclass, or kappa
coefficient for continuous paradigms.

This tutorial explore the paradigm object, with 3 examples of paradigm :
 - BaseMotorImagery
 - FilterBankMotorImagery
 - LeftRightImagery

"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import (LeftRightImagery, BaseMotorImagery,
                             FilterBankMotorImagery)

print(__doc__)

###############################################################################
# Base MotorImagery
# -------------
#
# First, lets take a example of the BaseMotorImagery paradigm.

paradigm = BaseMotorImagery()

print(paradigm.__doc__)

###############################################################################
# The function `get_data` allow you to access preprocessed data from a dataset.
# this function will return 3 objects. A numpy array containing the
# preprocessed EEG data, the labels, and a dataframe with metadata.

print(paradigm.get_data.__doc__)

###############################################################################
# Lets take the example of the BNCI2014001 dataset, known as the dataset IIa
# from the BCI competition IV. We will load the data from the subject 1.
# When calling `get_data`, the paradigm will retrieve the data from the
# specified list of subject, apply preprocessing (by default, a bandpass
# between 7 and 35 Hz), epoch the data (with interval specified by the dataset,
# unless superseeded by the paradigm) and return the corresponding objects.

dataset = BNCI2014001()
subjects = [1]

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

###############################################################################
# The epoched data is a 3D array, with epochs on the first dimension (here
# 576 trials), channels on the second (22 channels) and time sample on the last
# one.

print(X.shape)

###############################################################################
# Labels contains the labels corresponding to each trial. in the case of this
# dataset, we have the 4 type of motor imagery that was performed.

print(np.unique(y))

###############################################################################
# metadata

print(metadata.head())

###############################################################################
# Load our data

print(metadata.describe(include='all'))

###############################################################################
# Load our data

compatible_datasets = paradigm.datasets
print([dataset.code for dataset in compatible_datasets])

###############################################################################
# FilterBank MotorImagery
# -------------
#
# First we'll load the data we'll use in connectivity estimation. We'll use
# the sample MEG data provided with MNE.

paradigm = FilterBankMotorImagery()

print(paradigm.__doc__)

###############################################################################
# Load our data

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

print(X.shape)

###############################################################################
# LeftRight MotorImagery
# -------------
#
# First we'll load the data we'll use in connectivity estimation. We'll use
# the sample MEG data provided with MNE.

paradigm = LeftRightImagery()

print(paradigm.__doc__)

###############################################################################
# Load our data

compatible_datasets = paradigm.datasets
print([dataset.code for dataset in compatible_datasets])

###############################################################################
# Load our data

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

print(np.unique(y))
