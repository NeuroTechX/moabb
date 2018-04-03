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
# -----------------
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
# metadata have at least 3 columns, subject, session and run.
#
# - subject is the subject id of the corresponding trial
# - session is the session id. A session is a all the data recorded without
# removing the EEG cap.
# - run is the individual continuous recording made during a session. A Session
# may or may not contain multiple run.

print(metadata.head())

###############################################################################
# For this data, we have one subjecy, 2 sessions (2 different recording day)
# and 6 run per session.

print(metadata.describe(include='all'))

###############################################################################
# Paradigm object can also return the list of all dataset compatible. here
# it will return the list all the imagery datasets from the moabb.

compatible_datasets = paradigm.datasets
print([dataset.code for dataset in compatible_datasets])

###############################################################################
# FilterBank MotorImagery
# -----------------------
#
# FilterBankMotorImagery is the same paradigm, but with a different
# preprocessing. In this case, it apply a bank of 6 bandpass filter on the data
# before concatenating the output.

paradigm = FilterBankMotorImagery()

print(paradigm.__doc__)

###############################################################################
# therefore, the output X is a 4D array, with trial x channel x time x filter

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

print(X.shape)

###############################################################################
# LeftRight MotorImagery
# ----------------------
#
# LeftRightImagery is a variation over the BaseMotorImagery paradigm,
# restricted to left and right hand events.

paradigm = LeftRightImagery()

print(paradigm.__doc__)

###############################################################################
# the compatible dataset list is a subset of motor imagery dataset that
# contains at least left and right hand events.

compatible_datasets = paradigm.datasets
print([dataset.code for dataset in compatible_datasets])

###############################################################################
# So if we apply this this to our original dataset, it will only return trials
# corresponding to left and right hand motor imagination.

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

print(np.unique(y))
