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

This example explore the paradigm object, with 3 examples of paradigm :
 - BaseMotorImagery
 - FilterBankMotorImagery
 - LeftRightImagery

"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import (LeftRightImagery, BaseMotorImagery,
                             FilterBankMotorImagery)

###############################################################################
# Load our data
# -------------
#
# First we'll load the data we'll use in connectivity estimation. We'll use
# the sample MEG data provided with MNE.

paradigm = BaseMotorImagery()

print(paradigm.__doc__)

###############################################################################
# Load our data

print(paradigm.get_data.__doc__)

###############################################################################
# Load our data

dataset = BNCI2014001()
subjects = [1]

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

###############################################################################
# Load our data

print(X.shape)

###############################################################################
# Load our data

print(np.unique(y))

###############################################################################
# Load our data

print(metadata.head())

###############################################################################
# Load our data

print(metadata.describe(include='all'))

###############################################################################
# Load our data

compatible_datasets = paradigm.datasets
print([dataset.code for dataset in compatible_datasets])

###############################################################################
# Load our data
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
# Load our data
# -------------
#
# First we'll load the data we'll use in connectivity estimation. We'll use
# the sample MEG data provided with MNE.

paradigm = LeftRightImagery()

print(paradigm.__doc__)

###############################################################################
# Load our data

print(paradigm.datasets)

###############################################################################
# Load our data

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

print(np.unique(y))
