""" Example of P300 classification with different epoch size.

=======================================
Changing epoch size in P300 VR dataset
=======================================

This example shows how to extract the epochs from the P300-VR dataset of a given
subject and then classify them using Riemannian Geometry framework for BCI.
We compare the scores in the VR and PC conditions, using different epoch size.

This example demonstrates the use of `get_block_repetition`, which allows
to specify the experimental blocks and repetitions for analysis.
"""

# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
# Modified by: Gregoire Cattan <gcattan@hotmail.fr>
# License: BSD (3-clause)

import warnings

import numpy as np
import pandas as pd
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.datasets import Cattan2019_VR
from moabb.paradigms import P300


warnings.filterwarnings("ignore")

###############################################################################
# Initialization
# ---------------
#
# 1) Create an instance of the dataset.
# 2) Create an instance of a P300 paradigm.
#    By default filtering between 1-24 Hz
#    with epochs of length 1s.
#    In this example we will be modifying the length of the epochs, by
#    changing the `tmax` attribute of the paradigm.
# 3) Encode categorical variable (Target/NonTarget) to numerical values.
#    We will be using label encoding.

dataset = Cattan2019_VR()
paradigm = P300()
le = LabelEncoder().fit(["Target", "NonTarget"])

# change this to include more subjects
nsubjects = 2

###############################################################################
# Validation
# ---------------
#
# We will perform a 3-folds validation for each combination of
# tmax, subjects and experimental conditions (VR or PC).
#
# Not all the data will be used for this validation.
# The Cattan2019_VR dataset contains the data from a randomized experiment.
# We will only be using the two first repetitions of the 12 experimental blocks.
# Data will be selected thanks to the `get_block_repetition` method.

# Contains the score for all combination of tmax, subjects
# and experimental condition (VR or PC).
scores = []

# Init 3-folds validation.
kf = KFold(n_splits=3)

# Select the first two repetitions.
repetitions = [1, 2]

# Generate all possible arrangement with the 12 blocks.
blocks = np.arange(1, 12 + 1)

# run validation for each combination.
for tmax in [0.2, 1.0]:
    paradigm.tmax = tmax

    for subject in tqdm(dataset.subject_list[:nsubjects]):
        # Note: here we are adding `tmax` to scores_subject,
        # although `tmax` is defined outside the scope of this inner loop.
        # The reason behind is to facilitate the conversion from array to dataframe at the end.
        scores_subject = [tmax, subject]

        for condition in ["VR", "PC"]:
            print(f"subject {subject}, {condition}, tmax {tmax}")

            # Rather than creating a new instance depending on the condition,
            # let's change the attribute value to download the correct data.
            dataset.virtual_reality = condition == "VR"
            dataset.personal_computer = condition == "PC"

            auc = []

            # Split in training and testing blocks, and fit/predict.
            # This loop will run 3 times as we are using a 3-folds validation
            for train_idx, test_idx in kf.split(np.arange(12)):
                # Note the use of the `get_block_repetition` method,
                # to select the appropriate number of blocks and repetitions:
                # - 8 blocks for training, 4 for testing
                # - only the first two repetitions inside each blocks
                X_train, y_train, _ = dataset.get_block_repetition(
                    paradigm, [subject], blocks[train_idx], repetitions
                )

                X_test, y_test, _ = dataset.get_block_repetition(
                    paradigm, [subject], blocks[test_idx], repetitions
                )

                # We use riemannian geometry processing techniques with MDM algorithm.
                pipe = make_pipeline(ERPCovariances(estimator="lwf"), MDM())
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                # y_test and y_pred contains categorical variable (Target/NonTarget).
                # To use a metric, we need to convert target information to numerical values.
                y_test = le.transform(y_test)
                y_pred = le.transform(y_pred)

                # We use the roc_auc_score, which is a reliable metric for multi-class problem.
                auc.append(roc_auc_score(y_test, y_pred))

            # stock scores
            scores_subject.append(np.mean(auc))

        scores.append(scores_subject)

###############################################################################
# Display of the data
# ---------------
#
# Let's transform or array to a dataframe.
# We can then print it on the console, and
# plot the mean AUC as a function of the epoch length.

df = pd.DataFrame(scores, columns=["tmax", "subject", "VR", "PC"])

print(df)

df.groupby("tmax").mean().plot(
    y=["VR", "PC"], title="Mean AUC as a function of the epoch length"
)
