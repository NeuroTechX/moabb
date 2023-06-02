"""
=============================
Changing epoch size in P300 VR dataset
=============================

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

from moabb.datasets import VirtualReality
from moabb.paradigms import P300


warnings.filterwarnings("ignore")

# create dataset
dataset = VirtualReality()

# To encode classes into 0 and 1.
le = LabelEncoder().fit(["Target", "NonTarget"])

# get the paradigm
paradigm = P300()

# change this to include more subjects
nsubjects = 2

scores = []
for tmax in [0.2, 1.0]:
    paradigm.tmax = tmax

    for subject in tqdm(dataset.subject_list[:nsubjects]):
        scores_subject = [tmax, subject]

        for condition in ["VR", "PC"]:
            print(f"subject {subject}, {condition}, tmax {tmax}")

            # define the dataset instance
            dataset.virtual_reality = condition == "VR"
            dataset.personal_computer = condition == "PC"

            # cross validate with 3-folds validation.
            kf = KFold(n_splits=3)

            # There is 12 blocks of 5 repetitions.
            repetitions = [1, 2]  # Select the first two repetitions.
            blocks = np.arange(1, 12 + 1)

            auc = []

            # split in training and testing blocks, and fit/predict.
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

                pipe = make_pipeline(ERPCovariances(estimator="lwf"), MDM())
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                y_test = le.transform(y_test)
                y_pred = le.transform(y_pred)

                auc.append(roc_auc_score(y_test, y_pred))

            # stock scores
            scores_subject.append(np.mean(auc))

        scores.append(scores_subject)

df = pd.DataFrame(scores, columns=["tmax", "subject", "VR", "PC"])

print(df)

df.groupby("tmax").mean().plot(
    y=["VR", "PC"], title="Mean AUC as a function of the epoch size"
)
