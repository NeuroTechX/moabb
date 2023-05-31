from moabb.datasets import VirtualReality

import numpy as np
import pandas as pd

from moabb.paradigms import P300
from pyriemann.estimation import ERPCovariances
from pyriemann.classification import MDM

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from tqdm import tqdm

"""
=============================
Classification of the trials
=============================

This example shows how to extract the epochs from the P300-VR dataset of a given
subject and then classify them using Riemannian Geometry framework for BCI. 
We compare the scores in the VR and PC conditions, using different epoch size.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
# Modified by: Gregoire Cattan <gcattan@hotmail.fr>
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

# create dataset
dataset = VirtualReality()

# To encode classes into 0 and 1.
le = LabelEncoder().fit(["Target", "NonTarget"])

# get the paradigm
paradigm = P300()

# change this to include more subjects
nsubjects = 2

scores_epochs = []
for tmax in [0.2, 1.0]:

	paradigm.tmax = tmax

	scores = []
	for subject in tqdm(dataset.subject_list[:nsubjects]):
		scores_subject = [tmax, subject]

		for condition in ['VR', 'PC']:

			print(f"subject {subject}, {condition}, tmax {tmax}")

			# define the dataset instance
			dataset.virtual_reality = (condition == 'VR')
			dataset.personal_computer = (condition == 'PC')

            # cross validate with 3-folds validation.
			kf = KFold(n_splits = 3)
			
            # There is 12 blocks of 5 repetitions.
			repetitions = [1, 2] # Select the first two repetitions.
			blocks = np.arange(1, 12+1)
					
			auc = []

			for train_idx, test_idx in kf.split(np.arange(12)):

				# split in training and testing blocks
				X_train, y_train, _ = dataset.get_block_repetition(paradigm, [subject], blocks[train_idx], repetitions)

				X_test, y_test, _ = dataset.get_block_repetition(paradigm, [subject], blocks[test_idx], repetitions)

				pipe = make_pipeline(ERPCovariances(estimator='lwf'), MDM())
				pipe.fit(X_train, y_train)
				y_pred = pipe.predict(X_test)

				y_test = le.transform(y_test)
				y_pred= le.transform(y_pred)

				auc.append(roc_auc_score(y_test, y_pred))

			# stock scores
			scores_subject.append(np.mean(auc))

		scores.append(scores_subject)

	scores_epochs.append(scores)

print(scores_epochs)
df = pd.DataFrame(np.array(scores_epochs), columns=['tmax', 'subject', 'VR', 'PC'])
print(df)