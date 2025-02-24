"""Cross-dataset motor imagery classification example.

This example demonstrates how to perform cross-dataset evaluation using MOABB,
training on one dataset and testing on another.
"""

# Standard library imports
import logging
from typing import Any, List

import matplotlib.pyplot as plt

# Third-party imports
import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

# MOABB imports
from moabb import set_log_level
from moabb.analysis.plotting import score_plot
from moabb.datasets import BNCI2014001, Zhou2016
from moabb.evaluations.evaluations import CrossDatasetEvaluation
from moabb.paradigms import MotorImagery


# Configure logging
set_log_level("WARNING")
logging.getLogger('mne').setLevel(logging.ERROR)


def get_common_channels(datasets: List[Any]) -> List[str]:
    """Get channels that are available across all datasets.

    Parameters
    ----------
    datasets : List[Dataset]
        List of MOABB dataset objects to analyze

    Returns
    -------
    List[str]
        Sorted list of common channel names
    """
    all_channels = []
    for dataset in datasets:
        # Get a sample raw from each dataset
        subject = dataset.subject_list[0]
        raw_dict = dataset.get_data([subject])

        # Navigate through the nested dictionary structure
        subject_data = raw_dict[subject]
        first_session = list(subject_data.keys())[0]
        first_run = list(subject_data[first_session].keys())[0]
        raw = subject_data[first_session][first_run]

        all_channels.append(raw.ch_names)

    # Find common channels across all datasets
    common_channels = set.intersection(*map(set, all_channels))
    return sorted(list(common_channels))


def create_pipeline(common_channels: List[str]) -> Pipeline:
    """Create classification pipeline with CSP and SVM.

    Parameters
    ----------
    common_channels : List[str]
        List of channel names to use in the pipeline

    Returns
    -------
    Pipeline
        Sklearn pipeline for classification
    """
    def raw_to_data(X: np.ndarray) -> np.ndarray:
        """Convert raw MNE data to numpy array format.

        Parameters
        ----------
        X : np.ndarray or mne.io.Raw
            Input data to convert

        Returns
        -------
        np.ndarray
            Converted data array
        """
        if hasattr(X, 'get_data'):
            picks = mne.pick_channels(
                X.info['ch_names'],
                include=common_channels,
                ordered=True
            )
            data = X.get_data()
            if data.ndim == 2:
                data = data.reshape(1, *data.shape)
            data = data[:, picks, :]
            return data
        return X

    pipeline = Pipeline([
        ('to_array', FunctionTransformer(raw_to_data)),
        ('covariances', Covariances(estimator='oas')),
        ('csp', CSP(nfilter=4, log=True)),
        ('classifier', SVC(kernel='rbf', C=0.1))
    ])

    return pipeline


# Define datasets
train_dataset = BNCI2014001()
test_dataset = Zhou2016()

# Get common channels across datasets
common_channels = get_common_channels([train_dataset, test_dataset])
print(f"\nCommon channels across datasets: {common_channels}\n")

# Initialize the paradigm with common channels
paradigm = MotorImagery(
    channels=common_channels,
    n_classes=2,
    fmin=8,
    fmax=32
)

# Initialize the CrossDatasetEvaluation
evaluation = CrossDatasetEvaluation(
    paradigm=paradigm,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    hdf5_path="./res_test",
    save_model=True
)

# Run the evaluation
results = []
for result in evaluation.evaluate(
    dataset=None,
    pipelines={'CSP_SVM': create_pipeline(common_channels)}
):
    result['subject'] = 'all'
    print(f"Cross-dataset score: {result.get('score', 'N/A'):.3f}")
    results.append(result)

# Convert results to DataFrame and process
results_df = pd.DataFrame(results)
results_df['dataset'] = results_df['dataset'].apply(
    lambda x: x.__class__.__name__
)

# Print evaluation scores
print("\nCross-dataset evaluation scores:")
print(results_df[['dataset', 'score', 'time']])

# Plot the results
score_plot(results_df)
plt.show()
