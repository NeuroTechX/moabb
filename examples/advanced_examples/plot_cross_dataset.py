"""Cross-dataset motor imagery classification example.

This example demonstrates how to perform cross-dataset evaluation using MOABB,
training on one dataset and testing on another.
"""

# Standard library imports
import logging
from typing import List

# Third-party imports
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.io.cnt.cnt import RawCNT
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
logging.getLogger("mne").setLevel(logging.ERROR)


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
        if hasattr(X, "get_data"):
            picks = mne.pick_channels(
                X.info["ch_names"], include=common_channels, ordered=True
            )
            data = X.get_data()
            if data.ndim == 2:
                data = data.reshape(1, *data.shape)
            data = data[:, picks, :]
            return data
        return X

    pipeline = Pipeline(
        [
            ("to_array", FunctionTransformer(raw_to_data)),
            ("covariances", Covariances(estimator="oas")),
            ("csp", CSP(nfilter=4, log=True)),
            ("classifier", SVC(kernel="rbf", C=0.1)),
        ]
    )

    return pipeline


# Define datasets
train_dataset = BNCI2014001()
test_dataset = Zhou2016()

# Create a dictionary of datasets for easier handling
datasets_dict = {"train_dataset": train_dataset, "test_dataset": test_dataset}

# Get the list of channels from each dataset before matching
print("\nChannels before matching:")
for ds_name, ds in datasets_dict.items():
    try:
        # Load data for first subject to get channel information
        data = ds.get_data([ds.subject_list[0]])  # Get data for first subject
        first_subject = list(data.keys())[0]
        first_session = list(data[first_subject].keys())[0]
        first_run = list(data[first_subject][first_session].keys())[0]
        run_data = data[first_subject][first_session][first_run]

        if isinstance(run_data, (RawArray, RawCNT)):
            channels = run_data.info["ch_names"]
        else:
            # Assuming the channels are stored in the dataset class after loading
            channels = ds.channels
        print(f"{ds_name}: {channels}")
    except Exception as e:
        print(f"Error getting channels for {ds_name}: {str(e)}")

# Use MOABB's match_all for channel handling
print("\nMatching channels across datasets...")
paradigm = MotorImagery()

# Apply match_all to all datasets
all_datasets = list(datasets_dict.values())
paradigm.match_all(all_datasets, channel_merge_strategy="intersect")

# Get channels from all datasets after matching to ensure we have the correct intersection
all_channels_after_matching = []
print("\nChannels after matching:")
for i, (ds_name, _) in enumerate(datasets_dict.items()):
    ds = all_datasets[i]  # Get the matched dataset
    try:
        data = ds.get_data([ds.subject_list[0]])
        subject = list(data.keys())[0]
        session = list(data[subject].keys())[0]
        run = list(data[subject][session].keys())[0]
        run_data = data[subject][session][run]

        if isinstance(run_data, (RawArray, RawCNT)):
            channels = run_data.info["ch_names"]
        else:
            channels = ds.channels
        all_channels_after_matching.append(set(channels))
        print(f"{ds_name}: {channels}")
    except Exception as e:
        print(f"Error getting channels for {ds_name} after matching: {str(e)}")

# Get the intersection of all channel sets
common_channels = sorted(list(set.intersection(*all_channels_after_matching)))
print(f"\nCommon channels after matching: {common_channels}")
print(f"Number of common channels: {len(common_channels)}")

# Update the datasets_dict with the matched datasets
for i, (name, _) in enumerate(datasets_dict.items()):
    datasets_dict[name] = all_datasets[i]

train_dataset = datasets_dict["train_dataset"]
test_dataset = datasets_dict["test_dataset"]

# Initialize the paradigm with common channels
paradigm = MotorImagery(channels=common_channels, n_classes=2, fmin=8, fmax=32)

# Initialize the CrossDatasetEvaluation
evaluation = CrossDatasetEvaluation(
    paradigm=paradigm,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    hdf5_path="./res_test",
    save_model=True,
)

# Run the evaluation
results = []
for result in evaluation.evaluate(
    dataset=None, pipelines={"CSP_SVM": create_pipeline(common_channels)}
):
    result["subject"] = "all"
    print(f"Cross-dataset score: {result.get('score', 'N/A'):.3f}")
    results.append(result)

# Convert results to DataFrame and process
results_df = pd.DataFrame(results)
results_df["dataset"] = results_df["dataset"].apply(lambda x: x.__class__.__name__)

# Print evaluation scores
print("\nCross-dataset evaluation scores:")
print(results_df[["dataset", "score", "time"]])

# Plot the results
score_plot(results_df)
plt.show()
