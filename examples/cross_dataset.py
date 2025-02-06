from moabb import set_log_level
from moabb.datasets import BNCI2014001, Zhou2016
from moabb.paradigms import MotorImagery
from moabb.evaluations.evaluations import CrossDatasetEvaluation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from moabb.analysis.plotting import score_plot
import pandas as pd
import mne
import logging

# Configure logging - reduce verbosity
set_log_level("WARNING")  # Changed from "info" to "WARNING"
logging.getLogger('mne').setLevel(logging.ERROR)  # Reduce MNE logging

def get_common_channels(datasets):
    """Get channels that are available across all datasets."""
    all_channels = []
    for dataset in datasets:
        # Get a sample raw from each dataset
        subject = dataset.subject_list[0]
        raw_dict = dataset.get_data([subject])
        # Navigate through the nested dictionary structure
        subject_data = raw_dict[subject]  # Get subject's data
        first_session = list(subject_data.keys())[0]  # Get first session
        first_run = list(subject_data[first_session].keys())[0]  # Get first run
        raw = subject_data[first_session][first_run]  # Get raw data
        all_channels.append(raw.ch_names)
    
    # Find common channels across all datasets
    common_channels = set.intersection(*map(set, all_channels))
    return sorted(list(common_channels))

def create_pipeline(common_channels) -> Pipeline:
    """Create classification pipeline."""
    def raw_to_data(X):
        """Convert raw MNE data to numpy array format"""
        if hasattr(X, 'get_data'):
            # Get only common channels to ensure consistency
            picks = mne.pick_channels(X.info['ch_names'], 
                                    include=common_channels,
                                    ordered=True)
            data = X.get_data()
            if data.ndim == 2:
                data = data.reshape(1, *data.shape)
            data = data[:, picks, :]
            return data
        return X

    pipeline = Pipeline([
        ('to_array', FunctionTransformer(raw_to_data)),
        ('covariances', Covariances(estimator='oas')),
        ('csp', CSP(nfilter=4, log=True)),  # Changed n_components to nfilter, removed invalid parameters
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
    channels=common_channels,  # Use common channels
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
    pipelines={'CSP_SVM': create_pipeline(common_channels)},
    param_grid=None
):
    result['subject'] = 'all'
    print(f"Cross-dataset score: {result.get('score', 'N/A'):.3f}")
    results.append(result)

# Convert list of results to DataFrame
results_df = pd.DataFrame(results)
results_df['dataset'] = results_df['dataset'].apply(lambda x: x.__class__.__name__)

# Print evaluation scores
print("\nCross-dataset evaluation scores:")
print(results_df[['dataset', 'score', 'time']])

# Plot the results
score_plot(results_df)
plt.show()