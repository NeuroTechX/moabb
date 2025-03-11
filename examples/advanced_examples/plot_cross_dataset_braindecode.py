"""
Cross-Dataset Brain Decoding with Deep Learning
=============================================
This example shows how to train deep learning models on one dataset
and test on another using Braindecode.
"""

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.util import set_random_seeds
from braindecode.visualization import plot_confusion_matrix
from matplotlib.lines import Line2D
from mne.io import RawArray
from mne.io.cnt.cnt import RawCNT
from sklearn.metrics import confusion_matrix
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from moabb.datasets import (
    BNCI2014_001,
    BNCI2014_004,
    BNCI2015_001,
    Zhou2016,
)
from moabb.paradigms import MotorImagery


# Configure logging
logging.basicConfig(level=logging.WARNING)


def get_all_events(train_dataset, test_dataset):
    """Get all unique events across datasets."""
    # Get events from first subject of each dataset
    train_events = train_dataset.datasets[0].raw.annotations.description
    test_events = test_dataset.datasets[0].raw.annotations.description

    # Get all unique events
    all_events = sorted(list(set(train_events).union(set(test_events))))
    print(f"\nAll unique events across datasets: {all_events}\n")

    # Create event mapping (event description -> numerical ID)
    event_id = {str(event): idx for idx, event in enumerate(all_events)}
    print(f"Event mapping: {event_id}\n")

    return event_id


def create_fixed_windows(
    dataset: BaseConcatDataset,
    samples_before: int,
    samples_after: int,
    event_id: Dict[str, int],
) -> BaseConcatDataset:
    """Create windows with consistent size across datasets.

    Parameters
    ----------
    dataset : BaseConcatDataset
        Dataset to create windows from
    samples_before : int
        Number of samples before event
    samples_after : int
        Number of samples after event
    event_id : Dict[str, int]
        Mapping of event names to numerical IDs

    Returns
    -------
    BaseConcatDataset
        Windowed dataset
    """
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=-samples_before,
        trial_stop_offset_samples=samples_after,
        preload=True,
        mapping=event_id,
        window_size_samples=samples_before + samples_after,
        window_stride_samples=samples_before + samples_after,
    )


def standardize_windows(
    train_dataset: BaseConcatDataset,
    test_dataset: BaseConcatDataset,
    all_channels: List[str],
    event_id: Dict[str, int],
) -> Tuple[BaseConcatDataset, BaseConcatDataset, int, int, int]:
    """Standardize datasets with consistent preprocessing.

    Parameters
    ----------
    train_dataset : BaseConcatDataset
        Training dataset to standardize
    test_dataset : BaseConcatDataset
        Test dataset to standardize
    all_channels : List[str]
        List of all required channel names
    event_id : Dict[str, int]
        Mapping of event names to numerical IDs

    Returns
    -------
    Tuple[BaseConcatDataset, BaseConcatDataset, int, int, int]
        Processed training windows, test windows, window length,
        samples before and after
    """
    target_sfreq = 100  # Target sampling frequency

    print("\nInitial dataset properties:")
    for name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        for i, ds in enumerate(dataset.datasets):
            print(f"{name} dataset {i}:")
            print(f"  Sampling rate: {ds.raw.info['sfreq']} Hz")
            print(f"  Number of channels: {len(ds.raw.info['ch_names'])}")
            print(f"  Channel names: {ds.raw.info['ch_names']}")

    # Get the actual available channels (intersection of all datasets)
    available_channels = set(train_dataset.datasets[0].raw.info["ch_names"])
    for ds in train_dataset.datasets[1:] + test_dataset.datasets:
        available_channels = available_channels.intersection(ds.raw.info["ch_names"])
    available_channels = sorted(list(available_channels))

    print(
        f"\nCommon channels across all datasets ({len(available_channels)}): {available_channels}"
    )

    # Verify all datasets have the same number of channels
    for name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        for i, ds in enumerate(dataset.datasets):
            if len(ds.raw.info["ch_names"]) != len(available_channels):
                print(
                    f"Warning: {name} dataset {i} has {len(ds.raw.info['ch_names'])} channels, "
                    f"expected {len(available_channels)}"
                )

    # Define preprocessing pipeline using only available channels
    preprocessors = [
        Preprocessor("pick_channels", ch_names=available_channels, ordered=True),
        Preprocessor("resample", sfreq=target_sfreq),
        Preprocessor(lambda data: np.multiply(data, 1e6)),
        Preprocessor("filter", l_freq=4.0, h_freq=38.0),
        Preprocessor(
            exponential_moving_standardize, factor_new=1e-3, init_block_size=1000
        ),
    ]

    # Apply preprocessing
    print("\nApplying preprocessing...")
    preprocess(train_dataset, preprocessors, n_jobs=-1)
    preprocess(test_dataset, preprocessors, n_jobs=-1)

    # Verify channel counts after preprocessing
    print("\nVerifying channel counts after preprocessing:")
    for name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        for i, ds in enumerate(dataset.datasets):
            n_channels = len(ds.raw.info["ch_names"])
            print(f"{name} dataset {i} has {n_channels} channels")
            if n_channels != len(available_channels):
                raise ValueError(
                    f"Channel count mismatch in {name} dataset {i}: "
                    f"got {n_channels}, expected {len(available_channels)}"
                )

    # Define window parameters
    window_start = -0.5  # Start 0.5s before event
    window_duration = 4.0  # Window duration in seconds
    samples_before = int(abs(window_start) * target_sfreq)
    samples_after = int(window_duration * target_sfreq)

    # Standardize event durations
    for dataset in [train_dataset, test_dataset]:
        for ds in dataset.datasets:
            events = ds.raw.annotations[:-1]
            new_annotations = mne.Annotations(
                onset=events.onset,
                duration=np.zeros_like(events.duration),
                description=events.description,
            )
            ds.raw.set_annotations(new_annotations)

    # Create and validate windows
    print("\nCreating windows...")
    train_windows = create_fixed_windows(
        train_dataset, samples_before, samples_after, event_id
    )
    test_windows = create_fixed_windows(
        test_dataset, samples_before, samples_after, event_id
    )

    # Verify window shapes
    train_shape = train_windows[0][0].shape
    test_shape = test_windows[0][0].shape
    print(f"\nWindow shapes - Train: {train_shape}, Test: {test_shape}")

    if train_shape != test_shape:
        raise ValueError(
            f"Window shapes don't match: train={train_shape}, test={test_shape}"
        )

    window_length = train_shape[1]
    return train_windows, test_windows, window_length, samples_before, samples_after


# Load datasets with validation
def load_and_validate_dataset(dataset_class, subject_ids):
    """Load dataset and validate its contents.

    Parameters
    ----------
    dataset_class : MOABB Dataset class
        The dataset class to instantiate
    subject_ids : list
        List of subject IDs to include

    Returns
    -------
    dataset : MOABB Dataset
        The loaded and validated dataset
    """
    # Initialize the dataset
    dataset = dataset_class()
    dataset.subject_list = subject_ids

    print(f"\nValidating dataset: {dataset.__class__.__name__}")

    try:
        # Get data in MOABB format
        data = dataset.get_data()

        # Validate each subject's data
        for subject_id in subject_ids:
            print(f"\nSubject {subject_id}:")
            if subject_id not in data:
                print(f"No data found for subject {subject_id}")
                continue

            subject_data = data[subject_id]
            for session_name, session_data in subject_data.items():
                print(f"Session: {session_name}")
                for run_name, run_data in session_data.items():
                    print(f"Run: {run_name}")

                    # Handle both dictionary and MNE Raw object formats
                    if isinstance(run_data, (RawArray, RawCNT)):
                        data_array = run_data.get_data()
                        events = mne.events_from_annotations(run_data)[0]
                        print(f"Data shape: {data_array.shape}")
                        print(f"Number of events: {len(events)}")
                        if len(events) > 0:
                            print(f"Event types: {np.unique(events[:, -1])}")
                    elif isinstance(run_data, dict):
                        if "data" in run_data and "events" in run_data:
                            data_array = run_data["data"]
                            events_array = run_data["events"]
                            print(f"Data shape: {data_array.shape}")
                            print(f"Events shape: {events_array.shape}")
                            if events_array.size > 0:
                                print(f"Event types: {np.unique(events_array[:, -1])}")
                        else:
                            print("Warning: Run data missing required keys")
                    else:
                        print(f"Warning: Unexpected run_data type: {type(run_data)}")

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

    return dataset


# The conversion between MOABB and Braindecode formats is necessary because:
# 1. MOABB's match_all provides robust channel matching and interpolation
# 2. Braindecode's training pipeline expects its own data format
# 3. We need to preserve both the benefits of MOABB's preprocessing and Braindecode's training


def convert_moabb_to_braindecode(moabb_dataset, subject_ids, channels):
    """Convert MOABB dataset format to Braindecode format.

    Parameters
    ----------
    moabb_dataset : MOABB Dataset
        The MOABB dataset to convert
    subject_ids : list
        List of subject IDs to include
    channels : list
        List of channels to use

    Returns
    -------
    BaseConcatDataset
        Dataset in Braindecode format
    """
    # Get data in MOABB format
    moabb_data = moabb_dataset.get_data()

    # Create list to hold all raw objects with their subject IDs
    raw_objects = []
    descriptions = []  # To keep track of metadata for each raw object

    # Iterate through all subjects and their sessions/runs
    for subject_id in subject_ids:
        if subject_id not in moabb_data:
            print(f"Warning: No data found for subject {subject_id}")
            continue

        subject_data = moabb_data[subject_id]
        for session_name, session_data in subject_data.items():
            for run_name, run_data in session_data.items():
                try:
                    if isinstance(run_data, (RawArray, RawCNT)):
                        # If it's already an MNE Raw object, pick only the common channels
                        raw = run_data.copy().pick_channels(channels, ordered=True)
                        raw_objects.append(raw)
                        descriptions.append(
                            {
                                "subject": subject_id,
                                "session": session_name,
                                "run": run_name,
                                "dataset_name": moabb_dataset.__class__.__name__,
                            }
                        )
                    elif (
                        isinstance(run_data, dict)
                        and "data" in run_data
                        and "events" in run_data
                    ):
                        # If it's a dictionary, create MNE Raw object with only common channels
                        X = run_data["data"]
                        events = run_data["events"]

                        # Create MNE RawArray with only common channels
                        sfreq = moabb_dataset.interval[2]
                        info = mne.create_info(
                            ch_names=channels,
                            sfreq=sfreq,
                            ch_types=["eeg"] * len(channels),
                        )
                        raw = mne.io.RawArray(X.T, info)

                        # Convert events to annotations
                        onset = events[:, 0] / sfreq
                        duration = np.zeros_like(onset)
                        description = events[:, -1].astype(str)
                        annot = mne.Annotations(
                            onset=onset, duration=duration, description=description
                        )
                        raw.set_annotations(annot)

                        raw_objects.append(raw)
                        descriptions.append(
                            {
                                "subject": subject_id,
                                "session": session_name,
                                "run": run_name,
                                "dataset_name": moabb_dataset.__class__.__name__,
                            }
                        )
                    else:
                        print(
                            f"Warning: Invalid run data format for subject {subject_id}, session {session_name}, run {run_name}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Error processing run data for subject {subject_id}: {str(e)}"
                    )

    if not raw_objects:
        raise ValueError("No valid data found to convert")

    # Convert to Braindecode format with proper descriptions
    return BaseConcatDataset(
        [
            BaseDataset(raw, description=description)
            for raw, description in zip(raw_objects, descriptions)
        ]
    )


# Load datasets in MOABB format first
# This allows us to use MOABB's robust dataset handling and preprocessing
print("\nLoading training datasets...")
train_dataset_1_moabb = load_and_validate_dataset(BNCI2014_001, [1, 2, 3, 4])
train_dataset_2_moabb = load_and_validate_dataset(BNCI2015_001, [1, 2, 3, 4])
train_dataset_3_moabb = load_and_validate_dataset(BNCI2014_004, [1, 2, 3, 4])

print("\nLoading test dataset...")
test_dataset_moabb = load_and_validate_dataset(Zhou2016, [1, 2, 3])

# Use MOABB's match_all for channel handling
# This is a crucial step that:
# 1. Ensures all datasets have the same channels
# 2. Handles different channel names and positions
# 3. Interpolates missing channels when needed
# 4. Maintains data quality across datasets
paradigm = MotorImagery()
all_datasets = [
    train_dataset_1_moabb,
    train_dataset_2_moabb,
    train_dataset_3_moabb,
    test_dataset_moabb,
]

# Get the list of channels from each dataset before matching
print("\nChannels before matching:")
for ds in all_datasets:
    # Load data for first subject to get channel information
    data = ds.get_data([ds.subject_list[0]])  # Get data for first subject
    # Get first session and run
    first_subject = list(data.keys())[0]
    first_session = list(data[first_subject].keys())[0]
    first_run = list(data[first_subject][first_session].keys())[0]
    run_data = data[first_subject][first_session][first_run]

    if isinstance(run_data, (RawArray, RawCNT)):
        channels = run_data.info["ch_names"]
    else:
        # Assuming the channels are stored in the dataset class after loading
        channels = ds.channels
    print(f"{ds.__class__.__name__}: {channels}")

paradigm.match_all(all_datasets, channel_merge_strategy="intersect")

# Get channels from all datasets after matching to ensure we have the correct intersection
all_channels_after_matching = []
print("\nGetting channels from each dataset after matching:")
for ds in all_datasets:
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
    print(f"{ds.__class__.__name__}: {channels}")

# Get the intersection of all channel sets
common_channels = sorted(list(set.intersection(*all_channels_after_matching)))
print(f"\nActual common channels after matching: {common_channels}")
print(f"Number of common channels: {len(common_channels)}")

# Convert matched datasets to Braindecode format
print("\nConverting datasets to Braindecode format...")
print(f"Using {len(common_channels)} common channels: {common_channels}")

# Convert datasets using common channels
train_dataset_1 = convert_moabb_to_braindecode(
    train_dataset_1_moabb, [1, 2, 3, 4], common_channels
)
train_dataset_2 = convert_moabb_to_braindecode(
    train_dataset_2_moabb, [1, 2, 3, 4], common_channels
)
train_dataset_3 = convert_moabb_to_braindecode(
    train_dataset_3_moabb, [1, 2, 3, 4], common_channels
)
test_dataset = convert_moabb_to_braindecode(
    test_dataset_moabb, [1, 2, 3], common_channels
)

# Verify channel counts in converted datasets
print("\nVerifying channel counts in converted datasets:")
for name, dataset in [
    ("Train 1", train_dataset_1),
    ("Train 2", train_dataset_2),
    ("Train 3", train_dataset_3),
    ("Test", test_dataset),
]:
    for i, ds in enumerate(dataset.datasets):
        n_channels = len(ds.raw.info["ch_names"])
        print(f"{name} dataset {i} has {n_channels} channels")
        if n_channels != len(common_channels):
            raise ValueError(
                f"Channel count mismatch in {name} dataset {i}: "
                f"got {n_channels}, expected {len(common_channels)}"
            )

# Get all events across all datasets
train_events_1 = train_dataset_1.datasets[0].raw.annotations.description
train_events_2 = train_dataset_2.datasets[0].raw.annotations.description
train_events_3 = train_dataset_3.datasets[0].raw.annotations.description
test_events = test_dataset.datasets[0].raw.annotations.description

all_events = sorted(
    list(
        set(train_events_1)
        .union(set(train_events_2))
        .union(set(train_events_3))
        .union(set(test_events))
    )
)
print(f"\nAll unique events across datasets: {all_events}\n")

event_id = {str(event): idx for idx, event in enumerate(all_events)}
print(f"Event mapping: {event_id}\n")

# Define number of classes based on all unique events
n_classes = len(event_id)
print(f"Number of classes: {n_classes}\n")

# Process all training datasets using the common channels
print("\nProcessing training datasets...")
print(f"Using {len(common_channels)} common channels: {common_channels}")

# Process datasets one at a time to ensure consistent channel counts
train_windows_list = []
for i, (train_ds, name) in enumerate(
    [
        (train_dataset_1, "Dataset 1"),
        (train_dataset_2, "Dataset 2"),
        (train_dataset_3, "Dataset 3"),
    ]
):
    print(f"\nProcessing training {name}...")
    # Verify channel count before processing
    for ds in train_ds.datasets:
        if len(ds.raw.info["ch_names"]) != len(common_channels):
            print(
                f"Warning: {name} has {len(ds.raw.info['ch_names'])} channels before processing"
            )
            print(f"Current channels: {ds.raw.info['ch_names']}")

    # Process the dataset
    windows, _, _, _, _ = standardize_windows(
        train_ds, test_dataset, common_channels, event_id
    )

    # Verify window dimensions
    for w_idx, window in enumerate(windows):
        if window[0].shape[0] != len(common_channels):
            raise ValueError(
                f"Window {w_idx} in {name} has {window[0].shape[0]} channels, "
                f"expected {len(common_channels)}"
            )

    train_windows_list.append(windows)

print("\nProcessing test dataset...")
train_windows_test, test_windows, window_length, samples_before, samples_after = (
    standardize_windows(train_dataset_1, test_dataset, common_channels, event_id)
)

# Verify all window shapes before combining
print("\nVerifying window shapes:")
for i, windows in enumerate(train_windows_list):
    print(f"Training dataset {i+1} window shape: {windows[0][0].shape}")
print(f"Test dataset window shape: {test_windows[0][0].shape}")

# Combine all training windows
print("\nCombining training windows...")
combined_train_windows = BaseConcatDataset(train_windows_list)

# Verify combined dataset
print(f"Combined training set size: {len(combined_train_windows)}")
print(f"First window shape: {combined_train_windows[0][0].shape}")
print(f"Last window shape: {combined_train_windows[-1][0].shape}")

# Split training data using only the combined dataset
split = combined_train_windows.split("session")
train_set = split["0train"]
valid_set = split["1test"]

print(f"\nTraining set size: {len(train_set)}")
print(f"Validation set size: {len(valid_set)}")

# Setup compute device
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

# Set random seed
set_random_seeds(seed=20200220, cuda=cuda)

# Calculate model parameters based on standardized data
n_chans = len(common_channels)
input_window_samples = window_length  # 450 samples

# Create model with adjusted parameters for all classes
model = ShallowFBCSPNet(
    n_chans,
    n_classes,  # This will now be the total number of unique events
    n_times=input_window_samples,
    n_filters_time=40,
    filter_time_length=20,
    pool_time_length=35,
    pool_time_stride=7,
    final_conv_length="auto",
)

if cuda:
    model = model.cuda()

# Create and train classifier
clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=64,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=11)),
    ],
    device=device,
    classes=list(range(n_classes)),
)

# Train the model
_ = clf.fit(train_set, y=None, epochs=100)

# Get test labels from the windows
y_true = test_windows.get_metadata().target
y_pred = clf.predict(test_windows)
test_accuracy = np.mean(y_true == y_pred)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Generate confusion matrix for test set
confusion_mat = confusion_matrix(y_true, y_pred)

# Plot confusion matrix with all event names
plot_confusion_matrix(confusion_mat, class_names=list(event_id.keys()))
plt.show()

# Create visualization
fig = plt.figure(figsize=(10, 5))
plt.plot(clf.history[:, "train_loss"], label="Training Loss")
plt.plot(clf.history[:, "valid_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.show()

# Plot training curves
results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
df = pd.DataFrame(
    clf.history[:, results_columns],
    columns=results_columns,
    index=clf.history[:, "epoch"],
)

df = df.assign(
    train_misclass=100 - 100 * df.train_accuracy,
    valid_misclass=100 - 100 * df.valid_accuracy,
)

# Plot results
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ["train_loss", "valid_loss"]].plot(
    ax=ax1, style=["-", ":"], marker="o", color="tab:blue", legend=False
)

ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylabel("Loss", color="tab:blue")

ax2 = ax1.twinx()
df.loc[:, ["train_misclass", "valid_misclass"]].plot(
    ax=ax2, style=["-", ":"], marker="o", color="tab:red", legend=False
)
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylabel("Misclassification Rate [%]", color="tab:red")
ax2.set_ylim(ax2.get_ylim()[0], 85)

handles = []
handles.append(Line2D([0], [0], color="black", linestyle="-", label="Train"))
handles.append(Line2D([0], [0], color="black", linestyle=":", label="Valid"))
plt.legend(handles, [h.get_label() for h in handles])
plt.tight_layout()
