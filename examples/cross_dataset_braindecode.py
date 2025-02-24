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
from braindecode.datasets import BaseConcatDataset, MOABBDataset
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
from sklearn.metrics import confusion_matrix
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split


# Configure logging
logging.basicConfig(level=logging.WARNING)


def get_common_channels(train_dataset, test_dataset):
    """Get channels that are available across both datasets."""
    train_chans = train_dataset.datasets[0].raw.ch_names
    test_chans = test_dataset.datasets[0].raw.ch_names
    common_channels = sorted(list(set(train_chans).intersection(set(test_chans))))
    print(f"\nCommon channels across datasets: {common_channels}\n")
    return common_channels


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


def interpolate_missing_channels(
    raw_data: mne.io.Raw, all_channels: List[str]
) -> mne.io.Raw:
    """Interpolate missing channels using spherical spline interpolation.

    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw EEG data to process
    all_channels : List[str]
        List of all required channel names

    Returns
    -------
    mne.io.Raw
        Processed data with interpolated channels

    Raises
    ------
    TypeError
        If raw_data is not an MNE Raw object
    """
    if isinstance(raw_data, np.ndarray):
        return raw_data

    if not isinstance(raw_data, mne.io.Raw):
        raise TypeError("Expected MNE Raw object")

    missing_channels = [ch for ch in all_channels if ch not in raw_data.ch_names]
    existing_channels = raw_data.ch_names

    print("\nChannel Information:")
    print(f"Total channels needed: {len(all_channels)}")
    print(f"Existing channels: {len(existing_channels)}")
    print(f"Missing channels to interpolate: {len(missing_channels)}")

    if missing_channels:
        print("\nMissing channels:")
        for ch in missing_channels:
            print(f"- {ch}")

        # Mark missing channels as bad
        raw_data.info["bads"] = missing_channels

        # Add missing channels (temporarily with zeros)
        print("\nAdding temporary channels for interpolation...")
        raw_data.add_channels(
            [
                mne.io.RawArray(
                    np.zeros((1, len(raw_data.times))),
                    mne.create_info([ch], raw_data.info["sfreq"], ["eeg"]),
                )
                for ch in missing_channels
            ]
        )

        # Interpolate the bad channels
        print("Performing spherical spline interpolation...")
        raw_data.interpolate_bads(reset_bads=True)

        # Calculate and print interpolation statistics
        data_after = raw_data.get_data()
        for ch in missing_channels:
            ch_idx = raw_data.ch_names.index(ch)
            interpolated_data = data_after[ch_idx]
            stats = {
                "mean": np.mean(interpolated_data),
                "std": np.std(interpolated_data),
                "min": np.min(interpolated_data),
                "max": np.max(interpolated_data),
            }
            print(f"\nInterpolated channel {ch} statistics:")
            print(f"- Mean: {stats['mean']:.2f}")
            print(f"- Std: {stats['std']:.2f}")
            print(f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

        print("\nInterpolation complete.")
    else:
        print("No channels need interpolation.")

    return raw_data


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
            print(f"{name} dataset {i} sampling rate: {ds.raw.info['sfreq']} Hz")

    # Define preprocessing pipeline
    preprocessors = [
        Preprocessor(lambda raw: interpolate_missing_channels(raw, all_channels)),
        Preprocessor("pick_channels", ch_names=all_channels, ordered=True),
        Preprocessor("resample", sfreq=target_sfreq),
        Preprocessor(lambda data: np.multiply(data, 1e6)),
        Preprocessor("filter", l_freq=4.0, h_freq=38.0),
        Preprocessor(
            exponential_moving_standardize, factor_new=1e-3, init_block_size=1000
        ),
    ]

    # Apply preprocessing
    preprocess(train_dataset, preprocessors, n_jobs=-1)
    preprocess(test_dataset, preprocessors, n_jobs=-1)

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
    train_windows = create_fixed_windows(
        train_dataset, samples_before, samples_after, event_id
    )
    test_windows = create_fixed_windows(
        test_dataset, samples_before, samples_after, event_id
    )

    # Verify window shapes
    train_shape = train_windows[0][0].shape
    test_shape = test_windows[0][0].shape

    if train_shape != test_shape:
        raise ValueError(
            f"Window shapes don't match: train={train_shape}, test={test_shape}"
        )

    window_length = train_shape[1]
    return train_windows, test_windows, window_length, samples_before, samples_after


# Load datasets with validation
def load_and_validate_dataset(dataset_name, subject_ids):
    """Load dataset and validate its contents."""
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids)

    print(f"\nValidating dataset: {dataset_name}")
    for i, ds in enumerate(dataset.datasets):
        events = ds.raw.annotations
        print(f"\nSubject {i+1}:")  # Changed from subject_ids[i] to i+1
        print(f"Number of events: {len(events)}")
        print(f"Unique event types: {set(events.description)}")
        print(f"First 5 event timings: {events.onset[:5]}")
        print(f"Sample event descriptions: {list(events.description[:5])}")

    return dataset


# Load datasets with validation
print("\nLoading training datasets...")
train_dataset_1 = load_and_validate_dataset("BNCI2014_001", subject_ids=[1, 2, 3, 4])
train_dataset_2 = load_and_validate_dataset("BNCI2015_001", subject_ids=[1, 2, 3, 4])
train_dataset_3 = load_and_validate_dataset("BNCI2014_004", subject_ids=[1, 2, 3, 4])

print("\nLoading test dataset...")
test_dataset = load_and_validate_dataset("Zhou2016", subject_ids=[1, 2, 3])

# Verify datasets are different
print("\nVerifying dataset uniqueness...")
for name1, ds1 in [
    ("Train1", train_dataset_1),
    ("Train2", train_dataset_2),
    ("Train3", train_dataset_3),
    ("Test", test_dataset),
]:
    for name2, ds2 in [
        ("Train1", train_dataset_1),
        ("Train2", train_dataset_2),
        ("Train3", train_dataset_3),
        ("Test", test_dataset),
    ]:
        if name1 < name2:  # Compare each pair only once
            print(f"\nComparing {name1} vs {name2}:")
            # Compare first subject of each dataset
            events1 = ds1.datasets[0].raw.annotations
            events2 = ds2.datasets[0].raw.annotations
            print(f"Event counts: {len(events1)} vs {len(events2)}")
            print(
                f"Event types: {set(events1.description)} vs {set(events2.description)}"
            )
            print(f"First timing: {events1.onset[0]} vs {events2.onset[0]}")

# Get common channels across all datasets
train_chans_1 = train_dataset_1.datasets[0].raw.ch_names
train_chans_2 = train_dataset_2.datasets[0].raw.ch_names
train_chans_3 = train_dataset_3.datasets[0].raw.ch_names
test_chans = test_dataset.datasets[0].raw.ch_names

common_channels = sorted(
    list(
        set(train_chans_1)
        .intersection(set(train_chans_2))
        .intersection(set(train_chans_3))
        .intersection(set(test_chans))
    )
)
print(f"\nCommon channels across all datasets: {common_channels}\n")

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

# Process all training datasets
train_windows_1, _, _, _, _ = standardize_windows(
    train_dataset_1, test_dataset, common_channels, event_id
)
train_windows_2, _, _, _, _ = standardize_windows(
    train_dataset_2, test_dataset, common_channels, event_id
)
train_windows_3, _, _, _, _ = standardize_windows(
    train_dataset_3, test_dataset, common_channels, event_id
)
train_windows_test, test_windows, window_length, samples_before, samples_after = (
    standardize_windows(train_dataset_1, test_dataset, common_channels, event_id)
)

# Combine all training windows
combined_train_windows = BaseConcatDataset(
    [train_windows_1, train_windows_2, train_windows_3]
)

# Split training data using only the combined dataset
splitted = combined_train_windows.split("session")
train_set = splitted["0train"]
valid_set = splitted["1test"]

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
