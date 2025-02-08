"""
Cross-Dataset Brain Decoding with Deep Learning
=============================================
This example shows how to train deep learning models on one dataset
and test on another using Braindecode.
"""

from braindecode.datasets import MOABBDataset
from numpy import multiply
from braindecode.preprocessing import (Preprocessor,
                                     exponential_moving_standardize,
                                     preprocess,
                                     create_windows_from_events)
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix
import logging
import mne
from numpy import array
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING)

def get_common_channels(train_dataset, test_dataset):
    """Get channels that are available across both datasets."""
    train_chans = train_dataset.datasets[0].raw.ch_names
    test_chans = test_dataset.datasets[0].raw.ch_names
    common_channels = sorted(list(set(train_chans).intersection(set(test_chans))))
    print(f"\nCommon channels across datasets: {common_channels}\n")
    return common_channels

def get_common_events(train_dataset, test_dataset):
    """Get events that are available across both datasets."""
    # Get events from first subject of each dataset
    train_events = train_dataset.datasets[0].raw.annotations.description
    test_events = test_dataset.datasets[0].raw.annotations.description
    
    # Find common events
    common_events = sorted(list(set(train_events).intersection(set(test_events))))
    print(f"\nCommon events across datasets: {common_events}\n")
    
    # Create event mapping (event description -> numerical ID)
    event_id = {str(event): idx for idx, event in enumerate(common_events)}
    print(f"Event mapping: {event_id}\n")
    
    return event_id

def standardize_windows(train_dataset, test_dataset, common_channels, event_id):
    """Standardize datasets with consistent preprocessing."""
    # Define preprocessing parameters
    target_sfreq = 100  # Target sampling frequency
    
    print("\nInitial dataset properties:")
    for name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        for i, ds in enumerate(dataset.datasets):
            print(f"{name} dataset {i} sampling rate: {ds.raw.info['sfreq']} Hz")
    
    # Define preprocessors for standardization
    preprocessors = [
        Preprocessor('pick_channels', ch_names=common_channels, ordered=True),
        Preprocessor('resample', sfreq=target_sfreq),  # Standardize sampling rate
        Preprocessor(lambda data: multiply(data, 1e6)),  # Convert to microvolts
        Preprocessor('filter', l_freq=4., h_freq=38.),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Normalize
                    factor_new=1e-3, init_block_size=1000)
    ]
    
    # Apply preprocessing
    preprocess(train_dataset, preprocessors, n_jobs=-1)
    preprocess(test_dataset, preprocessors, n_jobs=-1)
    
    print("\nAfter resampling:")
    for name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        for i, ds in enumerate(dataset.datasets):
            print(f"{name} dataset {i} sampling rate: {ds.raw.info['sfreq']} Hz")
    
    # Fixed window parameters (in seconds)
    window_start = -0.5  # Start 0.5s before event
    window_duration = 4.0  # Increased from 3.0 to ensure enough samples for kernel
    
    # Convert to samples based on target frequency
    samples_before = int(abs(window_start) * target_sfreq)  # 50 samples
    samples_after = int(window_duration * target_sfreq)     # 400 samples
    
    print(f"\nWindow configuration:")
    print(f"Sampling frequency: {target_sfreq} Hz")
    print(f"Window: {window_start}s to {window_duration}s")
    print(f"Samples before: {samples_before}")
    print(f"Samples after: {samples_after}")
    print(f"Total window length: {samples_before + samples_after} samples")
    
    # Standardize event durations to 0 for both datasets
    for name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        for i, ds in enumerate(dataset.datasets):
            # Drop last event and standardize durations
            events = ds.raw.annotations[:-1]
            new_annotations = mne.Annotations(
                onset=events.onset,
                duration=np.zeros_like(events.duration),  # Set all durations to 0
                description=events.description
            )
            ds.raw.set_annotations(new_annotations)
            
            print(f"\n{name} dataset {i}:")
            print(f"Number of events: {len(ds.raw.annotations)}")
            print(f"Event timings: {ds.raw.annotations.onset[:5]}")
            print(f"Event durations: {ds.raw.annotations.duration[:5]}")
    
    # Create windows with explicit trial_stop_offset_samples
    def create_fixed_windows(dataset):
        return create_windows_from_events(
            dataset,
            trial_start_offset_samples=-samples_before,
            trial_stop_offset_samples=samples_after,
            preload=True,
            mapping=event_id,
            window_size_samples=samples_before + samples_after,  # Force window size
            window_stride_samples=samples_before + samples_after  # Add stride parameter
        )
    
    # Create windows
    train_windows = create_fixed_windows(train_dataset)
    test_windows = create_fixed_windows(test_dataset)
    
    # Verify window shapes
    train_shape = train_windows[0][0].shape
    test_shape = test_windows[0][0].shape
    print(f"\nActual window shapes:")
    print(f"Train windows: {train_shape}")
    print(f"Test windows: {test_shape}")
    print(f"Expected shape: (9, {samples_before + samples_after})")
    
    if train_shape != test_shape:
        print("\nWindow size mismatch analysis:")
        print(f"Train window size: {train_shape[1]} samples")
        print(f"Test window size: {test_shape[1]} samples")
        print(f"Difference: {train_shape[1] - test_shape[1]} samples")
        print(f"In seconds: {(train_shape[1] - test_shape[1])/target_sfreq:.2f}s")
        raise ValueError(f"Window shapes don't match: train={train_shape}, test={test_shape}")
    
    window_length = train_shape[1]
    print(f"Final window length: {window_length} samples")
    
    return train_windows, test_windows, window_length, samples_before, samples_after

# Load datasets
train_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[3, 4])
test_dataset = MOABBDataset(dataset_name="Zhou2016", subject_ids=[1])

# Get common channels and events
common_channels = get_common_channels(train_dataset, test_dataset)
event_id = get_common_events(train_dataset, test_dataset)
n_classes = len(event_id)

# Standardize datasets and get window length
train_windows, test_windows, window_length, samples_before, samples_after = standardize_windows(
    train_dataset, test_dataset, common_channels, event_id
)

# Split training data
splitted = train_windows.split('session')
train_set = splitted['0train']
valid_set = splitted['1test']

# Setup compute device
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

# Set random seed
set_random_seeds(seed=20200220, cuda=cuda)

# Calculate model parameters based on standardized data
n_chans = len(common_channels)
input_window_samples = window_length  # 450 samples

# Create model with adjusted parameters
model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    n_times=input_window_samples,  # Use n_times instead of deprecated input_window_samples
    n_filters_time=40,
    filter_time_length=20,  # Reduced from default
    pool_time_length=35,    # Reduced from default
    pool_time_stride=7,     # Reduced from default
    final_conv_length='auto'  # Let the model calculate the appropriate final conv length
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
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=11)),
    ],
    device=device,
    classes=list(range(n_classes)),
)

# Train the model
_ = clf.fit(train_set, y=None, epochs=12)

# Get test labels from the windows
y_true = test_windows.get_metadata().target
y_pred = clf.predict(test_windows)
test_accuracy = np.mean(y_true == y_pred)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Generate confusion matrix for test set
confusion_mat = confusion_matrix(y_true, y_pred)

# Plot confusion matrix with dynamic class names
plot_confusion_matrix(confusion_mat, class_names=list(event_id.keys()))
plt.show()

# Create visualization
fig = plt.figure(figsize=(10, 5))
plt.plot(clf.history[:, 'train_loss'], label='Training Loss')
plt.plot(clf.history[:, 'valid_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.show()

# Plot training curves
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                 index=clf.history[:, 'epoch'])

df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
               valid_misclass=100 - 100 * df.valid_accuracy)

# Plot results
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False)

ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylabel("Loss", color='tab:blue')

ax2 = ax1.twinx()
df.loc[:, ['train_misclass', 'valid_misclass']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red')
ax2.set_ylim(ax2.get_ylim()[0], 85)

handles = []
handles.append(Line2D([0], [0], color='black', linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles])
plt.tight_layout()
