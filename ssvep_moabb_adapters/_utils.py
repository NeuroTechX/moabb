"""Shared utilities for SSVEP dataset adapters."""

import logging

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray


log = logging.getLogger(__name__)

FIGSHARE_DL_URL = "https://ndownloader.figshare.com/files/"

# Tsinghua/Neuroscan 64-channel layout used by Wang2016, Liu2020BETA,
# Liu2022EldBETA, and Han2024Fatigue. Includes non-standard CB1/CB2 channels.
# fmt: off
TSINGHUA_64CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
    "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz",
    "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2",
    "CB2",
]
# fmt: on


def build_raw_from_epochs(
    data,
    ch_names,
    sfreq,
    event_ids,
    montage_name,
    *,
    scale=1e-6,
    buffer_samples=50,
    onset_sample=0,
):
    """Convert (n_trials, n_channels, n_samples) epoched data to continuous Raw.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_samples)
        Epoched EEG data (in original units, e.g. microvolts).
    ch_names : list of str
        EEG channel names (without "stim").
    sfreq : float
        Sampling frequency in Hz.
    event_ids : ndarray, shape (n_trials,)
        Integer event code for each trial.
    montage_name : str
        Name of a standard MNE montage (e.g. "standard_1005", "biosemi32").
    scale : float
        Scale factor to convert data to Volts (default 1e-6 for microvolts).
    buffer_samples : int
        Number of zero-padding samples between trials (default 50).
    onset_sample : int
        Sample index within each epoch to place the event marker (default 0).

    Returns
    -------
    raw : mne.io.RawArray
        Continuous raw data with EEG + stim channels.
    """
    n_trials, n_channels, n_samples = data.shape

    # De-mean each trial
    data = data - data.mean(axis=2, keepdims=True)

    # Build stim channel
    stim = np.zeros((n_trials, 1, n_samples))
    stim[:, 0, onset_sample] = event_ids

    # Combine scaled EEG + stim, add zero-padding buffers
    combined = np.concatenate([scale * data, stim], axis=1)
    n_total_ch = n_channels + 1

    if buffer_samples > 0:
        buff = np.zeros((n_trials, n_total_ch, buffer_samples))
        combined = np.concatenate([buff, combined, buff], axis=2)

    # Flatten trials into continuous data: (n_trials, n_ch, n_time) → (n_ch, n_trials*n_time)
    continuous = combined.transpose(1, 0, 2).reshape(n_total_ch, -1)

    ch_names_full = list(ch_names) + ["stim"]
    ch_types = ["eeg"] * n_channels + ["stim"]
    info = create_info(ch_names_full, sfreq, ch_types)
    raw = RawArray(data=continuous, info=info, verbose=False)
    montage = make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="ignore")
    return raw
