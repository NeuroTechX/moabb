"""Shared helpers for BNCI datasets."""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets import download as dl


BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"


def bnci_data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    """Download data file from URL."""
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


def make_raw(
    data,
    ch_names,
    ch_types,
    sfreq,
    *,
    verbose=None,
    montage="standard_1005",
    line_freq=50.0,
    meas_date=None,
    description=None,
):
    """Create RawArray and apply common BNCI metadata."""
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=data, info=info, verbose=verbose)

    if line_freq is not None:
        raw.info["line_freq"] = line_freq
    if montage:
        if isinstance(montage, str):
            montage = make_standard_montage(montage)
        raw.set_montage(montage, on_missing="ignore")
    if meas_date is not None:
        raw.set_meas_date(meas_date)
    if description:
        raw.info["description"] = description
    return raw


def validate_subject(subject, n_subjects, dataset_code):
    """Validate subject number with consistent error message.

    Parameters
    ----------
    subject : int
        Subject number to validate.
    n_subjects : int
        Total number of subjects in the dataset.
    dataset_code : str
        Dataset identifier for error messages.

    Raises
    ------
    ValueError
        If subject number is out of valid range.
    """
    if (subject < 1) or (subject > n_subjects):
        raise ValueError(
            f"Subject must be between 1 and {n_subjects} for {dataset_code}. "
            f"Got {subject}."
        )


def ensure_data_orientation(data, n_channels):
    """Ensure data is in (n_channels, n_samples) orientation.

    Parameters
    ----------
    data : ndarray
        2D array of EEG data.
    n_channels : int
        Expected number of channels.

    Returns
    -------
    data : ndarray
        Data in (n_channels, n_samples) orientation.
    """
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got {data.ndim}D array.")
    # Use channel count to determine orientation
    # If first dimension matches expected channels, assume correct orientation
    # If second dimension matches, transpose
    # Otherwise use heuristic: samples > channels
    if data.shape[0] == n_channels:
        return data
    elif data.shape[1] == n_channels:
        return data.T
    elif data.shape[0] > data.shape[1]:
        # More rows than columns, assume (samples, channels)
        return data.T
    return data


def convert_units(data, from_unit="uV", to_unit="V", channel_mask=None):
    """Convert data units with optional channel selection.

    Always returns a copy to avoid modifying the input data.

    Parameters
    ----------
    data : ndarray
        Data array to convert (not modified).
    from_unit : str
        Source unit. Currently supports "uV" (microvolts).
    to_unit : str
        Target unit. Currently supports "V" (volts).
    channel_mask : array-like or None
        Boolean mask or indices of channels to convert.
        If None, all channels are converted.

    Returns
    -------
    ndarray
        New array with converted units (float64).
    """
    if from_unit == "uV" and to_unit == "V":
        scale = 1e-6
    elif from_unit == to_unit:
        return data.copy()
    else:
        raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

    # Always create a copy to avoid modifying input
    result = data.astype(np.float64, copy=True)
    if channel_mask is None:
        result *= scale
    else:
        result[channel_mask] *= scale
    return result
