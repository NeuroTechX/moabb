"""Shared mixin for SSVEP datasets stored as .mat 4D tensors.

Extracts the common pattern from Wang2016 and Nakanishi2015: load a 4D .mat
tensor, transpose/reshape to (trials, channels, samples), build a stim
channel, add zero-padding buffers, and create an MNE RawArray.
"""

import logging

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat


log = logging.getLogger(__name__)


class SSVEPMatDatasetMixin:
    """Mixin that provides _load_mat_as_raw for epoched .mat SSVEP datasets.

    Subclasses should set the following class-level constants to configure
    the loading behaviour:

    Attributes
    ----------
    MAT_KEY : str
        Key in the .mat file to read. Default ``"data"``.
    MAT_AXES_ORDER : tuple of int
        Permutation indices to transpose the raw 4D matrix so that the result
        is ordered as ``(classes, blocks, channels, samples)``.
        Default ``(2, 3, 0, 1)`` (Wang2016 convention: original is
        ``[ch, samples, classes, blocks]``).
    CH_NAMES : list of str
        Channel names *including* a trailing ``"stim"`` entry.
    SFREQ : float
        Sampling frequency in Hz.
    MONTAGE_NAME : str
        MNE standard montage name. Default ``"standard_1005"``.
    SCALE_FACTOR : float
        Scaling factor applied to EEG data (e.g. 1e-6 for µV→V).
        Default ``1e-6``.
    BUFFER_SAMPLES : int
        Number of zero-padding samples inserted before/after each trial.
        Default ``50``.
    MONTAGE_ON_MISSING : str
        Behaviour when montage channels are missing. Default ``"ignore"``.
    SQUEEZE_MAT : bool
        Whether to pass ``squeeze_me=True`` to ``loadmat``. Default ``False``.
    STRIP_EXT : bool
        Whether to strip the last 4 characters from the file path before
        loading (Wang2016 quirk). Default ``False``.
    """

    MAT_KEY: str = "data"
    MAT_AXES_ORDER: tuple = (2, 3, 0, 1)
    CH_NAMES: list  # must include "stim" as last entry
    SFREQ: float
    MONTAGE_NAME: str = "standard_1005"
    SCALE_FACTOR: float = 1e-6
    BUFFER_SAMPLES: int = 50
    MONTAGE_ON_MISSING: str = "ignore"
    SQUEEZE_MAT: bool = False
    STRIP_EXT: bool = False

    def _load_mat_as_raw(self, fname, n_channels, n_samples, n_classes, n_blocks):
        """Load a .mat file and return an MNE RawArray.

        Parameters
        ----------
        fname : str or Path
            Path to the .mat file.
        n_channels : int
            Number of EEG channels (excluding stim).
        n_samples : int
            Number of time points per trial.
        n_classes : int
            Number of target classes.
        n_blocks : int
            Number of blocks/trials per class.

        Returns
        -------
        raw : mne.io.RawArray
            Continuous raw data with embedded stim channel.
        """
        load_path = str(fname)
        if self.STRIP_EXT:
            load_path = load_path[:-4]

        mat = loadmat(load_path, squeeze_me=self.SQUEEZE_MAT)

        data = np.transpose(mat[self.MAT_KEY], axes=self.MAT_AXES_ORDER)
        data = np.reshape(data, (-1, n_channels, n_samples))
        data = data - data.mean(axis=2, keepdims=True)

        # Build stim channel: event code at sample 0 of each trial
        raw_events = np.zeros((data.shape[0], 1, n_samples))
        raw_events[:, 0, 0] = np.array(
            [n_blocks * [i + 1] for i in range(n_classes)]
        ).flatten()

        data = np.concatenate([self.SCALE_FACTOR * data, raw_events], axis=1)

        # Add zero-padding buffers between trials
        log.warning(
            "Trial data de-meaned and concatenated with a buffer"
            " to create continuous data"
        )
        buff = (data.shape[0], n_channels + 1, self.BUFFER_SAMPLES)
        data = np.concatenate([np.zeros(buff), data, np.zeros(buff)], axis=2)

        ch_types = ["eeg"] * n_channels + ["stim"]
        info = create_info(self.CH_NAMES, self.SFREQ, ch_types)
        raw = RawArray(data=np.concatenate(list(data), axis=1), info=info, verbose=False)
        montage = make_standard_montage(self.MONTAGE_NAME)
        raw.set_montage(montage, on_missing=self.MONTAGE_ON_MISSING)
        return raw
