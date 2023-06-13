"""
================================
Spectral analysis of the trials
================================

This example shows how to extract the epochs from of a given
subject inside the HeadMountedDisplay dataset
and then do a spectral analysis of the signals.

"""

# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
# Modified by: Gregoire Cattan <gcattan@hotmail.fr>
# License: BSD (3-clause)

import warnings

import mne
import numpy as np
import matplotlib.pyplot as plt

from moabb.datasets import HeadMountedDisplay
from moabb.paradigms import RestingStateToP300Adapter
from scipy.signal import welch

warnings.filterwarnings("ignore")

###############################################################################
# Initialization
# ---------------
#
# 1) Create an instance of the dataset.
# 2) Create an instance of the resting state paradigm.
#    By default filtering between 10-50 Hz
#    with epochs from 1 to 35 s after tagging of the event.

dataset = HeadMountedDisplay()
paradigm = RestingStateToP300Adapter(events=["ON", "OFF"])

channel='Cz'
subject=1

###############################################################################
# Estimate power spectral density
# ---------------
# 1) Get first subject epochs
# 2) Use welch to estimate power spectral density

X, y = paradigm.get_data(dataset, [subject])
f, S = welch(X, axis=-1, nperseg=1024, fs=dataset.resample)

###############################################################################
# Display of the data
# ---------------
#
# plot the averaged PSD for each kind of label for the channel selected at the beginning of the script

fig, ax = plt.subplots(facecolor='white', figsize=(8.2, 5.1))
for condition in ['ON', 'OFF']:
	ax.plot(f, 10*np.log10(np.mean(S[paradigm.events == condition], axis=0)[dataset.chnames.index(channel)]), label=condition)
ax.set_xlim(0, dataset.fmax)
ax.set_ylim(-10, +15)
ax.set_ylabel('Spectrum Manitude (dB)', fontsize=14)
ax.set_xlabel('Frequency (Hz)', fontsize=14)
ax.set_title('PSD for Channel ' + channel, fontsize=16)
ax.legend()
fig.show()