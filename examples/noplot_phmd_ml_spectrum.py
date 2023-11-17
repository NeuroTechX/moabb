"""
================================
Spectral analysis of the trials
================================

This example demonstrates how to perform spectral
analysis on epochs extracted from a specific subject
within the :class:`moabb.datasets.Cattan2019_PHMD`  dataset.

"""

# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
# Modified by: Gregoire Cattan <gcattan@hotmail.fr>
# License: BSD (3-clause)

import warnings

import matplotlib.pyplot as plt
import numpy as np

from moabb.datasets import Cattan2019_PHMD
from moabb.paradigms import RestingStateToP300Adapter


warnings.filterwarnings("ignore")

###############################################################################
# Initialization
# ---------------
#
# 1) Specify the channel and subject to compute the power spectrum.
# 2) Create an instance of the :class:`moabb.datasets.Cattan2019_PHMD` dataset.
# 3) Create an instance of the :class:`moabb.paradigms.RestingStateToP300Adapter`  paradigm.
#    By default, the data is filtered between 1-35 Hz,
#    and epochs are extracted from 10 to 50 seconds after event tagging.

# Select channel and subject for the remaining of the example.
channel = "Cz"
subject = 1

dataset = Cattan2019_PHMD()
events = ["on", "off"]
paradigm = RestingStateToP300Adapter(events=events, channels=[channel])

###############################################################################
# Estimate Power Spectral Density
# ---------------
# 1) Obtain the epochs for the specified subject.
# 2) Use Welch's method to estimate the power spectral density.

f, S, _, y = paradigm.psd(subject, dataset)

###############################################################################
# Display of the data
# ---------------
#
# Plot the averaged Power Spectral Density (PSD) for each label condition,
# using the selected channel specified at the beginning of the script.

fig, ax = plt.subplots(facecolor="white", figsize=(8.2, 5.1))
for condition in events:
    mean_power = np.mean(S[y == condition], axis=0).flatten()
    ax.plot(f, 10 * np.log10(mean_power), label=condition)

ax.set_xlim(paradigm.fmin, paradigm.fmax)
ax.set_ylim(100, 135)
ax.set_ylabel("Spectrum Magnitude (dB)", fontsize=14)
ax.set_xlabel("Frequency (Hz)", fontsize=14)
ax.set_title("PSD for Channel " + channel, fontsize=16)
ax.legend()
fig.show()
