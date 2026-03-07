"""Lightweight channel-picking helper — no intra-package imports to avoid cycles."""

import mne


def pick_channels_for_modalities(info, return_all_modalities=False):
    """Pick channel indices: all non-stim if *return_all_modalities*, else EEG only."""
    if return_all_modalities:
        stim_picks = set(mne.pick_types(info=info, stim=True))
        return [i for i in range(len(info.ch_names)) if i not in stim_picks]
    return mne.pick_types(info=info, eeg=True, stim=False)
