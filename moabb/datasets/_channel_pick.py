"""Lightweight channel-picking helper — no intra-package imports to avoid cycles."""

import mne


def pick_channels_for_modalities(info, return_all_modalities=False):
    """Pick channel indices based on requested modalities.

    Parameters
    ----------
    info : mne.Info
        The measurement info.
    return_all_modalities : bool | dict
        - ``False`` (default): pick only EEG channels.
        - ``True``: pick all channels except stim.
        - ``dict``: passed as keyword arguments to :func:`mne.pick_types`,
          e.g. ``dict(eeg=True, eog=True, emg=False)``.  ``stim`` is
          always forced to ``False``.

    Returns
    -------
    picks : array-like of int
        Channel indices to keep.
    """
    if isinstance(return_all_modalities, dict):
        kwargs = dict(return_all_modalities)
        kwargs["stim"] = False
        return mne.pick_types(info=info, **kwargs)
    if return_all_modalities:
        stim_picks = set(mne.pick_types(info=info, stim=True))
        return [i for i in range(len(info.ch_names)) if i not in stim_picks]
    return mne.pick_types(info=info, eeg=True, stim=False)
