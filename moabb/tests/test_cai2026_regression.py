"""Regression tests for Cai2026 channel normalization."""

import mne
import numpy as np

from moabb.datasets.cai2026 import _EEG_CH_NAMES, _MISC_CH_NAMES, Cai2026


def _raw_with_channels(channel_names, channel_types=None):
    if channel_types is None:
        channel_types = ["eeg"] * len(channel_names)
    info = mne.create_info(channel_names, sfreq=100, ch_types=channel_types)
    # A distinct constant per channel proves reordering keeps each signal with
    # its channel name rather than merely renaming a positional array.
    data = np.repeat(
        np.arange(len(channel_names), dtype=float)[:, np.newaxis],
        repeats=100,
        axis=1,
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations([0.25], [0], ["left_hand"]))
    return raw


def test_canonical_26_channel_recording_is_retained_in_canonical_order():
    source_names = list(reversed(_EEG_CH_NAMES))
    raw = _raw_with_channels(source_names)
    expected_values = {
        name: raw.get_data(picks=[name])[0, 0] for name in _EEG_CH_NAMES
    }

    Cai2026._normalize_scalp_channels(raw)

    assert raw.ch_names == _EEG_CH_NAMES
    assert raw.get_channel_types() == ["eeg"] * len(_EEG_CH_NAMES)
    assert raw.annotations.description.tolist() == ["left_hand"]
    assert {
        name: raw.get_data(picks=[name])[0, 0] for name in _EEG_CH_NAMES
    } == expected_values


def test_mastoid_variant_drops_extra_eeg_channels_and_preserves_auxiliaries():
    source_names = ["M2", *_EEG_CH_NAMES[10:], "M1", *_EEG_CH_NAMES[:10]]
    source_names.extend(_MISC_CH_NAMES)
    channel_types = ["eeg"] * (len(source_names) - len(_MISC_CH_NAMES)) + [
        "misc"
    ] * len(_MISC_CH_NAMES)
    raw = _raw_with_channels(source_names, channel_types)
    expected_values = {
        name: raw.get_data(picks=[name])[0, 0]
        for name in [*_EEG_CH_NAMES, *_MISC_CH_NAMES]
    }

    Cai2026._normalize_scalp_channels(raw)

    assert raw.ch_names == [*_EEG_CH_NAMES, *_MISC_CH_NAMES]
    assert "M1" not in raw.ch_names
    assert "M2" not in raw.ch_names
    assert {
        name: raw.get_data(picks=[name])[0, 0]
        for name in [*_EEG_CH_NAMES, *_MISC_CH_NAMES]
    } == expected_values


def test_missing_required_scalp_channel_has_explicit_error():
    raw = _raw_with_channels(_EEG_CH_NAMES[:-1])

    with np.testing.assert_raises_regex(ValueError, "missing required channels.*O2"):
        Cai2026._normalize_scalp_channels(raw)
