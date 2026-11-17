"""Offline regressions: Brandl2020 channel positions.

16 of the 63 channels carry the Berlin lab's labels for intermediate sites
(CFC1-CFC8, CCP1-CCP8). No downloads: the raw is built from zeros with the
dataset's channel labels.
"""

import mne
import numpy as np

from moabb.datasets import Brandl2020
from moabb.datasets.brandl2020 import _BERLIN_TO_1005


_LABELS = Brandl2020.METADATA.acquisition.sensors


def _positions(raw):
    return raw.get_montage().get_positions()["ch_pos"]


def _brandl_raw():
    return Brandl2020()._make_raw(
        np.zeros((len(_LABELS), 100)), _LABELS, 1000.0, np.array([]), np.array([])
    )


def test_every_eeg_channel_is_placed():
    pos = _positions(_brandl_raw())
    assert sorted(pos) == sorted(_LABELS)
    assert not np.isnan(np.array(list(pos.values()))).any()


def test_berlin_labels_sit_where_the_standard_montage_puts_their_sites():
    sites = [_BERLIN_TO_1005.get(ch, ch) for ch in _LABELS]
    ref = mne.io.RawArray(
        np.zeros((len(sites), 10)), mne.create_info(sites, 1000.0, "eeg")
    )
    ref.set_montage("colin27_1005")

    placed, expected = _positions(_brandl_raw()), _positions(ref)
    for ch, site in zip(_LABELS, sites):
        np.testing.assert_allclose(placed[ch], expected[site], err_msg=ch)
