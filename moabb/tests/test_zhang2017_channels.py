"""Offline regressions: Zhang2017 never falls back to generic channel names.

Every released KI.rar file has 17 channels and an empty BCI2000
``ChannelNames`` parameter (EGI sensor order). Before, a file with another
channel count got ``EEG001``.. names, i.e. a channel set of its own, and a
file with header names got 10-20 labels. No downloads: BCI2000 reads are
synthetic.
"""

import numpy as np
import pytest

from moabb.datasets import zhang2017
from moabb.datasets.zhang2017 import _EGI32_NONFACIAL_ORDER, _bci2000_to_raw


def _fake_reader(n_channels, names=()):
    def read(fpath):
        signals = np.ones((n_channels, 3000))
        states = {"StimulusCode": np.zeros(3000)}
        params = {"ChannelNames": list(names), "SourceChGain": ["0.02466864"] * 17}
        return signals, states, 1000.0, params

    return read


def test_released_layout_gets_egi_names_and_positions(monkeypatch):
    monkeypatch.setattr(zhang2017, "_read_bci2000_file", _fake_reader(17))
    raw = _bci2000_to_raw("x.dat", {})
    assert raw.ch_names == _EGI32_NONFACIAL_ORDER
    assert raw.get_channel_types() == ["eeg"] * 17
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(17)])
    assert np.isfinite(pos).all() and (np.abs(pos).sum(axis=1) > 0).all()
    # SourceChGain (uV/count) scales the counts to volts.
    np.testing.assert_allclose(raw.get_data(), 0.02466864e-6)


@pytest.mark.parametrize("n_channels", [16, 18, 32])
def test_other_channel_counts_raise(monkeypatch, n_channels):
    monkeypatch.setattr(zhang2017, "_read_bci2000_file", _fake_reader(n_channels))
    with pytest.raises(ValueError, match="every released file has 17 channels"):
        _bci2000_to_raw("odd.dat", {})


def test_header_channel_names_raise(monkeypatch):
    names = [f"C{i}" for i in range(17)]
    monkeypatch.setattr(zhang2017, "_read_bci2000_file", _fake_reader(17, names))
    with pytest.raises(ValueError, match="C0"):
        _bci2000_to_raw("named.dat", {})


def test_empty_name_entries_count_as_no_names(monkeypatch):
    monkeypatch.setattr(zhang2017, "_read_bci2000_file", _fake_reader(17, [""] * 17))
    assert _bci2000_to_raw("x.dat", {}).ch_names == _EGI32_NONFACIAL_ORDER
