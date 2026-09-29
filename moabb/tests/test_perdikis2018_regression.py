"""Regression tests for Perdikis2018 GDF amplitude units and caching."""

import mne
import numpy as np

from moabb.datasets.perdikis2018 import _EEG_CHANNELS, Perdikis2018


def test_perdikis_gdf_microvolts_are_converted_to_volts(tmp_path, monkeypatch):
    """The GDF numeric payload is in uV although MNE treats it as volts."""
    sfreq = 512.0
    source_uv = np.linspace(-25.0, 25.0, int(4 * sfreq), dtype=np.float64)
    source_data = np.tile(source_uv, (len(_EEG_CHANNELS), 1))
    trigger = np.zeros((1, source_data.shape[1]))
    ch_names = [f"eeg:{idx}" for idx in range(1, 17)] + ["trigger:1"]
    info = mne.create_info(ch_names, sfreq, ["eeg"] * 16 + ["stim"])
    source_raw = mne.io.RawArray(np.vstack([source_data, trigger]), info, verbose="ERROR")
    source_raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 1.5, 2.5],
            duration=[0.0, 0.0, 0.0],
            description=["771", "773", "783"],
        )
    )
    monkeypatch.setattr(mne.io, "read_raw_gdf", lambda *args, **kwargs: source_raw)

    dataset = Perdikis2018()
    montage = mne.channels.make_standard_montage("standard_1005")
    raw = dataset._read_calibration_run(tmp_path / "calibration.gdf", montage)

    np.testing.assert_allclose(raw.get_data(), source_data * 1e-6, rtol=0, atol=0)
    assert raw.ch_names == _EEG_CHANNELS
    assert raw.n_times == source_data.shape[1]
    assert raw.info["sfreq"] == sfreq
    assert raw.annotations.description.tolist() == ["both_feet", "both_hands", "rest"]
    np.testing.assert_allclose(raw.annotations.onset, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(raw.annotations.duration, [0.0, 0.0, 0.0])
