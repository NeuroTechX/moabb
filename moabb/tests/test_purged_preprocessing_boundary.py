"""A documentary regression of the preprocessing support limitation.

The splitter protects epoch *raw extraction intervals*. A fixed IIR filter
applied to the entire recording first can transmit an impulse from a held-out
interval into a training interval. No claim is made that this is label leakage
or that the splitter is incorrect; it is a reason not to promise complete
post-filter signal independence.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from moabb.evaluations.splitters import PurgedEpochKFold


def test_raw_disjoint_epochs_can_still_share_prefilter_source_influence():
    sfreq = 100.0
    recording = np.zeros(600, dtype=np.float64)
    # Perturb ONLY in the held-out source interval [300, 400).
    changed = recording.copy()
    changed[305] = 1.0

    # MOABB's fixed Raw filter runs on continuous data before epoching.
    # A zero-phase IIR has nonlocal support, so this counterexample does
    # not rely on raw epoch windows sharing any physical samples.
    sos = butter(2, (2.0, 35.0), btype="bandpass", fs=sfreq, output="sos")
    baseline = sosfiltfilt(sos, recording)
    perturbed = sosfiltfilt(sos, changed)

    train = slice(200, 300)
    test = slice(300, 400)
    assert set(range(train.start, train.stop)).isdisjoint(range(test.start, test.stop))
    assert not np.any(recording[train] != changed[train])
    assert not np.any(recording[100:200] != changed[100:200])
    assert np.max(np.abs(perturbed[train] - baseline[train])) > 1e-7


def test_purged_split_uses_only_declared_epoch_intervals_not_iir_tail():
    events = np.arange(12) * 100
    y = np.tile([0, 1], 6)
    groups = np.empty((12, 3), dtype=object)
    groups[:, 0] = "one-run"
    groups[:, 1] = events
    groups[:, 2] = 100
    splitter = PurgedEpochKFold(n_splits=3)
    train, test = next(splitter.split(np.zeros((12, 1)), y, groups))
    assert len(train) > 0
    assert len(test) > 0
    for i in train:
        for j in test:
            assert events[i] + 100 <= events[j] or events[j] + 100 <= events[i]
    # This does NOT assert no pre-filter influence: a long-range IIR
    # convolution can couple different, sample-disjoint source intervals.
