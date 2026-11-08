"""Offline regression test for the Cho2017 per-block pre-cue demeaning (gh-1340)."""

from types import SimpleNamespace

import numpy as np

from moabb.datasets import Cho2017, gigadb


SFREQ = 100
N_BLOCKS = 10  # trials per class
N_PRE = 20  # fixation samples before the cue in every block
BLOCK = 70  # samples per tiled block
STEP = 50.0  # class-locked deviation after the cue, in file units


def _tiled_stream(sign, rng):
    """One class stream of tiled blocks: flat fixation, then a late step."""
    t = np.arange(BLOCK)
    block = np.where(t >= N_PRE + 5, sign * STEP, 0.0)
    stream = np.tile(block, N_BLOCKS)
    # a different DC level per block, as between tiled trials of a real file
    stream = stream + np.repeat(rng.normal(0, 200, N_BLOCKS), BLOCK)
    return np.tile(stream, (68, 1)) + rng.normal(0, 1, (68, stream.size))


def _fake_file(rng):
    event = np.zeros(N_BLOCKS * BLOCK)
    event[N_PRE::BLOCK] = 1
    return SimpleNamespace(
        imagery_left=_tiled_stream(-1, rng),
        imagery_right=_tiled_stream(+1, rng),
        imagery_event=event,
        srate=SFREQ,
    )


def _pre_post_means(eeg, stim, code):
    onsets = np.flatnonzero(stim == code)
    pre = np.array([eeg[:, o - N_PRE : o].mean() for o in onsets])
    post = np.array([eeg[:, o + 10 : o + BLOCK - N_PRE].mean() for o in onsets])
    return pre, post


def test_cho2017_pre_cue_mean_is_class_neutral(monkeypatch):
    fake = _fake_file(np.random.default_rng(0))
    monkeypatch.setattr(Cho2017, "data_path", lambda self, subject: "unused.mat")
    monkeypatch.setattr(gigadb, "loadmat", lambda *args, **kwargs: {"eeg": fake})

    raw = Cho2017()._get_single_subject_data(1)["0"]["0"]
    data = raw.get_data()
    eeg, stim = data[:64] * 1e6, data[-1]
    assert raw.n_times == 2 * N_BLOCKS * BLOCK + 500

    pre_l, post_l = _pre_post_means(eeg, stim, 1)
    pre_r, post_r = _pre_post_means(eeg, stim, 2)
    assert len(pre_l) == len(pre_r) == N_BLOCKS
    # every block is zero-mean over its own fixation period, in both classes
    np.testing.assert_allclose(pre_l, 0, atol=1e-6)
    np.testing.assert_allclose(pre_r, 0, atol=1e-6)
    # the post-cue class difference is kept
    np.testing.assert_allclose(post_r.mean() - post_l.mean(), 2 * STEP, atol=1)

    # the former per-class-stream demeaning gave the pre-cue segment a
    # class-dependent offset of opposite sign to the post-cue step
    old_l = fake.imagery_left - fake.imagery_left.mean(axis=1, keepdims=True)
    old_r = fake.imagery_right - fake.imagery_right.mean(axis=1, keepdims=True)
    cues = np.flatnonzero(fake.imagery_event)
    old_pre_l = np.array([old_l[:64, c - N_PRE : c].mean() for c in cues])
    old_pre_r = np.array([old_r[:64, c - N_PRE : c].mean() for c in cues])
    offset = old_pre_r.mean() - old_pre_l.mean()
    assert offset < -STEP  # about -2 * STEP * (BLOCK - N_PRE - 5) / BLOCK
