"""Regression tests for dataset-loader fixes.

Each test exercises a single bug that previously corrupted public datasets:

- ``test_bnci2020_002_layout`` -- ``BNCI2020_002`` reshape that produced an
  interleaved trial-fastest EEG layout instead of the trial-major layout that
  every per-trial marker assumed.
- ``test_fs_get_file_list_caches`` -- the Figshare file-list endpoint was
  hammered once per subject by the MAMEM loader; both the in-process
  ``lru_cache`` and the on-disk persistence below it should keep external
  hits at most once.
- ``test_mamem_filelist_disk_cache`` -- once the dataset has been downloaded
  the JSON-cached filelist must satisfy subsequent calls without reaching
  Figshare at all.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from moabb.datasets import download as _dl
from moabb.datasets.bnci.bnci_2020 import _convert_attention_shift
from moabb.datasets.ssvep_mamem import MAMEM1


def _fake_bciexp(n_channels=3, n_samples=4, n_trials=5):
    """Build a mock ``bciexp`` struct that mirrors the real .mat fields.

    ``data`` is shape ``(channels, samples, trials)`` and F-contiguous (the
    same layout ``scipy.io.loadmat`` returns), so the reshape fix is what
    actually drives the test.
    """
    rng = np.random.default_rng(0)
    data = np.asfortranarray(
        rng.standard_normal((n_channels, n_samples, n_trials)).astype(np.float64)
    )
    heog = rng.standard_normal((n_samples, n_trials)).astype(np.float64)
    veog = rng.standard_normal((n_samples, n_trials)).astype(np.float64)
    intentions = np.array(["yes", "no", "yes", "no", "yes"], dtype=object)
    return SimpleNamespace(
        data=data,
        heog=heog,
        veog=veog,
        srate=250.0,
        label=[f"E{i + 1}" for i in range(n_channels)],
        intention=intentions,
    )


def test_bnci2020_002_layout():
    """``_convert_attention_shift`` must produce a trial-major EEG layout.

    Regression for the reshape bug where ``bciexp.data.reshape(n_channels,
    -1)`` (default C-order on an F-contiguous source) produced
    ``flat[c, k] = data[c, k // n_trials, k % n_trials]`` instead of
    ``flat[c, t * n_samples + s] = data[c, s, t]`` -- which silently broke
    every per-trial epoch and made N2pc analyses chance-level.
    """
    bciexp = _fake_bciexp()
    fake_mat = {"bciexp": bciexp}

    with patch("moabb.datasets.bnci.bnci_2020.loadmat", return_value=fake_mat):
        raw, event_id = _convert_attention_shift("dummy.mat")

    n_channels, n_samples, n_trials = bciexp.data.shape
    eeg = raw.copy().pick("eeg").get_data()

    # Trial-major layout: every (c, s, t) triple lands at the slot the
    # per-trial stim markers expect.  Compare against the original .mat data
    # scaled to volts (the loader applies ``* 1e-6``).
    expected = bciexp.data.transpose(0, 2, 1).reshape(n_channels, -1) * 1e-6
    np.testing.assert_allclose(eeg, expected, rtol=0, atol=1e-15)

    # Stim markers must land at trial starts (sample = trial_idx * n_samples).
    stim = raw.copy().pick("stim").get_data()[0]
    nonzero_idx = np.flatnonzero(stim)
    np.testing.assert_array_equal(nonzero_idx, np.arange(n_trials) * n_samples)
    # 'yes' -> Target (2), 'no' -> NonTarget (1).
    assert event_id == {"NonTarget": 1, "Target": 2}
    np.testing.assert_array_equal(stim[nonzero_idx], [2, 1, 2, 1, 2])


def test_fs_get_file_list_caches():
    """The lru_cache on ``fs_get_file_list`` must short-circuit repeat calls."""
    _dl.fs_get_file_list.cache_clear()
    # Return a fresh list per call so we can detect cache hits via identity.
    side_effect = [
        [{"id": 1, "name": "a.mat", "supplied_md5": "0" * 32}],
        [{"id": 2, "name": "b.mat", "supplied_md5": "1" * 32}],
    ]
    with patch.object(_dl, "_fs_paginated_file_list", side_effect=side_effect) as page:
        first = _dl.fs_get_file_list(123456)
        second = _dl.fs_get_file_list(123456)
        third = _dl.fs_get_file_list(123456, version=2)
    # Same (article_id, version) => one network call.  Different version =>
    # one extra call.  Three logical lookups, two API hits.
    assert page.call_count == 2
    assert first is second
    assert third is not first
    _dl.fs_get_file_list.cache_clear()


def test_mamem_filelist_disk_cache(tmp_path: Path):
    """After a successful fetch the JSON cache feeds subsequent lookups.

    Once the dataset is on disk the loader must not contact Figshare again
    -- earlier behaviour re-issued the API call once per subject and
    triggered 403 rate limits from datacenter IPs.
    """
    _dl.fs_get_file_list.cache_clear()
    ds = MAMEM1()
    sentinel = [{"id": 99, "name": "U001ai.mat", "supplied_md5": "0" * 32}]

    cache_path = ds._filelist_cache_path(str(tmp_path))
    assert not Path(cache_path).exists()

    # First lookup hits the (mocked) API and persists the JSON.
    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list", return_value=sentinel
    ) as api:
        first = ds._load_or_fetch_filelist(str(tmp_path))
    assert api.call_count == 1
    assert first == sentinel
    assert Path(cache_path).exists()
    with open(cache_path) as f:
        assert json.load(f) == sentinel

    # Second lookup must read the JSON instead of calling the API.
    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list",
        side_effect=AssertionError("API should not be called when cache exists"),
    ):
        second = ds._load_or_fetch_filelist(str(tmp_path))
    assert second == sentinel

    # ``force_update=True`` bypasses the disk cache and re-hits the API.
    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list", return_value=sentinel
    ) as api:
        ds._load_or_fetch_filelist(str(tmp_path), force_update=True)
    assert api.call_count == 1
    _dl.fs_get_file_list.cache_clear()


def test_mamem_already_downloaded_does_not_ping_figshare(tmp_path: Path):
    """End-to-end: pre-existing data + cache = zero Figshare API calls.

    Simulates the realistic "user already has the dataset" workflow by
    pre-populating the on-disk filelist JSON, then iterating over
    :meth:`data_path` for several subjects.  The Figshare API must not be
    contacted at all -- not by ``fs_get_file_list`` and not by pooch (we
    pre-create local files matching the registry hashes so ``gb.fetch``
    short-circuits).
    """
    _dl.fs_get_file_list.cache_clear()
    ds = MAMEM1()

    # Build a tiny synthetic Figshare filelist covering 3 subjects, plus
    # local files matching each registry hash so pooch never downloads.
    filelist = []
    for sub_id in (1, 2, 3):
        for (
            run_letter
        ) in "ai":  # two runs per subject, naming convention U<sub>{a,i}{i}.mat
            payload = f"S{sub_id:02d}{run_letter}".encode()
            md5 = hashlib.md5(payload).hexdigest()
            file_id = sub_id * 100 + ord(run_letter)
            name = f"U0{sub_id:02d}{run_letter}i.mat"
            filelist.append({"id": file_id, "name": name, "supplied_md5": md5})
            # Pooch saves files with the file-id as their basename (the
            # registry key), so write the synthetic bytes to that path.
            (tmp_path / str(file_id)).write_bytes(payload)

    # Pre-populate the filelist cache exactly where ``data_path`` would
    # write it -- this is the post-download state.
    cache_path = Path(ds._filelist_cache_path(str(tmp_path)))
    cache_path.write_text(json.dumps(filelist))

    # Force ``_dataset_root`` to land in tmp_path so we don't touch the
    # user's real MNE data directory.
    with (
        patch.object(ds, "_dataset_root", return_value=str(tmp_path)),
        patch(
            "moabb.datasets.ssvep_mamem.fs_get_file_list",
            side_effect=AssertionError(
                "fs_get_file_list must not be called when the dataset is "
                "already downloaded and the filelist cache is present"
            ),
        ),
    ):
        # Iterating over multiple subjects must not contact Figshare even
        # once.  Each call performs a JSON read + pooch hash check.
        for sub_id in (1, 2, 3):
            paths = ds.data_path(sub_id)
            assert paths, f"Expected files for subject {sub_id}"
            for p in paths:
                assert Path(p).exists()
    _dl.fs_get_file_list.cache_clear()
