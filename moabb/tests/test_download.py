"""Tests to ensure that datasets download correctly using pytest."""

import inspect
import os
import sys
from pathlib import Path

import mne
import pytest
from mne import get_config, set_config

import moabb.datasets.brandl2020 as brandl
import moabb.datasets.download as dl
import moabb.datasets.romani_bf2025_erp as romani
from moabb.datasets.bbci_eeg_fnirs import BaseShin2017
from moabb.datasets.utils import dataset_list
from moabb.utils import set_download_dir


# Datasets that legitimately do not resolve their storage location through the
# shared ``get_dataset_path``/``MNE_DATA`` mechanism, so the config-change
# recovery guarantee does not apply to them:
#   * the fake datasets generate synthetic data and never download anything.
_PATH_RECOVERY_EXEMPT = {"FakeDataset", "FakeVirtualRealityDataset"}


def _get_events(raw):
    """Helper function to extract events from a raw object."""
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    if len(stim_channels) > 0:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, verbose=False)
    return events


@pytest.mark.download
@pytest.mark.parametrize(
    "dataset_class",
    [pytest.param(dataset, id=dataset.__name__) for dataset in dataset_list],
)
def test_nemar_download_equivalence(dl_data, tmp_path, dataset_class):
    if not dl_data:
        pytest.skip(
            "Skipping NEMAR download test by default. "
            "Run the test with option --dl-data to execute it."
        )

    kwargs = {}
    if "accept" in inspect.signature(dataset_class).parameters:
        kwargs["accept"] = True
    dataset = dataset_class(**kwargs)
    assert dataset.nemar_id is not None, (
        f"{dataset_class.__name__} has no NEMAR dataset ID"
    )
    subject = dataset.subject_list[0]
    dataset.download(subject_list=[subject], path=tmp_path)

    bids_root = tmp_path / f"MNE-{dataset.code.lower()}-data" / dataset.nemar_id
    description = bids_root / "dataset_description.json"
    assert description.is_file()
    assert dataset.nemar_id in description.read_text(encoding="utf-8")

    sessions = dataset._get_single_subject_data_from_nemar(subject, path=tmp_path)
    assert sessions
    for runs in sessions.values():
        assert runs
        for raw in runs.values():
            assert isinstance(raw, mne.io.BaseRaw)
            assert len(_get_events(raw)) > 0


@pytest.mark.parametrize("dataset", dataset_list)
def test_dataset_download(dl_data, dataset):
    """
    Test that a dataset downloads and returns data with the correct structure.

    For datasets of type BaseShin2017, we need to pass an "accept" flag.
    We then test that:

    - The returned data is a dict.
    - The keys of the dict match the subject list.
    - For each subject, sessions are provided as a dict and their number is at least as expected.
    - Each session contains runs that are dicts and each run is a valid MNE Raw object
      that contains at least one event.
    """
    if not dl_data:
        pytest.skip(
            "Skipping download tests by default. "
            "Run the test with option --dl-data to execute these tests."
        )

    # Some datasets (e.g., BaseShin2017) require explicit acceptance of terms.
    if isinstance(dataset(), BaseShin2017):
        obj = dataset(accept=True)
    else:
        obj = dataset()

    # Use only a subset of subjects for faster testing.
    subj = (0, 1)
    obj.subject_list = obj.subject_list[subj[0] : subj[1]]
    data = obj.get_data(obj.subject_list)

    # Check that the returned data is a dict.
    assert isinstance(data, dict), "Data returned by get_data is not a dict."

    # Check that the dictionary keys match the subject_list.
    assert list(data.keys()) == obj.subject_list, (
        "Data keys do not match the subject_list."
    )

    # For each subject, check the structure of sessions and runs.
    for subject, sessions in data.items():
        assert isinstance(sessions, dict), (
            f"Sessions for subject {subject} is not a dict."
        )
        assert len(sessions) >= obj.n_sessions, (
            f"Number of sessions for subject {subject} is less than expected."
        )

        for session, runs in sessions.items():
            assert isinstance(runs, dict), f"Runs for session {session} is not a dict."

            for run, raw in runs.items():
                assert isinstance(raw, mne.io.BaseRaw), (
                    f"Data for run {run} in session {session} is not an instance of mne.io.BaseRaw."
                )
                events = _get_events(raw)
                assert len(events) != 0, (
                    f"No events found in run {run} of session {session}."
                )


class _ProbeStop(Exception):
    """Sentinel raised to abort a download once the storage path is resolved."""

    def __init__(self, sign, resolved):
        super().__init__(sign)
        self.sign = sign
        self.resolved = resolved


@pytest.fixture
def _isolated_mne_config(tmp_path, monkeypatch):
    """Redirect MNE configuration writes to a temporary home directory."""
    fake_home = tmp_path / "mne-home"
    fake_home.mkdir()
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(fake_home))
    for key in tuple(os.environ):
        if key == "MNE_DATA" or key.startswith("MNE_DATASETS_"):
            monkeypatch.delenv(key)


# Dataset modules that did ``from .download import get_dataset_path`` hold their
# own reference to the original function, so the probe has to replace those
# bindings too. Which modules those are is static, so resolve it once at import
# instead of re-scanning all of ``dataset_list`` inside every parametrization.
_REBOUND_MODULES = tuple(
    module
    for module in {sys.modules[cls.__module__] for cls in dataset_list}
    if getattr(module, "get_dataset_path", None) is dl.get_dataset_path
)


def _install_path_probe(monkeypatch):
    """Replace ``get_dataset_path`` so it records the sign and aborts.

    The real ``get_dataset_path`` is still invoked, but a :class:`_ProbeStop`
    is raised immediately afterwards to avoid any network access.
    """
    real = dl.get_dataset_path

    def probe(sign, path):
        resolved = real(sign, path)
        raise _ProbeStop(sign, str(resolved))

    monkeypatch.setattr(dl, "get_dataset_path", probe)
    for module in _REBOUND_MODULES:
        monkeypatch.setattr(module, "get_dataset_path", probe)


def _resolve_dataset_path(dataset_class, download_dir):
    """Resolve a dataset's storage location for the given download directory."""
    set_download_dir(str(download_dir))
    kwargs = {}
    if "accept" in inspect.signature(dataset_class).parameters:
        kwargs["accept"] = True
    dataset = dataset_class(**kwargs)
    subject = dataset.subject_list[0]

    # Fill any required positional parameters that come before ``path`` in the
    # ``data_path`` signature (e.g. ``session`` or ``paradigm_type``) so that we
    # reach the path-resolution call for datasets with custom signatures.
    extra_args = []
    for name in inspect.signature(dataset.data_path).parameters:
        if name in ("self", "subject"):
            continue
        if name == "path":
            break
        # ``session``-like parameters need an integer (they are often used in
        # f-strings before the path is resolved); anything else only appears
        # after the probe fires, so an empty string is a safe placeholder.
        extra_args.append(1 if "session" in name else "")

    try:
        dataset.data_path(subject, *extra_args)
    except _ProbeStop as stop:
        return stop.resolved
    return None


@pytest.mark.parametrize(
    "dataset_class",
    [pytest.param(dataset, id=dataset.__name__) for dataset in dataset_list],
)
def test_download_dir_change_is_respected(
    dataset_class, tmp_path, monkeypatch, _isolated_mne_config
):
    """A change of download directory must be honoured by every dataset.

    Legacy MOABB releases persisted a ``MNE_DATASETS_<SIGN>_PATH`` config entry
    the first time a dataset was accessed. If that stale entry is not handled
    when the download directory changes, the dataset keeps pointing at the old
    location (see issue #1115). This integration test iterates over all
    datasets, mocks the download mechanism, and asserts that switching the
    download directory moves the resolved storage location.
    """
    if dataset_class.__name__ in _PATH_RECOVERY_EXEMPT:
        pytest.skip(
            f"{dataset_class.__name__} does not use the shared MNE_DATA path mechanism."
        )

    _install_path_probe(monkeypatch)

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()

    resolved_old = _resolve_dataset_path(dataset_class, old_dir)
    if resolved_old is None:
        pytest.skip(
            f"{dataset_class.__name__} does not resolve its path through "
            "get_dataset_path with a subject-only call."
        )
    resolved_new = _resolve_dataset_path(dataset_class, new_dir)

    assert resolved_old.startswith(str(old_dir)), (
        f"{dataset_class.__name__} did not resolve under the configured directory."
    )
    assert resolved_new is not None
    assert resolved_new.startswith(str(new_dir)), (
        f"{dataset_class.__name__} did not follow the new download directory; "
        f"resolved {resolved_new!r} instead of a path under {new_dir}."
    )
    assert not resolved_new.startswith(str(old_dir)), (
        f"{dataset_class.__name__} reverted to the old download directory."
    )


def test_get_dataset_path_does_not_persist_shared_path(tmp_path, _isolated_mne_config):
    """Dataset paths should inherit ``MNE_DATA`` without copying its value."""
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    key = "MNE_DATASETS_WEIBO_PATH"

    set_download_dir(str(old_dir))
    assert dl.get_dataset_path("WEIBO", None) == old_dir
    assert get_config(key, use_env=False) is None
    assert key not in os.environ

    set_download_dir(str(new_dir))
    assert dl.get_dataset_path("WEIBO", None) == new_dir
    assert get_config(key, use_env=False) is None
    assert key not in os.environ


def test_set_download_dir_removes_only_legacy_mirrors(
    tmp_path, monkeypatch, _isolated_mne_config
):
    """Changing ``MNE_DATA`` should remove stale mirrors, not explicit paths."""
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    custom_dir = tmp_path / "custom"
    env_dir = tmp_path / "environment-override"
    old_dir.mkdir()

    set_download_dir(str(old_dir))
    legacy_key = "MNE_DATASETS_WEIBO_PATH"
    set_config(legacy_key, str(old_dir) + os.sep)

    custom_key = "MNE_DATASETS_CUSTOM_PATH"
    set_config(custom_key, str(custom_dir))

    env_key = "MNE_DATASETS_BBCIFNIRS_PATH"
    set_config(env_key, str(old_dir), set_env=False)
    monkeypatch.setenv(env_key, str(env_dir))

    set_download_dir(str(new_dir))

    assert get_config("MNE_DATA") == str(new_dir)
    assert get_config(legacy_key, use_env=False) is None
    assert legacy_key not in os.environ
    assert dl.get_dataset_path("WEIBO", None) == new_dir

    assert get_config(custom_key, use_env=False) == str(custom_dir)
    assert get_config(custom_key) == str(custom_dir)

    assert get_config(env_key, use_env=False) is None
    assert get_config(env_key) == str(env_dir)
    assert os.environ[env_key] == str(env_dir)

    independent_dir = tmp_path / "independent"
    set_config(legacy_key, str(independent_dir), set_env=False)
    assert get_config(legacy_key) == str(independent_dir)


def test_romani_follows_download_dir_changes(tmp_path, monkeypatch, _isolated_mne_config):
    """Romani should use the shared path and forward ``force_update``."""
    calls = []

    def fake_fetch_dataset(
        *, dataset_params, path, processor, force_update, update_path=True
    ):
        calls.append(
            {
                "dataset_params": dataset_params,
                "path": path,
                "processor": processor,
                "force_update": force_update,
                "update_path": update_path,
            }
        )
        root = Path(path) if path is not None else tmp_path / "legacy-default"
        destination = root / romani.BF_folder_name
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    monkeypatch.setattr(romani, "fetch_dataset", fake_fetch_dataset)
    dataset = romani.RomaniBF2025ERP()
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    set_download_dir(str(old_dir))
    resolved_old = dataset.data_path(0)
    set_download_dir(str(new_dir))
    resolved_new = dataset.data_path(0, force_update=True)

    assert resolved_old == str(old_dir / romani.BF_folder_name)
    assert resolved_new == str(new_dir / romani.BF_folder_name)
    assert [Path(call["path"]) for call in calls] == [old_dir, new_dir]
    assert [call["force_update"] for call in calls] == [False, True]
    assert all(call["update_path"] is False for call in calls)
    assert all("config_key" not in call["dataset_params"] for call in calls)


# -- Brandl2020 / DepositOnce -------------------------------------------------
#
# DepositOnce answers HTTP 200 with an HTML app shell for files it cannot serve,
# so a status-code-only downloader caches that page under the .mat name. These
# tests pin the two things that stop it: the download goes to the REST host that
# actually serves bytes, and the result is rejected unless it really is a MAT
# file.


@pytest.mark.parametrize(
    "filename, uuid",
    [
        ("pp1.mat", "15de7e5f-8e0c-4804-aac4-7452c38baee5"),
        ("pp16.mat", "358eef6d-bffe-4562-91fb-0173deafa0ed"),
        ("mnt.mat", "6bc4de0d-4c65-4bd8-96b2-1f921a800ace"),
    ],
)
def test_brandl_bitstream_url_targets_rest_api(filename, uuid, tmp_path, monkeypatch):
    """The bitstream URL must use the API host and the file's UUID."""
    seen = {}

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None, fname=None):
        seen["url"], seen["fname"] = url, fname
        dest = tmp_path / fname
        dest.write_bytes(b"MATLAB 5.0 MAT-file")
        return str(dest)

    monkeypatch.setattr(brandl.dl, "data_dl", fake_data_dl)
    brandl._download_bitstream(filename, str(tmp_path), False, None)

    assert seen["url"] == (
        f"https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/{uuid}/content"
    )
    # Without an explicit name every bitstream would be stored as "content".
    assert seen["fname"] == filename


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            b"<!DOCTYPE html><html><head><title>DSpace",
            "the app shell served with HTTP 200",
        ),
        (b"", "an empty response"),
        (b"\x00\x01\x02\x03garbage", "an unrelated binary body"),
    ],
)
def test_brandl_rejects_and_removes_non_mat_download(body, reason, tmp_path, monkeypatch):
    """A non-MAT payload must raise and must not stay in the cache."""
    dest = tmp_path / "pp1.mat"

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None, fname=None):
        dest.write_bytes(body)
        return str(dest)

    monkeypatch.setattr(brandl.dl, "data_dl", fake_data_dl)

    with pytest.raises(RuntimeError, match="not a MATLAB file"):
        brandl._download_bitstream("pp1.mat", str(tmp_path), False, None)
    # A cached bad file would be served to every later call without a network hit.
    assert not dest.exists(), f"{reason} was left in the cache"


def test_brandl_unknown_bitstream_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="No DepositOnce bitstream UUID"):
        brandl._download_bitstream("pp99.mat", str(tmp_path), False, None)


@pytest.mark.parametrize(
    "fname, expected_parts",
    [
        ("pp1.mat", ("MNE-brandl2020-data", "pp1.mat")),
        ("mnt.mat", ("MNE-brandl2020-data", "mnt.mat")),
    ],
)
def test_data_dl_fname_overrides_url_derived_name(
    fname, expected_parts, tmp_path, monkeypatch
):
    """``fname`` names the destination for URLs that do not end in a filename."""
    captured = {}

    def fake_retrieve(url, known_hash, fname=None, path=None, **kwargs):
        captured["fname"], captured["path"] = fname, path
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"MATLAB")
        return str(dest)

    monkeypatch.setattr(dl, "retrieve", fake_retrieve)
    result = dl.data_dl(
        "https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/"
        "15de7e5f-8e0c-4804-aac4-7452c38baee5/content",
        "Brandl2020",
        path=str(tmp_path),
        fname=fname,
    )

    assert captured["fname"] == fname
    assert Path(result).parts[-2:] == expected_parts
