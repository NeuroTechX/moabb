"""Tests to ensure that datasets download correctly using pytest."""

import inspect
import sys

import mne
import pytest
from mne import get_config, set_config

import moabb.datasets.download as dl
from moabb.datasets.bbci_eeg_fnirs import BaseShin2017
from moabb.datasets.utils import dataset_list
from moabb.utils import set_download_dir


# Datasets that legitimately do not resolve their storage location through the
# shared ``get_dataset_path``/``MNE_DATA`` mechanism, so the config-change
# recovery guarantee does not apply to them:
#   * the fake datasets generate synthetic data and never download anything;
#   * ``RomaniBF2025ERP`` manages its own ``data_folder`` (temporary directory
#     or user-provided path) instead of the MNE config path system.
_PATH_RECOVERY_EXEMPT = {"FakeDataset", "FakeVirtualRealityDataset", "RomaniBF2025ERP"}


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
def _restore_mne_config():
    """Isolate the MNE config touched by the path tests.

    Snapshot the current config, clear any pre-existing
    ``MNE_DATASETS_*_PATH`` entries so each dataset starts from a clean
    "first access" state, then restore everything afterwards.
    """
    before = dict(get_config() or {})
    for key in list(before):
        if key.startswith("MNE_DATASETS_") and key.endswith("_PATH"):
            set_config(key, None)
    try:
        yield
    finally:
        after = dict(get_config() or {})
        # Reset keys that were added and restore keys that were modified.
        for key in set(after) | set(before):
            if before.get(key) != after.get(key):
                set_config(key, before.get(key))


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

    The real ``get_dataset_path`` is still invoked (so the per-dataset config
    key is persisted exactly as in production) but a :class:`_ProbeStop` is
    raised immediately afterwards to avoid any network access.
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
    dataset_class, tmp_path, monkeypatch, _restore_mne_config
):
    """A change of download directory must be honoured by every dataset.

    ``get_dataset_path`` persists a ``MNE_DATASETS_<SIGN>_PATH`` config entry the
    first time a dataset is accessed. If that entry is not refreshed when the
    download directory changes, the dataset keeps pointing at the old location
    (see issue #1115). This integration test iterates over all datasets, mocks
    the download mechanism, and asserts that switching the download directory
    moves the resolved storage location instead of reverting to the old one.
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


def test_set_download_dir_propagates_to_dataset_keys(tmp_path):
    """Changing the download dir must update per-dataset MNE config keys.

    ``get_dataset_path`` persists a ``MNE_DATASETS_<SIGN>_PATH`` entry mirroring
    ``MNE_DATA`` the first time a dataset is accessed. Changing the download
    directory afterwards must realign those entries, otherwise datasets keep
    pointing at the old location (see issue #1115).
    """
    keys = (
        "MNE_DATA",
        "MNE_DATASETS_WEIBO_PATH",
        "MNE_DATASETS_BBCIFNIRS_PATH",
        "MNE_DATASETS_CUSTOM_PATH",
    )
    original = {key: get_config(key) for key in keys}
    try:
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"
        old_dir.mkdir()

        set_download_dir(str(old_dir))
        # Simulate two datasets that mirrored MNE_DATA on first access ...
        set_config("MNE_DATASETS_WEIBO_PATH", str(old_dir))
        set_config("MNE_DATASETS_BBCIFNIRS_PATH", str(old_dir))
        # ... and a key that the user deliberately configured elsewhere.
        custom = str(tmp_path / "custom")
        set_config("MNE_DATASETS_CUSTOM_PATH", custom)

        set_download_dir(str(new_dir))

        assert get_config("MNE_DATA") == str(new_dir)
        assert get_config("MNE_DATASETS_WEIBO_PATH") == str(new_dir)
        assert get_config("MNE_DATASETS_BBCIFNIRS_PATH") == str(new_dir)
        # Independently-configured keys must be left untouched.
        assert get_config("MNE_DATASETS_CUSTOM_PATH") == custom
    finally:
        for key, value in original.items():
            set_config(key, value)
