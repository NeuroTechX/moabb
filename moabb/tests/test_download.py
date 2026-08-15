"""Tests to ensure that datasets download correctly using pytest."""

import inspect
import json
import os
import sys
from pathlib import Path, PurePath
from types import SimpleNamespace

import mne
import pytest
from mne import get_config, set_config

import moabb.datasets.download as dl
import moabb.datasets.romani_bf2025_erp as romani
from moabb.datasets import base as base_module
from moabb.datasets.bbci_eeg_fnirs import BaseShin2017
from moabb.datasets.fake import FakeDataset
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
    # `download()` now fetches NEMAR's `sourcedata/` (the original pre-BIDS
    # distribution), so drive the BIDS copy explicitly to keep covering the
    # BIDS loader below.
    dataset._download_nemar(subject=subject, path=tmp_path)

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


# --------------------------------------------------------------------------
# sourcedata/: NEMAR as a provider for the ORIGINAL, pre-BIDS distribution
# --------------------------------------------------------------------------


class _FakeNemar:
    """Stand-in for the ``nemar`` module that records calls and writes files.

    A recording function would be enough for the argument-forwarding checks,
    but the "published no sourcedata/" guard inspects the target directory, so
    the fake has to actually put files on disk. It honours ``include`` so a
    per-subject call that matches nothing behaves like the real client.
    """

    def __init__(self, files=("sourcedata/subject_01.mat",), subjects=None):
        self.files = files
        # {relative path -> subject}; None means the deposit predates the
        # subject field, which is the state every already-enriched deposit is
        # in today.
        self.subjects = subjects
        self.calls = []

    def download(self, **kwargs):
        self.calls.append(kwargs)
        target = Path(kwargs["target_dir"])
        include = kwargs.get("include")
        patterns = [include] if isinstance(include, str) else include
        for relative in self._served():
            if patterns is not None and not any(
                PurePath(relative).match(p) for p in patterns
            ):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == dl.SOURCEDATA_PROVENANCE:
                destination.write_text(json.dumps(self._provenance()))
            else:
                destination.write_bytes(b"x")

    def _served(self):
        return (dl.SOURCEDATA_PROVENANCE, *self.files)

    def _provenance(self):
        entries = []
        for relative in self.files:
            name = relative[len("sourcedata/") :]
            entry = {"file": name, "bytes": 1, "sha256": "0" * 64}
            if self.subjects is not None:
                entry["subject"] = self.subjects[relative]
            entries.append(entry)
        return {"dataset": "nm000341", "files": entries}


@pytest.fixture
def fake_nemar(monkeypatch):
    """Install a :class:`_FakeNemar` in place of the real client."""

    def _install(files=("sourcedata/subject_01.mat",), subjects=None):
        fake = _FakeNemar(files, subjects)
        monkeypatch.setattr(dl, "nemar", fake)
        return fake

    return _install


@pytest.fixture
def nemar_calls(monkeypatch, tmp_path):
    """Record what ``base`` asks of NEMAR, for both entry points.

    Both are patched so a test can assert the BIDS copy was *not* fetched,
    which is the whole point of preferring ``sourcedata/``.
    """
    calls = SimpleNamespace(sourcedata=[], bids=[], path=tmp_path)
    monkeypatch.setattr(base_module, "get_download_provider", lambda: "auto")
    monkeypatch.setattr(
        base_module,
        "nemar_sourcedata_dl",
        lambda *a, **k: calls.sourcedata.append(k) or str(tmp_path),
    )
    monkeypatch.setattr(
        base_module, "nemar_dl", lambda *a, **k: calls.bids.append(k) or str(tmp_path)
    )
    return calls


@pytest.mark.parametrize(
    ("kwargs", "expected", "absent"),
    [
        pytest.param(
            {},
            {"dataset": "nm000341", "scope": "sourcedata"},
            ("include",),
            id="sourcedata-scope-without-include",
        ),
        pytest.param(
            {"include": "sourcedata/subject_01.*"},
            {"include": "sourcedata/subject_01.*"},
            (),
            id="include-forwarded",
        ),
        pytest.param(
            {"force_update": True},
            {"trust_existing": False},
            (),
            id="force-update-distrusts-existing",
        ),
    ],
)
def test_nemar_sourcedata_dl_forwards_arguments(
    tmp_path, fake_nemar, kwargs, expected, absent
):
    """The scope must be ``sourcedata`` -- ``raw`` would fetch the BIDS copy.

    ``include`` is omitted rather than passed as ``None``, which would filter
    everything out instead of selecting everything.
    """
    fake = fake_nemar()

    dl.nemar_sourcedata_dl("nm000341", "Cattan2019-PHMD", path=tmp_path, **kwargs)

    call = fake.calls[0]
    assert expected.items() <= call.items()
    assert not set(absent) & set(call)


def test_nemar_sourcedata_dl_returns_the_sourcedata_directory(tmp_path, fake_nemar):
    fake_nemar()

    root = dl.nemar_sourcedata_dl("nm000341", "Cattan2019-PHMD", path=tmp_path)

    assert Path(root).name == "sourcedata"
    assert Path(root).is_dir()


@pytest.mark.parametrize(
    ("files", "include", "match"),
    [
        pytest.param((), None, "no sourcedata/", id="deposit-has-none"),
        pytest.param(
            ("sourcedata/subject_01.mat",),
            "sourcedata/subject_99.*",
            "no sourcedata/",
            id="subject-not-in-deposit",
        ),
    ],
)
def test_nemar_sourcedata_dl_raises_when_nothing_was_fetched(
    tmp_path, fake_nemar, files, include, match
):
    """An empty result must fail here, not surface later as an empty cache.

    The second case matters because ``download()`` calls this once per subject
    into a shared directory, so "the directory is non-empty" is true from the
    second subject on even when this subject matched nothing.
    """
    fake_nemar(files)

    with pytest.raises(dl.NemarDownloadError, match=match):
        dl.nemar_sourcedata_dl("nm000115", "Zhou2016", path=tmp_path, include=include)


def test_sourcedata_path_requires_a_nemar_id():
    """A dataset NEMAR does not mirror cannot serve its original files."""
    dataset = FakeDataset()
    dataset.nemar_id = None

    with pytest.raises(ValueError, match="declares no nemar_id"):
        dataset.sourcedata_path()


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        pytest.param(1, ["sourcedata/sub-A/eeg/a.eeg"], id="letters"),
        pytest.param(2, ["sourcedata/session1/s2/x.mat"], id="nested-session-dir"),
        pytest.param(
            3,
            ["sourcedata/S3_Session_1.mat", "sourcedata/S3_Session_2.mat"],
            id="two-files-one-subject",
        ),
    ],
)
def test_sourcedata_selects_a_subject_from_the_provenance(
    tmp_path, fake_nemar, subject, expected
):
    """Selection is data-driven, so any upstream layout works.

    These three shapes are all real: ``sub-A`` (nm000193 labels subjects with
    letters), ``session1/s2/`` (nm000273 splits the subject across two path
    components), and two files per subject (nm000339). No glob template spans
    them, and at least one deposit encodes no subject in the path at all --
    which is why the deposit is asked instead of guessed at.
    """
    files = {
        "sourcedata/sub-A/eeg/a.eeg": "1",
        "sourcedata/session1/s2/x.mat": "2",
        "sourcedata/S3_Session_1.mat": "3",
        "sourcedata/S3_Session_2.mat": "3",
    }
    fake = fake_nemar(tuple(files), subjects=files)

    dl.nemar_sourcedata_dl(
        "nm000341", "Cattan2019-PHMD", path=tmp_path, subject=subject
    )

    # First call fetches the manifest, second fetches that subject's files.
    assert fake.calls[0]["include"] == dl.SOURCEDATA_PROVENANCE
    assert sorted(fake.calls[-1]["include"]) == sorted(expected)


def test_sourcedata_reports_an_unknown_subject_with_the_known_ones(
    tmp_path, fake_nemar
):
    """A subject the deposit does not carry must say so, and say what it has."""
    files = {"sourcedata/sub-A/eeg/a.eeg": "1"}
    fake_nemar(tuple(files), subjects=files)

    with pytest.raises(dl.NemarDownloadError, match="lists no sourcedata for subject"):
        dl.nemar_sourcedata_dl(
            "nm000341", "Cattan2019-PHMD", path=tmp_path, subject=99
        )


def test_sourcedata_falls_back_to_the_whole_tree_without_subject_records(
    tmp_path, fake_nemar
):
    """Deposits enriched before the subject field still work, with a warning.

    Every deposit is in this state today, so this is the live path until the
    manifests are regenerated -- it must degrade to the whole tree rather than
    fail or silently fetch nothing.
    """
    fake = fake_nemar(("sourcedata/subject_01.mat",), subjects=None)

    with pytest.warns(RuntimeWarning, match="before its sourcedata manifest"):
        dl.nemar_sourcedata_dl(
            "nm000341", "Cattan2019-PHMD", path=tmp_path, subject=1
        )

    assert "include" not in fake.calls[-1]


@pytest.mark.parametrize(
    ("n_subjects", "subject_list", "expected"),
    [
        pytest.param(2, None, [1, 2], id="all-subjects"),
        pytest.param(3, [1, 2], [1, 2], id="explicit-subject-list"),
    ],
)
def test_download_selects_sourcedata_not_the_bids_copy(
    nemar_calls, n_subjects, subject_list, expected
):
    """``download()`` pulls the ORIGINAL distribution, never the BIDS copy.

    The BIDS copy is a re-encoding whose events and session/run labels differ
    from what each dataset's own loader produces, so substituting it would
    change results rather than only change where the bytes come from.
    """
    dataset = FakeDataset(n_subjects=n_subjects)
    dataset.nemar_id = "nm000341"

    dataset.download(subject_list=subject_list, path=nemar_calls.path)

    assert [call.get("subject") for call in nemar_calls.sourcedata] == expected
    assert nemar_calls.bids == []


def test_download_provider_upstream_skips_nemar(nemar_calls, monkeypatch):
    """``upstream`` must not touch NEMAR even when a nemar_id exists."""
    monkeypatch.setattr(base_module, "get_download_provider", lambda: "upstream")
    dataset = FakeDataset(n_subjects=1)
    dataset.nemar_id = "nm000341"

    dataset.download(path=nemar_calls.path)

    assert nemar_calls.sourcedata == []
    assert nemar_calls.bids == []


def test_download_provider_nemar_does_not_fall_back(monkeypatch, tmp_path):
    """Pinned to NEMAR, a failure surfaces instead of silently going upstream."""
    monkeypatch.setattr(base_module, "get_download_provider", lambda: "nemar")

    def _boom(*args, **kwargs):
        raise dl.NemarDownloadError("nemar is down")

    monkeypatch.setattr(base_module, "nemar_sourcedata_dl", _boom)
    dataset = FakeDataset(n_subjects=1)
    dataset.nemar_id = "nm000341"

    with pytest.raises(dl.NemarDownloadError, match="nemar is down"):
        dataset.download(path=tmp_path)


def test_download_provider_nemar_rejects_unmirrored_dataset(monkeypatch, tmp_path):
    """Pinning to NEMAR must explain why an unmirrored dataset cannot work."""
    monkeypatch.setattr(base_module, "get_download_provider", lambda: "nemar")
    dataset = FakeDataset(n_subjects=1)
    dataset.nemar_id = None

    with pytest.raises(dl.NemarDownloadError, match="pinned to 'nemar'"):
        dataset.download(path=tmp_path)


def test_download_falls_back_to_upstream_when_sourcedata_missing(monkeypatch, tmp_path):
    """A deposit without sourcedata/ must not block the download."""
    monkeypatch.setattr(base_module, "get_download_provider", lambda: "auto")

    def _no_sourcedata(*args, **kwargs):
        raise dl.NemarDownloadError("published no sourcedata/")

    monkeypatch.setattr(base_module, "nemar_sourcedata_dl", _no_sourcedata)
    dataset = FakeDataset(n_subjects=1)
    dataset.nemar_id = "nm000115"

    with pytest.warns(RuntimeWarning, match="falling back"):
        dataset.download(path=tmp_path)
