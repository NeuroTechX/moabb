"""Offline tests for the PhysioNet AWS Open Data mirror fallback (#1245)."""

from types import SimpleNamespace

import pytest

import moabb.datasets.download as dl


UP = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf"
MIRROR = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/S001/S001R04.edf"


def test_mirror_url_mapping():
    assert dl.physionet_s3_mirror(UP) == MIRROR
    # Only the /files/ tree is mirrored.
    assert dl.physionet_s3_mirror("https://physionet.org/content/eegmmidb/") is None
    assert dl.physionet_s3_mirror("https://archive.physionet.org/x.edf") is None
    assert dl.physionet_s3_mirror("https://zenodo.org/x.edf") is None
    assert dl.physionet_s3_mirror("https://physionet.org/files/../x") is None


@pytest.fixture
def calls(monkeypatch):
    state = SimpleNamespace(urls=[], fail_mirror=False)

    def fake_retrieve(url, known_hash, **kwargs):
        state.urls.append(url)
        if url.startswith(dl.PHYSIONET_S3_MIRROR_BASE) and state.fail_mirror:
            raise ConnectionError("mirror down")
        return f"{kwargs['path']}/{kwargs['fname']}"

    monkeypatch.setattr(dl, "retrieve", fake_retrieve)
    monkeypatch.setattr(dl, "_S3_MIRROR_FAILED", False)
    monkeypatch.delenv("MOABB_S3_MIRROR", raising=False)
    return state


def _fetch():
    return dl._retrieve_with_physionet_fallback(UP, fname="S001R04.edf", path="/d")


def test_mirror_first(calls):
    assert _fetch() == "/d/S001R04.edf"
    assert calls.urls == [MIRROR]


def test_falls_back_to_upstream(calls):
    calls.fail_mirror = True
    with pytest.warns(RuntimeWarning, match="falling back to upstream"):
        assert _fetch() == "/d/S001R04.edf"
    assert calls.urls == [MIRROR, UP]


def test_env_disables_mirror(calls, monkeypatch):
    monkeypatch.setenv("MOABB_S3_MIRROR", "0")
    _fetch()
    assert calls.urls == [UP]


def test_non_physionet_url_untouched(calls):
    dl._retrieve_with_physionet_fallback(
        "https://zenodo.org/x.edf", fname="x.edf", path="/d"
    )
    assert calls.urls == ["https://zenodo.org/x.edf"]


def test_mirror_skipped_after_first_failure(calls):
    calls.fail_mirror = True
    with pytest.warns(RuntimeWarning, match="falling back to upstream"):
        _fetch()
    _fetch()  # no second mirror attempt, no second warning
    assert calls.urls == [MIRROR, UP, UP]


def test_data_dl_routes_physionet_through_mirror(calls, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "get_dataset_path", lambda sign, path: str(tmp_path))
    monkeypatch.setattr(dl, "_store_lookup", lambda url, fname=None: None)
    dl.data_dl(UP, "PHYSIONET")
    assert calls.urls[0] == MIRROR
