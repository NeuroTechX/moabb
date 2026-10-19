"""Offline tests: Lee2021Mobile reads the ``/sourcedata/`` copy from OSF.

OSF project r7s9b stores every BrainVision file twice under the same file
name: the BIDS copy ``/sub-XX/ses-YY/eeg/`` (73 channels at 100 Hz) and the
original recording ``/sourcedata/sub-XX/ses-YY/eeg/`` (50 channels at 500 Hz).
The loader must download the second one whatever order OSF lists them in.
No test touches the network: the OSF API and the downloads are faked.
"""

from pathlib import Path

import pandas as pd
import pytest

from moabb.datasets import Lee2021Mobile_ERP, Lee2021Mobile_SSVEP
from moabb.datasets import download as dl


API = "https://api.osf.io/v2"
ROOT_LISTING = f"{API}/nodes/r7s9b/files/osfstorage/?page[size]=100"
STORAGE = "https://files.osf.io/v1/resources/r7s9b/providers/osfstorage"

SUB = "sub-11"
ERP = [f"{SUB}_ses-01_task-ERP_eeg{ext}" for ext in (".vhdr", ".eeg", ".vmrk")]
SSVEP = [f"{SUB}_ses-02_task-SSVEP_eeg{ext}" for ext in (".vhdr", ".eeg", ".vmrk")]


def _bids(name):
    ses = name.split("_")[1]
    return f"/{SUB}/{ses}/eeg/{name}"


def _source(name):
    return "/sourcedata" + _bids(name)


def _url(path):
    # Encode the OSF path in the fake download URL so tests can tell copies apart.
    return STORAGE + path


def _layout(bids_first=True, drop=()):
    """File paths of a small r7s9b-like project, in listing order."""
    bids = [_bids(n) for n in ERP + SSVEP] + [
        _bids(f"{SUB}_ses-01_task-ERP_events.tsv"),
        _bids(f"{SUB}_ses-01_task-ERP_channels.tsv"),
    ]
    source = [_source(n) for n in ERP + SSVEP] + [
        _source(f"{SUB}_ses-01_task-ERP_channels.tsv")
    ]
    root = ["/dataset_description.json", "/participants.tsv"]
    paths = (bids + source if bids_first else source + bids) + root
    return [p for p in paths if p not in drop]


def _metainfo(paths):
    return pd.DataFrame(
        [(p.rsplit("/", 1)[1], _url(p), p) for p in paths],
        columns=["filename", "url", "path"],
    )


def _fake_osf_api(paths, folder_first=(), page_size=None, with_paths=True):
    """Return ``{listing url: json}`` pages of the OSF v2 API for ``paths``.

    Children are listed in first-appearance order, except that folders named
    in ``folder_first`` are listed first in their parent. ``page_size`` splits
    listings into pages linked by ``links.next``. ``with_paths=False`` drops
    ``materialized_path`` from every item.
    """
    tree = {}
    for path in paths:
        node = tree
        parts = path.strip("/").split("/")
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = path

    pages = {}
    counter = iter(range(10_000))

    def item(name, value, prefix):
        attrs = {"name": name.rstrip("/")}
        if isinstance(value, dict):
            attrs["kind"] = "folder"
            href = f"{API}/files/folder{next(counter)}/"
            register(value, href + "?page[size]=100", prefix + name)
            entry = {"relationships": {"files": {"links": {"related": {"href": href}}}}}
            if with_paths:
                attrs["materialized_path"] = prefix + name
        else:
            attrs["kind"] = "file"
            entry = {"links": {"download": _url(value)}}
            if with_paths:
                attrs["materialized_path"] = value
        entry["attributes"] = attrs
        return entry

    def register(node, url, prefix):
        names = sorted(node, key=lambda n: n.rstrip("/") not in folder_first)
        items = [item(n, node[n], prefix) for n in names]
        size = page_size or max(len(items), 1)
        chunks = [items[i : i + size] for i in range(0, len(items), size)] or [[]]
        page_urls = [url] + [f"{url}&page={k + 2}" for k in range(len(chunks) - 1)]
        for k, chunk in enumerate(chunks):
            nxt = page_urls[k + 1] if k + 1 < len(chunks) else None
            pages[page_urls[k]] = {"data": chunk, "links": {"next": nxt}}

    register(tree, ROOT_LISTING, "/")
    return pages


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_osf(monkeypatch):
    """Serve fake OSF API pages through ``requests.get``."""

    def install(pages):
        requested = []

        def get(url, *args, **kwargs):
            requested.append(url)
            if url not in pages:
                raise AssertionError(f"unexpected OSF request {url}")
            return _Response(pages[url])

        monkeypatch.setattr(dl.requests, "get", get)
        return requested

    return install


@pytest.fixture
def fake_download(monkeypatch, tmp_path):
    """Replace downloads by writing the URL into the file; return the calls."""
    calls = []

    def download_if_missing(
        file_path, url, warn_missing=True, verbose=True, force_update=False
    ):
        calls.append((file_path, url, force_update))
        path = Path(file_path)
        if force_update or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(url)
        return file_path

    monkeypatch.setattr(dl, "download_if_missing", download_if_missing)
    monkeypatch.setattr(dl, "get_dataset_path", lambda sign, path: str(tmp_path))
    for cls in (Lee2021Mobile_ERP, Lee2021Mobile_SSVEP):
        monkeypatch.setattr(cls, "_metainfo_cache", None)
    return calls


def _local(tmp_path, name):
    ses = name.split("_")[1]
    return tmp_path / "MNE-lee2021mobile-data" / SUB / ses / "eeg" / name


def _source_of(tmp_path, name):
    """OSF path of the copy that landed in the local cache."""
    return _local(tmp_path, name).read_text().removeprefix(STORAGE)


# --- create_metainfo_osf ----------------------------------------------------


@pytest.mark.parametrize("page_size", [None, 2])
@pytest.mark.parametrize("with_paths", [True, False])
def test_metainfo_keeps_each_file_osf_path(fake_osf, page_size, with_paths):
    paths = _layout()
    fake_osf(_fake_osf_api(paths, page_size=page_size, with_paths=with_paths))

    metainfo = dl.create_metainfo_osf("r7s9b")

    assert list(metainfo.columns) == ["filename", "url", "path"]
    assert sorted(metainfo["path"]) == sorted(paths)
    assert (metainfo["filename"] == metainfo["path"].str.rsplit("/", n=1).str[1]).all()
    assert (metainfo["url"] == metainfo["path"].map(_url)).all()
    # Same file name, two paths: the listing no longer loses which is which.
    vhdr = metainfo[metainfo["filename"] == ERP[0]]
    assert sorted(vhdr["path"]) == [_source(ERP[0]), _bids(ERP[0])]


# --- Lee2021Mobile download selection ---------------------------------------


def test_reads_sourcedata_when_osf_lists_bids_copy_first(
    fake_osf, fake_download, tmp_path
):
    # LIFO walk: "sourcedata" listed before "sub-11" in the root page means
    # sub-11 is walked first, so the BIDS copy comes first in the metainfo.
    fake_osf(_fake_osf_api(_layout(), folder_first=("sourcedata",)))
    listing = dl.create_metainfo_osf("r7s9b")
    first_vhdr = listing[listing["filename"] == ERP[0]]["url"].iloc[0]
    assert first_vhdr == _url(_bids(ERP[0]))  # the order that broke develop

    vhdr_files = Lee2021Mobile_ERP().data_path(11)

    assert vhdr_files == [str(_local(tmp_path, ERP[0]))]
    for name in ERP + SSVEP:
        assert _source_of(tmp_path, name) == _source(name)
    # Sidecars: the /sourcedata/ copy when there is one, else the BIDS copy.
    assert _source_of(tmp_path, f"{SUB}_ses-01_task-ERP_channels.tsv") == _source(
        f"{SUB}_ses-01_task-ERP_channels.tsv"
    )
    assert _source_of(tmp_path, f"{SUB}_ses-01_task-ERP_events.tsv") == _bids(
        f"{SUB}_ses-01_task-ERP_events.tsv"
    )
    # No BIDS-copy BrainVision file was ever requested.
    eeg_urls = [u for _, u, _ in fake_download if u.endswith((".vhdr", ".eeg", ".vmrk"))]
    assert all(u.startswith(STORAGE + "/sourcedata/") for u in eeg_urls)
    assert (tmp_path / "MNE-lee2021mobile-data" / "dataset_description.json").exists()


@pytest.mark.parametrize("bids_first", [True, False])
@pytest.mark.parametrize("dataset_cls", [Lee2021Mobile_ERP, Lee2021Mobile_SSVEP])
def test_selection_does_not_depend_on_listing_order(
    monkeypatch, fake_download, tmp_path, bids_first, dataset_cls
):
    monkeypatch.setattr(
        dataset_cls,
        "_get_metainfo",
        classmethod(lambda cls: _metainfo(_layout(bids_first=bids_first))),
    )

    dataset_cls().data_path(11)

    for name in ERP + SSVEP:
        assert _source_of(tmp_path, name) == _source(name)


@pytest.mark.parametrize("missing", ERP)
def test_missing_sourcedata_file_fails_loudly(
    monkeypatch, fake_download, tmp_path, missing
):
    monkeypatch.setattr(
        Lee2021Mobile_ERP,
        "_get_metainfo",
        classmethod(lambda cls: _metainfo(_layout(drop=(_source(missing),)))),
    )
    Lee2021Mobile_ERP._metainfo_cache = "stale listing"

    with pytest.raises(FileNotFoundError, match=f"/sourcedata/{SUB}/") as err:
        Lee2021Mobile_ERP().data_path(11)

    assert missing in str(err.value)
    # Nothing was taken from the BIDS copy instead.
    assert not _local(tmp_path, missing).exists()
    # The (possibly incomplete) listing is dropped so a retry lists OSF again.
    assert Lee2021Mobile_ERP._metainfo_cache is None


def test_other_task_missing_from_sourcedata_does_not_block(
    monkeypatch, fake_download, tmp_path
):
    drop = tuple(_source(n) for n in SSVEP)
    monkeypatch.setattr(
        Lee2021Mobile_ERP,
        "_get_metainfo",
        classmethod(lambda cls: _metainfo(_layout(drop=drop))),
    )

    assert Lee2021Mobile_ERP().data_path(11) == [str(_local(tmp_path, ERP[0]))]
    # ... but its BIDS copy is not downloaded in place of the original.
    assert not any(_local(tmp_path, n).exists() for n in SSVEP)


def test_listing_without_paths_is_refused(monkeypatch, fake_download):
    listing = _metainfo(_layout()).drop(columns="path")
    monkeypatch.setattr(
        Lee2021Mobile_ERP, "_get_metainfo", classmethod(lambda cls: listing)
    )

    with pytest.raises(ValueError, match="no 'path' column"):
        Lee2021Mobile_ERP().data_path(11)


def test_existing_cache_is_used_as_is(monkeypatch, fake_download, tmp_path):
    local = _local(tmp_path, ERP[0])
    local.parent.mkdir(parents=True)
    local.write_text("cached")

    def no_listing(cls):
        raise AssertionError("a cached subject must not list OSF")

    monkeypatch.setattr(Lee2021Mobile_ERP, "_get_metainfo", classmethod(no_listing))

    assert Lee2021Mobile_ERP().data_path(11) == [str(local)]
    assert local.read_text() == "cached"
    assert fake_download == []


def test_force_update_replaces_a_cached_bids_copy(monkeypatch, fake_download, tmp_path):
    # A cache filled from the BIDS copy is repaired by force_update=True.
    for name in ERP:
        local = _local(tmp_path, name)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(_url(_bids(name)))
    monkeypatch.setattr(
        Lee2021Mobile_ERP, "_get_metainfo", classmethod(lambda cls: _metainfo(_layout()))
    )

    Lee2021Mobile_ERP().data_path(11, force_update=True)

    for name in ERP:
        assert _source_of(tmp_path, name) == _source(name)
