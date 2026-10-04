"""Tests for source fetching (HTTP mocked with a requests adapter)."""

import json
from pathlib import Path

import pytest
import requests
from requests.adapters import BaseAdapter

from scripts.paper_audit.fetch import DeepCeilingIndex, FetchResult, fetch_dataset
from scripts.paper_audit.inventory import DatasetRecord
from scripts.paper_audit.sources import (
    FetchError,
    HttpClient,
    classify_doi,
    find_related_paper,
    normalize_record,
)


class FakeAdapter(BaseAdapter):
    """Route requests by URL prefix; each route is a list of responses."""

    def __init__(self):
        super().__init__()
        self.routes: dict[str, list[tuple[int, bytes, dict]]] = {}
        self.calls: list[str] = []

    def add(self, prefix, status=200, body=b"", headers=None, json_body=None):
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
        if isinstance(body, str):
            body = body.encode()
        self.routes.setdefault(prefix, []).append((status, body, headers or {}))
        return self

    def send(self, request, **kwargs):
        self.calls.append(request.url)
        for prefix, queue in self.routes.items():
            if request.url.startswith(prefix):
                status, body, headers = queue[0] if len(queue) == 1 else queue.pop(0)
                resp = requests.Response()
                resp.status_code = status
                resp._content = body
                resp.headers.update(headers)
                resp.url = request.url
                resp.request = request
                return resp
        raise AssertionError(f"unexpected request: {request.url}")

    def close(self):
        pass


@pytest.fixture
def http():
    adapter = FakeAdapter()
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    sleeps: list[float] = []
    client = HttpClient(session=session, sleep=sleeps.append, clock=lambda: 0.0)
    client._test_sleeps = sleeps
    client._test_adapter = adapter
    return client


def _crossref_article(doi, title="A paper", year=2017):
    return {
        "status": "ok",
        "message": {
            "DOI": doi,
            "type": "journal-article",
            "title": [title],
            "author": [{"given": "Ann", "family": "Author"}],
            "issued": {"date-parts": [[year, 1, 1]]},
            "container-title": ["Journal"],
        },
    }


def _datacite_dataset(doi, related=None):
    return {
        "data": {
            "attributes": {
                "doi": doi,
                "titles": [{"title": "A dataset"}],
                "creators": [{"name": "Author, Ann"}],
                "publicationYear": 2021,
                "publisher": "Zenodo",
                "types": {"resourceTypeGeneral": "Dataset"},
                "relatedIdentifiers": related or [],
                "url": "https://zenodo.org/records/999",
            }
        }
    }


def _record(name="FakeDS", primary="10.1000/paper", paper=None, doc=None):
    return DatasetRecord(
        name=name,
        module="moabb.datasets.fake",
        primary_doi=primary,
        paper_dois=paper or [],
        docstring_dois=doc or [],
        declared={"n_subjects": 9},
    )


# --- pure helpers ---------------------------------------------------------


def test_classify_doi_prefixes_and_types():
    assert classify_doi("10.5281/zenodo.123") == "dataset"
    assert classify_doi("10.6084/m9.figshare.1") == "dataset"
    assert classify_doi("10.18112/openneuro.ds004362.v1.0.0") == "dataset"
    assert classify_doi("10.17605/osf.io/abc") == "dataset"
    assert classify_doi("10.7910/dvn/abc") == "dataset"
    assert classify_doi("10.17632/abc.1") == "dataset"
    assert classify_doi("10.13026/abc") == "dataset"
    assert classify_doi("10.1002/hbm.23730") == "paper"
    rec = normalize_record(
        "10.1000/x", datacite=_datacite_dataset("10.1000/x")["data"]["attributes"]
    )
    assert classify_doi("10.1000/x", rec) == "dataset"
    rec = normalize_record("10.1000/y", crossref=_crossref_article("10.1000/y"))
    assert classify_doi("10.1000/y", rec) == "paper"


def test_find_related_paper_shapes():
    datacite = normalize_record(
        "10.5281/zenodo.1",
        datacite=_datacite_dataset(
            "10.5281/zenodo.1",
            related=[
                {
                    "relationType": "IsSupplementTo",
                    "relatedIdentifierType": "DOI",
                    "relatedIdentifier": "10.1000/PAPER1",
                },
                {
                    "relationType": "HasVersion",
                    "relatedIdentifierType": "DOI",
                    "relatedIdentifier": "10.5281/zenodo.2",
                },
            ],
        )["data"]["attributes"],
    )
    datacite["repository"] = {
        "kind": "zenodo",
        "record": {
            "metadata": {
                "related_identifiers": [
                    {"relation": "isSupplementTo", "identifier": "10.1000/paper2"},
                    {"relation": "isVersionOf", "identifier": "10.5281/zenodo.0"},
                ],
                "references": ["Doe J. Title. doi:10.1000/paper3"],
            }
        },
        "description": {
            "ReferencesAndLinks": ["https://doi.org/10.1000/paper4", "no doi here"]
        },
        "figshare_references": ["https://doi.org/10.1000/paper5"],
    }
    assert find_related_paper(datacite) == [
        "10.1000/paper1",
        "10.1000/paper2",
        "10.1000/paper3",
        "10.1000/paper4",
        "10.1000/paper5",
    ]


# --- HTTP client ----------------------------------------------------------


def test_rate_limit_and_retry(http):
    adapter = http._test_adapter
    adapter.add("https://api.example.org/a", status=429, headers={"Retry-After": "2"})
    adapter.add("https://api.example.org/a", status=200, body=b"ok")
    adapter.add("https://api.example.org/b", status=200, body=b"ok")

    resp = http.get("https://api.example.org/a")
    assert resp.status_code == 200
    assert adapter.calls.count("https://api.example.org/a") == 2
    assert http._test_sleeps and max(http._test_sleeps) >= 2.0

    http.get("https://api.example.org/b")
    # Same host, clock frozen at 0 -> a full minimum interval must be slept.
    assert any(s >= http.min_interval for s in http._test_sleeps)
    assert [c["status"] for c in http.calls] == [429, 200, 200]


def test_retry_exhaustion_raises(http):
    adapter = http._test_adapter
    for _ in range(4):
        adapter.add("https://api.example.org/dead", status=503)
    with pytest.raises(FetchError):
        http.get("https://api.example.org/dead")


# --- fetch_dataset --------------------------------------------------------


def test_dataset_doi_without_paper_is_unsupported(http, tmp_path):
    adapter = http._test_adapter
    doi = "10.5281/zenodo.999"
    adapter.add("https://api.crossref.org/works/", status=404)
    adapter.add("https://api.datacite.org/dois/", json_body=_datacite_dataset(doi))
    adapter.add(
        "https://zenodo.org/api/records/999",
        json_body={
            "id": 999,
            "metadata": {"related_identifiers": [], "references": []},
            "files": [
                {
                    "key": "README.md",
                    "size": 120,
                    "links": {
                        "self": "https://zenodo.org/api/records/999/files/README.md/content"
                    },
                },
                {
                    "key": "sub-01_eeg.bdf",
                    "size": 5_000_000,
                    "links": {
                        "self": "https://zenodo.org/api/records/999/files/sub-01_eeg.bdf/content"
                    },
                },
            ],
        },
    )
    adapter.add(
        "https://zenodo.org/api/records/999/files/README.md/content",
        body=b"# Dataset\nRecorded from 12 participants at 256 Hz.\n",
    )

    result = fetch_dataset(_record(primary=doi), tmp_path, client=http)

    assert isinstance(result, FetchResult)
    assert result.access == "missing"
    assert result.unsupported is True
    assert result.paper_text_paths == []
    assert result.paper_dois == []
    assert [p.name for p in result.repository_files] == ["README.md"]
    assert not any("bdf" in c for c in adapter.calls)
    prov = json.loads(result.provenance.read_text())
    assert any("zenodo.org/api/records/999" in e["url"] for e in prov["requests"])
    assert "manual DOI assignment" in " ".join(result.notes)


def test_non_pdf_payload_not_cached(http, tmp_path):
    adapter = http._test_adapter
    doi = "10.1000/paper"
    adapter.add("https://api.crossref.org/works/", json_body=_crossref_article(doi))
    adapter.add(
        "https://api.unpaywall.org/v2/",
        json_body={
            "is_oa": True,
            "oa_status": "green",
            "best_oa_location": {"url_for_pdf": "https://publisher.example/x.pdf"},
            "oa_locations": [],
        },
    )
    adapter.add(
        "https://publisher.example/x.pdf",
        body=b"<html><body>Please log in</body></html>",
        headers={"Content-Type": "text/html"},
    )
    adapter.add(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json_body={"errCode": 404, "errMsg": "boom"},
    )

    result = fetch_dataset(_record(primary=doi), tmp_path, client=http)

    assert result.access == "closed"
    assert result.paper_text_paths == []
    assert list((tmp_path / "FakeDS").glob("paper-*")) == []
    prov = json.loads(result.provenance.read_text())
    outcomes = [e.get("outcome") for e in prov["requests"]]
    assert "not_pdf" in outcomes
    assert "europepmc_error" in outcomes


def test_europepmc_fulltext_fallback(http, tmp_path):
    adapter = http._test_adapter
    doi = "10.1000/oa"
    adapter.add("https://api.crossref.org/works/", json_body=_crossref_article(doi))
    adapter.add(
        "https://api.unpaywall.org/v2/", json_body={"is_oa": False, "oa_locations": []}
    )
    adapter.add(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json_body={"hitCount": 1, "resultList": {"result": [{"pmcid": "PMC123"}]}},
    )
    adapter.add(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML",
        body=b"<article><body><sec><title>Methods</title><p>Twenty subjects at 512 Hz.</p></sec></body></article>",
    )
    result = fetch_dataset(_record(primary=doi), tmp_path, client=http)
    assert result.access == "open"
    assert len(result.paper_text_paths) == 1
    text = result.paper_text_paths[0].read_text()
    assert "Methods" in text and "Twenty subjects at 512 Hz." in text


def test_reuses_deep_ceiling_markdown(http, tmp_path):
    repo = tmp_path / "deep-ceiling"
    (repo / "docs" / "paper" / "refs_md").mkdir(parents=True)
    (repo / "docs" / "paper" / "ref_dois.json").write_text(
        json.dumps({"bnci2014001_ds": {"doi": "10.3389/FNINS.2012.00055"}})
    )
    (repo / "docs" / "paper" / "refs_md" / "bnci2014001_ds.md").write_text(
        "## Review of the BCI competition IV\n9 subjects, 22 EEG channels\n"
    )
    index = DeepCeilingIndex.from_repo(repo)
    assert index.lookup("10.3389/fnins.2012.00055") is not None

    adapter = http._test_adapter
    adapter.add(
        "https://api.crossref.org/works/",
        json_body=_crossref_article("10.3389/fnins.2012.00055"),
    )
    cache = tmp_path / "cache"
    result = fetch_dataset(
        _record(name="BNCI2014_001", primary="10.3389/fnins.2012.00055"),
        cache,
        client=http,
        deep_ceiling=index,
    )
    assert result.access == "open"
    assert len(result.paper_text_paths) == 1
    assert "22 EEG channels" in result.paper_text_paths[0].read_text()
    assert not any("unpaywall" in c for c in adapter.calls)
    prov = json.loads(result.provenance.read_text())
    assert any(s.get("source") == "deep-ceiling" for s in prov["papers"])
    # Deep-ceiling repository must be untouched.
    assert sorted(p.name for p in (repo / "docs" / "paper" / "refs_md").iterdir()) == [
        "bnci2014001_ds.md"
    ]


def test_offline_mode_makes_no_requests(http, tmp_path):
    result = fetch_dataset(_record(), tmp_path, client=http, offline=True)
    assert http._test_adapter.calls == []
    assert result.access == "missing"


def test_cache_hit_skips_network(http, tmp_path):
    adapter = http._test_adapter
    doi = "10.1000/oa"
    adapter.add("https://api.crossref.org/works/", json_body=_crossref_article(doi))
    adapter.add(
        "https://api.unpaywall.org/v2/", json_body={"is_oa": False, "oa_locations": []}
    )
    adapter.add(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json_body={"hitCount": 1, "resultList": {"result": [{"pmcid": "PMC123"}]}},
    )
    adapter.add(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML",
        body=b"<article><body><p>Twenty subjects.</p></body></article>",
    )
    first = fetch_dataset(_record(primary=doi), tmp_path, client=http)
    n_calls = len(adapter.calls)
    second = fetch_dataset(_record(primary=doi), tmp_path, client=http)
    assert len(adapter.calls) == n_calls
    assert second.access == first.access == "open"
    assert [Path(p) for p in second.paper_text_paths] == first.paper_text_paths
