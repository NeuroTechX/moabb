"""DOI validation for MOABB dataset metadata.

Treats the DOI in each dataset class (``__init__``) as ground truth,
then validates that:

1. All DOIs (class, METADATA, docstring) have valid format.
2. Docstring DOIs are tracked in METADATA.
3. Every DOI resolves via doi.org content negotiation.
4. When class DOI and metadata DOI differ, they share at least one author.

Run all (including network tests)::

    python -m pytest moabb/tests/test_doi_validation.py --timeout=300 -v

Run only offline checks::

    python -m pytest moabb/tests/test_doi_validation.py -k "not network" -v
"""

import json
import pathlib
import re
import time
import unicodedata

import pytest
import requests

from moabb.datasets.metadata.schema import DatasetMetadata
from moabb.datasets.utils import dataset_list


_SKIP_CLASSES = {"FakeDataset", "FakeVirtualRealityDataset"}
_NON_DOI_PREFIXES = ("hal-", "tel-", "arXiv:")
_DATA_REPO_PREFIXES = (
    "10.5281/zenodo.",
    "10.7910/DVN/",
    "10.6084/m9.figshare.",
    "10.6094/",
    "10.5524/",
    "10.5061/dryad.",
    "10.34973/",
    "10.18115/",
    "10.48550/arXiv.",
    "10.35376/",
    "10.13026/",
)
_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
)
_DOI_RE = re.compile(r"^10\.\d{4,}/")
_REQUEST_DELAY = 0.15

_REAL_DATASETS = [
    cls
    for cls in dataset_list
    if cls.__name__ not in _SKIP_CLASSES
    and isinstance(getattr(cls, "METADATA", None), DatasetMetadata)
]


# -- helpers -----------------------------------------------------------------


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    for prefix in _DOI_URL_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _is_doi(value: str) -> bool:
    return bool(value and _DOI_RE.match(_normalize_doi(value) or ""))


def _extract_docstring_dois(cls) -> list[str]:
    doc = getattr(cls, "__doc__", "") or ""
    raw = re.findall(r"10\.\d{4,}/[^\s\]\">]+", doc)
    cleaned = []
    for d in raw:
        d = d.rstrip(".,;:)")
        d = d.rstrip("`")
        if ">`_" in d:
            d = d[: d.index(">")]
        d = d.rstrip("`_>")
        if d.endswith("/abstract"):
            d = d[: -len("/abstract")]
        cleaned.append(d)
    return list(dict.fromkeys(cleaned))


def _collect_dois(cls) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    try:
        instance = cls()
        result["init.doi"] = _normalize_doi(getattr(instance, "doi", None))
    except Exception:
        result["init.doi"] = None

    meta = getattr(cls, "METADATA", None)
    if isinstance(meta, DatasetMetadata):
        doc = getattr(meta, "documentation", None)
        if doc:
            result["metadata.doi"] = _normalize_doi(getattr(doc, "doi", None))
            result["metadata.associated_paper"] = _normalize_doi(
                getattr(doc, "associated_paper_doi", None)
            )
            for j, rd in enumerate(getattr(doc, "related_paper_dois", None) or []):
                result[f"metadata.related.{j}"] = _normalize_doi(rd)

    for i, doi in enumerate(_extract_docstring_dois(cls)):
        result[f"docstring.{i}"] = doi
    return result


_DOI_CACHE: dict[str, dict | None] = {}
_DOI_CACHE_PATH = pathlib.Path(__file__).parent / "doi_cache.json"
_DOI_CACHE_DIRTY = False
_UPDATE_DOI_CACHE = False
_NEWLY_RESOLVED_DOIS: list[str] = []


def _load_doi_cache():
    """Load persistent DOI cache from disk into _DOI_CACHE."""
    global _DOI_CACHE
    if _DOI_CACHE_PATH.exists():
        try:
            data = json.loads(_DOI_CACHE_PATH.read_text(encoding="utf-8"))
            # Skip the _metadata key
            _DOI_CACHE = {k: v for k, v in data.items() if k != "_metadata"}
        except (json.JSONDecodeError, OSError):
            _DOI_CACHE = {}


def _save_doi_cache():
    """Write _DOI_CACHE to disk as sorted JSON with metadata."""
    data = {"_metadata": {"total": len(_DOI_CACHE)}}
    data.update(dict(sorted(_DOI_CACHE.items())))
    _DOI_CACHE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _resolve_doi(doi: str) -> dict | None:
    global _DOI_CACHE_DIRTY
    if not _UPDATE_DOI_CACHE and doi in _DOI_CACHE:
        return _DOI_CACHE[doi]
    try:
        time.sleep(_REQUEST_DELAY)
        r = requests.get(
            f"https://doi.org/{doi}",
            headers={"Accept": "application/citeproc+json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in data.get("author", [])
        ]
        issued = data.get("issued", {}).get("date-parts", [[None]])
        year = issued[0][0] if issued and issued[0] and issued[0][0] else None
        result = {
            "title": data.get("title"),
            "authors": authors,
            "year": year,
            "doi": doi,
        }
    except Exception:
        result = None
    if result is not None:
        if doi not in _DOI_CACHE:
            _NEWLY_RESOLVED_DOIS.append(doi)
        _DOI_CACHE[doi] = result
        _DOI_CACHE_DIRTY = True
    return result


# Load persistent cache at import time
_load_doi_cache()


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _extract_surnames(authors: list[str]) -> set[str]:
    out = set()
    for a in authors or []:
        a = a.strip()
        if not a:
            continue
        if ", " in a:
            surname = a.split(",")[0].strip().lower()
        else:
            parts = a.split()
            if not parts:
                continue
            surname = parts[-1].strip(".").lower()
        out.add(_strip_accents(surname))
    return out


def _all_codebase_dois() -> set[str]:
    """Collect every unique resolvable DOI from all dataset classes (offline)."""
    all_dois: set[str] = set()
    for cls in _REAL_DATASETS:
        for doi in _collect_dois(cls).values():
            if doi and _is_doi(doi):
                all_dois.add(doi)
    return all_dois


# -- offline tests -----------------------------------------------------------

_ids = lambda c: c.__name__  # noqa: E731


@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_doi_format(dataset_class):
    dois = _collect_dois(dataset_class)
    if not any(dois.values()):
        pytest.skip("No DOIs found")
    invalid = [
        f"  {src} = {doi!r}"
        for src, doi in dois.items()
        if doi
        and not _is_doi(doi)
        and not any(doi.startswith(p) for p in _NON_DOI_PREFIXES)
    ]
    assert not invalid, f"{dataset_class.__name__}: invalid DOI format:\n" + "\n".join(
        invalid
    )


@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_docstring_dois_tracked(dataset_class):
    dois = _collect_dois(dataset_class)

    known = {
        dois.get(k)
        for k in ("metadata.doi", "metadata.associated_paper", "init.doi")
        if dois.get(k) and _is_doi(dois.get(k))
    }
    meta = getattr(dataset_class, "METADATA", None)
    if meta and getattr(meta, "documentation", None):
        data_url = getattr(meta.documentation, "data_url", None)
        if isinstance(data_url, str):
            m = re.search(r"10\.\d{4,}/[^\s]+", data_url)
            if m:
                known.add(m.group().rstrip(".,;:)"))
        for rd in getattr(meta.documentation, "related_paper_dois", None) or []:
            nd = _normalize_doi(rd)
            if nd and _is_doi(nd):
                known.add(nd)

    doc_dois = [dois[k] for k in sorted(dois) if k.startswith("docstring.") and dois[k]]
    if not doc_dois:
        pytest.skip("No DOIs in docstring")

    untracked = [
        d
        for d in doc_dois
        if d not in known
        and not any(d.startswith(p) for p in _DATA_REPO_PREFIXES)
        and not any(d in k or k in d for k in known)
    ]
    assert not untracked, (
        f"{dataset_class.__name__}: docstring DOIs not tracked in metadata: "
        f"{untracked}\n  Known: {known}"
    )


def test_doi_cache_complete():
    """Check that doi_cache.json contains every DOI found in the codebase."""
    codebase_dois = _all_codebase_dois()
    cached_dois = set(_DOI_CACHE.keys())
    missing = sorted(codebase_dois - cached_dois)
    assert not missing, (
        f"{len(missing)} DOI(s) found in codebase but missing from "
        f"{_DOI_CACHE_PATH.name}:\n"
        + "\n".join(f"  - {d}" for d in missing)
        + "\n\nTo fix, run:\n"
        "  python -m pytest moabb/tests/test_doi_validation.py "
        '-k "test_dois_resolve" --timeout=300 -v'
    )


@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_data_url_format(dataset_class):
    """Check that data_url values are well-formed and don't have broken suffixes."""
    meta = getattr(dataset_class, "METADATA", None)
    if not isinstance(meta, DatasetMetadata):
        pytest.skip("No METADATA")
    doc = getattr(meta, "documentation", None)
    if not doc or not doc.data_url:
        pytest.skip("No data_url")

    url = doc.data_url
    issues = []

    # Trailing /files/ is a common Zenodo mistake — the /files/ page doesn't
    # exist as a standalone URL; the record page is the correct landing page.
    if url.rstrip("/").endswith("/files"):
        issues.append(
            f"data_url ends with '/files/' — should be the record landing page: "
            f"{url.rstrip('/').rsplit('/files', 1)[0]}"
        )

    # Trailing slash on DOI URLs (doi.org doesn't need it and some resolvers choke)
    if url.startswith(("https://doi.org/", "http://doi.org/")) and url.endswith("/"):
        issues.append("DOI URL should not end with trailing slash")

    assert not issues, (
        f"{dataset_class.__name__}: data_url issues:\n"
        + "\n".join(f"  - {i}" for i in issues)
        + f"\n  current: {url}"
    )


@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_senior_author_in_paper(dataset_class):
    """Check that senior_author appears in the paper's author list.

    The senior/corresponding author is not always the last listed author
    (conventions vary across fields and teams), so this test only verifies
    that the ``senior_author`` is among the paper's authors at all.

    Datasets whose DOI resolves to a data repository (Zenodo, Figshare, etc.)
    rather than a paper are skipped because data repositories often list only
    the uploader, not the full author list.
    """
    meta = getattr(dataset_class, "METADATA", None)
    if not isinstance(meta, DatasetMetadata):
        pytest.skip("No METADATA")
    doc = getattr(meta, "documentation", None)
    if not doc or not doc.senior_author:
        pytest.skip("No senior_author")

    # Prefer associated_paper_doi (the actual paper), fall back to doi
    paper_doi = _normalize_doi(doc.associated_paper_doi) or _normalize_doi(doc.doi)
    if not paper_doi or not _is_doi(paper_doi):
        pytest.skip("No resolvable paper DOI")

    # Skip data-repository DOIs — they often list only the uploader
    if any(paper_doi.startswith(p) for p in _DATA_REPO_PREFIXES):
        pytest.skip(f"DOI {paper_doi} is a data repository, not a paper")

    cached = _DOI_CACHE.get(paper_doi)
    if not cached:
        pytest.skip(f"DOI {paper_doi} not in cache")

    cache_authors = cached.get("authors", [])
    if not cache_authors:
        pytest.skip("No authors in DOI cache entry")

    senior_surnames = _extract_surnames([doc.senior_author])
    all_surnames = _extract_surnames(cache_authors)

    if senior_surnames & all_surnames:
        return  # senior_author found in the paper's author list

    pytest.fail(
        f"{dataset_class.__name__}: senior_author {doc.senior_author!r} "
        f"is NOT in the paper's author list.\n"
        f"  DOI: {paper_doi}\n"
        f"  Paper authors: {cache_authors}"
    )


@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_investigators_match_doi_authors(dataset_class):
    """Check that the investigators list matches DOI-resolved authors.

    When both ``investigators`` and a resolvable paper DOI exist, verify
    that the last investigator matches the last DOI author (i.e. the
    investigators list preserves academic author order).
    """
    meta = getattr(dataset_class, "METADATA", None)
    if not isinstance(meta, DatasetMetadata):
        pytest.skip("No METADATA")
    doc = getattr(meta, "documentation", None)
    if not doc or not doc.investigators:
        pytest.skip("No investigators")

    paper_doi = _normalize_doi(doc.associated_paper_doi) or _normalize_doi(doc.doi)
    if not paper_doi or not _is_doi(paper_doi):
        pytest.skip("No resolvable paper DOI")

    if any(paper_doi.startswith(p) for p in _DATA_REPO_PREFIXES):
        pytest.skip(f"DOI {paper_doi} is a data repository")

    cached = _DOI_CACHE.get(paper_doi)
    if not cached:
        pytest.skip(f"DOI {paper_doi} not in cache")

    cache_authors = cached.get("authors", [])
    if len(cache_authors) <= 1:
        pytest.skip("Single-author paper, order comparison not meaningful")

    # When the investigators list has significantly more names than the DOI
    # author list, it likely includes additional dataset contributors beyond
    # the paper authors — skip ordering comparison in that case.
    if len(doc.investigators) > len(cache_authors) + 2:
        pytest.skip(
            f"investigators ({len(doc.investigators)}) much larger than "
            f"DOI authors ({len(cache_authors)}), likely includes extra contributors"
        )

    inv_last = doc.investigators[-1]
    doi_last = cache_authors[-1]
    inv_surnames = _extract_surnames([inv_last])
    doi_surnames = _extract_surnames([doi_last])

    if inv_surnames & doi_surnames:
        return

    # Check if investigators first matches DOI first (reversed order)
    inv_first = doc.investigators[0]
    doi_first = cache_authors[0]
    inv_first_surnames = _extract_surnames([inv_first])
    doi_first_surnames = _extract_surnames([doi_first])

    if inv_first_surnames & doi_first_surnames:
        # First authors match but last don't — likely different author ordering
        pytest.fail(
            f"{dataset_class.__name__}: investigators last author mismatch.\n"
            f"  investigators last: {inv_last!r}\n"
            f"  DOI last author:    {doi_last!r}\n"
            f"  DOI: {paper_doi}\n"
            f"  (First authors match, but last authors differ)"
        )


# -- network tests -----------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_dois_resolve(dataset_class):
    dois = _collect_dois(dataset_class)
    unique = sorted({d for d in dois.values() if d and _is_doi(d)})
    if not unique:
        pytest.skip("No DOIs to resolve")
    failures = []
    for doi in unique:
        result = _resolve_doi(doi)
        if result is None:
            failures.append(doi)
        elif not result.get("title"):
            failures.append(f"{doi} (no title)")
    assert not failures, f"{dataset_class.__name__}: DOIs failed to resolve: {failures}"


@pytest.mark.network
@pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=_ids)
def test_class_and_metadata_dois_share_authors(dataset_class):
    dois = _collect_dois(dataset_class)
    init_doi = dois.get("init.doi")
    if not init_doi or not _is_doi(init_doi):
        pytest.skip("No class DOI")

    meta_doi = dois.get("metadata.doi")
    if not meta_doi or not _is_doi(meta_doi) or meta_doi == init_doi:
        pytest.skip("No differing metadata.doi to compare")

    init_result = _resolve_doi(init_doi)
    if init_result is None:
        pytest.skip(f"Could not resolve init DOI {init_doi!r}")

    init_authors = _extract_surnames(init_result.get("authors"))

    meta_result = _resolve_doi(meta_doi)
    if meta_result is None:
        pytest.skip(f"Could not resolve metadata DOI {meta_doi!r}")

    meta_authors = _extract_surnames(meta_result.get("authors"))
    if init_authors & meta_authors:
        return

    pytest.fail(
        f"{dataset_class.__name__}: class DOI shares no authors with "
        f"metadata DOI.\n"
        f"  class DOI {init_doi}: {init_result.get('title')}\n"
        f"    authors: {init_result.get('authors')}\n"
        f"  metadata.doi {meta_doi}: {meta_result.get('title')}\n"
        f"    authors: {meta_result.get('authors')}"
    )
