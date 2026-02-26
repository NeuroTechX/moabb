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
    "10.34973/",
    "10.18115/",
    "10.48550/arXiv.",
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

    for i, doi in enumerate(_extract_docstring_dois(cls)):
        result[f"docstring.{i}"] = doi
    return result


_DOI_CACHE: dict[str, dict | None] = {}


def _resolve_doi(doi: str) -> dict | None:
    if doi in _DOI_CACHE:
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
    _DOI_CACHE[doi] = result
    return result


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
