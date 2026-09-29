"""HTTP client and source-specific helpers for the paper audit.

Every remote call goes through :class:`HttpClient`, which enforces a minimum
interval per host (default 1 request/second), retries transient failures with
exponential backoff (honouring ``Retry-After``) and logs each request so the
caller can persist provenance.

Silent-failure modes documented in ``~/.agents/skills/paper-lookup/references``
are handled explicitly:

* Europe PMC search errors arrive as HTTP 200 with ``errCode`` and no
  ``resultList``; ``fullTextXML`` returns a clean 404 when not open access.
* Unpaywall ``url_for_pdf`` may serve an HTML landing page with HTTP 200; the
  payload is only accepted when it starts with ``%PDF``.
* Figshare ``GET /articles`` is a listing, not a search; only ``/articles/{id}``
  is used here.
"""

from __future__ import annotations

import html
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests


log = logging.getLogger("paper_audit.sources")

DEFAULT_EMAIL = "b.aristimunha@gmail.com"
USER_AGENT = "moabb-paper-audit/0.1 (https://github.com/NeuroTechX/moabb; mailto:{email})"

CROSSREF_API = "https://api.crossref.org/works/"
DATACITE_API = "https://api.datacite.org/dois/"
UNPAYWALL_API = "https://api.unpaywall.org/v2/"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
ARXIV_PDF = "https://arxiv.org/pdf/"
ZENODO_API = "https://zenodo.org/api/records/"
FIGSHARE_API = "https://api.figshare.com/v2/articles/"
OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org/"
OSF_API = "https://api.osf.io/v2/nodes/"
DATAVERSE_API = "https://dataverse.harvard.edu/api/"
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/"

#: DOI prefixes that always denote data repositories.
DATASET_DOI_PREFIXES = (
    "10.5281/",  # Zenodo
    "10.6084/",  # Figshare
    "10.18112/",  # OpenNeuro
    "10.17605/",  # OSF
    "10.7910/",  # Harvard Dataverse
    "10.17632/",  # Mendeley Data
    "10.13026/",  # PhysioNet
    "10.5524/",  # GigaDB
    "10.5061/",  # Dryad
    "10.34973/",  # Radboud Data Repository
    "10.6094/",  # FreiDok (Univ. Freiburg)
    "10.35376/",  # UNIVERSITY data repositories (Bath)
    "10.18115/",  # HAL Data? (kept from test_doi_validation)
)

RELATED_PAPER_RELATIONS = {
    "issupplementto",
    "isdescribedby",
    "isreferencedby",
    "iscitedby",
    "isdocumentedby",
    "ispublishedin",
}

MAX_REPO_FILE_BYTES = 2 * 1024 * 1024
MAX_REPO_FILES = 12
MAX_PDF_BYTES = 60 * 1024 * 1024
TEXT_FILE_RE = re.compile(
    r"^(readme.*|license.*|changes|citation.*|dataset_description\.json|participants\.(tsv|json)"
    r"|.*\.(md|txt|json|tsv|csv|bib|rst|yaml|yml))$",
    re.IGNORECASE,
)
SIGNAL_EXT_RE = re.compile(
    r"\.(edf|bdf|gdf|mat|fif|set|fdt|npy|npz|h5|hdf5|eeg|vhdr|vmrk|xdf|zip|tar|gz|tgz|7z|rar"
    r"|nii|pkl|pickle|dat|bin|cnt|raw|snirf|mff|pdf|png|jpg|jpeg)$",
    re.IGNORECASE,
)
_DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,}/[^\s\]\">,;]+")


class FetchError(RuntimeError):
    """Raised when a request keeps failing after retries."""


class HttpClient:
    """Polite, rate-limited ``requests`` wrapper with a request log."""

    RETRY_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        session: requests.Session | None = None,
        email: str = DEFAULT_EMAIL,
        min_interval: float = 1.0,
        retries: int = 3,
        timeout: float = 45.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.session = session or requests.Session()
        self.email = email
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._last_by_host: dict[str, float] = {}
        self.calls: list[dict] = []
        self.user_agent = USER_AGENT.format(email=email)

    # -- rate limiting -----------------------------------------------------
    def _throttle(self, host: str) -> None:
        last = self._last_by_host.get(host)
        if last is not None:
            elapsed = self._clock() - last
            if elapsed < self.min_interval:
                self._sleep(self.min_interval - elapsed)
        self._last_by_host[host] = self._clock()

    def mark(self, outcome: str, **extra) -> None:
        """Annotate the most recent request with an outcome label."""
        if self.calls:
            self.calls[-1]["outcome"] = outcome
            self.calls[-1].update(extra)

    # -- requests ------------------------------------------------------------
    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | None = None,
        method: str = "GET",
        json_body: dict | None = None,
    ) -> requests.Response:
        host = urlsplit(url).netloc
        hdrs = {"User-Agent": self.user_agent, **(headers or {})}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle(host)
            entry = {
                "url": url,
                "params": params,
                "method": method,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "attempt": attempt,
            }
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=hdrs,
                    timeout=timeout or self.timeout,
                    json=json_body,
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                last_error = exc
                entry.update(status=None, error=f"{type(exc).__name__}: {exc}")
                self.calls.append(entry)
                self._sleep(2.0**attempt)
                continue
            entry.update(status=resp.status_code, bytes=len(resp.content or b""))
            self.calls.append(entry)
            if resp.status_code in self.RETRY_STATUSES and attempt < self.retries:
                retry_after = resp.headers.get("Retry-After")
                wait = 2.0**attempt
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                self._sleep(wait)
                continue
            if resp.status_code in self.RETRY_STATUSES:
                raise FetchError(f"{url} -> HTTP {resp.status_code} after retries")
            return resp
        raise FetchError(f"{url} failed after retries: {last_error}")

    def get_json(self, url: str, **kwargs) -> tuple[int, dict | list | None]:
        resp = self.get(url, **kwargs)
        if resp.status_code != 200:
            return resp.status_code, None
        try:
            return 200, resp.json()
        except ValueError:
            self.mark("invalid_json")
            return 200, None


# ---------------------------------------------------------------------------
# DOI metadata records
# ---------------------------------------------------------------------------


def crossref_work(client: HttpClient, doi: str) -> dict | None:
    status, data = client.get_json(
        CROSSREF_API + quote(doi, safe="/()"), params={"mailto": client.email}
    )
    if status != 200 or not isinstance(data, dict) or "message" not in data:
        client.mark("crossref_miss" if status == 404 else "crossref_error")
        return None
    client.mark("crossref_ok")
    return data["message"]


def datacite_doi(client: HttpClient, doi: str) -> dict | None:
    status, data = client.get_json(DATACITE_API + quote(doi, safe="/()"))
    attrs = (
        (data or {}).get("data", {}).get("attributes") if isinstance(data, dict) else None
    )
    if status != 200 or not attrs:
        client.mark("datacite_miss" if status == 404 else "datacite_error")
        return None
    client.mark("datacite_ok")
    return attrs


def _crossref_authors(msg: dict) -> list[str]:
    out = []
    for a in msg.get("author", []) or []:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x) or a.get(
            "name"
        )
        if name:
            out.append(name)
    return out


def _datacite_authors(attrs: dict) -> list[str]:
    out = []
    for c in attrs.get("creators", []) or []:
        name = c.get("name") or " ".join(
            x for x in [c.get("givenName"), c.get("familyName")] if x
        )
        if name:
            if "," in name:
                fam, _, giv = name.partition(",")
                name = f"{giv.strip()} {fam.strip()}".strip()
            out.append(name)
    return out


def normalize_record(
    doi: str, crossref: dict | None = None, datacite: dict | None = None
):
    """Flatten a Crossref message or DataCite attributes into a common shape."""
    rec: dict = {
        "doi": doi.lower(),
        "source": None,
        "type": None,
        "resource_type": None,
        "title": None,
        "authors": [],
        "year": None,
        "venue": None,
        "publisher": None,
        "url": None,
        "arxiv_id": None,
        "raw": None,
    }
    if crossref:
        msg = dict(crossref)
        msg.pop("reference", None)
        rec.update(
            source="crossref",
            type=msg.get("type"),
            resource_type=msg.get("type"),
            title=(msg.get("title") or [None])[0],
            authors=_crossref_authors(msg),
            venue=(msg.get("container-title") or [None])[0],
            publisher=msg.get("publisher"),
            url=msg.get("URL"),
            raw=msg,
        )
        for key in ("published-print", "published-online", "issued", "created"):
            parts = (msg.get(key) or {}).get("date-parts") or [[None]]
            if parts and parts[0] and parts[0][0]:
                rec["year"] = int(parts[0][0])
                break
        rel = msg.get("relation") or {}
        for items in rel.values():
            for it in items if isinstance(items, list) else []:
                ident = str(it.get("id", ""))
                m = re.search(r"arxiv\.(\d{4}\.\d{4,5}(v\d+)?)", ident, re.IGNORECASE)
                if m:
                    rec["arxiv_id"] = m.group(1)
    elif datacite:
        attrs = dict(datacite)
        types = attrs.get("types") or {}
        rec.update(
            source="datacite",
            type=types.get("resourceTypeGeneral"),
            resource_type=types.get("resourceType") or types.get("resourceTypeGeneral"),
            title=((attrs.get("titles") or [{}])[0] or {}).get("title"),
            authors=_datacite_authors(attrs),
            year=attrs.get("publicationYear"),
            venue=attrs.get("publisher"),
            publisher=attrs.get("publisher"),
            url=attrs.get("url"),
            raw=attrs,
        )
    m = re.match(r"10\.48550/arxiv\.(.+)$", doi, re.IGNORECASE)
    if m:
        rec["arxiv_id"] = m.group(1)
    return rec


def classify_doi(doi: str, record: dict | None = None) -> str:
    """Return ``"dataset"`` or ``"paper"`` for a DOI."""
    d = (doi or "").lower()
    if d.startswith(DATASET_DOI_PREFIXES):
        return "dataset"
    if record:
        t = str(record.get("type") or "").lower()
        if record.get("source") == "crossref" and t in {"dataset", "component"}:
            return "dataset"
        if record.get("source") == "datacite":
            if t in {"dataset", "collection", "software", "physicalobject", "image"}:
                return "dataset"
            if t in {"text", "journalarticle", "preprint", "conferencepaper", "report"}:
                return "paper"
    return "paper"


def _dois_in_text(text: str) -> list[str]:
    from scripts.paper_audit.inventory import normalize_doi

    out = []
    for raw in _DOI_IN_TEXT_RE.findall(text or ""):
        nd = normalize_doi(raw.rstrip(".)"))
        if nd:
            out.append(nd)
    return out


def find_related_paper(dataset_record_json) -> list[str]:
    """Paper DOIs linked from a dataset record (DataCite/Zenodo/Figshare/OpenNeuro)."""
    from scripts.paper_audit.inventory import normalize_doi

    rec = dataset_record_json
    if isinstance(rec, (str, Path)):
        rec = json.loads(Path(rec).read_text(encoding="utf-8"))
    own = normalize_doi(rec.get("doi"))
    found: list[str] = []

    def add(candidate):
        nd = normalize_doi(candidate)
        if nd and nd != own and nd not in found and classify_doi(nd) != "dataset":
            found.append(nd)

    raw = rec.get("raw") or {}
    if rec.get("source") == "datacite":
        for ri in raw.get("relatedIdentifiers") or []:
            if str(ri.get("relatedIdentifierType", "")).upper() != "DOI":
                continue
            if str(ri.get("relationType", "")).lower() in RELATED_PAPER_RELATIONS:
                add(ri.get("relatedIdentifier"))
    elif rec.get("source") == "crossref":
        for rel_name, items in (raw.get("relation") or {}).items():
            if rel_name.lower().replace("-", "") in RELATED_PAPER_RELATIONS:
                for it in items if isinstance(items, list) else []:
                    if str(it.get("id-type", "")).lower() == "doi":
                        add(it.get("id"))

    repo = rec.get("repository") or {}
    zen = (repo.get("record") or {}).get("metadata") or {}
    for ri in zen.get("related_identifiers") or []:
        if str(ri.get("relation", "")).lower() in RELATED_PAPER_RELATIONS:
            add(ri.get("identifier"))
    for ref in zen.get("references") or []:
        text = ref if isinstance(ref, str) else json.dumps(ref)
        for d in _dois_in_text(text):
            add(d)
    desc = repo.get("description") or {}
    for ref in desc.get("ReferencesAndLinks") or []:
        for d in _dois_in_text(str(ref)):
            add(d)
    for ref in repo.get("figshare_references") or []:
        for d in _dois_in_text(str(ref)):
            add(d)
    for d in _dois_in_text(str(desc.get("HowToAcknowledge") or "")):
        add(d)
    return found


# ---------------------------------------------------------------------------
# Paper full text
# ---------------------------------------------------------------------------


def unpaywall_pdf_urls(client: HttpClient, doi: str) -> list[str]:
    status, data = client.get_json(
        UNPAYWALL_API + quote(doi, safe="/()"), params={"email": client.email}
    )
    if status != 200 or not isinstance(data, dict):
        client.mark("unpaywall_miss" if status == 404 else "unpaywall_error")
        return []
    urls: list[str] = []
    locs = []
    if data.get("best_oa_location"):
        locs.append(data["best_oa_location"])
    locs.extend(data.get("oa_locations") or [])
    for loc in locs:
        u = (loc or {}).get("url_for_pdf") or None
        if u and u not in urls:
            urls.append(u)
    client.mark("unpaywall_ok" if urls else "unpaywall_no_pdf", is_oa=data.get("is_oa"))
    return urls


def download_pdf(client: HttpClient, url: str, dest: Path) -> bool:
    """Download ``url`` into ``dest`` only when the payload is a real PDF."""
    try:
        resp = client.get(
            url, headers={"Accept": "application/pdf,*/*;q=0.8"}, timeout=90
        )
    except FetchError as exc:
        log.warning("pdf download failed: %s", exc)
        return False
    body = resp.content or b""
    if resp.status_code != 200:
        client.mark("pdf_http_error")
        return False
    if not body.startswith(b"%PDF"):
        client.mark("not_pdf")
        return False
    if len(body) > MAX_PDF_BYTES:
        client.mark("pdf_too_large")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    client.mark("pdf_ok")
    return True


def pdf_to_text(pdf: Path, txt: Path, pdftotext: str = "pdftotext") -> bool:
    try:
        subprocess.run(
            [pdftotext, "-layout", str(pdf), str(txt)],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("pdftotext failed for %s: %s", pdf, exc)
        return False
    if (
        not txt.exists()
        or len(txt.read_text(encoding="utf-8", errors="replace").strip()) < 500
    ):
        return False
    return True


def europepmc_pmcid(client: HttpClient, doi: str) -> str | None:
    status, data = client.get_json(
        EUROPEPMC_API + "search",
        params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"},
    )
    if status != 200 or not isinstance(data, dict) or "resultList" not in data:
        client.mark("europepmc_error")
        return None
    results = (data.get("resultList") or {}).get("result") or []
    for r in results:
        if r.get("pmcid"):
            client.mark("europepmc_pmcid", pmcid=r["pmcid"])
            return r["pmcid"]
    client.mark("europepmc_no_pmcid")
    return None


def jats_to_text(xml: str) -> str:
    """Strip JATS tags, keeping section titles as Markdown headings."""
    s = re.sub(r"<!--.*?-->", "", xml, flags=re.S)
    s = re.sub(r"<(title)[^>]*>", "\n## ", s)
    s = re.sub(r"</title>", "\n", s)
    s = re.sub(r"</(p|sec|abstract|caption|tr|table-wrap|fig|list-item|ref)>", "\n", s)
    s = re.sub(r"<(td|th)[^>]*>", " | ", s)
    s = re.sub(r"<break\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip() + "\n"


def _html_to_text(page: str) -> str:
    """Very small HTML -> text conversion (scripts/styles dropped)."""
    s = re.sub(
        r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I
    )
    s = re.sub(r"<(h[1-6])[^>]*>", "\n## ", s, flags=re.I)
    s = re.sub(r"</(h[1-6]|p|div|li|tr|section|article|dd|dt)>", "\n", s, flags=re.I)
    s = re.sub(r"<(td|th)[^>]*>", " | ", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip() + "\n"


def europepmc_fulltext(client: HttpClient, pmcid: str) -> str | None:
    resp = client.get(EUROPEPMC_API + f"{pmcid}/fullTextXML")
    if resp.status_code == 404:
        client.mark("europepmc_no_fulltext")
        return None
    if resp.status_code != 200:
        client.mark("europepmc_error")
        return None
    xml = resp.text or ""
    if "<body" not in xml:
        client.mark("europepmc_empty_body")
        return None
    client.mark("europepmc_fulltext_ok")
    return jats_to_text(xml)


def arxiv_pdf(client: HttpClient, arxiv_id: str, dest: Path) -> bool:
    return download_pdf(client, ARXIV_PDF + arxiv_id, dest)


# ---------------------------------------------------------------------------
# Data repositories
# ---------------------------------------------------------------------------


def repository_kind(doi: str, url: str | None = None) -> tuple[str, str] | None:
    """Return ``(kind, identifier)`` for a repository DOI, else ``None``."""
    d = doi.lower()
    u = (url or "").lower()
    m = re.match(r"10\.5281/zenodo\.(\d+)", d)
    if m:
        return "zenodo", m.group(1)
    m = re.match(r"10\.6084/m9\.figshare\.(\d+)", d)
    if m:
        return "figshare", m.group(1)
    m = re.match(r"10\.18112/openneuro\.(ds\d+)\.v([\w.]+)", d)
    if m:
        return "openneuro", f"{m.group(1)}:{m.group(2)}"
    m = re.match(r"10\.17605/osf\.io/(\w+)", d)
    if m:
        return "osf", m.group(1)
    m = re.match(r"10\.7910/dvn/(\w+)", d)
    if m:
        return "dataverse", doi
    m = re.match(r"10\.17632/(\w+)\.(\d+)", d)
    if m:
        return "mendeley", f"{m.group(1)}:{m.group(2)}"
    if "zenodo.org/record" in u:
        m = re.search(r"zenodo\.org/records?/(\d+)", u)
        if m:
            return "zenodo", m.group(1)
    return None


def _is_text_file(name: str, size: int | None) -> bool:
    base = name.rsplit("/", 1)[-1]
    if SIGNAL_EXT_RE.search(base):
        return False
    if size is not None and size > MAX_REPO_FILE_BYTES:
        return False
    return bool(TEXT_FILE_RE.match(base))


def _save_text(client: HttpClient, url: str, dest: Path, headers=None) -> bool:
    try:
        resp = client.get(url, headers=headers, timeout=60)
    except FetchError:
        return False
    if resp.status_code != 200:
        client.mark("repo_file_http_error")
        return False
    body = resp.content or b""
    if len(body) > MAX_REPO_FILE_BYTES:
        client.mark("repo_file_too_large")
        return False
    lowered = body[:200].lower()
    if b"<!doctype html" in lowered or b"<html" in lowered:
        client.mark("repo_file_html")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    client.mark("repo_file_ok")
    return True


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.rsplit("/", 1)[-1])[:120]


def fetch_repository(client: HttpClient, doi: str, record: dict, out_dir: Path) -> dict:
    """Fetch the repository record and small text files for a dataset DOI.

    Returns a dict ``{"kind", "id", "record", "description", "files": [...],
    "figshare_references": [...]}`` that is also stored under
    ``record["repository"]`` by the caller.
    """
    kind_id = repository_kind(doi, record.get("url"))
    info: dict = {
        "kind": None,
        "id": None,
        "record": None,
        "description": None,
        "files": [],
    }
    descs = (
        (record.get("raw") or {}).get("descriptions")
        if record.get("source") == "datacite"
        else None
    )
    if descs:
        text = "\n\n".join(
            str(d.get("description", "")) for d in descs if isinstance(d, dict)
        )
        if text.strip():
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / "datacite_description.txt"
            p.write_text(jats_to_text(text), encoding="utf-8")
            info["files"].append(str(p))
    if not kind_id:
        # Generic public landing page (e.g. PhysioNet, GigaDB, KiltHub): keep the
        # page text as repository evidence; no file listing is attempted.
        url = record.get("url")
        if url and url.startswith("http"):
            info.update(kind="landing-page", id=url)
            try:
                resp = client.get(url, headers={"Accept": "text/html"}, timeout=60)
            except FetchError:
                resp = None
            if resp is not None and resp.status_code == 200 and resp.text.strip():
                client.mark("landing_page_ok")
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / "landing_page.txt"
                p.write_text(_html_to_text(resp.text), encoding="utf-8")
                info["files"].append(str(p))
            elif resp is not None:
                client.mark("landing_page_http_error")
        else:
            info["note"] = "no repository handler for this DOI"
        return info
    kind, ident = kind_id
    info.update(kind=kind, id=ident)
    candidates: list[
        tuple[str, str, int | None, dict | None]
    ] = []  # name,url,size,headers

    if kind == "zenodo":
        status, data = client.get_json(ZENODO_API + ident)
        if status == 200 and isinstance(data, dict):
            client.mark("zenodo_ok")
            info["record"] = {
                "id": data.get("id"),
                "doi": data.get("doi"),
                "conceptdoi": data.get("conceptdoi"),
                "metadata": {
                    k: v
                    for k, v in (data.get("metadata") or {}).items()
                    if k
                    in {
                        "title",
                        "publication_date",
                        "description",
                        "license",
                        "related_identifiers",
                        "references",
                        "creators",
                        "keywords",
                        "resource_type",
                        "version",
                    }
                },
            }
            for f in data.get("files") or []:
                name = f.get("key") or f.get("filename") or ""
                url = (f.get("links") or {}).get("self") or (f.get("links") or {}).get(
                    "download"
                )
                if url:
                    candidates.append((name, url, f.get("size"), None))
            desc = (data.get("metadata") or {}).get("description")
            if desc:
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / "zenodo_description.txt"
                p.write_text(jats_to_text(desc), encoding="utf-8")
                info["files"].append(str(p))
        else:
            client.mark("zenodo_error")
    elif kind == "figshare":
        status, data = client.get_json(FIGSHARE_API + ident)
        if status == 200 and isinstance(data, dict):
            client.mark("figshare_ok")
            info["record"] = {
                k: data.get(k)
                for k in (
                    "id",
                    "title",
                    "doi",
                    "defined_type_name",
                    "description",
                    "license",
                    "published_date",
                    "authors",
                    "references",
                    "categories",
                    "tags",
                )
            }
            info["figshare_references"] = data.get("references") or []
            for f in data.get("files") or []:
                if f.get("download_url"):
                    candidates.append(
                        (f.get("name", ""), f["download_url"], f.get("size"), None)
                    )
            # Figshare descriptions are HTML; keep a text copy as a repo file.
            desc = data.get("description")
            if desc:
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / "figshare_description.txt"
                p.write_text(jats_to_text(desc), encoding="utf-8")
                info["files"].append(str(p))
        else:
            client.mark("figshare_error")
    elif kind == "openneuro":
        ds, tag = ident.split(":", 1)
        query = {
            "query": "query($d:ID!,$t:String!){snapshot(datasetId:$d,tag:$t){"
            "id tag description{Name ReferencesAndLinks HowToAcknowledge License DatasetDOI Authors}"
            " files{filename size urls directory}}}",
            "variables": {"d": ds, "t": tag},
        }
        try:
            resp = client.get(OPENNEURO_GRAPHQL, method="POST", json_body=query)
            data = resp.json() if resp.status_code == 200 else None
        except (FetchError, ValueError):
            data = None
        snap = ((data or {}).get("data") or {}).get("snapshot") if data else None
        if snap:
            client.mark("openneuro_ok")
            info["description"] = snap.get("description") or {}
            info["record"] = {"id": snap.get("id"), "tag": snap.get("tag")}
            for f in snap.get("files") or []:
                if f.get("directory"):
                    continue
                urls = f.get("urls") or []
                url = urls[0] if urls else OPENNEURO_S3 + f"{ds}/{f.get('filename')}"
                candidates.append((f.get("filename", ""), url, f.get("size"), None))
        else:
            client.mark("openneuro_graphql_error")
            for name in (
                "dataset_description.json",
                "README",
                "README.md",
                "participants.tsv",
                "participants.json",
                "CHANGES",
            ):
                candidates.append((name, OPENNEURO_S3 + f"{ds}/{name}", None, None))
    elif kind == "osf":
        status, data = client.get_json(OSF_API + f"{ident}/files/osfstorage/")
        if status == 200 and isinstance(data, dict):
            client.mark("osf_ok")
            info["record"] = {"id": ident, "n_entries": len(data.get("data") or [])}
            for f in data.get("data") or []:
                attrs = f.get("attributes") or {}
                if attrs.get("kind") != "file":
                    continue
                url = (f.get("links") or {}).get("download")
                if url:
                    candidates.append(
                        (attrs.get("name", ""), url, attrs.get("size"), None)
                    )
        else:
            client.mark("osf_error")
    elif kind == "dataverse":
        status, data = client.get_json(
            DATAVERSE_API + "datasets/:persistentId/",
            params={"persistentId": f"doi:{doi}"},
        )
        d = (data or {}).get("data") if isinstance(data, dict) else None
        if status == 200 and d:
            client.mark("dataverse_ok")
            ver = d.get("latestVersion") or {}
            info["record"] = {
                "id": d.get("id"),
                "license": (ver.get("license") or {}).get("name")
                if isinstance(ver.get("license"), dict)
                else ver.get("license"),
                "citation": (ver.get("metadataBlocks") or {})
                .get("citation", {})
                .get("fields"),
            }
            for f in ver.get("files") or []:
                df = f.get("dataFile") or {}
                if df.get("id"):
                    candidates.append(
                        (
                            df.get("filename", ""),
                            DATAVERSE_API + f"access/datafile/{df['id']}",
                            df.get("filesize"),
                            None,
                        )
                    )
        else:
            client.mark("dataverse_error")
    elif kind == "mendeley":
        mid, ver = ident.split(":", 1)
        status, data = client.get_json(
            MENDELEY_API + f"{mid}/files", params={"folder_id": "root", "version": ver}
        )
        if status == 200 and isinstance(data, list):
            client.mark("mendeley_ok")
            info["record"] = {"id": mid, "version": ver, "n_files": len(data)}
            for f in data:
                url = (f.get("content_details") or {}).get("download_url")
                if url:
                    candidates.append((f.get("filename", ""), url, f.get("size"), None))
        else:
            client.mark("mendeley_error")

    n = 0
    for name, url, size, headers in candidates:
        if n >= MAX_REPO_FILES:
            break
        if not _is_text_file(name, size):
            continue
        dest = out_dir / _safe_name(name)
        if _save_text(client, url, dest, headers=headers):
            info["files"].append(str(dest))
            n += 1
            if (
                dest.name == "dataset_description.json"
                and info.get("description") is None
            ):
                try:
                    info["description"] = json.loads(dest.read_text(encoding="utf-8"))
                except ValueError:
                    pass
    return info
