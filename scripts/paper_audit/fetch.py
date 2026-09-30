"""Fetch primary sources (papers and repository text files) for a dataset.

Cache layout (``cache_root``)::

    _doi/<doi-slug>/record.json      normalized Crossref/DataCite record
    _doi/<doi-slug>/status.json      access + request log for this DOI
    _doi/<doi-slug>/paper.txt        full text when obtained
    _doi/<doi-slug>/paper.pdf        original PDF when downloaded
    _doi/<doi-slug>/repo/<file>      repository text files (<= 2 MB each)
    <dataset>/record.json            dataset-level summary of DOIs/sources
    <dataset>/provenance.json        endpoints, timestamps, hashes
    <dataset>/paper-<doi-slug>.txt   copies of the paper texts
    <dataset>/repo/<doi-slug>/<file> copies of repository files

The per-DOI store is shared by datasets (and checkouts) citing the same DOI, so
each remote document is fetched once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.paper_audit import sources
from scripts.paper_audit.inventory import DatasetRecord, load_inventory, normalize_doi
from scripts.paper_audit.sources import FetchError, HttpClient


log = logging.getLogger("paper_audit.fetch")

DEFAULT_CACHE = Path.home() / "Projects" / "moabb" / ".paper-audit"
DEFAULT_DEEP_CEILING = Path.home() / "Projects" / "papers" / "deep-ceiling"


def doi_slug(doi: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")[:150]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class DeepCeilingIndex:
    """Map DOI -> already converted Markdown from the deep-ceiling paper repo."""

    def __init__(self, mapping: dict[str, Path]):
        self.mapping = mapping

    @classmethod
    def from_repo(cls, repo: Path | str | None) -> "DeepCeilingIndex":
        mapping: dict[str, Path] = {}
        if repo is None:
            return cls(mapping)
        repo = Path(repo)
        ref_dois = repo / "docs" / "paper" / "ref_dois.json"
        refs_md = repo / "docs" / "paper" / "refs_md"
        if ref_dois.exists():
            data = json.loads(ref_dois.read_text(encoding="utf-8"))
            for key, entry in data.items():
                doi = normalize_doi(
                    (entry or {}).get("doi") if isinstance(entry, dict) else entry
                )
                md = refs_md / f"{key}.md"
                if doi and md.exists() and md.stat().st_size > 0:
                    mapping.setdefault(doi, md)
        return cls(mapping)

    def lookup(self, doi: str) -> Path | None:
        nd = normalize_doi(doi)
        return self.mapping.get(nd) if nd else None


@dataclass
class FetchResult:
    name: str
    access: str  # open | closed | missing
    paper_text_paths: list[Path] = field(default_factory=list)
    repository_files: list[Path] = field(default_factory=list)
    record_json: Path | None = None
    provenance: Path | None = None
    paper_dois: list[str] = field(default_factory=list)
    dataset_dois: list[str] = field(default_factory=list)
    unsupported: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "access": self.access,
            "paper_text_paths": [str(p) for p in self.paper_text_paths],
            "repository_files": [str(p) for p in self.repository_files],
            "record_json": str(self.record_json) if self.record_json else None,
            "provenance": str(self.provenance) if self.provenance else None,
            "paper_dois": self.paper_dois,
            "dataset_dois": self.dataset_dois,
            "unsupported": self.unsupported,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FetchResult":
        return cls(
            name=d["name"],
            access=d["access"],
            paper_text_paths=[Path(p) for p in d.get("paper_text_paths", [])],
            repository_files=[Path(p) for p in d.get("repository_files", [])],
            record_json=Path(d["record_json"]) if d.get("record_json") else None,
            provenance=Path(d["provenance"]) if d.get("provenance") else None,
            paper_dois=d.get("paper_dois", []),
            dataset_dois=d.get("dataset_dois", []),
            unsupported=d.get("unsupported", False),
            notes=d.get("notes", []),
        )

    @classmethod
    def from_cache(cls, cache_root: Path | str, name: str) -> "FetchResult | None":
        """Rebuild the result of a previous ``fetch_dataset`` from its cache dir."""
        ds_dir = Path(cache_root) / name
        record_json = ds_dir / "record.json"
        if not record_json.exists():
            return None
        d = json.loads(record_json.read_text(encoding="utf-8"))
        dois = d.get("dois", {})
        return cls(
            name=name,
            access=d.get("access", "missing"),
            paper_text_paths=[Path(p) for p in d.get("paper_text_paths", [])],
            repository_files=[Path(p) for p in d.get("repository_files", [])],
            record_json=record_json,
            provenance=ds_dir / "provenance.json",
            paper_dois=[k for k, v in dois.items() if v.get("kind") == "paper"],
            dataset_dois=[k for k, v in dois.items() if v.get("kind") == "dataset"],
            unsupported=d.get("unsupported", False),
            notes=d.get("notes", []),
        )


# ---------------------------------------------------------------------------
# Per-DOI resolution
# ---------------------------------------------------------------------------


class _DoiStore:
    def __init__(self, cache_root: Path, doi: str):
        self.doi = doi
        self.dir = cache_root / "_doi" / doi_slug(doi)
        self.record_json = self.dir / "record.json"
        self.status_json = self.dir / "status.json"
        self.paper_txt = self.dir / "paper.txt"
        self.paper_pdf = self.dir / "paper.pdf"
        self.repo_dir = self.dir / "repo"

    def load(self) -> tuple[dict | None, dict | None]:
        rec = (
            json.loads(self.record_json.read_text())
            if self.record_json.exists()
            else None
        )
        st = (
            json.loads(self.status_json.read_text())
            if self.status_json.exists()
            else None
        )
        return rec, st


def _resolve_record(client: HttpClient, doi: str) -> tuple[dict, bool]:
    """Crossref first, DataCite second. Returns (record, network_ok)."""
    try:
        cr = sources.crossref_work(client, doi)
    except FetchError as exc:
        log.warning("crossref failed for %s: %s", doi, exc)
        return sources.normalize_record(doi), False
    if cr:
        return sources.normalize_record(doi, crossref=cr), True
    try:
        dc = sources.datacite_doi(client, doi)
    except FetchError as exc:
        log.warning("datacite failed for %s: %s", doi, exc)
        return sources.normalize_record(doi), False
    return sources.normalize_record(doi, datacite=dc), True


def _fetch_paper_text(
    client: HttpClient, doi: str, record: dict, store: _DoiStore
) -> tuple[str | None, list[str]]:
    """Try Unpaywall PDF -> Europe PMC -> arXiv. Returns (source, errors).

    Each step is attempted even when a previous one raised, so a transient
    Europe PMC 500 does not hide an arXiv copy; errors are all recorded.
    """
    errors: list[str] = []
    try:
        for url in sources.unpaywall_pdf_urls(client, doi)[:3]:
            if sources.download_pdf(client, url, store.paper_pdf):
                if sources.pdf_to_text(store.paper_pdf, store.paper_txt):
                    return "unpaywall", errors
                client.mark("pdf_no_text")
                store.paper_txt.unlink(missing_ok=True)
    except FetchError as exc:
        errors.append(str(exc))
        log.warning("unpaywall step failed for %s: %s", doi, exc)
    try:
        pmcid = sources.europepmc_pmcid(client, doi)
        if pmcid:
            text = sources.europepmc_fulltext(client, pmcid)
            if text and text.strip():
                store.paper_txt.write_text(text, encoding="utf-8")
                return "europepmc", errors
    except FetchError as exc:
        errors.append(str(exc))
        log.warning("europepmc step failed for %s: %s", doi, exc)
    try:
        arxiv_id = record.get("arxiv_id")
        if arxiv_id and sources.arxiv_pdf(client, arxiv_id, store.paper_pdf):
            if sources.pdf_to_text(store.paper_pdf, store.paper_txt):
                return "arxiv", errors
            client.mark("pdf_no_text")
            store.paper_txt.unlink(missing_ok=True)
    except FetchError as exc:
        errors.append(str(exc))
        log.warning("arxiv step failed for %s: %s", doi, exc)
    return None, errors


def resolve_doi(
    client: HttpClient | None,
    doi: str,
    cache_root: Path,
    deep_ceiling: DeepCeilingIndex | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> tuple[dict, dict]:
    """Resolve one DOI into the per-DOI store; returns (record, status)."""
    store = _DoiStore(cache_root, doi)
    rec, st = store.load()
    if rec and st and not refresh and st.get("complete"):
        return rec, st
    store.dir.mkdir(parents=True, exist_ok=True)
    call_start = len(client.calls) if client else 0
    network_ok = True
    if rec is None:
        if offline or client is None:
            rec = sources.normalize_record(doi)
            network_ok = False
        else:
            rec, network_ok = _resolve_record(client, doi)
    kind = sources.classify_doi(doi, rec)
    status = {
        "doi": doi,
        "kind": kind,
        "access": "missing",
        "paper_source": None,
        "paper_path": None,
        "repository": None,
        "related_paper_dois": [],
        "errors": [],
        "fetched_at": _now(),
        "complete": False,
    }
    if not network_ok:
        status["errors"].append("metadata lookup unavailable (offline or network error)")

    if kind == "paper":
        md = deep_ceiling.lookup(doi) if deep_ceiling else None
        if md is not None:
            shutil.copyfile(md, store.paper_txt)
            status.update(
                access="open",
                paper_source="deep-ceiling",
                paper_origin=str(md),
                paper_path=str(store.paper_txt),
            )
        elif store.paper_txt.exists() and st and st.get("paper_source"):
            status.update(
                access="open",
                paper_source=st["paper_source"],
                paper_path=str(store.paper_txt),
            )
        elif offline or client is None:
            status["access"] = "missing"
        else:
            src, errs = _fetch_paper_text(client, doi, rec, store)
            status["errors"].extend(errs)
            if src:
                status.update(
                    access="open", paper_source=src, paper_path=str(store.paper_txt)
                )
            else:
                # All lookups were consulted; a transient error only marks the
                # DOI incomplete (retried next run); the paper is closed for now.
                status["access"] = "closed"
    else:  # dataset DOI
        if offline or client is None:
            info = {"kind": None, "files": [], "note": "offline"}
        else:
            try:
                info = sources.fetch_repository(client, doi, rec, store.repo_dir)
            except FetchError as exc:
                info = {"kind": None, "files": [], "note": str(exc)}
                status["errors"].append(str(exc))
        rec["repository"] = {k: v for k, v in info.items() if k != "files"}
        status["repository"] = {"kind": info.get("kind"), "files": info.get("files", [])}
        status["related_paper_dois"] = sources.find_related_paper(rec)
        status["access"] = "n/a"

    status["requests"] = list(client.calls[call_start:]) if client else []
    status["complete"] = network_ok and not status["errors"]
    if status["paper_path"]:
        status["paper_sha256"] = _sha256(Path(status["paper_path"]))
    _write_json(store.record_json, rec)
    _write_json(store.status_json, status)
    return rec, status


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------


def fetch_dataset(
    record: DatasetRecord,
    cache_root: Path | str,
    session=None,
    client: HttpClient | None = None,
    deep_ceiling: DeepCeilingIndex | None = None,
    offline: bool = False,
    refresh: bool = False,
    email: str = sources.DEFAULT_EMAIL,
) -> FetchResult:
    """Fetch all sources for one dataset and write its cache directory."""
    cache_root = Path(cache_root)
    if client is None and not offline:
        client = HttpClient(session=session, email=email)
    ds_dir = cache_root / record.name
    ds_dir.mkdir(parents=True, exist_ok=True)

    queue = list(record.all_dois)
    seen: set[str] = set()
    result = FetchResult(name=record.name, access="missing")
    doi_summaries: dict[str, dict] = {}
    prov_requests: list[dict] = []
    prov_papers: list[dict] = []
    prov_repos: list[dict] = []
    network_errors = 0

    while queue:
        doi = queue.pop(0)
        if doi in seen:
            continue
        seen.add(doi)
        rec, st = resolve_doi(
            client,
            doi,
            cache_root,
            deep_ceiling=deep_ceiling,
            offline=offline,
            refresh=refresh,
        )
        doi_summaries[doi] = {
            "kind": st["kind"],
            "access": st["access"],
            "title": rec.get("title"),
            "year": rec.get("year"),
            "authors": rec.get("authors"),
            "type": rec.get("type"),
            "source": rec.get("source"),
            "paper_source": st.get("paper_source"),
            "errors": st.get("errors"),
        }
        prov_requests.extend(st.get("requests") or [])
        if st["errors"]:
            network_errors += 1
        store = _DoiStore(cache_root, doi)
        if st["kind"] == "paper":
            result.paper_dois.append(doi)
            if st["access"] == "open" and store.paper_txt.exists():
                dest = ds_dir / f"paper-{doi_slug(doi)}.txt"
                shutil.copyfile(store.paper_txt, dest)
                result.paper_text_paths.append(dest)
                prov_papers.append(
                    {
                        "doi": doi,
                        "source": st.get("paper_source"),
                        "origin": st.get("paper_origin"),
                        "path": str(dest),
                        "sha256": _sha256(dest),
                    }
                )
        else:
            result.dataset_dois.append(doi)
            files = []
            for f in (st.get("repository") or {}).get("files", []):
                src = Path(f)
                if not src.exists():
                    continue
                dest = ds_dir / "repo" / doi_slug(doi) / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                result.repository_files.append(dest)
                files.append({"path": str(dest), "sha256": _sha256(dest)})
            prov_repos.append(
                {
                    "doi": doi,
                    "kind": (st.get("repository") or {}).get("kind"),
                    "files": files,
                    "related_paper_dois": st.get("related_paper_dois", []),
                }
            )
            for rd in st.get("related_paper_dois", []):
                if rd not in seen and rd not in queue:
                    queue.append(rd)

    # Access verdict for the dataset as a whole.
    if result.paper_text_paths:
        result.access = "open"
    elif result.paper_dois:
        closed = [d for d in result.paper_dois if doi_summaries[d]["access"] == "closed"]
        if closed:
            result.access = "closed"
            result.notes.append(
                "no open-access full text found for: "
                + ", ".join(closed)
                + " (not bypassed)"
            )
        else:
            result.access = "missing"
            result.notes.append(
                "paper lookups failed (offline or network error); rerun later"
            )
    else:
        result.access = "missing"
        if result.dataset_dois:
            result.unsupported = True
            result.notes.append(
                "DOI(s) resolve to a data repository with no linked paper; "
                "needs manual DOI assignment: " + ", ".join(result.dataset_dois)
            )
        else:
            result.unsupported = True
            result.notes.append("no DOI declared; needs manual DOI assignment")

    record_json = ds_dir / "record.json"
    _write_json(
        record_json,
        {
            "name": record.name,
            "module": record.module,
            "primary_doi": record.primary_doi,
            "dois": doi_summaries,
            "access": result.access,
            "unsupported": result.unsupported,
            "notes": result.notes,
            "paper_text_paths": [str(p) for p in result.paper_text_paths],
            "repository_files": [str(p) for p in result.repository_files],
        },
    )
    provenance = ds_dir / "provenance.json"
    _write_json(
        provenance,
        {
            "dataset": record.name,
            "generated": _now(),
            "offline": offline,
            "user_agent": client.user_agent if client else None,
            "requests": prov_requests,
            "papers": prov_papers,
            "repositories": prov_repos,
        },
    )
    result.record_json = record_json
    result.provenance = provenance
    return result


def fetch_inventory(
    records: list[DatasetRecord],
    cache_root: Path,
    deep_ceiling: DeepCeilingIndex | None = None,
    offline: bool = False,
    client: HttpClient | None = None,
    refresh: bool = False,
    progress=None,
) -> dict[str, FetchResult]:
    results: dict[str, FetchResult] = {}
    for i, rec in enumerate(records, 1):
        if progress:
            progress(i, len(records), rec.name)
        results[rec.name] = fetch_dataset(
            rec,
            cache_root,
            client=client,
            deep_ceiling=deep_ceiling,
            offline=offline,
            refresh=refresh,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch primary sources for an inventory")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--deep-ceiling", type=Path, default=DEFAULT_DEEP_CEILING)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="fetch-results.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    records = load_inventory(args.inventory)
    if args.classes:
        records = [r for r in records if r.name in set(args.classes)]
    index = DeepCeilingIndex.from_repo(args.deep_ceiling)
    log.info("deep-ceiling index: %d DOIs", len(index.mapping))
    client = None if args.offline else HttpClient()
    results = fetch_inventory(
        records,
        args.cache,
        deep_ceiling=index,
        offline=args.offline,
        client=client,
        refresh=args.refresh,
        progress=lambda i, n, name: log.info("[%d/%d] %s", i, n, name),
    )
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r.access] = counts.get(r.access, 0) + 1
    print(json.dumps(counts))
    if args.out:
        _write_json(args.out, {k: v.to_dict() for k, v in results.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
