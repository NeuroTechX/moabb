# Paper-grounded metadata audit

Reproducible tool that fetches every dataset's primary sources (papers and
repository text files), compares MOABB's declared `METADATA` against them with
verbatim quotes, and writes per-dataset evidence for fixing agents.

Design: `docs/superpowers/specs/2026-09-30-paper-metadata-audit-design.md`.
Plan: `docs/superpowers/plans/2026-09-30-paper-metadata-audit.md`.

## Pipeline

| step | module | input → output |
|---|---|---|
| 1 | `inventory.py` | checkout path → `inventory.json` (one record per dataset **class** with `METADATA`; imported in a subprocess with `PYTHONPATH=<checkout>`, so PR worktrees are audited without installing them) |
| 2 | `fetch.py` / `sources.py` | DOIs → cached records, paper full text, repository text files, `provenance.json` |
| 3 | `compare.py` / `checks.py` | declared values + cached text → `evidence.json`, `report.md` |
| 4 | `summary.py` / `run.py` | all reports → `audit-summary.csv`, `audit-summary.md`, `run-stats.json` |

```bash
PY=/path/to/python   # needs requests + the MOABB dependencies of the audited checkout
export PYTHONPATH=$PWD

# whole checkout
$PY -m scripts.paper_audit.run --checkout . --cache ~/Projects/moabb/.paper-audit \
    --out ~/Projects/moabb/.paper-audit/reports/develop

# PR worktree, only classes whose module does not exist on develop
$PY -m scripts.paper_audit.run --checkout ../mi-batch-1 --new-vs . \
    --out ~/Projects/moabb/.paper-audit/reports/mi-batch-1 --pr-map pr_map.json

# individual steps
$PY -m scripts.paper_audit.inventory --checkout . --out inventory.json
$PY -m scripts.paper_audit.fetch --inventory inventory.json --classes BNCI2014_001 [--offline]
$PY -m scripts.paper_audit.compare --inventory inventory.json --out reports [--agent-notes notes.json]
$PY -m scripts.paper_audit.summary --reports reports --pr-map pr_map.json
```

Tests: `$PY -m pytest scripts/paper_audit/tests -q` (HTTP is mocked with a
`requests.adapters.BaseAdapter`; the golden tests skip when the local cache
is absent).

## Sources and endpoints

All requests go through `sources.HttpClient`: ≤1 request/second per host,
3 retries with exponential backoff honouring `Retry-After`, User-Agent
`moabb-paper-audit/0.1 (...; mailto:b.aristimunha@gmail.com)`. Every request
(URL, params, status, outcome, timestamp, bytes) is logged in the per-dataset
`provenance.json`.

| endpoint | use | silent-failure handling |
|---|---|---|
| `https://api.crossref.org/works/{doi}?mailto=` | record (type, title, authors, year, arXiv relation) | 404 → DataCite |
| `https://api.datacite.org/dois/{doi}` | record for repository / arXiv DOIs; `relatedIdentifiers`, `descriptions` | |
| `https://api.unpaywall.org/v2/{doi}?email=` | best OA PDF URL(s) | payload accepted only when it starts with `%PDF` (`not_pdf` otherwise); 60 MB cap |
| `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:"…"` | PMCID | 200 with `errCode` / no `resultList` → `europepmc_error` |
| `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML` | JATS full text | 404 = not OA; 200 without `<body>` → `europepmc_empty_body`, not cached |
| `https://arxiv.org/pdf/{id}` | preprint PDF when the DOI is `10.48550/arXiv.*` or Crossref lists an arXiv relation | PDF magic check |
| `https://zenodo.org/api/records/{id}` | record, `related_identifiers`, `references`, description, text files | |
| `https://api.figshare.com/v2/articles/{id}` | record, `references`, description, text files | only the `/articles/{id}` endpoint (listing endpoints are not searches) |
| `https://openneuro.org/crn/graphql` (`snapshot` query) / `s3.amazonaws.com/openneuro.org` | `dataset_description.json` (`ReferencesAndLinks`), README, participants.* | S3 fallback when GraphQL fails |
| `https://api.osf.io/v2/nodes/{id}/files/osfstorage/` | root text files | |
| `https://dataverse.harvard.edu/api/datasets/:persistentId/` + `access/datafile/{id}` | files | |
| `https://data.mendeley.com/public-api/datasets/{id}/files` | files | |
| DataCite `url` (e.g. PhysioNet, GigaDB, KiltHub) | public landing page kept as `landing_page.txt` | HTML → text |

Constraints: never bypass paywalls (closed papers → `access: closed`);
repository files are text only (`README*`, `*.md/.txt/.json/.tsv/.csv/.bib`,
≤ 2 MB, max 12 per record); signal files are never downloaded. Papers already
converted for the deep-ceiling manuscript
(`~/Projects/papers/deep-ceiling/docs/paper/refs_md`, mapped through
`ref_dois.json`) are reused read-only and recorded as `paper_source:
deep-ceiling`.

PDF → text uses `pdftotext -layout` (poppler); JATS is stripped keeping
section titles as `## ` headings.

## Cache layout (`~/Projects/moabb/.paper-audit`)

```
_doi/<doi-slug>/{record.json,status.json,paper.txt,paper.pdf,repo/*}   shared per DOI
<Dataset>/{record.json,provenance.json,paper-<doi-slug>.txt,repo/<doi-slug>/*}
reports/<checkout>/{inventory.json,audit-summary.csv,audit-summary.md,run-stats.json,<Dataset>/{evidence.json,report.md}}
```

Per-DOI stores are reused across datasets and checkouts; a DOI whose lookup
hit a network error (`complete: false`) is retried on the next run,
`--refresh` re-resolves everything.

## Verdicts

Every row has `field, moabb_value, source_value, quote, source_file,
locator, verdict, confidence, note`. Quotes are verbatim excerpts and are
re-verified by whitespace-normalised substring search before a row is kept
(`compare.verify_quote`); agent-authored rows (`--agent-notes`) that fail
this check are dropped and logged.

* `match` – declared value supported by a source and contradicted by none;
* `mismatch` – sources give value(s) and none supports the declared one;
* `conflict` – one source supports the value and another contradicts it
  (both quotes kept: primary in `quote`, secondary in `note`);
* `unsupported` – nothing extractable for the field (also used when MOABB
  declares `None` but a source has a candidate: see `note`);
* `source_missing` – no source text at all (closed access, missing DOI).

Deterministic extractors (`checks.py`): subjects, sampling rate (kHz→Hz),
channels, sessions, runs, trials, reference, ground, hardware brands,
band-pass filters, line frequency, licence; keyword rules for paradigm and
class labels; phrase search for institution/country; DOI-record checks for
`publication_year` and `investigators` (surname overlap ≥ 50 %). Fields such
as `interval`, per-class trial counts or unit conversions need the agent
reading pass (`--agent-notes`).

Datasets whose DOI(s) resolve only to a data repository with no linked paper
are flagged `unsupported_doi` and listed under "Needs manual DOI assignment"
in `audit-summary.md`; no paper is ever guessed.

## Full run 2026-09-30

Filled in by `run.py` (`run-stats.json` per checkout); see the section at the
end of this file.
