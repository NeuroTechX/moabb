# Paper-grounded dataset metadata audit — design

Date: 2026-09-30. Requested by Bruno Aristimunha. Auto-approved execution with
parallel agents; no PR merges, NEMAR publication, or laptop signal downloads.

## Intent

For every MOABB dataset class on `develop` (87 modules with `METADATA`) and the
41 modules added by the 14 open PRs (#1178, #1186, #1188–#1199), obtain the
primary source documents, compare MOABB's declared metadata and docstrings
against them, and fix confirmed discrepancies with quoted evidence.

Success: a reproducible per-dataset evidence report, and branches whose
metadata either matches the sources or explicitly documents why it cannot.

## What already exists (reused, not rebuilt)

- `~/Projects/papers/deep-ceiling/scripts/fetch_refs_markdown.py` and
  `docs/paper/ref_dois.json` (105 DOIs): Unpaywall → PDF → Markdown via
  pymupdf4llm. 62 papers already converted (`docs/paper/refs_md/`).
- `docs/paper/refs_download_report.csv`: 35 `no_oa` (mostly dataset DOIs where
  Unpaywall is the wrong tool), 7 `not_pdf`, 1 download failure.
- `analysis/dataset_metadata_audit_20260929/release_sealed/`: 99-dataset
  catalog with measured cache/header evidence and named conflicts. Its protocol
  states it is *not* primary-publication revalidation — that is this task.
- `scripts/dataset_search/resolver.py`: repository/accession detection.
- `~/.agents/skills/paper-lookup/`: reference notes for Crossref, Unpaywall,
  Europe PMC, arXiv, Zenodo, Figshare APIs, including their silent-failure modes.
- MOABB: `moabb/tests/doi_cache.json` (157 DOI → title/authors/year) and
  `moabb/tests/test_doi_validation.py`.

## Architecture

New MOABB tooling under `scripts/paper_audit/` (tracked, reusable):

1. `inventory.py` — enumerates dataset classes from a checkout (import by
   PYTHONPATH), emits `inventory.json`: class, module, primary DOI,
   `associated_paper_doi`, `related_paper_dois`, docstring DOIs, `data_url`,
   and the declared metadata fields under audit.
2. `fetch.py` — for each DOI, keyed by DOI, rate-limited (≤1 req/s per host),
   cached under `~/Projects/moabb/.paper-audit/<class>/`:
   - Crossref/DataCite record → `record.json` (title, authors, year, venue, type).
   - If the DOI is a *dataset* DOI (Zenodo, Figshare, OpenNeuro, OSF, Dataverse,
     Mendeley, PhysioNet): fetch the repository record and its small text files
     (README, dataset_description.json, participants.tsv/json, task-*_eeg.json,
     channels/electrodes TSV when ≤2 MB). Locate the associated paper via the
     record's `relatedIdentifiers` / `references`; fetch that paper too.
   - If the DOI is a *paper* DOI: Unpaywall best OA location → PDF; fall back to
     Europe PMC full text (JATS) and arXiv. Convert to text with `pdftotext`
     (installed) or pymupdf4llm when available. Reuse existing deep-ceiling
     Markdown when the DOI matches. Paywalled → `access: closed`, recorded, not
     bypassed.
   - `provenance.json` per dataset: endpoints, parameters, timestamps, hashes.
3. `compare.py` — builds `evidence.json` + `report.md` per dataset. Each row:
   `field`, `moabb_value`, `source_value`, `quote`, `source_file`, `locator`
   (page/section/line), `verdict` ∈ {match, mismatch, unsupported,
   source_missing}, `confidence`. Audited fields: n_subjects, sessions_per_subject,
   runs, channel counts/types, sampling_rate, reference/ground, hardware,
   filters, line_freq, trial/epoch window, n_trials & class labels, paradigm
   classification (imagery vs execution/observation), license, DOI/title/
   authors/year consistency (with `doi_cache.json`), institution/country.
   The comparison is machine-assisted: deterministic regexes/table parsing for
   numeric fields, plus an LLM-agent reading pass **that must produce a verbatim
   quote for every mismatch**; quotes are re-verified by substring search in
   the cached text before the row is accepted.
4. `summary.py` — aggregates all datasets into `audit-summary.csv/md`
   (per-PR grouping, counts by verdict, unresolved decisions).

## Fixing workflow

- Agents are grouped by PR branch using the existing worktrees under
  `~/Projects/moabb/.worktrees/`; develop-only datasets get a dedicated branch
  `audit/metadata-paper-check` from `origin/develop` (new PR at the end).
- A change is allowed only when the report row has `verdict=mismatch` with a
  verified quote. No edits from memory. Loader *behaviour* changes (units,
  windows, labels) additionally require the existing synthetic regression
  tests to be updated, and are flagged `needs_real_data_check` for Voyager.
- Paper-vs-released-data conflicts (e.g. paper says 100 trials, files hold 74)
  are documented in the docstring/metadata notes, never silently resolved.
- Each agent runs: focused tests + `test_metadata.py` + `test_doi_validation.py`
  offline subset + pre-commit on touched files; commits on the branch with the
  evidence path in the message; normal push. No rebases/force pushes/merges.
- Output per group: `.worktrees/paper-audit-20260930/group-<name>.md` with
  changed fields, evidence, tests, CI run IDs, and a "needs Bruno decision" list.

## Error handling

- Network failure or rate limit: retry with backoff (3×), then mark
  `source_missing` with the error; never fabricate.
- Ambiguous paper identity (DOI resolves to a repository, not a paper, and no
  related paper found): `unsupported`, listed for manual DOI assignment.
- Conflicting sources (paper vs repository README): both quotes recorded;
  verdict `conflict`, routed to the decision list.

## Testing

- Unit tests for `fetch.py` (mocked HTTP: Crossref, Unpaywall, Zenodo, OpenNeuro
  shapes incl. the known 200-with-empty-body failure modes) and `compare.py`
  (fixture text with known values; quote verification rejects unquoted rows).
- Golden test on three datasets with already-converted papers (BNCI2014_001,
  Schirrmeister2017, Ma2022) asserting expected verdict rows.

## Out of scope

Signal downloads, benchmark reruns, NEMAR publication, PR merges, rewriting
git history, and any change to the deep-ceiling manuscript or sealed release.
