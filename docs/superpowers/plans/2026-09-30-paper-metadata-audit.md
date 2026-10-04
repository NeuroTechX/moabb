# Paper-grounded metadata audit — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/paper_audit/`, a reproducible tool that fetches every dataset's primary sources, compares MOABB's declared metadata against them with verbatim quotes, and produces per-dataset evidence used by fixing agents.

**Architecture:** Four small standard-library-plus-`requests` modules (inventory → fetch → compare → summary), disk cache keyed by DOI under `~/Projects/moabb/.paper-audit/`, deterministic numeric checks plus agent reading passes whose quotes are re-verified against cached text.

**Tech Stack:** Python 3.11+, `requests`, `pdftotext` (poppler, installed), optional `pymupdf4llm`; Crossref, DataCite, Unpaywall, Europe PMC, Zenodo, Figshare, OpenNeuro, OSF, Dataverse APIs.

**Spec:** `docs/superpowers/specs/2026-09-30-paper-metadata-audit-design.md`

## Global Constraints

- Rate limit ≤1 request/second per host; 3 retries with exponential backoff; contact email in User-Agent (`mailto:b.aristimunha@gmail.com`, already used by the deep-ceiling fetcher).
- Never bypass paywalls; closed papers recorded as `access: closed`.
- Every mismatch row carries a verbatim quote found by substring search in the cached text; otherwise the row is rejected.
- No signal-file downloads: repository text files only, each ≤ 2 MB.
- Reuse `~/Projects/papers/deep-ceiling/docs/paper/refs_md/*.md` when the DOI matches (map via `docs/paper/ref_dois.json`); never modify the deep-ceiling repository.
- Fix commits only on the existing PR branches or `audit/metadata-paper-check`; no rebases, force pushes, or merges.

## Review Focus

1. DOI resolving to a repository record (Zenodo/OpenNeuro/Figshare) with no paper linked — must yield `unsupported` + manual list, never a fabricated paper (Task 2 test `test_dataset_doi_without_paper_is_unsupported`).
2. HTTP 200 with empty/HTML body from Unpaywall/PMC (documented silent failures) — must not be cached as a paper (Task 2 test `test_non_pdf_payload_not_cached`).
3. A paper stating design counts (e.g. 100 trials/session) while the release holds fewer — verdict `conflict`, both quotes kept (Task 3 test `test_conflict_keeps_both_quotes`).
4. Docstring DOIs that differ only in case or prefix from metadata DOIs — normalize before comparison (Task 1 test `test_doi_normalization`).
5. Class present in a PR branch but not in develop — inventory must run against a given checkout path, not the installed package (Task 1 test `test_inventory_uses_checkout_path`).

---

### Task 1: Inventory

**Files:**
- Create: `scripts/paper_audit/__init__.py`, `scripts/paper_audit/inventory.py`
- Test: `scripts/paper_audit/tests/test_inventory.py`

**Interfaces:**
- Produces: `build_inventory(checkout: Path, classes: list[str] | None = None) -> list[DatasetRecord]` where `DatasetRecord` is a dataclass with `name`, `module`, `primary_doi`, `paper_dois: list[str]`, `docstring_dois: list[str]`, `data_url`, `declared: dict` (the audited metadata fields flattened: `n_subjects`, `sessions_per_subject`, `runs_per_session`, `sampling_rate`, `channel_types`, `reference`, `ground`, `hardware`, `filters`, `line_freq`, `interval`, `n_trials`, `class_labels`, `paradigm`, `license`, `institution`, `country`, `publication_year`, `investigators`).
- `normalize_doi(s: str) -> str | None` (lowercase, strip `https://doi.org/`, `doi:`; reuse the regex from `moabb/tests/test_doi_validation.py`).
- CLI: `python -m scripts.paper_audit.inventory --checkout PATH --out inventory.json`.

- [ ] Step 1: Write `test_doi_normalization` (three spellings → one) and `test_inventory_uses_checkout_path` (temp checkout containing a fake dataset module registered in a stub `moabb/datasets/__init__.py` is found; installed package is not imported — assert via `sys.modules` spy).
- [ ] Step 2: Run tests; expect failures.
- [ ] Step 3: Implement using a subprocess with `PYTHONPATH=checkout` that imports `moabb.datasets`, iterates classes with `METADATA`, and prints JSON (keeps the parent interpreter clean).
- [ ] Step 4: Run tests; expect pass. Run on develop: expect 87 records; on `.worktrees/mi-batch-1`: expect 91.
- [ ] Step 5: Commit `feat(audit): dataset metadata inventory`.

### Task 2: Fetch

**Files:**
- Create: `scripts/paper_audit/fetch.py`, `scripts/paper_audit/sources.py`
- Test: `scripts/paper_audit/tests/test_fetch.py` (HTTP mocked with `responses` or a local `requests` adapter)

**Interfaces:**
- Consumes: `DatasetRecord`.
- Produces: `fetch_dataset(record, cache_root: Path, session) -> FetchResult` with `paper_text_paths: list[Path]`, `repository_files: list[Path]`, `record_json: Path`, `access: {"open","closed","missing"}`, `provenance: Path`.
- `classify_doi(doi) -> {"paper","dataset"}` from the Crossref/DataCite `type` and known prefixes (`10.5281`, `10.6084`, `10.18112`, `10.17605`, `10.7910`, `10.17632`, `10.13026`).
- `find_related_paper(dataset_record_json) -> list[str]` from DataCite `relatedIdentifiers` (`IsSupplementTo`, `IsDescribedBy`, `IsReferencedBy`) and Zenodo/Figshare/OpenNeuro `references`/`ReferencesAndLinks`.
- Cache layout: `<cache_root>/<name>/{record.json, provenance.json, paper-<doi-slug>.txt, repo/<file>}`.

- [ ] Step 1: Write `test_dataset_doi_without_paper_is_unsupported`, `test_non_pdf_payload_not_cached`, `test_reuses_deep_ceiling_markdown`, `test_rate_limit_and_retry` (mock 429 then 200).
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement. PDF→text via `subprocess.run(["pdftotext", "-layout", pdf, txt])`; Europe PMC full text via `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML` (strip tags; keep section titles); arXiv via export API when the Crossref record carries an arXiv relation.
- [ ] Step 4: Run tests; expect pass. Dry-run on three datasets with existing Markdown (BNCI2014_001, Schirrmeister2017, Ma2022) and confirm reuse without network.
- [ ] Step 5: Commit `feat(audit): source fetching with provenance`.

### Task 3: Compare

**Files:**
- Create: `scripts/paper_audit/compare.py`, `scripts/paper_audit/checks.py`
- Test: `scripts/paper_audit/tests/test_compare.py`

**Interfaces:**
- Consumes: `DatasetRecord`, `FetchResult`.
- Produces: `compare_dataset(record, fetched) -> list[EvidenceRow]`; `EvidenceRow(field, moabb_value, source_value, quote, source_file, locator, verdict, confidence, note)`; `write_report(rows, out_dir)` writing `evidence.json` and `report.md`.
- `verify_quote(quote: str, text_path: Path) -> bool` (whitespace-normalized substring search).
- `propose_class_name(record, fetched) -> EvidenceRow` implementing the spec's
  Naming rule: `official_name` from the record title/README when it matches
  `\b[A-Z][A-Za-z0-9]{2,}\b` acronym patterns explicitly presented as the
  dataset's name ("the XYZ dataset", "XYZ:"), else `Surname + Year` from the
  Crossref/DataCite first author (accents stripped, CamelCase for compound
  surnames). Only emitted for classes absent from `develop` (use the
  `pr_map.json` produced by the parent, `'develop'` entries skipped).
- Deterministic extractors in `checks.py`: `find_subjects(text)`, `find_sampling_rate(text)`, `find_channels(text)`, `find_sessions(text)`, `find_trials(text)`, `find_reference(text)`, `find_license(text)`; each returns `list[(value, quote, locator)]`.
- Agent reading pass: `compare.py --agent-notes notes.json` accepts rows authored by a reading agent; rows failing `verify_quote` are dropped and logged.

- [ ] Step 1: Write tests: fixture text with "20 healthy participants", "sampled at 512 Hz", "64 EEG channels"; assert match/mismatch verdicts against declared values; `test_conflict_keeps_both_quotes`; `test_unverified_quote_rejected`; `test_propose_class_name` (Crossref first author "Pérez-Blanco" 2026 → `PerezBlanco2026`; README "the MILimbEEG dataset" → official `MILimbEEG`; develop class → no row).
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement extractors with unit-aware regexes (kHz→Hz, "sixty-four"→64 for 1–100) and the report writer (Markdown table sorted by verdict severity).
- [ ] Step 4: Run tests; expect pass. Golden check on the three dry-run datasets; keep the outputs under `scripts/paper_audit/tests/golden/`.
- [ ] Step 5: Commit `feat(audit): evidence comparison and reports`.

### Task 4: Summary and runner

**Files:**
- Create: `scripts/paper_audit/summary.py`, `scripts/paper_audit/run.py`, `scripts/paper_audit/README.md`
- Test: `scripts/paper_audit/tests/test_summary.py`

**Interfaces:**
- `python -m scripts.paper_audit.run --checkout PATH --cache ~/Projects/moabb/.paper-audit --out <dir> [--classes A B]` runs inventory → fetch → compare and writes `audit-summary.csv` (dataset, pr, access, n_match, n_mismatch, n_unsupported, n_conflict, report_path) and `audit-summary.md`.
- `--pr-map pr_map.json` (class → PR number) for grouping; produced by the parent from the 14 refreshed heads.

- [ ] Step 1: Write `test_summary_counts` from two synthetic evidence files.
- [ ] Step 2: Run; expect failure. Implement. Run; expect pass.
- [ ] Step 3: Full run over develop + the 14 PR checkouts (128 datasets) using the cache; record runtime and access counts in README.
- [ ] Step 4: Commit `feat(audit): runner and summary`; open PR `audit/metadata-paper-check` containing only the tool (fixes come in separate commits per branch).

### Task 5: Fixing agents (parallel, one per PR group)

**Files:** existing worktrees under `~/Projects/moabb/.worktrees/`; develop-only datasets in a new worktree for `audit/metadata-paper-check`.

- [ ] Step 1: Parent generates `pr_map.json` and dispatches one agent per group (same grouping as the 2026-09-30 branch refresh plus one develop group of 87 split in three).
- [ ] Step 2: Each agent, per dataset: read `report.md`; for `mismatch` rows apply the fix in metadata/docstring/summary CSV; for `class_name` rows with a confident proposal, rename the class, module, code, registry, docs, tests and CSV rows consistently on the PR branch (never on develop classes); for loader behaviour rows, update synthetic tests and tag `needs_real_data_check`; for `conflict`/`unsupported` rows add a docstring note and list them in the group report.
- [ ] Step 3: Gates per branch: focused tests, `test_metadata.py`, offline `test_doi_validation.py`, pre-commit on touched files, `git diff --check`.
- [ ] Step 4: Commit per dataset (`fix(<Dataset>): align metadata with <doi> (paper audit)`), normal push, record CI run IDs.
- [ ] Step 5: Group report `.worktrees/paper-audit-20260930/group-<name>.md` with a "needs Bruno decision" list.

### Task 6: Consolidation

- [ ] Parent verifies all pushed heads, aggregates group reports, cross-checks that no two groups edited the same shared file inconsistently (`summary_imagery.csv`, `doi_cache.json`), watches exact-head CI, and updates the front brief.
