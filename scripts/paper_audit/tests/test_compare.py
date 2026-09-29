"""Tests for deterministic checks, quote verification and reports."""

import json
from pathlib import Path

import pytest

from scripts.paper_audit import checks
from scripts.paper_audit.compare import (
    EvidenceRow,
    apply_agent_notes,
    compare_dataset,
    verify_quote,
    write_report,
)
from scripts.paper_audit.fetch import FetchResult
from scripts.paper_audit.inventory import DatasetRecord


PAPER = """\
## Methods

We recorded EEG from 20 healthy participants (12 female, mean age 24 years).
Signals were sampled at 512 Hz using a BioSemi ActiveTwo system with 64 EEG
channels and referenced to the left mastoid. Data were band-pass filtered
between 0.5 and 100 Hz with a 50 Hz notch. Each subject completed two sessions
on different days; every session comprised 100 trials per session of
left hand and right hand motor imagery, split over six runs.
The dataset is released under a CC BY 4.0 license.
"""

README = """\
# Release notes

Because of recording problems only 74 trials per session are provided.
Sampling rate: 512 Hz. Participants: 20.
"""


def _declared(**over):
    base = {
        "n_subjects": 20,
        "sessions_per_subject": 2,
        "runs_per_session": 6,
        "sampling_rate": 512.0,
        "channel_types": {"eeg": 64},
        "reference": "left mastoid",
        "ground": None,
        "hardware": "BioSemi ActiveTwo",
        "filters": "bandpass 0.5-100 Hz, 50 Hz notch",
        "line_freq": 50.0,
        "interval": [0, 4],
        "n_trials": 100,
        "class_labels": ["left_hand", "right_hand"],
        "paradigm": "imagery",
        "license": "CC-BY-4.0",
        "institution": None,
        "country": None,
        "publication_year": 2020,
        "investigators": ["Ann Author", "Bob Builder"],
    }
    base.update(over)
    return base


def _fixture(tmp_path, declared=None, with_readme=True, with_paper=True):
    ds = tmp_path / "cache" / "FixtureDS"
    ds.mkdir(parents=True)
    paper = ds / "paper-10_1000_fixture.txt"
    papers = []
    if with_paper:
        paper.write_text(PAPER)
        papers.append(paper)
    repo_files = []
    if with_readme:
        rd = ds / "repo" / "10_5281_zenodo_1" / "README.md"
        rd.parent.mkdir(parents=True)
        rd.write_text(README)
        repo_files.append(rd)
    (ds / "record.json").write_text(
        json.dumps(
            {
                "name": "FixtureDS",
                "dois": {
                    "10.1000/fixture": {
                        "kind": "paper",
                        "year": 2020,
                        "authors": ["Ann Author", "Bob Builder", "Cy Person"],
                    }
                },
            },
            indent=2,
        )
    )
    record = DatasetRecord(
        name="FixtureDS",
        module="moabb.datasets.fixture",
        primary_doi="10.1000/fixture",
        declared=_declared(**(declared or {})),
    )
    fetched = FetchResult(
        name="FixtureDS",
        access="open" if with_paper else "missing",
        paper_text_paths=papers,
        repository_files=repo_files,
        record_json=ds / "record.json",
        paper_dois=["10.1000/fixture"],
    )
    return record, fetched, ds


# --- extractors -----------------------------------------------------------


def test_find_subjects_numeric_and_words():
    hits = checks.find_subjects(PAPER)
    assert [h[0] for h in hits] == [20]
    assert "20 healthy participants" in hits[0][1]
    assert checks.find_subjects("Nine subjects took part.")[0][0] == 9
    assert checks.find_subjects("sixty-four participants")[0][0] == 64
    assert checks.find_subjects("N = 32 volunteers")[0][0] == 32
    assert checks.find_subjects("accuracy difference for one subject") == []
    assert checks.find_subjects("recorded from each subject") == []


def test_find_sampling_rate_units():
    assert checks.find_sampling_rate(PAPER)[0][0] == 512.0
    assert checks.find_sampling_rate("sampling frequency of 1 kHz")[0][0] == 1000.0
    assert checks.find_sampling_rate("digitized at 2.048 kHz")[0][0] == 2048.0
    # An unrelated frequency must not be reported as sampling rate.
    assert checks.find_sampling_rate("band-pass filtered at 8-30 Hz") == []


def test_find_channels_sessions_trials_runs():
    assert checks.find_channels(PAPER)[0][0] == 64
    assert checks.find_channels("a 22-channel EEG montage")[0][0] == 22
    assert checks.find_sessions(PAPER)[0][0] == 2
    assert [h[0] for h in checks.find_trials(PAPER)] == [100]
    assert checks.find_trials(README)[0][0] == 74
    assert checks.find_runs(PAPER)[0][0] == 6


def test_find_reference_license_filters_line_freq():
    ref = checks.find_reference(PAPER)
    assert ref and "left mastoid" in ref[0][0]
    lic = checks.find_license(PAPER)
    assert lic and lic[0][0] == "cc-by-4.0"
    flt = checks.find_filters(PAPER)
    assert flt and flt[0][0] == (0.5, 100.0)
    assert checks.find_line_freq(PAPER)[0][0] == 50.0


def test_locator_reports_page_and_line():
    text = "first page\n\x0csecond page line one\nsampled at 256 Hz\n"
    hits = checks.find_sampling_rate(text)
    assert hits[0][2] == "p2:l2"


# --- quote verification ---------------------------------------------------


def test_verify_quote_whitespace_normalized(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("Signals were\n   sampled   at 512 Hz using\na system.")
    assert verify_quote("sampled at 512 Hz using a system", p)
    assert not verify_quote("sampled at 256 Hz", p)
    assert not verify_quote("", p)


# --- compare_dataset --------------------------------------------------------


def _row(rows, field):
    out = [r for r in rows if r.field == field]
    assert out, f"no row for {field}"
    return out[0]


def test_match_and_mismatch_verdicts(tmp_path):
    record, fetched, ds = _fixture(tmp_path, with_readme=False)
    rows = compare_dataset(record, fetched)
    assert all(isinstance(r, EvidenceRow) for r in rows)
    assert _row(rows, "n_subjects").verdict == "match"
    assert _row(rows, "sampling_rate").verdict == "match"
    assert _row(rows, "channel_types.eeg").verdict == "match"
    assert _row(rows, "sessions_per_subject").verdict == "match"
    assert _row(rows, "runs_per_session").verdict == "match"
    assert _row(rows, "reference").verdict == "match"
    assert _row(rows, "hardware").verdict == "match"
    assert _row(rows, "license").verdict == "match"
    assert _row(rows, "filters").verdict == "match"
    assert _row(rows, "line_freq").verdict == "match"
    assert _row(rows, "publication_year").verdict == "match"
    assert _row(rows, "investigators").verdict == "match"
    assert _row(rows, "paradigm").verdict == "match"
    assert _row(rows, "class_labels").verdict == "match"
    # every row with a quote is verifiable against its source file
    for r in rows:
        if r.quote:
            assert verify_quote(r.quote, ds / r.source_file), r

    record.declared["n_subjects"] = 19
    record.declared["sampling_rate"] = 256.0
    rows = compare_dataset(record, fetched)
    r = _row(rows, "n_subjects")
    assert r.verdict == "mismatch"
    assert r.source_value == 20 and "20 healthy participants" in r.quote
    assert r.moabb_value == 19
    assert _row(rows, "sampling_rate").verdict == "mismatch"
    assert _row(rows, "sampling_rate").source_value == 512.0


def test_unsupported_and_source_missing(tmp_path):
    record, fetched, ds = _fixture(tmp_path, with_readme=False)
    record.declared["ground"] = "AFz"
    rows = compare_dataset(record, fetched)
    assert _row(rows, "ground").verdict == "unsupported"
    assert _row(rows, "ground").quote == ""

    record, fetched, ds = _fixture(tmp_path / "b", with_readme=False, with_paper=False)
    rows = compare_dataset(record, fetched)
    text_rows = [r for r in rows if r.field not in ("publication_year", "investigators")]
    assert text_rows and {r.verdict for r in text_rows} == {"source_missing"}
    # DOI-record checks still work without full text
    assert _row(rows, "publication_year").verdict == "match"


def test_conflict_keeps_both_quotes(tmp_path):
    record, fetched, ds = _fixture(tmp_path)
    rows = compare_dataset(record, fetched)
    r = _row(rows, "n_trials")
    assert r.verdict == "conflict"
    joined = r.quote + " || " + r.note
    assert "100 trials per session" in joined
    assert "74 trials per session" in joined
    assert "README.md" in r.note
    # both sources agree on subjects -> plain match
    assert _row(rows, "n_subjects").verdict == "match"


def test_undeclared_field_is_skipped_unless_source_has_value(tmp_path):
    record, fetched, ds = _fixture(tmp_path, with_readme=False)
    record.declared["license"] = None
    rows = compare_dataset(record, fetched)
    r = _row(rows, "license")
    assert r.verdict == "unsupported"
    assert r.moabb_value is None and r.source_value == "cc-by-4.0"
    assert "not declared" in r.note


# --- agent notes ------------------------------------------------------------


def test_unverified_quote_rejected(tmp_path):
    record, fetched, ds = _fixture(tmp_path, with_readme=False)
    notes = [
        {
            "field": "interval",
            "moabb_value": [0, 4],
            "source_value": [0, 3],
            "quote": "every session comprised 100 trials per session",
            "source_file": "paper-10_1000_fixture.txt",
            "locator": "l7",
            "verdict": "mismatch",
            "confidence": "medium",
            "note": "agent",
        },
        {
            "field": "interval",
            "moabb_value": [0, 4],
            "source_value": [1, 5],
            "quote": "this sentence does not exist in the paper",
            "source_file": "paper-10_1000_fixture.txt",
            "verdict": "mismatch",
        },
        {
            "field": "interval",
            "moabb_value": [0, 4],
            "source_value": [1, 5],
            "quote": "",
            "source_file": "paper-10_1000_fixture.txt",
            "verdict": "mismatch",
        },
    ]
    accepted, rejected = apply_agent_notes(notes, ds)
    assert len(accepted) == 1 and accepted[0].source_value == [0, 3]
    assert accepted[0].note.startswith("agent")
    assert len(rejected) == 2
    assert {r["reason"] for r in rejected} == {"quote_not_found", "empty_quote"}


# --- class naming rule ------------------------------------------------------


def test_propose_class_name(tmp_path):
    from scripts.paper_audit.compare import propose_class_name

    # Rule 2: first author "Pérez-Blanco", 2026 -> PerezBlanco2026
    record, fetched, ds = _fixture(tmp_path / "a", with_readme=False)
    (ds / "record.json").write_text(
        json.dumps(
            {
                "name": "FixtureDS",
                "dois": {
                    "10.1000/fixture": {
                        "kind": "paper",
                        "year": 2026,
                        "authors": ["Juan Pérez-Blanco", "Ann Author"],
                    }
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    row = propose_class_name(record, fetched, is_new=True)
    assert row.field == "class_name"
    assert row.source_value == "PerezBlanco2026"
    assert row.verdict == "mismatch"
    assert verify_quote(row.quote, ds / row.source_file)
    record.name = "PerezBlanco2026_MI"
    assert propose_class_name(record, fetched, is_new=True).verdict == "match"

    # Rule 1: README presents an official name
    record, fetched, ds = _fixture(tmp_path / "b", with_readme=True)
    readme = fetched.repository_files[0]
    readme.write_text(
        "# Data\nWe release the MILimbEEG dataset for upper-limb decoding.\n"
    )
    row = propose_class_name(record, fetched, is_new=True)
    assert row.source_value == "MILimbEEG"
    assert row.verdict == "unsupported" and "needs decision" in row.note
    assert verify_quote(row.quote, ds / row.source_file)
    record.name = "MILimbEEG2024"
    assert propose_class_name(record, fetched, is_new=True).verdict == "match"

    # develop class -> no row
    assert propose_class_name(record, fetched, is_new=False) is None
    rows = compare_dataset(record, fetched)
    assert not [r for r in rows if r.field == "class_name"]
    rows = compare_dataset(record, fetched, is_new=True)
    assert [r for r in rows if r.field == "class_name"]


# --- report -----------------------------------------------------------------


def test_write_report_sorted_by_severity(tmp_path):
    record, fetched, ds = _fixture(tmp_path)
    record.declared["sampling_rate"] = 256.0
    rows = compare_dataset(record, fetched)
    out = tmp_path / "out"
    paths = write_report(rows, out, dataset="FixtureDS", access="open")
    evidence = json.loads((out / "evidence.json").read_text())
    assert paths["evidence"] == out / "evidence.json"
    assert {
        "field",
        "moabb_value",
        "source_value",
        "quote",
        "source_file",
        "locator",
        "verdict",
        "confidence",
        "note",
    } <= set(evidence["rows"][0])
    md = (out / "report.md").read_text()
    order = [v for v in ("mismatch", "conflict", "unsupported", "match") if v in md]
    positions = [md.index(f"| {v} |") for v in order if f"| {v} |" in md]
    assert positions == sorted(positions)
    assert md.index("| mismatch |") < md.index("| match |")
    assert "FixtureDS" in md


# --- golden datasets (needs the local cache; skipped otherwise) -------------

GOLDEN = Path(__file__).parent / "golden"
CACHE = Path.home() / "Projects" / "moabb" / ".paper-audit"


@pytest.mark.parametrize("name", ["BNCI2014_001", "Schirrmeister2017", "Ma2022"])
def test_golden_quotes_verify(name):
    golden = GOLDEN / name / "evidence.json"
    if not golden.exists():
        pytest.skip("golden output missing")
    if not (CACHE / name).exists():
        pytest.skip("local paper cache missing")
    data = json.loads(golden.read_text())
    assert data["dataset"] == name
    for row in data["rows"]:
        if row["quote"]:
            assert verify_quote(row["quote"], CACHE / name / row["source_file"]), row
    verdicts = {r["field"]: r["verdict"] for r in data["rows"]}
    expected = {
        # Schirrmeister2017 keeps dataset details in the Supporting Information,
        # which is not part of the main-text Markdown: unsupported, never mismatch.
        "BNCI2014_001": {
            "n_subjects": "match",
            "sampling_rate": "match",
            "reference": "match",
        },
        "Schirrmeister2017": {
            "n_subjects": "unsupported",
            "sampling_rate": "unsupported",
        },
        "Ma2022": {
            "n_subjects": "match",
            "sampling_rate": "match",
            "sessions_per_subject": "match",
        },
    }[name]
    for field, verdict in expected.items():
        assert verdicts[field] == verdict, (field, verdicts[field])
    assert verdicts["publication_year"] == "match"
    assert verdicts["investigators"] == "match"
