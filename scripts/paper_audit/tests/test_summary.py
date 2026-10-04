"""Tests for the audit summary."""

import csv
import json

from scripts.paper_audit.summary import summarize, write_summary


def _evidence(out, name, access, verdicts, unsupported_doi=False):
    d = out / name
    d.mkdir(parents=True)
    rows = [
        {
            "field": f"f{i}",
            "moabb_value": 1,
            "source_value": 2,
            "quote": "q" if v != "unsupported" else "",
            "source_file": "paper.txt",
            "locator": "l1",
            "verdict": v,
            "confidence": "high",
            "note": "",
        }
        for i, v in enumerate(verdicts)
    ]
    d.joinpath("evidence.json").write_text(
        json.dumps(
            {
                "dataset": name,
                "access": access,
                "counts": {},
                "unsupported_doi": unsupported_doi,
                "notes": ["needs manual DOI assignment"] if unsupported_doi else [],
                "rows": rows,
            }
        )
    )
    d.joinpath("report.md").write_text("# r\n")


def test_summary_counts(tmp_path):
    reports = tmp_path / "reports"
    _evidence(reports, "DsA", "open", ["match", "match", "mismatch", "conflict"])
    _evidence(
        reports, "DsB", "missing", ["unsupported", "source_missing"], unsupported_doi=True
    )

    rows = summarize(reports, pr_map={"DsA": 1188})
    by = {r["dataset"]: r for r in rows}
    assert by["DsA"]["pr"] == 1188 and by["DsB"]["pr"] == "develop"
    assert by["DsA"]["access"] == "open"
    assert (by["DsA"]["n_match"], by["DsA"]["n_mismatch"], by["DsA"]["n_conflict"]) == (
        2,
        1,
        1,
    )
    assert by["DsA"]["n_unsupported"] == 0
    assert by["DsB"]["n_unsupported"] == 1 and by["DsB"]["n_source_missing"] == 1
    assert by["DsB"]["manual_doi"] is True
    assert by["DsA"]["report_path"].endswith("DsA/report.md")

    paths = write_summary(rows, tmp_path / "out")
    with open(paths["csv"], newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [r["dataset"] for r in csv_rows] == ["DsA", "DsB"]
    assert csv_rows[0]["n_mismatch"] == "1"
    assert {
        "dataset",
        "pr",
        "access",
        "n_match",
        "n_mismatch",
        "n_unsupported",
        "n_conflict",
        "report_path",
    } <= set(csv_rows[0])
    md = paths["md"].read_text()
    assert "| DsA |" in md and "DsB" in md
    assert "manual DOI" in md
    assert "open: 1" in md and "missing: 1" in md
