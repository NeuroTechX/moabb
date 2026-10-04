"""Aggregate per-dataset evidence into ``audit-summary.csv`` / ``.md``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.paper_audit.compare import VERDICT_ORDER


CSV_COLUMNS = [
    "dataset",
    "pr",
    "access",
    "n_match",
    "n_mismatch",
    "n_unsupported",
    "n_conflict",
    "n_source_missing",
    "manual_doi",
    "primary_doi",
    "report_path",
]


def summarize(
    reports_root: Path | str, pr_map: dict[str, int | str] | None = None
) -> list[dict]:
    """Read every ``<reports_root>/<dataset>/evidence.json`` into summary rows."""
    reports_root = Path(reports_root)
    pr_map = pr_map or {}
    rows = []
    for evidence in sorted(reports_root.glob("*/evidence.json")):
        data = json.loads(evidence.read_text(encoding="utf-8"))
        name = data.get("dataset") or evidence.parent.name
        counts = Counter(r.get("verdict") for r in data.get("rows", []))
        rows.append(
            {
                "dataset": name,
                "pr": pr_map.get(name, "develop"),
                "access": data.get("access", ""),
                "n_match": counts.get("match", 0),
                "n_mismatch": counts.get("mismatch", 0),
                "n_unsupported": counts.get("unsupported", 0),
                "n_conflict": counts.get("conflict", 0),
                "n_source_missing": counts.get("source_missing", 0),
                "manual_doi": bool(data.get("unsupported_doi", False)),
                "primary_doi": data.get("primary_doi") or "",
                "notes": data.get("notes") or [],
                "report_path": str(evidence.parent / "report.md"),
            }
        )
    rows.sort(key=lambda r: (str(r["pr"]), r["dataset"]))
    return rows


def write_summary(
    rows: list[dict], out_dir: Path | str, title: str = "Paper audit summary"
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "audit-summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    access = Counter(r["access"] for r in rows)
    totals = {v: sum(r.get(f"n_{v}", 0) for r in rows) for v in VERDICT_ORDER}
    lines = [f"# {title}", ""]
    lines.append(f"- datasets: {len(rows)}")
    lines.append("- access: " + ", ".join(f"{k}: {v}" for k, v in sorted(access.items())))
    lines.append("- verdict totals: " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    lines.append("")

    manual = [r for r in rows if r["manual_doi"]]
    lines.append("## Needs manual DOI assignment (DOI resolves to no paper)")
    lines.append("")
    if manual:
        for r in manual:
            lines.append(
                f"- {r['dataset']} (pr {r['pr']}): {r['primary_doi']} — "
                + "; ".join(r.get("notes") or [])
            )
    else:
        lines.append("- none")
    lines.append("")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["pr"])].append(r)
    for pr in sorted(groups, key=lambda p: (p != "develop", p)):
        grp = groups[pr]
        lines.append(f"## PR {pr} ({len(grp)} datasets)")
        lines.append("")
        lines.append(
            "| dataset | access | match | mismatch | conflict | unsupported | source_missing | report |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(
            grp, key=lambda x: (-x["n_mismatch"], -x["n_conflict"], x["dataset"])
        ):
            lines.append(
                f"| {r['dataset']} | {r['access']} | {r['n_match']} | {r['n_mismatch']} | {r['n_conflict']} | "
                f"{r['n_unsupported']} | {r['n_source_missing']} | {r['report_path']} |"
            )
        lines.append("")
    md_path = out_dir / "audit-summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "access": dict(access), "totals": totals}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize paper-audit reports")
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pr-map", type=Path, default=None)
    args = parser.parse_args(argv)
    pr_map = json.loads(args.pr_map.read_text()) if args.pr_map else None
    rows = summarize(args.reports, pr_map)
    res = write_summary(rows, args.out or args.reports)
    print(
        json.dumps(
            {"datasets": len(rows), "access": res["access"], "totals": res["totals"]}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
