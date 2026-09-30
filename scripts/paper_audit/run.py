"""End-to-end runner: inventory -> fetch -> compare -> summary.

Example::

    python -m scripts.paper_audit.run --checkout . \\
        --cache ~/Projects/moabb/.paper-audit --out .paper-audit/reports/develop

    # PR worktree, only classes whose module is absent from develop
    python -m scripts.paper_audit.run --checkout .worktrees/mi-batch-1 \\
        --new-vs ~/Projects/moabb/.worktrees/paper-audit --out .../reports/mi-batch-1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from scripts.paper_audit.compare import compare_dataset, load_agent_notes, write_report
from scripts.paper_audit.fetch import (
    DEFAULT_CACHE,
    DEFAULT_DEEP_CEILING,
    DeepCeilingIndex,
    fetch_dataset,
)
from scripts.paper_audit.inventory import build_inventory, write_inventory
from scripts.paper_audit.sources import HttpClient
from scripts.paper_audit.summary import summarize, write_summary


log = logging.getLogger("paper_audit.run")


def _new_modules(records, base_checkout: Path, python: str | None) -> set[str]:
    """Class names whose module is not present in ``base_checkout``."""
    base = build_inventory(base_checkout, python=python)
    base_modules = {r.module for r in base}
    return {r.name for r in records if r.module not in base_modules}


def run(
    checkout: Path,
    out: Path,
    cache: Path = DEFAULT_CACHE,
    classes: list[str] | None = None,
    pr_map: dict | None = None,
    deep_ceiling: Path | None = DEFAULT_DEEP_CEILING,
    offline: bool = False,
    refresh: bool = False,
    new_vs: Path | None = None,
    python: str | None = None,
    agent_notes_dir: Path | None = None,
    title: str | None = None,
) -> dict:
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = build_inventory(checkout, classes=classes, python=python)
    if new_vs is not None:
        keep = _new_modules(records, Path(new_vs), python)
        records = [r for r in records if r.name in keep]
    write_inventory(records, out / "inventory.json")
    log.info("inventory: %d classes from %s", len(records), checkout)

    index = DeepCeilingIndex.from_repo(deep_ceiling)
    client = None if offline else HttpClient()
    stats = {"access": {}, "requests": 0, "datasets": len(records)}
    for i, rec in enumerate(records, 1):
        log.info("[%d/%d] %s", i, len(records), rec.name)
        fetched = fetch_dataset(
            rec,
            cache,
            client=client,
            deep_ceiling=index,
            offline=offline,
            refresh=refresh,
        )
        stats["access"][fetched.access] = stats["access"].get(fetched.access, 0) + 1
        notes = None
        if agent_notes_dir and (Path(agent_notes_dir) / f"{rec.name}.json").exists():
            notes = load_agent_notes(Path(agent_notes_dir) / f"{rec.name}.json")
        is_new = (new_vs is not None) or (
            pr_map is not None and str(pr_map.get(rec.name, "develop")) != "develop"
        )
        rows = compare_dataset(rec, fetched, agent_notes=notes, is_new=is_new)
        write_report(
            rows,
            out / rec.name,
            dataset=rec.name,
            access=fetched.access,
            extra={
                "module": rec.module,
                "primary_doi": rec.primary_doi,
                "paper_dois": fetched.paper_dois,
                "dataset_dois": fetched.dataset_dois,
                "unsupported_doi": fetched.unsupported,
                "notes": fetched.notes,
                "sources": [str(p) for p in fetched.paper_text_paths]
                + [str(p) for p in fetched.repository_files],
                "provenance": str(fetched.provenance),
            },
        )
    if client:
        stats["requests"] = len(client.calls)
        hosts: dict[str, int] = {}
        for c in client.calls:
            host = c["url"].split("/")[2]
            hosts[host] = hosts.get(host, 0) + 1
        stats["requests_by_host"] = hosts
    rows = summarize(out, pr_map)
    res = write_summary(rows, out, title=title or f"Paper audit summary: {checkout}")
    stats["totals"] = res["totals"]
    stats["runtime_s"] = round(time.time() - t0, 1)
    stats["checkout"] = str(checkout)
    stats["summary_csv"] = str(res["csv"])
    (out / "run-stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    log.info("done in %.0fs: %s", stats["runtime_s"], stats["access"])
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the paper-grounded metadata audit")
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument(
        "--pr-map", type=Path, default=None, help="JSON {class: PR number}"
    )
    parser.add_argument("--deep-ceiling", type=Path, default=DEFAULT_DEEP_CEILING)
    parser.add_argument(
        "--new-vs",
        type=Path,
        default=None,
        help="only classes whose module is absent from this checkout",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--refresh", action="store_true", help="re-resolve DOIs already cached"
    )
    parser.add_argument(
        "--python", default=None, help="interpreter for the inventory subprocess"
    )
    parser.add_argument(
        "--agent-notes-dir",
        type=Path,
        default=None,
        help="directory with <Dataset>.json agent rows",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    pr_map = json.loads(args.pr_map.read_text()) if args.pr_map else None
    stats = run(
        args.checkout,
        args.out,
        cache=args.cache,
        classes=args.classes,
        pr_map=pr_map,
        deep_ceiling=args.deep_ceiling,
        offline=args.offline,
        refresh=args.refresh,
        new_vs=args.new_vs,
        python=args.python,
        agent_notes_dir=args.agent_notes_dir,
        title=args.title,
    )
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
