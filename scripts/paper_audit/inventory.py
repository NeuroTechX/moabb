"""Enumerate MOABB dataset classes and their declared metadata from a checkout.

The checkout is imported in a *subprocess* with ``PYTHONPATH`` pointing at it,
so the inventory reflects the given working tree (e.g. a PR worktree) and never
the package installed in the calling interpreter.

CLI::

    python -m scripts.paper_audit.inventory --checkout PATH --out inventory.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
)
# Same shape as moabb/tests/test_doi_validation.py::_DOI_RE.
_DOI_RE = re.compile(r"^10\.\d{4,}/\S+$")
_DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,}/[^\s\]\">]+")


def normalize_doi(value: str | None) -> str | None:
    """Return a lowercase bare DOI (``10.xxxx/...``) or ``None``."""
    if not value:
        return None
    s = str(value).strip()
    lowered = s.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            s = s[len(prefix) :].strip()
            lowered = s.lower()
            break
    s = s.strip().rstrip(".,;:)")
    if not s or not _DOI_RE.match(s):
        return None
    return s.lower()


def extract_docstring_dois(doc: str | None) -> list[str]:
    """Extract normalized DOIs from a docstring, mirroring test_doi_validation."""
    raw = _DOI_IN_TEXT_RE.findall(doc or "")
    cleaned: list[str] = []
    for d in raw:
        d = d.rstrip(".,;:)")
        d = d.rstrip("`")
        if ">`_" in d:
            d = d[: d.index(">")]
        d = d.rstrip("`_>")
        if d.endswith("/abstract"):
            d = d[: -len("/abstract")]
        nd = normalize_doi(d)
        if nd:
            cleaned.append(nd)
    return list(dict.fromkeys(cleaned))


@dataclass
class DatasetRecord:
    """One dataset class with the metadata fields under audit."""

    name: str
    module: str
    primary_doi: str | None
    paper_dois: list[str] = field(default_factory=list)
    docstring_dois: list[str] = field(default_factory=list)
    data_url: str | None = None
    declared: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @property
    def all_dois(self) -> list[str]:
        out: list[str] = []
        for d in [self.primary_doi, *self.paper_dois, *self.docstring_dois]:
            nd = normalize_doi(d)
            if nd and nd not in out:
                out.append(nd)
        return out


# ---------------------------------------------------------------------------
# Child script executed inside the checkout.  It only relies on duck typing so
# that it works both on real MOABB trees and on minimal test stubs.
# ---------------------------------------------------------------------------
_CHILD_SCRIPT = r"""
import inspect, json, sys, warnings
warnings.filterwarnings("ignore")

import moabb.datasets as db

try:
    from moabb.utils import aliases_list
    deprecated = {a[0] for a in aliases_list}
except Exception:
    deprecated = set()

wanted = set(json.loads(sys.argv[1])) if sys.argv[1] != "null" else None


def g(obj, *path):
    for p in path:
        if obj is None:
            return None
        obj = getattr(obj, p, None)
    return obj


def jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        if isinstance(v, (set, tuple)):
            return [jsonable(x) for x in v]
        if hasattr(v, "__dict__"):
            return {k: jsonable(x) for k, x in vars(v).items()}
        return str(v)


records = []
for name, cls in inspect.getmembers(db, inspect.isclass):
    meta = getattr(cls, "METADATA", None)
    if meta is None or not all(
        hasattr(meta, a) for a in ("acquisition", "participants", "experiment")
    ):
        continue
    if name in deprecated or name in {"FakeDataset", "FakeVirtualRealityDataset"}:
        continue
    if wanted is not None and name not in wanted:
        continue
    doc = getattr(meta, "documentation", None)
    inst = None
    inst_error = None
    try:
        inst = cls()
    except Exception as exc:  # pragma: no cover - dataset specific
        inst_error = f"{type(exc).__name__}: {exc}"

    exp = g(meta, "experiment")
    ds = g(meta, "data_structure")
    n_trials = g(ds, "n_trials")
    if n_trials is None:
        tpc = g(exp, "trials_per_class") or g(ds, "n_trials_per_class")
        n_trials = tpc if tpc else None
    class_labels = g(exp, "class_labels")
    if not class_labels and g(exp, "events"):
        class_labels = list(g(exp, "events").keys())

    declared = {
        "n_subjects": g(meta, "participants", "n_subjects"),
        "sessions_per_subject": g(meta, "sessions_per_subject"),
        "runs_per_session": g(meta, "runs_per_session"),
        "sampling_rate": g(meta, "acquisition", "sampling_rate"),
        "channel_types": g(meta, "acquisition", "channel_types"),
        "reference": g(meta, "acquisition", "reference"),
        "ground": g(meta, "acquisition", "ground"),
        "hardware": g(meta, "acquisition", "hardware"),
        "filters": g(meta, "acquisition", "filters"),
        "line_freq": g(meta, "acquisition", "line_freq"),
        "interval": g(inst, "interval"),
        "n_trials": n_trials,
        "class_labels": class_labels,
        "paradigm": g(exp, "paradigm") or g(inst, "paradigm"),
        "license": g(doc, "license"),
        "institution": g(doc, "institution"),
        "country": g(doc, "country"),
        "publication_year": g(doc, "publication_year"),
        "investigators": g(doc, "investigators"),
    }
    extras = {
        "instance_doi": g(inst, "doi"),
        "instance_error": inst_error,
        "instance_n_subjects": len(g(inst, "subject_list") or []) or None,
        "instance_n_sessions": g(inst, "n_sessions"),
        "instance_paradigm": g(inst, "paradigm"),
        "instance_events": g(inst, "event_id"),
        "n_classes": g(exp, "n_classes"),
        "trial_duration": g(exp, "trial_duration"),
        "task_type": g(exp, "task_type"),
        "sensors_declared": len(g(meta, "acquisition", "sensors") or []) or None,
        "repository": g(doc, "repository"),
        "senior_author": g(doc, "senior_author"),
        "sessions": g(meta, "sessions"),
        "file_format": g(meta, "file_format"),
        "source_file": inspect.getsourcefile(cls),
    }
    records.append(
        {
            "name": name,
            "module": cls.__module__,
            "primary_doi": g(doc, "doi") or g(inst, "doi"),
            "paper_dois": [
                d
                for d in [g(doc, "associated_paper_doi"), *(g(doc, "related_paper_dois") or [])]
                if d
            ],
            "docstring": inspect.getdoc(cls) or "",
            "data_url": g(doc, "data_url") or (g(meta, "external_links") or {}).get("source"),
            "declared": {k: jsonable(v) for k, v in declared.items()},
            "extras": {k: jsonable(v) for k, v in extras.items()},
        }
    )

json.dump(records, sys.stdout)
"""


def build_inventory(
    checkout: Path | str,
    classes: list[str] | None = None,
    python: str | None = None,
    timeout: int = 600,
) -> list[DatasetRecord]:
    """Enumerate dataset classes with ``METADATA`` inside ``checkout``.

    Parameters
    ----------
    checkout : Path
        Root of a MOABB working tree (contains ``moabb/``).
    classes : list of str, optional
        Restrict to these class names.
    python : str, optional
        Interpreter for the child process (defaults to ``sys.executable``).
    """
    checkout = Path(checkout).resolve()
    if not (checkout / "moabb").is_dir():
        raise FileNotFoundError(f"no moabb package under {checkout}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(checkout)
    env.setdefault("MNE_USE_CUDA", "false")
    # A stray cwd import path must not shadow the requested checkout.
    proc = subprocess.run(
        [
            python or sys.executable,
            "-c",
            _CHILD_SCRIPT,
            json.dumps(sorted(classes)) if classes is not None else "null",
        ],
        cwd=str(checkout),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"inventory subprocess failed for {checkout}:\n{proc.stderr[-4000:]}"
        )
    payload = json.loads(proc.stdout)
    records: list[DatasetRecord] = []
    for item in payload:
        docstring = item.pop("docstring", "")
        primary = normalize_doi(item.get("primary_doi"))
        paper = []
        for d in item.get("paper_dois", []):
            nd = normalize_doi(d)
            if nd and nd != primary and nd not in paper:
                paper.append(nd)
        item["primary_doi"] = primary
        item["paper_dois"] = paper
        item["docstring_dois"] = extract_docstring_dois(docstring)
        records.append(DatasetRecord.from_dict(item))
    records.sort(key=lambda r: r.name)
    return records


def write_inventory(records: list[DatasetRecord], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.to_dict() for r in records], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_inventory(path: Path) -> list[DatasetRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetRecord.from_dict(d) for d in data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--python", default=None)
    args = parser.parse_args(argv)
    records = build_inventory(args.checkout, classes=args.classes, python=args.python)
    write_inventory(records, args.out)
    modules = {r.module for r in records}
    print(f"{len(records)} classes in {len(modules)} modules -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
