"""Compare declared MOABB metadata against fetched sources.

Produces :class:`EvidenceRow` objects, each carrying a verbatim quote that is
re-verified by whitespace-normalized substring search in the cached source
text. Verdicts:

``match``
    the declared value is supported by at least one source and contradicted
    by none;
``mismatch``
    sources give a value and none of them supports the declared one;
``conflict``
    one source supports the declared value while another contradicts it
    (both quotes are kept: primary in ``quote``, secondary in ``note``);
``unsupported``
    no source contains an extractable value for the field;
``source_missing``
    no source text at all (closed access / missing DOI).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.paper_audit import checks
from scripts.paper_audit.fetch import DEFAULT_CACHE, FetchResult
from scripts.paper_audit.inventory import DatasetRecord, load_inventory


log = logging.getLogger("paper_audit.compare")

VERDICT_ORDER = ("mismatch", "conflict", "unsupported", "source_missing", "match")
COUNTRY_NAMES = {
    "AT": ["Austria"],
    "AU": ["Australia"],
    "BE": ["Belgium"],
    "BR": ["Brazil", "Brasil"],
    "CA": ["Canada"],
    "CH": ["Switzerland"],
    "CN": ["China"],
    "CZ": ["Czech"],
    "DE": ["Germany", "Deutschland"],
    "DK": ["Denmark"],
    "ES": ["Spain", "España"],
    "FI": ["Finland"],
    "FR": ["France"],
    "GB": ["United Kingdom", "UK", "England", "Scotland", "Wales"],
    "GR": ["Greece"],
    "HK": ["Hong Kong"],
    "IN": ["India"],
    "IR": ["Iran"],
    "IT": ["Italy", "Italia"],
    "JP": ["Japan"],
    "KR": ["Korea"],
    "MX": ["Mexico", "México"],
    "NL": ["Netherlands"],
    "NO": ["Norway"],
    "PL": ["Poland"],
    "PT": ["Portugal"],
    "RU": ["Russia"],
    "SE": ["Sweden"],
    "SG": ["Singapore"],
    "TR": ["Turkey", "Türkiye"],
    "TW": ["Taiwan"],
    "US": ["United States", "USA", "U.S.A"],
}


@dataclass
class EvidenceRow:
    field: str
    moabb_value: Any
    source_value: Any
    quote: str
    source_file: str
    locator: str
    verdict: str
    confidence: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


def _norm_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def verify_quote(quote: str, text_path: Path | str) -> bool:
    """True if ``quote`` occurs in the file after whitespace normalization."""
    if not quote or not str(quote).strip():
        return False
    p = Path(text_path)
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8", errors="replace")
    return _norm_ws(quote) in _norm_ws(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sources(fetched: FetchResult, base: Path) -> list[tuple[str, str, str]]:
    """Return [(relative_name, text, kind)] for all readable source files."""
    out = []
    for p in fetched.paper_text_paths:
        p = Path(p)
        if p.exists():
            out.append(
                (_rel(p, base), p.read_text(encoding="utf-8", errors="replace"), "paper")
            )
    for p in fetched.repository_files:
        p = Path(p)
        if p.exists():
            out.append(
                (_rel(p, base), p.read_text(encoding="utf-8", errors="replace"), "repo")
            )
    return out


def _rel(p: Path, base: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p)


def _num_eq(a, b, rel=0.01) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), 1e-9)


def _numbers_in(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_numbers_in(v))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_numbers_in(v))
        return out
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]


def _tokens(s: str) -> set[str]:
    stop = {
        "the",
        "and",
        "reference",
        "referenced",
        "electrode",
        "electrodes",
        "to",
        "at",
        "a",
        "an",
        "of",
    }
    return {
        t
        for t in re.findall(r"[a-z0-9]+", str(s).lower())
        if t not in stop and len(t) > 1
    }


def _aggregate(field, declared, per_source, eq, confidence="high", display=lambda v: v):
    """Turn per-source candidate hits into one EvidenceRow."""
    matched: list[tuple[str, tuple]] = []
    unmatched: list[tuple[str, list]] = []
    all_hits: list[tuple[str, tuple]] = []
    for src, hits in per_source:
        if not hits:
            continue
        for h in hits:
            all_hits.append((src, h))
        ok = [h for h in hits if eq(h[0], declared)]
        if ok:
            matched.append((src, ok[0]))
        else:
            unmatched.append((src, hits))

    if declared is None:
        if not all_hits:
            return None
        src, h = all_hits[0]
        others = sorted({str(display(x[1][0])) for x in all_hits})
        return EvidenceRow(
            field,
            None,
            display(h[0]),
            h[1],
            src,
            h[2],
            "unsupported",
            "low",
            "not declared in MOABB; source candidates: " + ", ".join(others[:8]),
        )
    if not all_hits:
        return EvidenceRow(
            field,
            declared,
            None,
            "",
            "",
            "",
            "unsupported",
            "low",
            "no candidate value found in sources",
        )
    distinct = sorted({str(display(x[1][0])) for x in all_hits})
    if matched and not unmatched:
        src, h = matched[0]
        conf = confidence if len(distinct) == 1 else "medium"
        note = (
            ""
            if len(distinct) == 1
            else "other values mentioned: " + ", ".join(distinct[:8])
        )
        return EvidenceRow(
            field, declared, display(h[0]), h[1], src, h[2], "match", conf, note
        )
    if matched and unmatched:
        src, h = matched[0]
        osrc, ohits = unmatched[0]
        oh = ohits[0]
        note = f'conflict: {osrc}:{oh[2]} says {display(oh[0])!r}: "{oh[1]}"' + (
            "; supported by " + ", ".join(s for s, _ in matched)
        )
        return EvidenceRow(
            field, declared, display(h[0]), h[1], src, h[2], "conflict", "medium", note
        )
    # nothing supports the declared value
    counts: dict[str, int] = {}
    for _, h in all_hits:
        counts[str(display(h[0]))] = counts.get(str(display(h[0])), 0) + 1
    best_val = max(counts, key=counts.get)
    src, h = next(x for x in all_hits if str(display(x[1][0])) == best_val)
    conf = confidence if len(distinct) == 1 else "low"
    note = (
        ""
        if len(distinct) == 1
        else "candidates: " + ", ".join(f"{k} (x{v})" for k, v in counts.items())
    )
    return EvidenceRow(
        field, declared, display(h[0]), h[1], src, h[2], "mismatch", conf, note
    )


# ---------------------------------------------------------------------------
# Field comparisons
# ---------------------------------------------------------------------------


def _run(extractor, sources):
    return [(name, extractor(text)) for name, text, _kind in sources]


def _compare_numeric(field, declared, sources, extractor, accept=None, confidence="high"):
    accepted = [declared] if accept is None else accept
    accepted = [a for a in accepted if a is not None]

    def eq(v, _d):
        return any(_num_eq(v, a) for a in accepted)

    row = _aggregate(field, declared, _run(extractor, sources), eq, confidence)
    return row


def _compare_string_tokens(field, declared, sources, extractor, confidence="medium"):
    def eq(v, d):
        if d is None:
            return False
        tv, td = _tokens(v), _tokens(d)
        return (
            bool(tv & td)
            or str(v).lower() in str(d).lower()
            or str(d).lower() in str(v).lower()
        )

    return _aggregate(field, declared, _run(extractor, sources), eq, confidence)


def _compare_hardware(declared, sources):
    def eq(v, d):
        return d is not None and str(v).lower() in str(d).lower()

    per = _run(checks.find_hardware, sources)
    if declared:
        phrases = _hardware_phrases(declared)
        merged = []
        for (n, hits), (_, text, _kind) in zip(per, sources):
            ph = []
            for phrase in phrases:
                ph.extend(checks.find_phrase(text, phrase))
            merged.append((n, ph + hits))
        per = merged
    return _aggregate("hardware", declared, per, eq, "medium")


_GENERIC_HW = {
    "channel",
    "channels",
    "system",
    "amplifier",
    "cap",
    "eeg",
    "with",
    "and",
    "the",
    "wireless",
    "active",
    "electrodes",
}


def _hardware_phrases(declared: str) -> list[str]:
    """Full string, then comma-separated chunks, then their leading brand words."""
    out = [declared]
    for chunk in re.split(r"[,;/()]+", declared):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk != declared:
            out.append(chunk)
        words = [
            w
            for w in chunk.split()
            if w.lower() not in _GENERIC_HW and not re.match(r"^\d", w)
        ]
        if len(words) >= 2:
            out.append(" ".join(words[:2]))
        elif words and len(words[0]) > 3:
            out.append(words[0])
    return list(dict.fromkeys(out))


def _compare_filters(declared, sources):
    nums = _numbers_in(declared)

    def eq(v, _d):
        lo, hi = v
        return any(_num_eq(lo, n) for n in nums) and any(_num_eq(hi, n) for n in nums)

    return _aggregate(
        "filters",
        declared,
        _run(checks.find_filters, sources),
        eq,
        "low",
        display=lambda v: f"{v[0]}-{v[1]} Hz",
    )


def _compare_license(declared, sources):
    nd = checks.normalize_license(declared)

    def eq(v, _d):
        if nd is None:
            return False
        return v == nd or (v and nd and (v.startswith(nd) or nd.startswith(v)))

    return _aggregate(
        "license",
        nd if declared else None,
        _run(checks.find_license, sources),
        eq,
        "medium",
    )


def _compare_paradigm(declared, sources):
    if not declared:
        return None
    key = str(declared).lower()
    pattern = checks.PARADIGM_KEYWORDS.get(key)
    if pattern is None:
        return EvidenceRow(
            "paradigm",
            declared,
            None,
            "",
            "",
            "",
            "unsupported",
            "low",
            "no keyword rule for this paradigm",
        )
    for name, text, _ in sources:
        hits = checks.find_keyword(text, pattern)
        if hits:
            h = hits[0]
            return EvidenceRow(
                "paradigm",
                declared,
                h[0],
                h[1],
                name,
                h[2],
                "match",
                "medium",
                "keyword presence only",
            )
    # Any other paradigm keyword present?
    for other, pat in checks.PARADIGM_KEYWORDS.items():
        if other == key:
            continue
        for name, text, _ in sources:
            hits = checks.find_keyword(text, pat)
            if hits:
                h = hits[0]
                return EvidenceRow(
                    "paradigm",
                    declared,
                    other,
                    h[1],
                    name,
                    h[2],
                    "mismatch",
                    "low",
                    f"no '{key}' keywords found; '{other}' keywords present",
                )
    return EvidenceRow(
        "paradigm",
        declared,
        None,
        "",
        "",
        "",
        "unsupported",
        "low",
        "no paradigm keywords found",
    )


def _compare_class_labels(declared, sources):
    if not declared:
        return None
    found: dict[str, tuple] = {}
    missing = []
    for label in declared:
        pat = checks.LABEL_SYNONYMS.get(label) or re.escape(str(label).replace("_", " "))
        hit = None
        for name, text, _ in sources:
            hits = checks.find_keyword(text, pat)
            if hits:
                hit = (name, hits[0])
                break
        if hit:
            found[label] = hit
        else:
            missing.append(label)
    if not found:
        return EvidenceRow(
            "class_labels",
            declared,
            None,
            "",
            "",
            "",
            "unsupported",
            "low",
            "no class label mentioned in sources",
        )
    name, h = next(iter(found.values()))
    if missing:
        return EvidenceRow(
            "class_labels",
            declared,
            list(found),
            h[1],
            name,
            h[2],
            "unsupported",
            "low",
            "labels not found in sources: " + ", ".join(map(str, missing)),
        )
    return EvidenceRow(
        "class_labels",
        declared,
        list(found),
        h[1],
        name,
        h[2],
        "match",
        "medium",
        "all labels mentioned",
    )


def _compare_phrase(field, declared, sources, aliases=None):
    if not declared:
        return None
    phrases = [declared, *(aliases or [])]
    for phrase in phrases:
        for name, text, _ in sources:
            hits = checks.find_phrase(text, phrase)
            if hits:
                h = hits[0]
                return EvidenceRow(
                    field,
                    declared,
                    h[0],
                    h[1],
                    name,
                    h[2],
                    "match",
                    "medium",
                    "phrase found",
                )
    return EvidenceRow(
        field,
        declared,
        None,
        "",
        "",
        "",
        "unsupported",
        "low",
        "phrase not found in sources",
    )


def _record_dois(fetched: FetchResult) -> dict:
    if fetched.record_json and Path(fetched.record_json).exists():
        return json.loads(Path(fetched.record_json).read_text(encoding="utf-8")).get(
            "dois", {}
        )
    return {}


def _compare_year(record: DatasetRecord, fetched: FetchResult, base: Path):
    declared = record.declared.get("publication_year")
    if declared is None:
        return None
    dois = _record_dois(fetched)
    src_file = (
        _rel(Path(fetched.record_json), base) if fetched.record_json else "record.json"
    )
    candidates = [(d, info.get("year")) for d, info in dois.items() if info.get("year")]
    if not candidates:
        return EvidenceRow(
            "publication_year",
            declared,
            None,
            "",
            "",
            "",
            "unsupported",
            "low",
            "no DOI record with a year",
        )
    primary = [(d, y) for d, y in candidates if d == record.primary_doi]
    ordered = primary + [c for c in candidates if c not in primary]
    for d, y in ordered:
        if int(y) == int(declared):
            note = (
                ""
                if d == record.primary_doi
                else f"year of related DOI {d} (primary differs: {dict(ordered).get(record.primary_doi)})"
            )
            return EvidenceRow(
                "publication_year",
                declared,
                y,
                f'"year": {y}',
                src_file,
                f"dois.{d}",
                "match",
                "high",
                note,
            )
    d, y = ordered[0]
    return EvidenceRow(
        "publication_year",
        declared,
        y,
        f'"year": {y}',
        src_file,
        f"dois.{d}",
        "mismatch",
        "high",
        f"Crossref/DataCite year of {d}",
    )


def _surname(name: str) -> str:
    parts = re.findall(
        r"[A-Za-zÀ-ÿ'\-]+",
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode(),
    )
    return parts[-1].lower() if parts else ""


def _compare_investigators(record: DatasetRecord, fetched: FetchResult, base: Path):
    declared = record.declared.get("investigators")
    if not declared:
        return None
    dois = _record_dois(fetched)
    src_file = (
        _rel(Path(fetched.record_json), base) if fetched.record_json else "record.json"
    )
    ordered = [record.primary_doi] + [d for d in dois if d != record.primary_doi]
    best = None
    for d in ordered:
        authors = (dois.get(d) or {}).get("authors") or []
        if not authors:
            continue
        rec_surnames = {_surname(a) for a in authors}
        decl_surnames = [_surname(a) for a in declared]
        overlap = [s for s in decl_surnames if s in rec_surnames]
        frac = len(overlap) / max(len(decl_surnames), 1)
        if best is None or frac > best[0]:
            first_match = next((a for a in authors if _surname(a) in overlap), authors[0])
            best = (frac, d, first_match, len(authors))
        if frac >= 0.5:
            break
    if best is None:
        return EvidenceRow(
            "investigators",
            declared,
            None,
            "",
            "",
            "",
            "unsupported",
            "low",
            "no author list in DOI records",
        )
    frac, d, author, n_auth = best
    verdict = "match" if frac >= 0.5 else "mismatch"
    return EvidenceRow(
        "investigators",
        declared,
        f"{n_auth} authors in {d}",
        f'"{author}"',
        src_file,
        f"dois.{d}.authors",
        verdict,
        "medium",
        f"{frac:.0%} of declared surnames found in record authors",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_dataset(
    record: DatasetRecord, fetched: FetchResult, agent_notes: list[dict] | None = None
) -> list[EvidenceRow]:
    base = Path(fetched.record_json).parent if fetched.record_json else Path(".")
    sources = _load_sources(fetched, base)
    d = record.declared
    rows: list[EvidenceRow] = []

    if not sources:
        for field in checks.EXTRACTORS.keys() | {
            "paradigm",
            "class_labels",
            "institution",
            "country",
        }:
            key = field.split(".")[0]
            if d.get(key) is None:
                continue
            rows.append(
                EvidenceRow(
                    field,
                    d.get(key)
                    if key != "channel_types"
                    else (d.get("channel_types") or {}).get("eeg"),
                    None,
                    "",
                    "",
                    "",
                    "source_missing",
                    "low",
                    f"no source text (access={fetched.access})",
                )
            )
        for r in (
            _compare_year(record, fetched, base),
            _compare_investigators(record, fetched, base),
        ):
            if r:
                rows.append(r)
        return _sorted(rows)

    ch = d.get("channel_types") or {}
    eeg = ch.get("eeg") if isinstance(ch, dict) else None
    total = (
        sum(v for v in ch.values() if isinstance(v, (int, float)))
        if isinstance(ch, dict)
        else None
    )
    n_trials = d.get("n_trials")
    trial_accept = None
    if isinstance(n_trials, dict):
        trial_accept = [v for v in n_trials.values() if isinstance(v, (int, float))]
    elif isinstance(n_trials, str):
        trial_accept = _numbers_in(n_trials)

    candidates = [
        _compare_numeric(
            "n_subjects", d.get("n_subjects"), sources, checks.find_subjects
        ),
        _compare_numeric(
            "sampling_rate", d.get("sampling_rate"), sources, checks.find_sampling_rate
        ),
        _compare_numeric(
            "channel_types.eeg",
            eeg,
            sources,
            checks.find_channels,
            accept=[eeg, total],
            confidence="medium",
        ),
        _compare_numeric(
            "sessions_per_subject",
            d.get("sessions_per_subject"),
            sources,
            checks.find_sessions,
            confidence="medium",
        ),
        _compare_numeric(
            "runs_per_session",
            d.get("runs_per_session"),
            sources,
            checks.find_runs,
            confidence="medium",
        ),
        _compare_numeric(
            "n_trials",
            n_trials,
            sources,
            checks.find_trials,
            accept=trial_accept,
            confidence="low",
        ),
        _compare_string_tokens(
            "reference", d.get("reference"), sources, checks.find_reference
        ),
        _compare_string_tokens("ground", d.get("ground"), sources, checks.find_ground),
        _compare_hardware(d.get("hardware"), sources),
        _compare_filters(d.get("filters"), sources),
        _compare_numeric(
            "line_freq",
            d.get("line_freq"),
            sources,
            checks.find_line_freq,
            confidence="medium",
        ),
        _compare_license(d.get("license"), sources),
        _compare_paradigm(d.get("paradigm"), sources),
        _compare_class_labels(d.get("class_labels"), sources),
        _compare_phrase("institution", d.get("institution"), sources),
        _compare_phrase(
            "country",
            d.get("country"),
            sources,
            aliases=COUNTRY_NAMES.get(str(d.get("country") or "").upper()),
        ),
        _compare_year(record, fetched, base),
        _compare_investigators(record, fetched, base),
    ]
    rows.extend(r for r in candidates if r is not None)
    # Rows with declared value but no sources only for undeclared: drop noise.
    rows = [
        r
        for r in rows
        if not (
            r.moabb_value is None
            and r.verdict == "unsupported"
            and r.field
            not in (
                "license",
                "reference",
                "ground",
                "hardware",
                "n_subjects",
                "sampling_rate",
                "channel_types.eeg",
                "sessions_per_subject",
                "runs_per_session",
                "n_trials",
            )
        )
    ]

    # Defensive: every quote must verify against its file.
    checked = []
    for r in rows:
        if r.quote and not verify_quote(r.quote, base / r.source_file):
            log.warning(
                "dropping row with unverifiable quote: %s %s", record.name, r.field
            )
            continue
        checked.append(r)
    rows = checked

    if agent_notes:
        accepted, rejected = apply_agent_notes(agent_notes, base)
        rows.extend(accepted)
        for rj in rejected:
            log.warning("rejected agent note for %s: %s", record.name, rj)
    return _sorted(rows)


def _sorted(rows: list[EvidenceRow]) -> list[EvidenceRow]:
    return sorted(
        rows,
        key=lambda r: (
            VERDICT_ORDER.index(r.verdict) if r.verdict in VERDICT_ORDER else 99,
            r.field,
        ),
    )


# ---------------------------------------------------------------------------
# Agent notes
# ---------------------------------------------------------------------------


def load_agent_notes(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["rows"] if isinstance(data, dict) and "rows" in data else data


def apply_agent_notes(
    notes: list[dict], base_dir: Path
) -> tuple[list[EvidenceRow], list[dict]]:
    """Accept agent-authored rows only when their quote verifies."""
    accepted, rejected = [], []
    for n in notes:
        quote = (n.get("quote") or "").strip()
        src = n.get("source_file") or ""
        if not quote:
            rejected.append({**n, "reason": "empty_quote"})
            continue
        if not src or not verify_quote(quote, Path(base_dir) / src):
            rejected.append({**n, "reason": "quote_not_found"})
            continue
        verdict = n.get("verdict") or "unsupported"
        if verdict not in VERDICT_ORDER:
            rejected.append({**n, "reason": "bad_verdict"})
            continue
        accepted.append(
            EvidenceRow(
                field=n.get("field", "?"),
                moabb_value=n.get("moabb_value"),
                source_value=n.get("source_value"),
                quote=quote,
                source_file=src,
                locator=n.get("locator") or "",
                verdict=verdict,
                confidence=n.get("confidence") or "medium",
                note="agent: " + str(n.get("note") or ""),
            )
        )
    return accepted, rejected


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _md_cell(v) -> str:
    s = (
        json.dumps(v, ensure_ascii=False)
        if isinstance(v, (dict, list))
        else ("" if v is None else str(v))
    )
    return s.replace("|", "\\|").replace("\n", " ")


def write_report(
    rows: list[EvidenceRow],
    out_dir: Path,
    dataset: str = "",
    access: str = "",
    extra: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _sorted(rows)
    counts = {v: sum(1 for r in rows if r.verdict == v) for v in VERDICT_ORDER}
    payload = {
        "dataset": dataset,
        "access": access,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
        **(extra or {}),
        "rows": [r.to_dict() for r in rows],
    }
    evidence = out_dir / "evidence.json"
    evidence.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [f"# Paper audit: {dataset}", ""]
    lines.append(f"- access: `{access}`")
    for k, v in (extra or {}).items():
        lines.append(f"- {k}: {_md_cell(v)}")
    lines.append("- counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines.append("")
    lines.append(
        "| verdict | field | MOABB | source | confidence | quote | source_file:locator | note |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        loc = f"{r.source_file}:{r.locator}" if r.source_file else ""
        lines.append(
            f"| {r.verdict} | {_md_cell(r.field)} | {_md_cell(r.moabb_value)} | {_md_cell(r.source_value)} | {r.confidence} | {_md_cell(r.quote)} | {_md_cell(loc)} | {_md_cell(r.note)} |"
        )
    lines.append("")
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return {"evidence": evidence, "report": report, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare declared metadata with cached sources"
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--out", required=True, type=Path, help="reports root; one dir per dataset"
    )
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument(
        "--agent-notes",
        type=Path,
        default=None,
        help="JSON rows for one dataset (use with a single --classes)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    records = load_inventory(args.inventory)
    if args.classes:
        records = [r for r in records if r.name in set(args.classes)]
    notes = load_agent_notes(args.agent_notes) if args.agent_notes else None
    for rec in records:
        fetched = FetchResult.from_cache(args.cache, rec.name)
        if fetched is None:
            log.warning("no cache for %s; run fetch first", rec.name)
            continue
        rows = compare_dataset(rec, fetched, agent_notes=notes)
        res = write_report(
            rows,
            args.out / rec.name,
            dataset=rec.name,
            access=fetched.access,
            extra={"primary_doi": rec.primary_doi, "notes": fetched.notes},
        )
        log.info("%s: %s", rec.name, res["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
