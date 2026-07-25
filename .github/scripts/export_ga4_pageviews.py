#!/usr/bin/env python3
"""Export GA4 dataset page view counts to a static JSON snapshot.

This script queries GA4 Data API for dataset page paths and writes a compact
JSON payload consumed by the Sphinx dataset card extension.

Environment variables:
    GA4_PROPERTY_ID               GA4 property id (numeric string)
    GA4_SERVICE_ACCOUNT_JSON      Service account JSON payload (string)
    GA4_SERVICE_ACCOUNT_FILE      Optional service account JSON file path
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


DATASET_PATH_PATTERNS = [
    re.compile(r"^/docs/generated/moabb\.datasets\.([A-Za-z0-9_]+)\.html/?$"),
    re.compile(r"^/generated/moabb\.datasets\.([A-Za-z0-9_]+)\.html/?$"),
    re.compile(r"^/docs/generated/moabb\.datasets\.([A-Za-z0-9_]+)/?$"),
    re.compile(r"^/generated/moabb\.datasets\.([A-Za-z0-9_]+)/?$"),
]


def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset name for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _build_canonical_name_map() -> dict[str, str]:
    """Map normalized dataset names to canonical class names."""
    mapping: dict[str, str] = {}
    try:
        from moabb.datasets.utils import dataset_list

        for ds_cls in dataset_list:
            name = ds_cls.__name__
            mapping[_normalize_dataset_name(name)] = name
    except Exception:
        # Keep empty map; we still export raw names as fallback.
        pass
    return mapping


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_snapshot(
    output_path: Path,
    *,
    property_id: str | None,
    counts: dict[str, dict[str, int]] | None = None,
    status: str,
    reason: str,
) -> None:
    payload: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "source": "ga4-data-api",
        "property_id": property_id or "",
        "status": status,
        "reason": reason,
        "counts": counts or {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_dataset_class(page_path: str) -> str | None:
    clean = page_path.split("?", 1)[0].split("#", 1)[0]
    for pattern in DATASET_PATH_PATTERNS:
        match = pattern.match(clean)
        if match:
            return match.group(1)
    return None


def _canonical_dataset_name(name: str, canonical_map: dict[str, str]) -> str:
    """Return canonical dataset name when an alias is detected."""
    key = _normalize_dataset_name(name)
    return canonical_map.get(key, name)


def _merge_counts_by_canonical(
    counts: dict[str, int], canonical_map: dict[str, str]
) -> dict[str, int]:
    """Merge duplicate aliases into canonical dataset names."""
    merged: dict[str, int] = {}
    for name, value in counts.items():
        canonical = _canonical_dataset_name(name, canonical_map)
        merged[canonical] = merged.get(canonical, 0) + int(value)
    return merged


def _run_report(
    *,
    property_id: str,
    access_token: str,
    start_date: str,
    end_date: str,
    timeout_seconds: float,
) -> dict[str, int]:
    import requests

    url = (
        f"https://analyticsdata.googleapis.com/v1beta/properties/{property_id}:runReport"
    )
    payload = {
        "dateRanges": [{"startDate": start_date, "endDate": end_date}],
        "dimensions": [{"name": "pagePath"}],
        "metrics": [{"name": "screenPageViews"}],
        "dimensionFilter": {
            "orGroup": {
                "expressions": [
                    {
                        "filter": {
                            "fieldName": "pagePath",
                            "stringFilter": {
                                "matchType": "BEGINS_WITH",
                                "value": "/docs/generated/moabb.datasets.",
                            },
                        }
                    },
                    {
                        "filter": {
                            "fieldName": "pagePath",
                            "stringFilter": {
                                "matchType": "BEGINS_WITH",
                                "value": "/generated/moabb.datasets.",
                            },
                        }
                    },
                ]
            }
        },
        "keepEmptyRows": False,
        "limit": "100000",
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if response.status_code >= 400:
        text = response.text[:1000]
        raise RuntimeError(f"GA4 runReport failed ({response.status_code}): {text}")

    data = response.json()
    rows = data.get("rows", [])
    per_dataset: dict[str, int] = {}
    for row in rows:
        dims = row.get("dimensionValues") or []
        mets = row.get("metricValues") or []
        if not dims or not mets:
            continue
        page_path = dims[0].get("value", "")
        ds_name = _extract_dataset_class(page_path)
        if not ds_name:
            continue
        try:
            value = int(mets[0].get("value", "0"))
        except ValueError:
            continue
        per_dataset[ds_name] = per_dataset.get(ds_name, 0) + value
    return per_dataset


def _run_report_daily(
    *,
    property_id: str,
    access_token: str,
    start_date: str,
    end_date: str,
    timeout_seconds: float,
    canonical_map: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Return per-dataset daily views: {dataset: {YYYYMMDD: views}}."""
    import requests

    url = (
        f"https://analyticsdata.googleapis.com/v1beta/properties/{property_id}:runReport"
    )
    payload = {
        "dateRanges": [{"startDate": start_date, "endDate": end_date}],
        "dimensions": [{"name": "pagePath"}, {"name": "date"}],
        "metrics": [{"name": "screenPageViews"}],
        "dimensionFilter": {
            "orGroup": {
                "expressions": [
                    {
                        "filter": {
                            "fieldName": "pagePath",
                            "stringFilter": {
                                "matchType": "BEGINS_WITH",
                                "value": "/docs/generated/moabb.datasets.",
                            },
                        }
                    },
                    {
                        "filter": {
                            "fieldName": "pagePath",
                            "stringFilter": {
                                "matchType": "BEGINS_WITH",
                                "value": "/generated/moabb.datasets.",
                            },
                        }
                    },
                ]
            }
        },
        "keepEmptyRows": False,
        "limit": "250000",
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if response.status_code >= 400:
        text = response.text[:1000]
        raise RuntimeError(f"GA4 daily runReport failed ({response.status_code}): {text}")

    data = response.json()
    rows = data.get("rows", [])
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        dims = row.get("dimensionValues") or []
        mets = row.get("metricValues") or []
        if len(dims) < 2 or not mets:
            continue
        page_path = dims[0].get("value", "")
        day = dims[1].get("value", "")
        ds_name = _extract_dataset_class(page_path)
        if not ds_name or not day:
            continue
        ds_name = _canonical_dataset_name(ds_name, canonical_map)
        try:
            value = int(mets[0].get("value", "0"))
        except ValueError:
            continue
        bucket = out.setdefault(ds_name, {})
        bucket[day] = bucket.get(day, 0) + value
    return out


def _build_weekly_series(
    per_dataset_daily: dict[str, dict[str, int]], *, n_weeks: int = 12
) -> dict[str, list[int]]:
    """Aggregate daily counts into n_weeks rolling 7-day bins."""
    end_day = date.today() - timedelta(days=1)
    start_day = end_day - timedelta(days=(n_weeks * 7 - 1))

    series: dict[str, list[int]] = {}
    for ds_name, daily in per_dataset_daily.items():
        values = []
        for week in range(n_weeks):
            bucket_start = start_day + timedelta(days=week * 7)
            total = 0
            for offset in range(7):
                key = (bucket_start + timedelta(days=offset)).strftime("%Y%m%d")
                total += int(daily.get(key, 0))
            values.append(total)
        series[ds_name] = values
    return series


def _build_access_token(service_account_file: str) -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/analytics.readonly"],
    )
    creds.refresh(Request())
    if not creds.token:
        raise RuntimeError("Failed to acquire GA4 access token from service account.")
    return creds.token


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GA4 dataset page views to docs static JSON."
    )
    parser.add_argument(
        "--output",
        default="docs/source/_static/analytics/pageviews.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--property-id",
        default=os.environ.get("GA4_PROPERTY_ID", "").strip(),
        help="GA4 property id (defaults to GA4_PROPERTY_ID env).",
    )
    parser.add_argument(
        "--service-account-file",
        default=os.environ.get("GA4_SERVICE_ACCOUNT_FILE", "").strip(),
        help="Service account JSON path (defaults to GA4_SERVICE_ACCOUNT_FILE env).",
    )
    parser.add_argument(
        "--service-account-json",
        default=os.environ.get("GA4_SERVICE_ACCOUNT_JSON", "").strip(),
        help="Raw service account JSON payload (defaults to GA4_SERVICE_ACCOUNT_JSON env).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=25.0,
        help="HTTP timeout for GA4 requests.",
    )
    parser.add_argument(
        "--fail-on-error", action="store_true", help="Exit non-zero if GA export fails."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)

    property_id = args.property_id.strip()
    service_account_file = args.service_account_file.strip()
    service_account_json = args.service_account_json.strip()

    if not property_id or (not service_account_file and not service_account_json):
        _write_snapshot(
            output_path,
            property_id=property_id or None,
            status="disabled",
            reason="missing GA4_PROPERTY_ID or service account credentials",
        )
        print(f"[ga4] Wrote empty snapshot to {output_path} (GA credentials missing).")
        return 0

    temp_file: NamedTemporaryFile | None = None
    try:
        if not service_account_file and service_account_json:
            temp_file = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            temp_file.write(service_account_json)
            temp_file.flush()
            temp_file.close()
            service_account_file = temp_file.name

        canonical_map = _build_canonical_name_map()
        access_token = _build_access_token(service_account_file)
        last30 = _run_report(
            property_id=property_id,
            access_token=access_token,
            start_date="30daysAgo",
            end_date="yesterday",
            timeout_seconds=args.timeout_seconds,
        )
        last30 = _merge_counts_by_canonical(last30, canonical_map)
        all_time = _run_report(
            property_id=property_id,
            access_token=access_token,
            # GA4 API rejects dates <= 2015-08-13.
            start_date="2015-08-14",
            end_date="yesterday",
            timeout_seconds=args.timeout_seconds,
        )
        all_time = _merge_counts_by_canonical(all_time, canonical_map)
        daily_12w = _run_report_daily(
            property_id=property_id,
            access_token=access_token,
            start_date="84daysAgo",
            end_date="yesterday",
            timeout_seconds=args.timeout_seconds,
            canonical_map=canonical_map,
        )
        weekly_12 = _build_weekly_series(daily_12w, n_weeks=12)

        merged: dict[str, dict[str, int]] = {}
        for name in set(last30) | set(all_time) | set(weekly_12):
            merged[name] = {
                "last30": int(last30.get(name, 0)),
                "all_time": int(all_time.get(name, 0)),
                "weekly_12": [int(v) for v in weekly_12.get(name, [0] * 12)],
            }

        _write_snapshot(
            output_path,
            property_id=property_id,
            counts=dict(sorted(merged.items())),
            status="ok",
            reason="ga4 export successful",
        )
        print(
            f"[ga4] Exported page views for {len(merged)} dataset pages to {output_path}."
        )
        return 0
    except Exception as exc:
        _write_snapshot(
            output_path, property_id=property_id, status="error", reason=str(exc)
        )
        print(f"[ga4] WARNING: {exc}")
        print(f"[ga4] Wrote fallback snapshot to {output_path}.")
        return 1 if args.fail_on_error else 0
    finally:
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
