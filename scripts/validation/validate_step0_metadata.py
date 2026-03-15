"""Step 0: Metadata consistency checks (no data download needed).

For each dataset, verifies:
  - Metadata exists in DATASET_METADATA_CATALOG
  - Required fields are populated (paradigm, n_subjects, n_channels, sampling_rate)
  - n_subjects matches len(subject_list)
  - paradigm string matches dataset.paradigm attribute
  - event_id keys match metadata experiment.events keys (if both exist)
  - n_classes matches len(event_id) or metadata.experiment.n_classes

Runs fully in parallel via multiprocessing.

Usage:
    python -m scripts.validation.validate_step0_metadata
"""

import multiprocessing as mp
import traceback
import warnings

import pandas as pd
from scripts.validation import DEFAULT_WORKERS, all_work_items, ensure_output_dir


def _check_one(args):
    """Check metadata consistency for a single dataset."""
    ds_name, paradigm_key = args
    warnings.filterwarnings("ignore")

    checks = {
        "dataset": ds_name,
        "paradigm": paradigm_key,
        "has_metadata": False,
        "n_subjects_match": None,
        "paradigm_match": None,
        "events_match": None,
        "n_classes_match": None,
        "has_doi": None,
        "has_n_channels": None,
        "has_sampling_rate": None,
        "issues": [],
        "status": "FAIL",
        "error": None,
    }

    try:
        import moabb.datasets as ds_module
        from moabb.datasets.metadata import DATASET_METADATA_CATALOG

        # Instantiate dataset
        ds_cls = getattr(ds_module, ds_name, None)
        if ds_cls is None:
            checks["issues"].append(f"Class {ds_name} not found in moabb.datasets")
            checks["error"] = "class_not_found"
            checks["issues"] = "; ".join(checks["issues"])
            return checks

        ds = ds_cls()

        # ── Check metadata exists ───────────────────────────────────────
        meta = DATASET_METADATA_CATALOG.get(ds_name)
        if meta is None:
            checks["issues"].append("No metadata in DATASET_METADATA_CATALOG")
            checks["error"] = "no_metadata"
            checks["issues"] = "; ".join(checks["issues"])
            return checks
        checks["has_metadata"] = True

        # ── n_subjects ──────────────────────────────────────────────────
        n_subj_meta = meta.participants.n_subjects
        n_subj_actual = len(ds.subject_list)
        checks["n_subjects_match"] = n_subj_meta == n_subj_actual
        if not checks["n_subjects_match"]:
            checks["issues"].append(
                f"n_subjects: metadata={n_subj_meta}, actual={n_subj_actual}"
            )

        # ── paradigm string ─────────────────────────────────────────────
        meta_paradigm = meta.experiment.paradigm
        ds_paradigm = ds.paradigm
        checks["paradigm_match"] = meta_paradigm == ds_paradigm
        if not checks["paradigm_match"]:
            checks["issues"].append(
                f"paradigm: metadata='{meta_paradigm}', dataset='{ds_paradigm}'"
            )

        # ── events ──────────────────────────────────────────────────────
        ds_events = set(ds.event_id.keys()) if hasattr(ds, "event_id") else set()
        meta_events = (
            set(meta.experiment.events.keys()) if meta.experiment.events else set()
        )
        if ds_events and meta_events:
            checks["events_match"] = ds_events == meta_events
            if not checks["events_match"]:
                missing_in_meta = ds_events - meta_events
                missing_in_ds = meta_events - ds_events
                if missing_in_meta:
                    checks["issues"].append(
                        f"events in dataset but not metadata: {missing_in_meta}"
                    )
                if missing_in_ds:
                    checks["issues"].append(
                        f"events in metadata but not dataset: {missing_in_ds}"
                    )
        elif ds_events and not meta_events:
            checks["events_match"] = None
            checks["issues"].append("metadata.experiment.events is empty")
        # else: can't check

        # ── n_classes ───────────────────────────────────────────────────
        n_classes_ds = len(ds.event_id) if hasattr(ds, "event_id") else None
        n_classes_meta = meta.experiment.n_classes
        if n_classes_ds is not None and n_classes_meta is not None:
            checks["n_classes_match"] = n_classes_ds == n_classes_meta
            if not checks["n_classes_match"]:
                checks["issues"].append(
                    f"n_classes: metadata={n_classes_meta}, dataset={n_classes_ds}"
                )

        # ── DOI ─────────────────────────────────────────────────────────
        if meta.documentation:
            checks["has_doi"] = meta.documentation.doi is not None
            if not checks["has_doi"]:
                checks["issues"].append("No DOI in metadata")
        else:
            checks["has_doi"] = False
            checks["issues"].append("No documentation metadata")

        # ── acquisition fields ──────────────────────────────────────────
        checks["has_n_channels"] = (
            meta.acquisition.n_channels is not None and meta.acquisition.n_channels > 0
        )
        if not checks["has_n_channels"]:
            checks["issues"].append("n_channels missing or zero")

        checks["has_sampling_rate"] = (
            meta.acquisition.sampling_rate is not None
            and meta.acquisition.sampling_rate > 0
        )
        if not checks["has_sampling_rate"]:
            checks["issues"].append("sampling_rate missing or zero")

        # ── Overall status ──────────────────────────────────────────────
        critical = [
            checks["has_metadata"],
            checks["n_subjects_match"],
            checks["paradigm_match"],
            checks["has_n_channels"],
            checks["has_sampling_rate"],
        ]
        if all(v is True for v in critical):
            if checks["issues"]:
                checks["status"] = "WARN"
            else:
                checks["status"] = "PASS"
        else:
            checks["status"] = "FAIL"

        checks["issues"] = "; ".join(checks["issues"]) if checks["issues"] else ""
        return checks

    except Exception:
        checks["error"] = traceback.format_exc().splitlines()[-1]
        checks["issues"] = checks["error"]
        return checks


def main():
    work = all_work_items()
    n_workers = min(DEFAULT_WORKERS, len(work))
    print(
        f"Step 0: Metadata consistency for {len(work)} datasets ({n_workers} workers)\n"
    )

    results = []
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for checks in pool.imap_unordered(_check_one, work):
            tag = checks["status"]
            name = checks["dataset"]
            issues = checks.get("issues", "")
            detail = f"  ({issues})" if issues else ""
            print(f"  {tag:4s} {name:35s}{detail}")
            results.append(checks)

    # ── Summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values(["paradigm", "dataset"])
    out = ensure_output_dir()

    csv_path = out / "step0_metadata.csv"
    df.to_csv(csv_path, index=False)

    n_pass = (df["status"] == "PASS").sum()
    n_warn = (df["status"] == "WARN").sum()
    n_fail = (df["status"] == "FAIL").sum()
    print(f"\n{'='*60}")
    print(f"Step 0 Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print(f"Results saved to {csv_path}")

    if n_fail > 0:
        print("\nFailed datasets:")
        for _, row in df[df["status"] == "FAIL"].iterrows():
            print(f"  {row['dataset']}: {row['issues']}")

    return df


if __name__ == "__main__":
    main()
