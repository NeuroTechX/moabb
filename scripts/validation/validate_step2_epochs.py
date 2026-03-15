"""Step 2: Epoch-level integrity checks (1 subject per dataset).

For each dataset, extracts epochs via the paradigm and verifies:
  - Paradigm.get_data() succeeds
  - Number of trials per class (total and per-class breakdown)
  - No flat channels (std > 0 for each EEG channel across epochs)
  - Epoch duration matches dataset.interval
  - Labels match dataset.event_id keys
  - X shape is (n_trials, n_channels, n_times) with all dims > 0

Runs fully in parallel via multiprocessing.

Usage:
    python -m scripts.validation.validate_step2_epochs
"""

import multiprocessing as mp
import traceback
import warnings

import numpy as np
import pandas as pd
from scripts.validation import DEFAULT_WORKERS, all_work_items, ensure_output_dir


def _make_paradigm(key):
    """Create a fresh paradigm instance (must be inside worker for pickling)."""
    from moabb.paradigms import P300, SSVEP, MotorImagery

    return {
        "imagery": lambda: MotorImagery(resample=128),
        "p300": lambda: P300(resample=128),
        "ssvep": lambda: SSVEP(resample=128),
    }[key]()


def _check_one(args):
    """Extract epochs and run integrity checks."""
    ds_name, paradigm_key = args
    warnings.filterwarnings("ignore")

    row = {
        "dataset": ds_name,
        "paradigm": paradigm_key,
        "epoch_ok": False,
        "n_trials": None,
        "n_channels_epoch": None,
        "n_times": None,
        "n_classes_actual": None,
        "class_counts": None,
        "n_flat_channels": None,
        "epoch_duration_s": None,
        "expected_duration_s": None,
        "duration_match": None,
        "labels_match_events": None,
        "min_class_count": None,
        "max_class_count": None,
        "class_balance_ratio": None,
        "issues": [],
        "status": "FAIL",
        "error": None,
    }

    try:
        import mne

        import moabb.datasets as ds_module

        mne.set_log_level("ERROR")

        ds_cls = getattr(ds_module, ds_name, None)
        if ds_cls is None:
            row["error"] = f"Class {ds_name} not found"
            row["issues"].append(row["error"])
            row["issues"] = "; ".join(row["issues"])
            return row

        # Load first subject
        first_subj = ds_cls().subject_list[0]
        ds = ds_cls(subjects=[first_subj])

        # Create paradigm and extract data
        paradigm = _make_paradigm(paradigm_key)
        X, labels, meta_df = paradigm.get_data(ds, subjects=[first_subj])
        row["epoch_ok"] = True

        # ── Shape checks ────────────────────────────────────────────────
        row["n_trials"] = X.shape[0]
        row["n_channels_epoch"] = X.shape[1]
        row["n_times"] = X.shape[2]

        if X.shape[0] == 0:
            row["issues"].append("Zero trials extracted")
            row["issues"] = "; ".join(row["issues"])
            return row

        # ── Class distribution ──────────────────────────────────────────
        unique_labels, counts = np.unique(labels, return_counts=True)
        row["n_classes_actual"] = len(unique_labels)
        row["class_counts"] = str(dict(zip(unique_labels.tolist(), counts.tolist())))
        row["min_class_count"] = int(counts.min())
        row["max_class_count"] = int(counts.max())
        row["class_balance_ratio"] = round(float(counts.min()) / float(counts.max()), 3)

        if row["min_class_count"] < 2:
            row["issues"].append(
                f"Class with <2 trials: min_count={row['min_class_count']}"
            )

        # ── Labels vs event_id ──────────────────────────────────────────
        ds_event_keys = set(ds.event_id.keys()) if hasattr(ds, "event_id") else set()
        actual_labels = set(unique_labels.tolist())
        if ds_event_keys:
            row["labels_match_events"] = actual_labels.issubset(ds_event_keys)
            if not row["labels_match_events"]:
                extra = actual_labels - ds_event_keys
                row["issues"].append(f"Labels not in event_id: {extra}")

        # ── Flat channel detection ──────────────────────────────────────
        # Compute std per channel across all trials
        # X shape: (n_trials, n_channels, n_times)
        ch_stds = X.std(axis=(0, 2))  # std across trials and time for each channel
        n_flat = int((ch_stds == 0).sum())
        row["n_flat_channels"] = n_flat
        if n_flat > 0:
            row["issues"].append(f"{n_flat} flat channel(s) detected")

        # ── Epoch duration ──────────────────────────────────────────────
        sfreq = 128  # We resampled to 128; assumes native sfreq >= 128
        actual_dur = X.shape[2] / sfreq
        row["epoch_duration_s"] = round(actual_dur, 3)

        ds_interval = getattr(ds, "interval", None)
        if ds_interval is not None:
            expected_dur = ds_interval[1] - ds_interval[0]
            row["expected_duration_s"] = round(expected_dur, 3)
            # Allow small tolerance for resampling rounding
            row["duration_match"] = abs(actual_dur - expected_dur) < 0.1
            if not row["duration_match"]:
                row["issues"].append(
                    f"epoch duration: actual={actual_dur:.3f}s, "
                    f"expected={expected_dur:.3f}s"
                )

        # ── NaN/Inf in epochs ───────────────────────────────────────────
        if np.any(np.isnan(X)):
            row["issues"].append("NaN in epoch data")
        if np.any(np.isinf(X)):
            row["issues"].append("Inf in epoch data")

        # ── Overall status ──────────────────────────────────────────────
        critical = [
            row["n_trials"] > 0,
            row["n_flat_channels"] == 0,
            not np.any(np.isnan(X)),
            not np.any(np.isinf(X)),
        ]
        if all(critical):
            row["status"] = "PASS" if not row["issues"] else "WARN"
        else:
            row["status"] = "FAIL"

        row["issues"] = "; ".join(row["issues"]) if row["issues"] else ""
        return row

    except Exception:
        row["error"] = traceback.format_exc().splitlines()[-1]
        row["issues"] = row["error"]
        return row


def main():
    work = all_work_items()
    n_workers = min(DEFAULT_WORKERS, len(work))
    print(f"Step 2: Epoch integrity for {len(work)} datasets ({n_workers} workers)\n")

    results = []
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_check_one, work):
            tag = row["status"]
            name = row["dataset"]
            n_trials = row.get("n_trials", "?")
            n_cls = row.get("n_classes_actual", "?")
            bal = row.get("class_balance_ratio", "?")
            flat = row.get("n_flat_channels", "?")
            issues = row.get("issues", "")
            detail = f"  ({issues})" if issues else ""
            print(
                f"  {tag:4s} {name:35s} "
                f"trials={n_trials:<6} classes={n_cls:<3} "
                f"balance={bal:<6} flat={flat}{detail}"
            )
            results.append(row)

    # ── Summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values(["paradigm", "dataset"])
    out = ensure_output_dir()

    csv_path = out / "step2_epochs.csv"
    df.to_csv(csv_path, index=False)

    n_pass = (df["status"] == "PASS").sum()
    n_warn = (df["status"] == "WARN").sum()
    n_fail = (df["status"] == "FAIL").sum()
    print(f"\n{'='*60}")
    print(f"Step 2 Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print(f"Results saved to {csv_path}")

    if n_fail > 0:
        print("\nFailed datasets:")
        for _, r in df[df["status"] == "FAIL"].iterrows():
            print(f"  {r['dataset']}: {r['issues']}")

    return df


if __name__ == "__main__":
    main()
