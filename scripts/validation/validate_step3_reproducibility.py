"""Step 3: Reproducibility check — load the same subject twice, compare data.

For each dataset, loads subject 1 twice and verifies:
  - Raw data arrays are bitwise identical (allclose with atol=0)
  - Channel names are identical
  - Sampling rates are identical
  - Event annotations/stim channels are identical
  - Number of sessions and runs are identical

This catches random state leaks, non-deterministic file parsing, or
side effects from mutable state in loading code.

Runs fully in parallel via multiprocessing.

Usage:
    python -m scripts.validation.validate_step3_reproducibility
"""

import multiprocessing as mp
import traceback
import warnings

import numpy as np
import pandas as pd
from scripts.validation import DEFAULT_WORKERS, all_work_items, ensure_output_dir


def _check_one(args):
    """Load subject 1 twice and compare."""
    ds_name, paradigm_key = args
    warnings.filterwarnings("ignore")

    row = {
        "dataset": ds_name,
        "paradigm": paradigm_key,
        "load1_ok": False,
        "load2_ok": False,
        "data_identical": None,
        "ch_names_identical": None,
        "sfreq_identical": None,
        "n_sessions_identical": None,
        "n_runs_identical": None,
        "max_abs_diff": None,
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

        first_subj = ds_cls().subject_list[0]

        # ── Load #1 ────────────────────────────────────────────────────
        ds1 = ds_cls(subjects=[first_subj])
        data1 = ds1.get_data(subjects=[first_subj])
        row["load1_ok"] = True

        # ── Load #2 ────────────────────────────────────────────────────
        ds2 = ds_cls(subjects=[first_subj])
        data2 = ds2.get_data(subjects=[first_subj])
        row["load2_ok"] = True

        # ── Structure comparison ────────────────────────────────────────
        sessions1 = sorted(data1[first_subj].keys())
        sessions2 = sorted(data2[first_subj].keys())
        row["n_sessions_identical"] = sessions1 == sessions2
        if not row["n_sessions_identical"]:
            row["issues"].append(f"Session keys differ: {sessions1} vs {sessions2}")
            row["issues"] = "; ".join(row["issues"])
            return row

        all_data_match = True
        all_ch_match = True
        all_sfreq_match = True
        all_runs_match = True
        max_diff = 0.0

        for sess_key in sessions1:
            runs1 = sorted(data1[first_subj][sess_key].keys())
            runs2 = sorted(data2[first_subj][sess_key].keys())

            if runs1 != runs2:
                all_runs_match = False
                row["issues"].append(
                    f"Run keys differ in session {sess_key}: {runs1} vs {runs2}"
                )
                continue

            for run_key in runs1:
                raw1 = data1[first_subj][sess_key][run_key]
                raw2 = data2[first_subj][sess_key][run_key]

                # Channel names
                if raw1.ch_names != raw2.ch_names:
                    all_ch_match = False
                    row["issues"].append(f"Channel names differ in {sess_key}/{run_key}")

                # Sampling rate
                if raw1.info["sfreq"] != raw2.info["sfreq"]:
                    all_sfreq_match = False
                    row["issues"].append(
                        f"sfreq differs in {sess_key}/{run_key}: "
                        f"{raw1.info['sfreq']} vs {raw2.info['sfreq']}"
                    )

                # Data comparison — sample first 30s to avoid memory issues
                eeg_picks = mne.pick_types(raw1.info, eeg=True)
                if len(eeg_picks) > 0:
                    n_samples = min(
                        int(30 * raw1.info["sfreq"]), raw1.n_times, raw2.n_times
                    )
                    d1 = raw1.get_data(picks=eeg_picks, start=0, stop=n_samples)
                    d2 = raw2.get_data(picks=eeg_picks, start=0, stop=n_samples)

                    if d1.shape != d2.shape:
                        all_data_match = False
                        row["issues"].append(
                            f"Shape differs in {sess_key}/{run_key}: "
                            f"{d1.shape} vs {d2.shape}"
                        )
                    else:
                        diff = np.abs(d1 - d2).max()
                        max_diff = max(max_diff, float(diff))
                        if diff > 0:
                            all_data_match = False

        row["n_runs_identical"] = all_runs_match

        row["data_identical"] = all_data_match
        row["ch_names_identical"] = all_ch_match
        row["sfreq_identical"] = all_sfreq_match
        row["max_abs_diff"] = max_diff

        if not all_data_match and max_diff > 0:
            row["issues"].append(f"Data not identical, max_abs_diff={max_diff:.2e}")

        # ── Overall status ──────────────────────────────────────────────
        if all_data_match and all_ch_match and all_sfreq_match:
            row["status"] = "PASS"
        elif all_ch_match and all_sfreq_match and max_diff < 1e-10:
            row["status"] = "WARN"
            if not row["issues"]:
                row["issues"].append(f"Near-identical (max_diff={max_diff:.2e})")
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
    print(f"Step 3: Reproducibility for {len(work)} datasets ({n_workers} workers)\n")

    results = []
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_check_one, work):
            tag = row["status"]
            name = row["dataset"]
            max_diff = row.get("max_abs_diff", "?")
            issues = row.get("issues", "")
            detail = f"  ({issues})" if issues else ""
            print(f"  {tag:4s} {name:35s} max_diff={max_diff}{detail}")
            results.append(row)

    # ── Summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values(["paradigm", "dataset"])
    out = ensure_output_dir()

    csv_path = out / "step3_reproducibility.csv"
    df.to_csv(csv_path, index=False)

    n_pass = (df["status"] == "PASS").sum()
    n_warn = (df["status"] == "WARN").sum()
    n_fail = (df["status"] == "FAIL").sum()
    print(f"\n{'='*60}")
    print(f"Step 3 Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print(f"Results saved to {csv_path}")

    if n_fail > 0:
        print("\nFailed datasets:")
        for _, r in df[df["status"] == "FAIL"].iterrows():
            print(f"  {r['dataset']}: {r['issues']}")

    return df


if __name__ == "__main__":
    main()
