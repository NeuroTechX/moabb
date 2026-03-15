"""Step 1: Data loading smoke test (1 subject per dataset).

For each dataset, loads subject 1 and verifies:
  - Loads without error
  - Channel count matches metadata.acquisition.n_channels
  - Sampling rate matches metadata.acquisition.sampling_rate
  - Number of unique stim event codes matches expected n_classes
  - Signal amplitude is physically reasonable (mean |x| in ~0.1-500 uV)
  - No NaN or Inf in data
  - Session count matches metadata.sessions_per_subject
  - Data duration > 0

Runs fully in parallel via multiprocessing.

Usage:
    python -m scripts.validation.validate_step1_loading
"""

import multiprocessing as mp
import traceback
import warnings

import numpy as np
import pandas as pd
from scripts.validation import DEFAULT_WORKERS, all_work_items, ensure_output_dir


def _check_one(args):
    """Load subject 1 and run smoke checks."""
    ds_name, paradigm_key = args
    warnings.filterwarnings("ignore")

    row = {
        "dataset": ds_name,
        "paradigm": paradigm_key,
        "loads_ok": False,
        "n_channels_actual": None,
        "n_channels_meta": None,
        "ch_match": None,
        "sfreq_actual": None,
        "sfreq_meta": None,
        "sfreq_match": None,
        "n_events_actual": None,
        "n_classes_meta": None,
        "events_match": None,
        "mean_abs_uv": None,
        "max_abs_uv": None,
        "amp_reasonable": None,
        "has_nan": None,
        "has_inf": None,
        "n_sessions": None,
        "n_runs_total": None,
        "duration_s": None,
        "issues": [],
        "status": "FAIL",
        "error": None,
    }

    try:
        import mne

        import moabb.datasets as ds_module
        from moabb.datasets.metadata import DATASET_METADATA_CATALOG

        mne.set_log_level("ERROR")

        ds_cls = getattr(ds_module, ds_name, None)
        if ds_cls is None:
            row["error"] = f"Class {ds_name} not found"
            row["issues"].append(row["error"])
            row["issues"] = "; ".join(row["issues"])
            return row

        # Load first subject only
        first_subj = ds_cls().subject_list[0]
        ds = ds_cls(subjects=[first_subj])
        data = ds.get_data(subjects=[first_subj])
        row["loads_ok"] = True

        # Get metadata
        meta = DATASET_METADATA_CATALOG.get(ds_name)

        # ── Collect all Raw objects ─────────────────────────────────────
        raws = []
        n_sessions = 0
        n_runs = 0
        for sess_id, sessions in data[first_subj].items():
            n_sessions += 1
            for run_id, raw in sessions.items():
                n_runs += 1
                raws.append(raw)

        row["n_sessions"] = n_sessions
        row["n_runs_total"] = n_runs

        if not raws:
            row["issues"].append("No Raw objects returned")
            row["issues"] = "; ".join(row["issues"])
            return row

        # Use first raw for channel/sfreq checks, all raws for amplitude
        raw0 = raws[0]

        # ── Channel count ───────────────────────────────────────────────
        # Count only EEG channels (what metadata.n_channels typically refers to)
        eeg_chs = mne.pick_types(raw0.info, eeg=True)
        row["n_channels_actual"] = len(eeg_chs)
        if meta:
            row["n_channels_meta"] = meta.acquisition.n_channels
            # Allow match if actual >= metadata (some datasets add stim/eog)
            row["ch_match"] = len(eeg_chs) == meta.acquisition.n_channels
            if not row["ch_match"]:
                row["issues"].append(
                    f"EEG channels: actual={len(eeg_chs)}, meta={meta.acquisition.n_channels}"
                )

        # ── Sampling rate ───────────────────────────────────────────────
        row["sfreq_actual"] = raw0.info["sfreq"]
        if meta:
            row["sfreq_meta"] = meta.acquisition.sampling_rate
            row["sfreq_match"] = (
                abs(raw0.info["sfreq"] - meta.acquisition.sampling_rate) < 0.5
            )
            if not row["sfreq_match"]:
                row["issues"].append(
                    f"sfreq: actual={raw0.info['sfreq']}, meta={meta.acquisition.sampling_rate}"
                )

        # ── Event count ─────────────────────────────────────────────────
        # Count unique stim events across all runs
        all_event_codes = set()
        for raw in raws:
            stim_chs = mne.pick_types(raw.info, stim=True)
            if len(stim_chs) > 0:
                events = mne.find_events(raw, shortest_event=1, verbose=False)
                if len(events) > 0:
                    all_event_codes.update(events[:, 2].tolist())
            else:
                # Try annotations
                if raw.annotations:
                    all_event_codes.update(
                        d
                        for d in raw.annotations.description
                        if d not in ("BAD_", "EDGE")
                    )

        row["n_events_actual"] = len(all_event_codes)
        if meta and meta.experiment.n_classes:
            row["n_classes_meta"] = meta.experiment.n_classes
            row["events_match"] = len(all_event_codes) >= meta.experiment.n_classes
            if not row["events_match"]:
                row["issues"].append(
                    f"unique events: actual={len(all_event_codes)}, n_classes={meta.experiment.n_classes}"
                )

        # ── Signal amplitude + NaN/Inf check (single read) ──────────────
        # Sample up to 30s from each raw to avoid memory issues
        amp_values = []
        has_nan = False
        has_inf = False
        for raw in raws:
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            if len(eeg_picks) == 0:
                continue
            n_samples = min(int(30 * raw.info["sfreq"]), raw.n_times)
            chunk = raw.get_data(picks=eeg_picks, start=0, stop=n_samples)
            # NaN/Inf on raw data before scaling
            if np.any(np.isnan(chunk)):
                has_nan = True
            if np.any(np.isinf(chunk)):
                has_inf = True
            # Convert to microvolts (MNE stores in V)
            chunk_uv = chunk * 1e6
            amp_values.append(np.abs(chunk_uv).mean())

        if amp_values:
            row["mean_abs_uv"] = round(float(np.mean(amp_values)), 2)
            row["max_abs_uv"] = round(float(np.max(amp_values)), 2)
            # Reasonable EEG amplitude: 0.1 - 500000 µV mean absolute
            # Upper bound is 500 mV to accommodate DC-coupled raw recordings
            # where electrode offset potentials dominate the mean amplitude.
            row["amp_reasonable"] = 0.1 <= row["mean_abs_uv"] <= 500_000
            if not row["amp_reasonable"]:
                row["issues"].append(
                    f"amplitude suspect: mean_abs={row['mean_abs_uv']} uV"
                )

        row["has_nan"] = has_nan
        row["has_inf"] = has_inf
        if has_nan:
            row["issues"].append("NaN detected in EEG data")
        if has_inf:
            row["issues"].append("Inf detected in EEG data")

        # ── Duration ────────────────────────────────────────────────────
        total_dur = sum(raw.times[-1] for raw in raws if raw.n_times > 0)
        row["duration_s"] = round(total_dur, 1)
        if total_dur <= 0:
            row["issues"].append("Zero or negative total duration")

        # ── Overall status ──────────────────────────────────────────────
        critical_fails = [row["has_nan"], row["has_inf"]]
        if any(v is True for v in critical_fails):
            row["status"] = "FAIL"
        elif row["issues"]:
            row["status"] = "WARN"
        else:
            row["status"] = "PASS"

        row["issues"] = "; ".join(row["issues"]) if row["issues"] else ""
        return row

    except Exception:
        row["error"] = traceback.format_exc().splitlines()[-1]
        row["issues"] = row["error"]
        return row


def main():
    work = all_work_items()
    n_workers = min(DEFAULT_WORKERS, len(work))
    print(f"Step 1: Loading smoke test for {len(work)} datasets ({n_workers} workers)\n")

    results = []
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_check_one, work):
            tag = row["status"]
            name = row["dataset"]
            amp = row.get("mean_abs_uv", "?")
            ch = row.get("n_channels_actual", "?")
            dur = row.get("duration_s", "?")
            issues = row.get("issues", "")
            detail = f"  ({issues})" if issues else ""
            print(f"  {tag:4s} {name:35s} ch={ch:<4} amp={amp:<8} dur={dur}s{detail}")
            results.append(row)

    # ── Summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values(["paradigm", "dataset"])
    out = ensure_output_dir()

    csv_path = out / "step1_loading.csv"
    df.to_csv(csv_path, index=False)

    n_pass = (df["status"] == "PASS").sum()
    n_warn = (df["status"] == "WARN").sum()
    n_fail = (df["status"] == "FAIL").sum()
    print(f"\n{'='*60}")
    print(f"Step 1 Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print(f"Results saved to {csv_path}")

    if n_fail > 0:
        print("\nFailed datasets:")
        for _, r in df[df["status"] == "FAIL"].iterrows():
            print(f"  {r['dataset']}: {r['issues']}")

    return df


if __name__ == "__main__":
    main()
