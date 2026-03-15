"""Step 4: Cross-dataset summary report.

Aggregates results from Steps 0-3 (and benchmark if available) into a
single unified report. Identifies outliers and produces a final
PASS/WARN/FAIL per dataset.

Usage:
    python -m scripts.validation.validate_step4_report
"""

import pandas as pd
from scripts.validation import DATASET_NAMES, ensure_output_dir


def _load_csv(path):
    """Load a CSV if it exists, else return None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    out = ensure_output_dir()

    # ── Load all step results ───────────────────────────────────────
    step0 = _load_csv(out / "step0_metadata.csv")
    step1 = _load_csv(out / "step1_loading.csv")
    step2 = _load_csv(out / "step2_epochs.csv")
    step3 = _load_csv(out / "step3_reproducibility.csv")
    bench = _load_csv(out.parent / "benchmark_new_datasets.csv")

    # ── Build master dataset list ───────────────────────────────────
    all_datasets = []
    for pkey, names in DATASET_NAMES.items():
        for name in names:
            all_datasets.append({"dataset": name, "paradigm": pkey})
    master = pd.DataFrame(all_datasets)

    # ── Merge step results ──────────────────────────────────────────
    def _merge_status(master_df, step_df, step_name):
        if step_df is None:
            master_df[step_name] = "N/A"
            master_df[f"{step_name}_issues"] = ""
            return master_df
        step_slim = step_df[["dataset", "status", "issues"]].rename(
            columns={"status": step_name, "issues": f"{step_name}_issues"}
        )
        return master_df.merge(step_slim, on="dataset", how="left").fillna(
            {step_name: "N/A", f"{step_name}_issues": ""}
        )

    master = _merge_status(master, step0, "step0_meta")
    master = _merge_status(master, step1, "step1_load")
    master = _merge_status(master, step2, "step2_epoch")
    master = _merge_status(master, step3, "step3_repro")

    # ── Merge numeric columns from step1 and step2 ─────────────────
    if step1 is not None:
        cols = [
            "dataset",
            "n_channels_actual",
            "sfreq_actual",
            "mean_abs_uv",
            "duration_s",
        ]
        cols = [c for c in cols if c in step1.columns]
        master = master.merge(step1[cols], on="dataset", how="left")

    if step2 is not None:
        cols = [
            "dataset",
            "n_trials",
            "n_classes_actual",
            "class_balance_ratio",
            "n_flat_channels",
            "epoch_duration_s",
        ]
        cols = [c for c in cols if c in step2.columns]
        master = master.merge(step2[cols], on="dataset", how="left")

    # ── Merge benchmark results ─────────────────────────────────────
    if bench is not None:
        # benchmark CSV may have dataset as index or column
        if "dataset" not in bench.columns and bench.index.name != "dataset":
            bench = bench.reset_index()
        if "dataset" not in bench.columns:
            bench = bench.rename(columns={bench.columns[0]: "dataset"})
        bench_cols = ["dataset"]
        for c in ["mean_score", "theoretical", "adjusted_05", "above_chance"]:
            if c in bench.columns:
                bench_cols.append(c)
        if len(bench_cols) > 1:
            master = master.merge(bench[bench_cols], on="dataset", how="left")

    # ── Compute overall status ──────────────────────────────────────
    step_cols = ["step0_meta", "step1_load", "step2_epoch", "step3_repro"]

    def _overall(row):
        statuses = [row.get(c, "N/A") for c in step_cols]
        if any(s == "FAIL" for s in statuses):
            return "FAIL"
        if any(s == "WARN" for s in statuses):
            return "WARN"
        if all(s in ("PASS", "N/A") for s in statuses):
            return "PASS"
        return "UNKNOWN"

    master["overall"] = master.apply(_overall, axis=1)

    # ── Collect all issues ──────────────────────────────────────────
    issue_cols = [c for c in master.columns if c.endswith("_issues")]

    def _all_issues(row):
        parts = []
        for c in issue_cols:
            v = row.get(c, "")
            if v and str(v) != "nan":
                parts.append(str(v))
        return "; ".join(parts) if parts else ""

    master["all_issues"] = master.apply(_all_issues, axis=1)

    # ── Print report ────────────────────────────────────────────────
    print("=" * 100)
    print("VALIDATION REPORT — Cross-Dataset Summary")
    print("=" * 100)

    for pkey in ["imagery", "p300", "ssvep"]:
        subset = master[master["paradigm"] == pkey].sort_values("dataset")
        n_pass = (subset["overall"] == "PASS").sum()
        n_warn = (subset["overall"] == "WARN").sum()
        n_fail = (subset["overall"] == "FAIL").sum()
        print(
            f"\n--- {pkey.upper()} ({len(subset)} datasets: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL) ---\n"
        )

        display_cols = [
            "dataset",
            "overall",
            "step0_meta",
            "step1_load",
            "step2_epoch",
            "step3_repro",
        ]
        extra = []
        for c in [
            "n_channels_actual",
            "n_trials",
            "mean_abs_uv",
            "class_balance_ratio",
            "mean_score",
        ]:
            if c in subset.columns:
                extra.append(c)
        display_cols.extend(extra)

        print(subset[display_cols].to_string(index=False))
        print()

    # ── Outlier detection ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("OUTLIER DETECTION")
    print("=" * 100)

    if "mean_abs_uv" in master.columns:
        amp = master["mean_abs_uv"].dropna()
        if len(amp) > 0:
            q1, q3 = amp.quantile(0.25), amp.quantile(0.75)
            iqr = q3 - q1
            outliers = master[
                (master["mean_abs_uv"] < q1 - 1.5 * iqr)
                | (master["mean_abs_uv"] > q3 + 1.5 * iqr)
            ]
            if len(outliers) > 0:
                print("\nAmplitude outliers (IQR method):")
                for _, r in outliers.iterrows():
                    print(f"  {r['dataset']:35s} mean_abs={r['mean_abs_uv']:.1f} uV")
            else:
                print("\nNo amplitude outliers detected.")

    if "class_balance_ratio" in master.columns:
        imbalanced = master[master["class_balance_ratio"] < 0.5]
        if len(imbalanced) > 0:
            print("\nClass-imbalanced datasets (min/max < 0.5):")
            for _, r in imbalanced.iterrows():
                print(f"  {r['dataset']:35s} balance={r['class_balance_ratio']:.3f}")

    if "n_flat_channels" in master.columns:
        flat = master[master["n_flat_channels"] > 0]
        if len(flat) > 0:
            print("\nDatasets with flat channels:")
            for _, r in flat.iterrows():
                print(f"  {r['dataset']:35s} flat_ch={int(r['n_flat_channels'])}")

    # ── Save ────────────────────────────────────────────────────────
    csv_path = out / "step4_full_report.csv"
    master.to_csv(csv_path, index=False)

    print(f"\n{'='*100}")
    total_pass = (master["overall"] == "PASS").sum()
    total_warn = (master["overall"] == "WARN").sum()
    total_fail = (master["overall"] == "FAIL").sum()
    print(
        f"OVERALL: {len(master)} datasets — "
        f"{total_pass} PASS, {total_warn} WARN, {total_fail} FAIL"
    )
    print(f"Full report saved to {csv_path}")

    if total_fail > 0:
        print("\nDatasets requiring attention:")
        for _, r in master[master["overall"] == "FAIL"].iterrows():
            print(f"  {r['dataset']:35s} {r['all_issues']}")

    return master


if __name__ == "__main__":
    main()
