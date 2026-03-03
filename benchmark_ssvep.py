"""Benchmark SSVEP datasets against paper-reported results.

Uses MOABB CrossSessionEvaluation for multi-session datasets, and manual
leave-one-block-out for single-session datasets (WithinSessionEvaluation
hardcodes n_folds=5, but our datasets have only 4 blocks = 4 trials/class).
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from ssvep_moabb_adapters import (
    Dong2023,
    Han2024Fatigue,
    Liu2020BETA,
    Liu2022EldBETA,
)

from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import SSVEP
from moabb.pipelines import SSVEP_CCA, SSVEP_TRCA, SSVEP_eCCA


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

N_SUBJECTS = 3


def build_freq_map(event_id):
    """Map frequency-string labels to float Hz values."""
    return {name: float(name) for name in event_id.keys()}


def make_pipelines(freq_map):
    """Build pipeline dict with CCA, eCCA, and TRCA."""
    return {
        "CCA": SSVEP_CCA(n_harmonics=3, freq_map=freq_map),
        "eCCA": SSVEP_eCCA(n_harmonics=3, freq_map=freq_map),
        "TRCA": SSVEP_TRCA(n_fbands=5, is_ensemble=True),
    }


def run_within_session_manual(name, dataset_cls, fmin, fmax, n_blocks, n_subjects):
    """Manual leave-one-block-out for single-session datasets.

    MOABB's WithinSessionEvaluation hardcodes n_folds=5, but Liu2020BETA
    and Dong2023 have only 4 blocks (4 trials/class), so we use GroupKFold
    to properly implement leave-one-block-out cross-validation.
    """
    ds = dataset_cls()
    subjects = ds.subject_list[:n_subjects]

    paradigm = SSVEP(fmin=fmin, fmax=fmax, n_classes=None)
    freq_map = build_freq_map(ds.event_id)
    pipelines = make_pipelines(freq_map)

    results = []

    for subj in subjects:
        print(f"  Subject {subj}...", end=" ", flush=True)
        t0 = time.time()

        epochs, labels, meta = paradigm.get_data(ds, subjects=[subj], return_epochs=True)

        # Block indices: data ordered as target1_b0, target1_b1, ..., targetN_b(K-1)
        block_ids = np.array([i % n_blocks for i in range(len(labels))])
        gkf = GroupKFold(n_splits=n_blocks)

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(epochs, labels, block_ids)
        ):
            X_train, y_train = epochs[train_idx], labels[train_idx]
            X_test, y_test = epochs[test_idx], labels[test_idx]

            for pipe_name, clf in pipelines.items():
                try:
                    clf_copy = clf.__class__(**clf.get_params())
                    clf_copy.fit(X_train, y_train)
                    y_pred = clf_copy.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results.append(
                        {
                            "dataset": name,
                            "subject": subj,
                            "pipeline": pipe_name,
                            "score": acc,
                        }
                    )
                except Exception as e:
                    print(f"\n    {pipe_name} fold {fold_idx}: {e}")

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def run_cross_session(name, dataset_cls, fmin, fmax, n_subjects):
    """Cross-session evaluation using MOABB's CrossSessionEvaluation.

    Uses mne_labels=True to keep original frequency string labels
    (needed by CCA/eCCA freq_map), and return_epochs=True for
    SSVEP classifiers that require MNE Epochs.
    """
    ds = dataset_cls()
    ds.subject_list = ds.subject_list[:n_subjects]

    paradigm = SSVEP(fmin=fmin, fmax=fmax, n_classes=None)
    freq_map = build_freq_map(ds.event_id)
    pipelines = make_pipelines(freq_map)

    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=[ds],
        random_state=42,
        overwrite=True,
        return_epochs=True,
        mne_labels=True,
    )
    return evaluation.process(pipelines)


# ── Dataset configurations ──────────────────────────────────────────
WITHIN_CONFIGS = [
    (
        "Liu2020BETA",
        Liu2020BETA,
        7,
        50,
        4,
        "Leave-1-block-out, 9ch, 130ms lat, Nh=5 | "
        "CCA ~40%, eCCA ~85%, TRCA ~80% (at 2s, 70 subj)",
    ),
    (
        "Dong2023",
        Dong2023,
        7,
        50,
        4,
        "Leave-1-block-out, 8ch, 160ms lat | "
        "FBCCA 76.9%, eTRCA 63.5% (at 4s, 59 subj)",
    ),
]

CROSS_CONFIGS = [
    (
        "Liu2022EldBETA",
        Liu2022EldBETA,
        7,
        50,
        "Leave-1-block-out (7 blocks), 9ch, 140ms lat | "
        "CCA ~85%, eCCA 86.2%, eTRCA 86.8% (at 1s, 100 subj)",
    ),
    (
        "Han2024Fatigue",
        Han2024Fatigue,
        7,
        50,
        "Train->Fatigue cross-session, 10ch, 140ms lat | "
        "TRCA-FS 92.4% low-freq (at 2s, 24 subj)",
    ),
]

all_results = []

# ── Within-session benchmarks (manual CV) ───────────────────────────
for name, cls, fmin, fmax, n_blocks, paper_ref in WITHIN_CONFIGS:
    print(f"\n{'='*60}")
    print(f"{name} (Manual leave-1-block-out, {n_blocks} blocks)")
    print(f"  Paper: {paper_ref}")
    print(f"{'='*60}")

    try:
        df = run_within_session_manual(name, cls, fmin, fmax, n_blocks, N_SUBJECTS)
        df["dataset_name"] = name
        all_results.append(df)

        summary = df.groupby("pipeline")["score"].agg(["mean", "std"])
        print(f"\n  Results (n={N_SUBJECTS} subjects):")
        for pipe, row in summary.iterrows():
            print(f"    {pipe:6s}: {row['mean']:.1%} +/- {row['std']:.1%}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

# ── Cross-session benchmarks (MOABB evaluation) ─────────────────────
for name, cls, fmin, fmax, paper_ref in CROSS_CONFIGS:
    print(f"\n{'='*60}")
    print(f"{name} (CrossSession, leave-1-session-out)")
    print(f"  Paper: {paper_ref}")
    print(f"{'='*60}")

    try:
        results = run_cross_session(name, cls, fmin, fmax, N_SUBJECTS)
        results["dataset_name"] = name
        all_results.append(results)

        summary = results.groupby("pipeline")["score"].agg(["mean", "std"])
        print(f"\n  Results (n={N_SUBJECTS} subjects):")
        for pipe, row in summary.iterrows():
            print(f"    {pipe:6s}: {row['mean']:.1%} +/- {row['std']:.1%}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

# ── Summary table ───────────────────────────────────────────────────
if all_results:
    df = pd.concat(all_results, ignore_index=True)

    print("\n\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':20s} | {'CCA':>15s} | {'eCCA':>15s} | {'TRCA':>15s}")
    print("-" * 80)

    all_names = [c[0] for c in WITHIN_CONFIGS] + [c[0] for c in CROSS_CONFIGS]
    all_refs = [c[-1] for c in WITHIN_CONFIGS] + [c[-1] for c in CROSS_CONFIGS]

    for name, paper_ref in zip(all_names, all_refs):
        ds_df = df[df["dataset_name"] == name]
        if ds_df.empty:
            print(f"{name:20s} | {'FAIL':>15s} | {'FAIL':>15s} | {'FAIL':>15s}")
            continue

        parts = []
        for pipe in ["CCA", "eCCA", "TRCA"]:
            pipe_df = ds_df[ds_df["pipeline"] == pipe]
            if not pipe_df.empty:
                m, s = pipe_df["score"].mean(), pipe_df["score"].std()
                parts.append(f"{m:.1%}+/-{s:.1%}")
            else:
                parts.append("N/A")

        print(f"{name:20s} | {parts[0]:>15s} | {parts[1]:>15s} | {parts[2]:>15s}")

    print()
    print("Paper references (full-dataset, full-window results):")
    for name, ref in zip(all_names, all_refs):
        print(f"  {name:20s}: {ref}")
    print()
    print("NOTE: Our results may differ from papers due to:")
    print("  - Using all channels (papers use 9-11 selected channels)")
    print("  - No visual latency shift (papers add 130-160ms)")
    print("  - Nh=3 harmonics (papers use Nh=5)")
    print(f"  - Only {N_SUBJECTS} subjects (papers use all subjects)")

    df.to_csv("benchmark_ssvep_results.csv", index=False)
    print("\nFull results saved to benchmark_ssvep_results.csv")
