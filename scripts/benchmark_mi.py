"""Benchmark new MI datasets using MOABB's evaluation framework.

Uses WithinSessionEvaluation with Tangent Space + LR pipeline
on subject 1 of each dataset, with adjusted chance levels.
"""

import traceback
import warnings

import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from moabb.analysis.chance_level import chance_by_chance
from moabb.datasets import (
    Brandl2020,
    Chang2025,
    HefmiIch2025,
    Kaya2018,
    Kumar2024,
    Ma2020,
    Rozado2015,
    Wairagkar2018,
    Yi2025,
    Zhang2017,
    Zuo2025,
)
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


warnings.filterwarnings("ignore")

DATASETS = [
    Kaya2018,
    Zhang2017,
    Rozado2015,
    Kumar2024,
    Brandl2020,
    Ma2020,
    Wairagkar2018,
    Chang2025,
    HefmiIch2025,
    Yi2025,
    Zuo2025,
]

pipelines = {
    "TS+LR": make_pipeline(
        Covariances(estimator="lwf"),
        TangentSpace(metric="riemann"),
        LogisticRegression(solver="lbfgs", max_iter=1000),
    ),
}

all_results = []

for ds_cls in DATASETS:
    ds = ds_cls(subjects=[ds_cls().subject_list[0]])
    name = ds.code
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    try:
        paradigm = MotorImagery()
        evaluation = WithinSessionEvaluation(
            paradigm=paradigm,
            datasets=[ds],
            overwrite=True,
            random_state=42,
        )
        results = evaluation.process(pipelines)
        all_results.append(results)

        chance = chance_by_chance(results)
        adj = list(chance[name]["adjusted"].values())[0]
        print(results[["dataset", "session", "score", "samples"]].to_string())
        print(
            f"\n  Mean: {results['score'].mean():.3f} | "
            f"Chance: {chance[name]['theoretical']:.3f} | "
            f"Adjusted (p<0.05): {adj:.3f}"
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

# Summary
if all_results:
    df = pd.concat(all_results, ignore_index=True)
    chance_all = chance_by_chance(df)

    print(f"\n\n{'='*80}")
    print("SUMMARY: WithinSessionEvaluation — TS+LR (Subject 1)")
    print(f"{'='*80}")
    summary = (
        df.groupby("dataset")
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            sessions=("session", "nunique"),
            channels=("channels", "first"),
            n_classes=("n_classes", "first"),
            samples=("samples", "first"),
        )
        .round(3)
    )
    summary["theoretical"] = summary.index.map(
        lambda d: chance_all[d]["theoretical"]
    ).round(3)
    summary["adjusted_05"] = summary.index.map(
        lambda d: list(chance_all[d]["adjusted"].values())[0]
    ).round(3)
    summary["above_chance"] = summary["mean_score"] > summary["adjusted_05"]
    summary = summary[
        [
            "mean_score",
            "std_score",
            "theoretical",
            "adjusted_05",
            "above_chance",
            "sessions",
            "channels",
            "n_classes",
            "samples",
        ]
    ]
    print(summary.to_string())
