"""
Example of the transfer-learning-oriented
CrossSubjectTargetAwareEvaluation.

This example compares a standard TS + LR pipeline with an RPA + TS + LR
pipeline. Riemannian Procrustes Alignment is used here as a simple example of
a target-aware transfer-learning method: it can use source-subject structure and,
when allowed by the evaluation mode, unlabeled target covariance data for
alignment.

The script can demonstrate 4 of the 6 available modes by changing `cs_mode`:

* HOS_SOURCE_ONLY_BLOCKWISE:
  No target data is used during adaptation. RPA aligns only the source
  subjects, so this should usually be weaker than target-adaptive modes.

* HOS_UNLABELED_20P:
  The first 20% of the held-out target subject is provided without labels
  for target-domain alignment. The remaining 80% is evaluated.

* HOS_UNLABELED_50P:
  The first 50% of the held-out target subject is provided without labels
  for target-domain alignment. The remaining 50% is evaluated.

* HOS_LABELED_20P:
  The first 20% of the held-out target subject is provided as labeled
  calibration data. In this example, RPA ignores the labels and uses only
  the covariance distribution for alignment, so HOS_LABELED_20P and
  HOS_UNLABELED_20P should give very similar or identical results for the
  RPA step.

The example does not demonstrate HOS_SOURCE_ONLY_TRIALWISE or
HOS_UNLABELED_100P.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moabb.datasets import Weibo2014
from moabb.evaluations.cross_subject_target_aware_evaluation import (
    CrossSubjectTargetAwareEvaluation,
    CsMode,
)
from moabb.paradigms import LeftRightImagery


# ---------------------------------------------------------------------
# SPD helpers
# ---------------------------------------------------------------------
def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def nearest_spd_jitter(C: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    C = symmetrize(np.asarray(C, dtype=float))
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    return symmetrize((vecs * vals) @ vecs.T)


def safe_mean_riemann(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)

    if X.ndim != 3 or len(X) == 0:
        raise ValueError(
            "safe_mean_riemann expects shape (n_matrices, n_channels, n_channels)."
        )

    X = np.stack([nearest_spd_jitter(C) for C in X], axis=0)
    return nearest_spd_jitter(mean_riemann(X))


def powerm_spd(C: np.ndarray, power: float, eps: float = 1e-12) -> np.ndarray:
    C = nearest_spd_jitter(C)
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    return symmetrize((vecs * (vals**power)) @ vecs.T)


def batch_congruence(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    X = np.asarray(X)

    return np.stack([nearest_spd_jitter(A @ C @ A.T) for C in X], axis=0)


# ---------------------------------------------------------------------
# Riemannian Procrustes Alignment transformer
# ---------------------------------------------------------------------
class RiemannianProcrustesAlignment(BaseEstimator, TransformerMixin):
    """
    Riemannian Procrustes Alignment for covariance matrices.

    Parameters
    ----------
    reference_subject : str or None, default="auto"
        Reference domain used for alignment.

        - "auto": choose the source subject whose mean covariance is closest
          to the global source mean.
        - "global" or None: use the global source mean directly.
        - any other value: use that source subject as the reference subject.
          The value is converted to str and must exist among the source
          subjects seen during fit.

    alignment_strength : float, default=1.0
        Strength of the alignment transform.

        - 0.0: no alignment.
        - 1.0: full Procrustes-style alignment.
        - values between 0.0 and 1.0: partial alignment.

        The transform is computed as:
        A = M_ref ** (0.5 * alignment_strength)
            @ M_domain ** (-0.5 * alignment_strength)

    use_global_transform_for_unseen : bool, default=True
        If True, unseen data without an explicit target transform is aligned
        using the transform from the global source mean to the reference mean.
        If False, unseen data is returned after SPD cleanup only.

    verbose : bool, default=False
        If True, print diagnostic information during fit, including the
        selected reference subject, number of source subjects, and whether
        target data was used.
    """

    def __init__(
        self,
        reference_subject: Optional[str] = "auto",
        alignment_strength: float = 1.0,
        use_global_transform_for_unseen: bool = True,
        verbose: bool = False,
    ):
        self.reference_subject = reference_subject
        self.alignment_strength = alignment_strength
        self.use_global_transform_for_unseen = use_global_transform_for_unseen
        self.verbose = verbose

    def _make_transform(self, M_ref: np.ndarray, M_source: np.ndarray) -> np.ndarray:
        alpha = float(self.alignment_strength)

        if alpha == 0.0:
            return np.eye(M_ref.shape[0])

        return powerm_spd(M_ref, 0.5 * alpha) @ powerm_spd(M_source, -0.5 * alpha)

    def fit(
        self,
        X,
        y=None,
        subjects=None,
        cs_mode=None,
        X_target_unlabeled=None,
        X_target_labeled=None,
        y_target_labeled=None,
        **fit_params,
    ):
        """
        Input
        -----
        X : ndarray, shape (n_trials, n_channels, n_channels)

        Fit metadata
        ------------
        subjects : array-like
            Source subject ID for each source training trial.

        cs_mode : str, optional
            Cross-subject evaluation mode name, for example:
            "HOS_SOURCE_ONLY_BLOCKWISE", "HOS_UNLABELED_20P",
            "HOS_LABELED_20P".

        X_target_unlabeled : ndarray, optional
            Unlabeled covariance matrices from the held-out target subject.
            If provided, RPA uses them to estimate the target-domain alignment
            transform.

        X_target_labeled : ndarray, optional
            Labeled covariance matrices from the held-out target subject.
            If provided and no unlabeled target data is provided, RPA uses
            their covariance distribution for alignment. The labels themselves
            are not used.

        y_target_labeled : ndarray, optional
            Labels for X_target_labeled. RPA accepts this argument for
            evaluator compatibility but does not use the labels directly.
        """
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "RiemannianProcrustesAlignment expects covariance matrices "
                f"with shape (n_trials, n_channels, n_channels). Got {X.shape}."
            )

        if X.shape[1] != X.shape[2]:
            raise ValueError("Covariance matrices must be square.")

        self.cs_mode_ = cs_mode
        self.n_channels_ = int(X.shape[1])

        if subjects is None:
            warnings.warn(
                "subjects was not provided to RPA.fit(). "
                "Using one global source domain only.",
                RuntimeWarning,
            )
            subjects = np.array(["source"] * len(X), dtype=str)
        else:
            subjects = np.asarray(subjects).astype(str)

        if len(subjects) != len(X):
            raise ValueError("X and subjects must have the same length.")

        self.source_subjects_ = np.unique(subjects).astype(str)
        self.source_means_: Dict[str, np.ndarray] = {}

        for s in self.source_subjects_:
            idx = subjects == s
            self.source_means_[str(s)] = safe_mean_riemann(X[idx])

        self.global_source_mean_ = safe_mean_riemann(X)

        # Choose reference domain.
        if self.reference_subject is None or self.reference_subject == "global":
            self.reference_subject_ = "global"
            self.reference_mean_ = self.global_source_mean_

        elif self.reference_subject == "auto":
            distances = {}

            for s, M_s in self.source_means_.items():
                distances[s] = float(np.linalg.norm(M_s - self.global_source_mean_))

            self.reference_subject_ = min(distances, key=distances.get)
            self.reference_mean_ = self.source_means_[self.reference_subject_]

        else:
            self.reference_subject_ = str(self.reference_subject)

            if self.reference_subject_ not in self.source_means_:
                raise ValueError(
                    f"reference_subject={self.reference_subject_!r} is not "
                    f"in source subjects {list(self.source_means_.keys())}."
                )

            self.reference_mean_ = self.source_means_[self.reference_subject_]

        # Source-subject transforms.
        self.source_transforms_: Dict[str, np.ndarray] = {}

        for s, M_s in self.source_means_.items():
            self.source_transforms_[s] = self._make_transform(self.reference_mean_, M_s)

        self.global_transform_ = self._make_transform(
            self.reference_mean_, self.global_source_mean_
        )

        # The evaluation mode decides which target data is provided.
        # Prefer explicit unlabeled target data when available. Otherwise,
        # use labeled target covariance data for alignment, but ignore labels.
        X_target_for_alignment = None
        target_source_kind = "none"

        if X_target_unlabeled is not None and len(X_target_unlabeled) > 0:
            X_target_for_alignment = X_target_unlabeled
            target_source_kind = "unlabeled"

        elif X_target_labeled is not None and len(X_target_labeled) > 0:
            X_target_for_alignment = X_target_labeled
            target_source_kind = "labeled"

        self.target_transform_ = None
        self.has_target_data_ = False
        self.target_source_kind_ = target_source_kind

        if X_target_for_alignment is not None:
            X_target_for_alignment = np.asarray(X_target_for_alignment)

            if X_target_for_alignment.ndim != 3:
                raise ValueError(
                    "Target data for RPA must have shape "
                    "(n_trials, n_channels, n_channels)."
                )

            if (
                X_target_for_alignment.shape[1] != self.n_channels_
                or X_target_for_alignment.shape[2] != self.n_channels_
            ):
                raise ValueError(
                    "Target covariance matrices have incompatible shape. "
                    f"Expected ({self.n_channels_}, {self.n_channels_}), "
                    f"got {X_target_for_alignment.shape[1:]}."
                )

            target_mean = safe_mean_riemann(X_target_for_alignment)

            self.target_transform_ = self._make_transform(
                self.reference_mean_, target_mean
            )
            self.has_target_data_ = True

        # Stored for sklearn Pipeline fit_transform.
        self._fit_subjects_ = subjects.copy()
        self._n_fit_samples_ = int(len(X))

        if self.verbose:
            n_unlab = 0 if X_target_unlabeled is None else len(X_target_unlabeled)
            n_lab = 0 if X_target_labeled is None else len(X_target_labeled)

            print(
                "RPA.fit | "
                f"cs_mode={self.cs_mode_}, "
                f"reference_subject={self.reference_subject_}, "
                f"n_source_subjects={len(self.source_subjects_)}, "
                f"alignment_strength={self.alignment_strength}, "
                f"has_target_data={self.has_target_data_}, "
                f"target_source_kind={self.target_source_kind_}, "
                f"n_target_unlabeled={n_unlab}, "
                f"n_target_labeled={n_lab}",
                flush=True,
            )

        return self

    def transform(self, X, subjects=None):
        self._check_is_fitted()

        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "RiemannianProcrustesAlignment expects covariance matrices "
                f"with shape (n_trials, n_channels, n_channels). Got {X.shape}."
            )

        if X.shape[1] != self.n_channels_ or X.shape[2] != self.n_channels_:
            raise ValueError(
                f"Expected matrices with shape ({self.n_channels_}, "
                f"{self.n_channels_}), got {X.shape[1:]}."
            )

        # Explicit source-subject transform.
        if subjects is not None:
            subjects = np.asarray(subjects).astype(str)

            if len(subjects) != len(X):
                raise ValueError("X and subjects must have the same length.")

            X_out = np.empty_like(X, dtype=float)

            for s in np.unique(subjects):
                s = str(s)
                idx = subjects == s

                if s not in self.source_transforms_:
                    raise ValueError(
                        f"Unknown source subject {s!r}. Available: "
                        f"{list(self.source_transforms_.keys())}"
                    )

                X_out[idx] = batch_congruence(X[idx], self.source_transforms_[s])

            return X_out

        # sklearn Pipeline training transform: no subjects are passed to
        # transform(), but the length matches the fitted training data.
        if hasattr(self, "_fit_subjects_") and len(X) == self._n_fit_samples_:
            X_out = np.empty_like(X, dtype=float)

            for s in np.unique(self._fit_subjects_):
                s = str(s)
                idx = self._fit_subjects_ == s
                X_out[idx] = batch_congruence(X[idx], self.source_transforms_[s])

            return X_out

        # Test / unseen data.
        if self.target_transform_ is not None:
            return batch_congruence(X, self.target_transform_)

        if self.use_global_transform_for_unseen:
            return batch_congruence(X, self.global_transform_)

        return np.stack([nearest_spd_jitter(C) for C in X], axis=0)

    def fit_transform(
        self,
        X,
        y=None,
        subjects=None,
        cs_mode=None,
        X_target_unlabeled=None,
        X_target_labeled=None,
        y_target_labeled=None,
        **fit_params,
    ):
        return self.fit(
            X,
            y=y,
            subjects=subjects,
            cs_mode=cs_mode,
            X_target_unlabeled=X_target_unlabeled,
            X_target_labeled=X_target_labeled,
            y_target_labeled=y_target_labeled,
            **fit_params,
        ).transform(X, subjects=subjects)

    def _check_is_fitted(self):
        if not hasattr(self, "reference_mean_"):
            raise RuntimeError("RiemannianProcrustesAlignment is not fitted.")


# ---------------------------------------------------------------------
# Demo pipelines
# ---------------------------------------------------------------------


def make_pipelines():
    ts_lr = make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=42
        ),
    )

    rpa_ts_lr = make_pipeline(
        Covariances(estimator="oas"),
        RiemannianProcrustesAlignment(
            reference_subject="auto",
            alignment_strength=1.0,
            use_global_transform_for_unseen=True,
            verbose=True,
        ),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=42
        ),
    )

    return {"TS + LR": ts_lr, "RPA + TS + LR": rpa_ts_lr}


# ---------------------------------------------------------------------
# Result summaries
# ---------------------------------------------------------------------
def normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()

    if "dataset" in results.columns:
        results["dataset"] = results["dataset"].apply(
            lambda d: d.code if hasattr(d, "code") else str(d)
        )

    return results


def summarize_per_dataset_pipeline(results: pd.DataFrame) -> pd.DataFrame:
    """
    One row per dataset and pipeline.
    """
    results = normalize_results(results)

    summary = (
        results.groupby(["dataset", "pipeline"], as_index=False)
        .agg(
            n_folds=("score", "count"),
            mean_ROC_AUC=("score", "mean"),
            std_ROC_AUC=("score", "std"),
        )
        .sort_values(["dataset", "mean_ROC_AUC"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return summary


def summarize_global_pipeline(summary: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pipeline.

    The global mean is computed over dataset means, not over all folds.
    This avoids giving more weight to datasets with more subjects/sessions.
    """
    global_summary = (
        summary.groupby("pipeline", as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            total_folds=("n_folds", "sum"),
            mean_ROC_AUC_over_datasets=("mean_ROC_AUC", "mean"),
            std_ROC_AUC_over_datasets=("mean_ROC_AUC", "std"),
        )
        .sort_values("mean_ROC_AUC_over_datasets", ascending=False)
        .reset_index(drop=True)
    )

    return global_summary


def make_dataset_pipeline_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table: rows are datasets, columns are pipelines.
    Useful for quick visual comparison.
    """
    table = summary.pivot(
        index="dataset", columns="pipeline", values="mean_ROC_AUC"
    ).reset_index()

    table.columns.name = None
    return table


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    dataset = Weibo2014()  # BNCI2014_001()

    # Uncomment for a fast smoke test.
    # dataset.subject_list = [1, 2, 3, 4, 5]

    datasets = [dataset]

    paradigm = LeftRightImagery(fmin=8.0, fmax=32.0, resample=128)

    pipelines = make_pipelines()

    # Good RPA examples:
    #
    # CsMode.HOS_UNLABELED_20P:
    #     First 20% of the held-out target subject is used without labels
    #     for target alignment. Remaining 80% is evaluated.
    #
    # CsMode.HOS_LABELED_20P:
    #     First 20% of the held-out target subject is passed separately as
    #     labeled calibration data. RPA may use its covariance distribution
    #     for alignment; a final classifier may use the labels if it supports
    #     X_target_labeled/y_target_labeled.
    #
    # CsMode.HOS_SOURCE_ONLY_BLOCKWISE:
    #     No target adaptation. This is closest to standard MOABB
    #     cross-subject block prediction.
    #
    cs_mode = (
        CsMode.HOS_UNLABELED_20P
    )  # RPA aligns using the first 20% target data without labels.
    # cs_mode = CsMode.HOS_UNLABELED_50P # RPA aligns using the first 20% target data without labels.
    # cs_mode = CsMode.HOS_LABELED_20P # RPA aligns using the first 20% target data, ignoring labels.
    # cs_mode = CsMode.HOS_SOURCE_ONLY_BLOCKWISE # RPA uses source-subject alignment only; no target adaptation.

    evaluation = CrossSubjectTargetAwareEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        cs_mode=cs_mode,
        n_jobs=4,  # 1 while debugging verbose RPA output
        overwrite=True,
        random_state=42,
    )

    results = evaluation.process(pipelines=pipelines)
    results = normalize_results(results)

    useful_cols = ["dataset", "subject", "session", "pipeline", "score"]

    useful_cols = [c for c in useful_cols if c in results.columns]

    print("\nRaw results:")
    print(results[useful_cols].to_string(index=False))

    summary = summarize_per_dataset_pipeline(results)

    print("\nPer-dataset / per-pipeline summary:")
    print(summary.to_string(index=False))

    global_summary = summarize_global_pipeline(summary)

    print("\nGlobal per-pipeline summary:")
    print(global_summary.to_string(index=False))


if __name__ == "__main__":
    main()
