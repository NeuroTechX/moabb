"""
=========================================================================
Cross-subject transfer learning example using several CrossSubject modes
=========================================================================

Target-aware transfer methods such as Riemannian Alignment / RPA need two
things that are now available in CrossSubjectEvaluation:

1. which source subject each training trial belongs to;
2. controlled access to an unlabeled slice of the held-out target subject.

The available cross-subject modes are defined and documented in
:mod:`moabb.evaluations.protocols`.

This example demonstrates how these modes extend the standard
:class:`~moabb.evaluations.CrossSubjectEvaluation` to support transfer-learning
methods.

This example runs the standard source-only ``TRAIN`` mode as a baseline, then
runs RPA with the currently supported unlabeled target-adaptation modes.

RPA itself is intentionally restricted here to the currently supported unlabeled
target-adaptation modes. The example shows how to select and benchmark those
modes against the baseline. It also demonstrates how an estimator can reject
protocol modes that are not appropriate or not currently supported by the
example, such as source-only, labeled target-calibration modes, or standalone
``TRAIN_TRIALWISE``.

"""

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import CrossSubjectEvaluation
from moabb.evaluations.protocols import CrossSubjectMode
from moabb.paradigms import LeftRightImagery


# ---------------------------------------------------------------------------
# pyriemann's ``Covariances`` is stateless and does not mark itself fitted,
# which ``Pipeline(transform_input=...)`` requires. It calls ``transform`` on
# the sub-pipeline, which runs ``check_is_fitted``.
#
# This small wrapper should no longer be needed with pyRiemann version > 0.11.
# ---------------------------------------------------------------------------
class Covariances_(Covariances):
    def __sklearn_is_fitted__(self):
        return True


RPA_COMPATIBLE_MODES = {
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_50P,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_FULL,
}


# ---------------------------------------------------------------------------
# A minimal target-aware step: Riemannian Alignment.
#
# It consumes:
#   * ``subjects``: source-subject label for each training trial;
#   * ``X_target_unlabeled``: unlabeled target calibration covariances;
#   * ``cs_mode``: the selected cross-subject protocol.
#
# The estimator refuses modes that do not match the currently implemented RPA
# assumption: unlabeled target adaptation.
# ---------------------------------------------------------------------------
class RiemannianAlignment(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, subjects=None, X_target_unlabeled=None, cs_mode=None):
        if cs_mode is None:
            raise ValueError(
                "RiemannianAlignment requires `cs_mode` fit metadata. "
                "Use set_fit_request(cs_mode=True)."
            )

        cs_mode = CrossSubjectMode(cs_mode)
        self.cs_mode_ = cs_mode

        if cs_mode not in RPA_COMPATIBLE_MODES:
            allowed = ", ".join(
                mode.value for mode in sorted(RPA_COMPATIBLE_MODES, key=lambda m: m.value)
            )
            raise ValueError(
                "RiemannianAlignment / RPA supports only the currently implemented "
                "unlabeled target-adaptation modes in this example. "
                f"Got {cs_mode.value!r}. "
                "Modes combining unlabeled target calibration with trialwise scoring "
                "could be meaningful, but are not currently represented by the "
                "available CrossSubjectMode presets. "
                f"Allowed modes are: {allowed}."
            )

        if subjects is None:
            raise ValueError(
                "RiemannianAlignment requires source-subject metadata. "
                "Use set_fit_request(subjects=True)."
            )

        if X_target_unlabeled is None or len(X_target_unlabeled) == 0:
            raise ValueError(
                "RiemannianAlignment requires unlabeled target calibration data. "
                "Use CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P, "
                "TRAIN_AND_TARGET_UNLABELED_50P, or "
                "TRAIN_AND_TARGET_UNLABELED_FULL."
            )

        X = np.asarray(X)
        self.fit_subjects_ = np.asarray(subjects)

        # One whitening reference per source subject.
        self.source_refs_ = {
            subject: invsqrtm(mean_riemann(X[self.fit_subjects_ == subject]))
            for subject in np.unique(self.fit_subjects_)
        }

        # ``X_target_unlabeled`` arrives as covariances thanks to transform_input.
        self.target_ref_ = invsqrtm(mean_riemann(np.asarray(X_target_unlabeled)))

        return self

    def transform(self, X):
        X = np.asarray(X)

        # Training data: align each source subject with its own reference.
        if len(X) == len(self.fit_subjects_):
            out = np.empty_like(X)
            for subject, ref in self.source_refs_.items():
                mask = self.fit_subjects_ == subject
                out[mask] = ref @ X[mask] @ ref
            return out

        # Held-out target data at prediction time: align with target reference.
        return self.target_ref_ @ X @ self.target_ref_


def run_mode(dataset, paradigm, mode, pipeline_name, pipeline):
    print("\n" + "=" * 78)
    print(f"Running mode: {mode.value}")
    print(f"Pipeline    : {pipeline_name}")
    print("=" * 78)

    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        cs_mode=mode,
        overwrite=True,
        suffix=f"rpa_example_{mode.value}_{pipeline_name}",
    )

    results = evaluation.process({pipeline_name: pipeline})
    view = results[["subject", "session", "pipeline", "score"]]
    print(view.to_string(index=False))
    return results


if __name__ == "__main__":
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=4, n_sessions=2, seed=42
    )
    paradigm = LeftRightImagery()

    all_results = []

    # ------------------------------------------------------------------
    # 1. Standard source-only baseline.
    #
    # This is the old ordinary CrossSubjectEvaluation behavior:
    # train on source subjects, test on the held-out subject, no target
    # calibration data.
    # ------------------------------------------------------------------
    baseline_results = run_mode(
        dataset=dataset,
        paradigm=paradigm,
        mode=CrossSubjectMode.TRAIN,  # set benchmark mode
        pipeline_name="SourceOnly+TS+LR",
        pipeline=make_pipeline(
            Covariances_("oas"), TangentSpace("riemann"), LogisticRegression(max_iter=500)
        ),
    )
    all_results.append(baseline_results)

    # ------------------------------------------------------------------
    # 2. RPA with compatible unlabeled target-adaptation modes.
    #
    # These modes differ in how much of the held-out target subject is made
    # available as unlabeled calibration/adaptation data.
    # ------------------------------------------------------------------
    for mode in sorted(RPA_COMPATIBLE_MODES, key=lambda m: m.value):
        with config_context(enable_metadata_routing=True):
            align = RiemannianAlignment().set_fit_request(
                subjects=True, X_target_unlabeled=True, cs_mode=True
            )

        results = run_mode(
            dataset=dataset,
            paradigm=paradigm,
            mode=mode,
            pipeline_name="RPA+TS+LR",
            pipeline=make_pipeline(
                Covariances_("oas"),
                align,
                TangentSpace("riemann"),
                LogisticRegression(max_iter=500),
                transform_input=["X_target_unlabeled"],
            ),
        )
        all_results.append(results)

    # ------------------------------------------------------------------
    # 3. One intentionally incompatible mode.
    #
    # RPA is not a labeled target-calibration method. This block shows the
    # explicit error message produced by the estimator.
    # ------------------------------------------------------------------
    try:
        with config_context(enable_metadata_routing=True):
            align = RiemannianAlignment().set_fit_request(
                subjects=True, X_target_unlabeled=True, cs_mode=True
            )

        run_mode(
            dataset=dataset,
            paradigm=paradigm,
            mode=CrossSubjectMode.TRAIN_AND_TARGET_LABELED_20P,
            pipeline_name="RPA+TS+LR",
            pipeline=make_pipeline(
                Covariances_("oas"),
                align,
                TangentSpace("riemann"),
                LogisticRegression(max_iter=500),
                transform_input=["X_target_unlabeled"],
            ),
        )
    except ValueError as err:
        print("\nExpected error for incompatible mode:")
        print(err)

    # ------------------------------------------------------------------
    # Optional compact comparison.
    # ------------------------------------------------------------------
    summary = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 78)
    print("Compact comparison")
    print("=" * 78)
    print(
        summary.groupby("pipeline")["score"]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )
