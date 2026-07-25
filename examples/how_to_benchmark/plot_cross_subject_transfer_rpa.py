"""
==========================================================
Cross-subject transfer with RPA-style covariance alignment
==========================================================

Target-aware transfer methods such as Riemannian alignment need information
that an ordinary source-only evaluation deliberately hides. This tutorial shows
how :class:`~moabb.evaluations.CrossSubjectEvaluation` can grant that
information under an explicit, reproducible protocol.

By the end of the tutorial, you will know how to:

1. choose a named :class:`~moabb.evaluations.protocols.CrossSubjectMode`;
2. route source-subject identifiers and permitted target data to one pipeline
   step;
3. use scikit-learn's ``transform_input`` so source and target data reach that
   step in the same representation;
4. make an estimator reject protocols that violate its assumptions; and
5. report the access protocol together with the benchmark score.

The worked method is a compact, pedagogical alignment stage inspired by
Riemannian Procrustes Analysis (RPA). It whitens covariance matrices around
source- and target-domain Riemannian means. It is intended to demonstrate the
MOABB interface, not to reproduce every step or variant of a published RPA
algorithm.

We compare the standard source-only ``TRAIN`` baseline with three protocols
that allow unlabeled target adaptation. A final, intentionally incompatible run
demonstrates the failure mode.

"""

# Authors: Anton Andonov <toncho11@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import CrossSubjectEvaluation
from moabb.evaluations.protocols import CrossSubjectMode
from moabb.paradigms import LeftRightImagery


###############################################################################
# Define an alignment transformer
# -------------------------------
#
# pyriemann's ``Covariances`` is stateless and does not mark itself fitted,
# which ``Pipeline(transform_input=...)`` requires. It calls ``transform`` on
# the sub-pipeline, which runs ``check_is_fitted``.
#
# This small wrapper should no longer be needed with pyRiemann version > 0.11.


class Covariances_(Covariances):
    def __sklearn_is_fitted__(self):
        return True


RPA_COMPATIBLE_MODES = {
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_50P,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_FULL,
}


###############################################################################
# Compute domain-specific references
# -----------------------------------
#
# EEG covariance matrices are symmetric positive definite (SPD), so their
# geometry is not Euclidean. For each source subject :math:`s`, the transformer
# estimates the affine-invariant Riemannian mean :math:`G_s` and applies
#
# .. math::
#
#    C_i^\prime = G_s^{-1/2} C_i G_s^{-1/2}.
#
# It estimates a separate target reference :math:`G_t` from only the unlabeled
# target trials permitted by the protocol, and uses the analogous mapping for
# target trials. Centering every domain around the identity reduces
# subject-specific covariance shifts while never using target labels.
#
# The fit method consumes three routed fields:
#
# * ``subjects`` identifies the source domain of every training trial;
# * ``X_target_unlabeled`` contains the permitted target calibration slice;
# * ``cs_mode`` states the benchmark contract explicitly.
#
# Requesting ``cs_mode`` is useful defensive design: the transformer can reject
# source-only or labeled protocols rather than silently running a different
# algorithm.
#
# Notice the separate ``fit_transform`` and ``transform`` implementations.
# During pipeline fitting, ``fit_transform`` aligns source trials with their
# own subject references. Later, ``transform`` is reserved for held-out target
# trials and uses the target reference. This explicit lifecycle is safer than
# guessing the domain from the number or shape of incoming trials.


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

    def fit_transform(
        self,
        X,
        y=None,
        subjects=None,
        X_target_unlabeled=None,
        cs_mode=None,
        **fit_params,
    ):
        """Fit the references and align the source trials used for training."""
        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit metadata: {names}.")
        self.fit(
            X,
            y,
            subjects=subjects,
            X_target_unlabeled=X_target_unlabeled,
            cs_mode=cs_mode,
        )
        X = np.asarray(X)
        out = np.empty_like(X)
        for subject, ref in self.source_refs_.items():
            mask = self.fit_subjects_ == subject
            out[mask] = ref @ X[mask] @ ref
        return out

    def transform(self, X):
        """Align held-out target trials with the target reference."""
        X = np.asarray(X)
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

    results = evaluation.process({pipeline_name: pipeline}).assign(mode=mode.value)
    view = results[["subject", "session", "pipeline", "score"]]
    print(view.to_string(index=False))
    return results


###############################################################################
# Follow the metadata through the pipeline
# ----------------------------------------
#
# ``X_target_unlabeled`` starts as raw epochs because the evaluation owns the
# split. The alignment step, however, receives source *covariances* as its main
# ``X``. Passing raw target epochs directly would therefore be a representation
# error.
#
# Scikit-learn's ``Pipeline(transform_input=["X_target_unlabeled"])`` solves
# this: the routed target data traverses the same fitted pipeline prefix as the
# source data before it reaches the consuming step. In this example both
# branches pass through ``Covariances_``. Learned prefixes such as PCA or
# feature selection would likewise use the clone fitted inside the current
# cross-validation fold, preventing leakage.

fig, ax = plt.subplots(figsize=(12, 4))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")

workflow_nodes = {
    "source": (0.03, 0.68, "Source epochs\n+ subject IDs", "#d8ecff"),
    "target": (0.03, 0.18, "Permitted unlabeled\ntarget epochs", "#fff0cc"),
    "source_cov": (0.29, 0.68, "Covariances_\n(source X)", "#d8ecff"),
    "target_cov": (0.29, 0.18, "Same pipeline prefix\n(target metadata)", "#fff0cc"),
    "alignment": (
        0.58,
        0.43,
        "RiemannianAlignment\nsource + target references",
        "#e7ddff",
    ),
    "classifier": (0.84, 0.43, "Tangent space\n+ classifier", "#dff4df"),
}
for x, y, label, color in workflow_nodes.values():
    ax.text(
        x,
        y,
        label,
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": color, "edgecolor": "0.35"},
    )

workflow_arrows = [
    ((0.20, 0.68), (0.285, 0.68)),
    ((0.20, 0.18), (0.285, 0.18)),
    ((0.46, 0.68), (0.575, 0.53)),
    ((0.46, 0.18), (0.575, 0.43)),
    ((0.78, 0.48), (0.835, 0.48)),
]
for start, end in workflow_arrows:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": "0.3",
            "lw": 1.8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )

ax.text(
    0.30,
    0.04,
    'transform_input=["X_target_unlabeled"] keeps both representations aligned',
    fontsize=10,
    color="0.25",
)
ax.set_title("Fit-time data and metadata flow", pad=12)
fig.tight_layout()
plt.show()

###############################################################################
# Compare the protocol contracts
# ------------------------------
#
# The mode controls the estimator's access, not merely the split size.
#
# .. list-table::
#    :header-rows: 1
#
#    * - Mode
#      - Target data visible at fit
#      - Scored target data
#    * - ``TRAIN``
#      - none
#      - full target block, blockwise
#    * - ``...UNLABELED_20P``
#      - first 20%, labels withheld
#      - remaining 80%, blockwise
#    * - ``...UNLABELED_50P``
#      - first 50%, labels withheld
#      - remaining 50%, blockwise
#    * - ``...UNLABELED_FULL``
#      - full target block, labels withheld
#      - same full block, transductively
#
# The ``FULL`` mode is transductive: all target samples may inform an unlabeled
# target reference before that same block is scored. Its score answers a
# different scientific question from an inductive source-only score. Always
# report the mode name with the number.


###############################################################################
# Configure a small benchmark
# ---------------------------
#
# A deterministic fake dataset keeps this tutorial fast and download-free. Its
# scores are illustrative; the purpose is to show the transfer-learning API.

dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=4, n_sessions=2, seed=42)
paradigm = LeftRightImagery()
all_results = []

###############################################################################
# Run the source-only baseline
# ----------------------------
#
# ``TRAIN`` is the ordinary cross-subject protocol: train on source subjects,
# test on the held-out subject, and expose no target calibration data.

baseline_results = run_mode(
    dataset=dataset,
    paradigm=paradigm,
    mode=CrossSubjectMode.TRAIN,
    pipeline_name="SourceOnly+TS+LR",
    pipeline=make_pipeline(
        Covariances_("oas"), TangentSpace("riemann"), LogisticRegression(max_iter=500)
    ),
)
all_results.append(baseline_results)

###############################################################################
# Run target-aware RPA
# --------------------
#
# The three compatible modes differ only in how much unlabeled data from the
# held-out target is available for estimating the target reference. Scikit-learn
# routes both that data and the source-subject identifiers to the alignment step.

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

###############################################################################
# Reject an incompatible protocol
# -------------------------------
#
# RPA uses unlabeled target data. A labeled-calibration preset therefore fails
# with an explicit protocol error instead of silently changing the method.

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

###############################################################################
# Compare the successful protocols
# --------------------------------
#
# Each point below is one reported cross-subject/session result. The diamond is
# the mean. The second panel shows how many target trials remain per held-out
# subject after calibration. This matters: changing the access budget can also
# change the evaluation set, so a score without its protocol is ambiguous.
#
# Because ``FakeDataset`` contains no simulated transfer effect, tiny ranking
# differences are noise and should not be interpreted as evidence that one mode
# is better. On real data, repeat the comparison across datasets and perform the
# usual MOABB statistical analysis.

summary = pd.concat(all_results, ignore_index=True)
protocol_summary = (
    summary.groupby(["mode", "pipeline"], as_index=False)
    .agg(
        mean_score=("score", "mean"),
        score_std=("score", "std"),
        n_results=("score", "size"),
        scored_trials=("samples_test", "sum"),
    )
    .fillna({"score_std": 0.0})
)
protocol_summary["scored_trials_per_target"] = protocol_summary["scored_trials"] / len(
    dataset.subject_list
)
print(protocol_summary.to_string(index=False))

mode_order = [
    CrossSubjectMode.TRAIN.value,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P.value,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_50P.value,
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_FULL.value,
]
mode_labels = ["Source only", "20% unlabeled", "50% unlabeled", "Full transductive"]
colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]

fig, (ax_score, ax_trials) = plt.subplots(
    1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.7, 1]}
)
for position, (mode, label, color) in enumerate(
    zip(mode_order, mode_labels, colors, strict=True)
):
    values = summary.loc[summary["mode"] == mode, "score"].to_numpy()
    offsets = np.linspace(-0.12, 0.12, len(values))
    ax_score.scatter(
        position + offsets, values, color=color, alpha=0.65, s=35, label=f"{label} folds"
    )
    ax_score.scatter(
        position,
        values.mean(),
        marker="D",
        color=color,
        edgecolor="black",
        s=90,
        zorder=3,
    )

ordered_summary = protocol_summary.set_index("mode").loc[mode_order]
ax_score.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax_score.set(
    xticks=range(len(mode_labels)),
    xticklabels=mode_labels,
    ylabel="ROC AUC",
    ylim=(0.4, 0.6),
    title="Fold-level scores and means",
)
ax_score.tick_params(axis="x", rotation=20)
ax_score.grid(axis="y", alpha=0.25)

ax_trials.bar(
    mode_labels,
    ordered_summary["scored_trials_per_target"],
    color=colors,
    edgecolor="black",
)
ax_trials.set(
    ylabel="Target trials scored per held-out subject", title="Evaluation-set size"
)
ax_trials.tick_params(axis="x", rotation=20)
ax_trials.grid(axis="y", alpha=0.25)

fig.tight_layout()
plt.show()
