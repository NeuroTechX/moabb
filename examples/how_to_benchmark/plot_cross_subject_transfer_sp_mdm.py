"""
========================================================
Trialwise cross-subject learning with subject prototypes
========================================================

Some cross-subject methods need to know which source subject each training trial
belongs to, even when they never adapt to the target subject. This tutorial
builds a subject-aware feature extractor and places it inside an official
one-trial-at-a-time MOABB protocol.

By the end of the tutorial, you will know how to:

1. route source-subject identifiers to a pipeline step during ``fit``;
2. construct one Riemannian class prototype per source subject;
3. turn a covariance matrix into a fixed-length distance feature vector;
4. enforce trialwise target prediction through
   :class:`~moabb.evaluations.protocols.CrossSubjectMode`; and
5. make a method reject a target-access protocol it was not designed for.

The feature extractor requests two fit metadata fields:

* ``subjects``: the source-subject label of each training trial;
* ``cs_mode``: the selected cross-subject protocol preset.

``TRAIN_TRIALWISE`` means: train only on source subjects, expose no target
calibration data, and call prediction on one held-out target trial at a time.
That final restriction matters for estimators that could otherwise calculate
statistics from an entire target batch. The baseline uses ``TRAIN`` and
therefore receives the target test block normally.

"""

# Authors: Anton Andonov <toncho11@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.mean import mean_riemann
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import CrossSubjectEvaluation
from moabb.evaluations.protocols import CrossSubjectMode
from moabb.paradigms import LeftRightImagery


###############################################################################
# Define subject-prototype features
# ---------------------------------
#
# Let :math:`G_{s,k}` be the affine-invariant Riemannian mean of the covariance
# matrices from source subject :math:`s` and class :math:`k`. A covariance
# matrix :math:`C` is represented by
#
# .. math::
#
#    \phi(C) =
#    [d_R(C, G_{1,1}), \ldots, d_R(C, G_{S,K})],
#
# where :math:`d_R` is the affine-invariant Riemannian distance. With
# :math:`S` source subjects and :math:`K` classes, a complete fold therefore
# produces :math:`S \times K` features. The subsequent standardization prevents
# a high-variance prototype distance from dominating the linear SVM.


class SubjectPrototypeDistanceFeatures(TransformerMixin, BaseEstimator):
    """Distances to subject-specific class prototypes.

    The transformer learns one Riemannian class prototype per source subject.
    Each trial is then represented by its distance to every available
    subject/class prototype.

    This is a subject-aware, source-only transformer. In this example it is
    intentionally restricted to ``CrossSubjectMode.TRAIN_TRIALWISE``.
    """

    def fit(self, X, y=None, subjects=None, cs_mode=None):
        if cs_mode is None:
            raise ValueError(
                "SubjectPrototypeDistanceFeatures requires `cs_mode` fit metadata. "
                "Use set_fit_request(cs_mode=True)."
            )

        cs_mode = CrossSubjectMode(cs_mode)
        self.cs_mode_ = cs_mode

        if cs_mode != CrossSubjectMode.TRAIN_TRIALWISE:
            raise ValueError(
                "SubjectPrototypeDistanceFeatures supports only "
                f"{CrossSubjectMode.TRAIN_TRIALWISE.value!r} in this example. "
                f"Got {cs_mode.value!r}. "
                "This transformer is demonstrated as a subject-aware source-only "
                "method scored one target trial at a time."
            )

        if subjects is None:
            raise ValueError(
                "SubjectPrototypeDistanceFeatures requires source-subject metadata. "
                "Use set_fit_request(subjects=True)."
            )

        if y is None:
            raise ValueError("SubjectPrototypeDistanceFeatures requires labels `y`.")

        X = np.asarray(X)
        y = np.asarray(y)
        subjects = np.asarray(subjects)

        self.classes_ = np.unique(y)
        self.subjects_ = np.unique(subjects)

        # We compute one prototype per (source subject, class), not just one
        # prototype per subject. Otherwise the features would describe only
        # subject/domain similarity and would not directly encode
        # class-discriminative distances.
        self.prototype_keys_ = []
        self.prototypes_ = {}

        for subject in self.subjects_:
            subject_mask = subjects == subject

            for klass in self.classes_:
                mask = subject_mask & (y == klass)

                if np.any(mask):
                    key = (subject, klass)
                    self.prototype_keys_.append(key)
                    self.prototypes_[key] = mean_riemann(X[mask])

        if not self.prototype_keys_:
            raise ValueError("No subject/class prototypes could be estimated.")

        return self

    def transform(self, X):
        X = np.asarray(X)
        features = np.empty((len(X), len(self.prototype_keys_)))

        for i, cov in enumerate(X):
            for j, key in enumerate(self.prototype_keys_):
                features[i, j] = distance_riemann(cov, self.prototypes_[key])

        return features


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
        suffix=f"subject_prototype_distance_{mode.value}_{pipeline_name}",
    )

    results = evaluation.process({pipeline_name: pipeline}).assign(mode=mode.value)
    view = results[["subject", "session", "pipeline", "score"]]
    print(view.to_string(index=False))
    return results


###############################################################################
# Follow the fit-time information flow
# ------------------------------------
#
# ``subjects`` is metadata: it describes the rows of ``X`` but is not itself a
# model feature. MOABB routes it only to the step that explicitly requests it.
# The covariance estimator, scaler, and classifier remain ordinary
# scikit-learn components. At prediction time, a target covariance needs no
# subject identifier; it is compared with every learned source prototype.

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")

workflow_nodes = [
    (0.03, "Source covariances\n+ subject/class IDs", "#d8ecff"),
    (0.28, "Riemannian mean\nfor each (subject, class)", "#e7ddff"),
    (0.56, "Distances to all\nsource prototypes", "#fff0cc"),
    (0.80, "StandardScaler\n+ linear SVM", "#dff4df"),
]
for x, label, color in workflow_nodes:
    ax.text(
        x,
        0.52,
        label,
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "0.35"},
    )
for start, end in [(0.21, 0.275), (0.49, 0.555), (0.73, 0.795)]:
    ax.annotate(
        "",
        xy=(end, 0.52),
        xytext=(start, 0.52),
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
    0.29,
    0.17,
    "subjects and cs_mode are routed only during fit",
    fontsize=10,
    color="0.25",
)
ax.set_title("Subject-aware feature construction", pad=12)
fig.tight_layout()
plt.show()

###############################################################################
# Understand the trialwise access guarantee
# -----------------------------------------
#
# ``TRAIN`` and ``TRAIN_TRIALWISE`` use exactly the same source-only fit
# information. They differ in what an estimator may observe during prediction:
# the ordinary mode passes the target block to a prediction method in one call,
# whereas the trialwise mode wraps the fitted estimator and makes one call per
# target trial. Predictions are then concatenated before MOABB computes the
# fold-level metric. This prevents accidental use of target-batch means,
# normalization statistics, or other transductive information.

fig, ax = plt.subplots(figsize=(11, 3.2))
trial_positions = np.arange(6)
for y, color in [(1.0, "#d8ecff"), (0.0, "#fff0cc")]:
    ax.scatter(
        trial_positions,
        np.full_like(trial_positions, y, dtype=float),
        marker="s",
        s=900,
        color=color,
        edgecolor="0.35",
    )
    for position in trial_positions:
        ax.text(position, y, f"T{position + 1}", ha="center", va="center", fontsize=9)

ax.plot([-0.45, 5.45], [1.48, 1.48], color="#4c78a8", linewidth=2)
ax.plot([-0.45, -0.45], [1.35, 1.48], color="#4c78a8", linewidth=2)
ax.plot([5.45, 5.45], [1.35, 1.48], color="#4c78a8", linewidth=2)
ax.text(2.5, 1.62, "one prediction call sees the block", ha="center")
ax.text(2.5, -0.48, "six prediction calls, each seeing one trial", ha="center")
ax.set(
    xlim=(-1.2, 6.2),
    ylim=(-0.75, 1.9),
    yticks=[0, 1],
    yticklabels=["TRAIN_TRIALWISE", "TRAIN"],
    title="Target information available to each prediction call",
)
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
plt.show()


###############################################################################
# Configure a small benchmark
# ---------------------------
#
# A deterministic fake dataset makes the tutorial executable without a
# download. The scores are illustrative; the protocol and routing are the focus.

dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=4, n_sessions=2, seed=42)
paradigm = LeftRightImagery()
all_results = []

###############################################################################
# Run the source-only baseline
# ----------------------------
#
# ``TRAIN`` is the ordinary cross-subject protocol and scores the target block
# normally.

baseline_results = run_mode(
    dataset=dataset,
    paradigm=paradigm,
    mode=CrossSubjectMode.TRAIN,
    pipeline_name="SourceOnlyMDM",
    pipeline=make_pipeline(Covariances("oas"), MDM(metric="riemann")),
)
all_results.append(baseline_results)

###############################################################################
# Route source-subject metadata
# -----------------------------
#
# The feature extractor requests source-subject identifiers during ``fit``.
# ``TRAIN_TRIALWISE`` then scores the held-out target one trial at a time,
# without exposing target calibration data.
#
# ``set_fit_request`` declares what the estimator can consume; it does not grant
# access by itself. MOABB first applies the selected protocol and routes only
# metadata that the protocol permits. Requesting ``cs_mode`` lets the estimator
# verify that contract explicitly.

with config_context(enable_metadata_routing=True):
    subject_prototype_features = SubjectPrototypeDistanceFeatures().set_fit_request(
        subjects=True, cs_mode=True
    )

prototype_svm_results = run_mode(
    dataset=dataset,
    paradigm=paradigm,
    mode=CrossSubjectMode.TRAIN_TRIALWISE,
    pipeline_name="SubjectPrototypeSVM",
    pipeline=make_pipeline(
        Covariances("oas"),
        subject_prototype_features,
        StandardScaler(),
        LinearSVC(dual=False, max_iter=5000),
    ),
)
all_results.append(prototype_svm_results)

###############################################################################
# Reject an incompatible protocol
# -------------------------------
#
# This source-only transformer does not adapt to target data. A target
# calibration preset therefore raises an explicit error. This is preferable to
# accepting extra target information silently and later reporting the result
# under the wrong benchmark category.

try:
    with config_context(enable_metadata_routing=True):
        bad_mode_features = SubjectPrototypeDistanceFeatures().set_fit_request(
            subjects=True, cs_mode=True
        )

    run_mode(
        dataset=dataset,
        paradigm=paradigm,
        mode=CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P,
        pipeline_name="SubjectPrototypeSVM",
        pipeline=make_pipeline(
            Covariances("oas"),
            bad_mode_features,
            StandardScaler(),
            LinearSVC(dual=False, max_iter=5000),
        ),
    )
except ValueError as err:
    print("\nExpected error for incompatible mode:")
    print(err)

###############################################################################
# Compare the successful protocols
# --------------------------------
#
# The lines pair results from the same target subject and session. The diamond
# marks the mean for each method. This is a tutorial comparison, not an
# algorithmic claim: the pipelines differ in both their classifier and their
# prediction protocol. Use the *same* pipeline under both modes when the
# scientific question is solely whether blockwise access changes its result.
#
# ``FakeDataset`` contains no class signal, so scores should fluctuate around
# chance. Its role is to make the complete API example fast and reproducible.

summary = pd.concat(all_results, ignore_index=True)
result_summary = (
    summary.groupby(["mode", "pipeline"], as_index=False)
    .agg(
        mean_score=("score", "mean"),
        score_std=("score", "std"),
        n_results=("score", "size"),
        mean_fit_time_s=("time", "mean"),
    )
    .fillna({"score_std": 0.0})
)
print(result_summary.to_string(index=False))

paired_scores = summary.pivot(
    index=["subject", "session"], columns="pipeline", values="score"
)
pipeline_order = ["SourceOnlyMDM", "SubjectPrototypeSVM"]
pipeline_labels = ["Source-only MDM\nTRAIN", "Subject prototypes + SVM\nTRAIN_TRIALWISE"]
colors = ["#4c78a8", "#f58518"]

fig, ax = plt.subplots(figsize=(8, 5))
for scores in paired_scores[pipeline_order].to_numpy():
    ax.plot([0, 1], scores, color="0.75", linewidth=1, alpha=0.8, zorder=1)
for position, (pipeline, color) in enumerate(zip(pipeline_order, colors, strict=True)):
    values = paired_scores[pipeline].to_numpy()
    offsets = np.linspace(-0.07, 0.07, len(values))
    ax.scatter(position + offsets, values, color=color, alpha=0.7, s=40, zorder=2)
    ax.scatter(
        position,
        values.mean(),
        marker="D",
        color=color,
        edgecolor="black",
        s=100,
        zorder=3,
    )

ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
ax.set(
    xticks=[0, 1],
    xticklabels=pipeline_labels,
    ylabel="ROC AUC",
    ylim=(0.4, 0.6),
    title="Paired target-subject/session results",
)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()
