r"""
=====================================================
Geometry-Aware recentering for cross-session transfer
=====================================================
EEG covariance statistics drift from session to session (electrode
repositioning, impedance changes, baseline arousal): the same mental task
produces differently-shaped data on each recording. A classical Riemannian
tangent-space pipeline (:class:`pyriemann.tangentspace.TangentSpace`) is
already a strong baseline for motor-imagery decoding, but its tangent-space
reference point is normally *frozen* at training time — so it does not
correct for this drift. Setting ``tsupdate=True`` re-estimates that reference
point, unsupervised, from each new (unlabelled) evaluation batch: a
**single, label-free correction** cheap enough to put in front of any linear
classifier [1]_.

In a controlled, feature-matched benchmark across eight public MOABB
motor-imagery datasets, Rahimipour, Yang & Van Hulle (in preparation) [1]_
show that this recentering step accounts for a large, statistically decisive
cross-session gain (Cohen's d = 1.06-1.50, all p_FDR < 1.1e-12) over its
recentering-free twin, while the same two pipelines are statistically
indistinguishable within-session (d ~ 0) — a double dissociation showing the
gain is attributable to recentering specifically, not to the choice of final
classifier. The same study found that substantially more complex deep
sequence models (a bidirectional Mamba mixture-of-experts, an SPDNet-style
network), given the *same* covariance features and a fair training budget,
did not recover this gain and in fact underperformed the simple recentering
pipeline in both protocols.

This example reproduces the core comparison — recentering on vs. off — on
the workhorse tangent-space + logistic-regression pipeline, following the
within/cross double-dissociation design of [1]_.

Note
----
Unlike :class:`moabb.datasets.preprocessing.EuclideanAlignment`, which
whitens raw *trials* before any covariance step, this example acts on the
tangent-space *reference point* used to linearise the SPD manifold — the
mechanism is closely related to :class:`pyriemann.transfer.TLCenter`
(matrix/tangent-vector recentering for transfer learning), but is expressed
here directly via ``TangentSpace(tsupdate=True)`` on a single target session,
matching how the study in [1]_ evaluates it under MOABB's
:class:`~moabb.evaluations.CrossSessionEvaluation` /
:class:`~moabb.evaluations.WithinSessionEvaluation` protocols, with no
domain-encoding machinery required.
"""

# Authors: Meysam Rahimipour <rahimipour.2110739@studenti.uniroma1.it>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


moabb.set_log_level("info")
mne.set_log_level("WARNING")  # keep the gallery output readable

###############################################################################
# Build the two pipelines
# ------------------------
#
# Both pipelines are identical except for one flag: whether the tangent-space
# reference point is re-estimated at transform time (``tsupdate=True``,
# "Geometry-Aware") or frozen from the training data ("TS + LR", the
# recentering-free twin). Holding the classifier and covariance estimator
# fixed isolates recentering as the only difference between the two.


def make_geometry_aware(tsupdate):
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann", tsupdate=tsupdate),
        LogisticRegression(max_iter=1000, C=1.0),
    )


pipelines = {
    "Geometry-Aware (recenter)": make_geometry_aware(tsupdate=True),
    "TS + LR (no recenter)": make_geometry_aware(tsupdate=False),
}

###############################################################################
# Cross-session evaluation: where recentering should matter
# ------------------------------------------------------------
#
# :class:`~moabb.evaluations.CrossSessionEvaluation` trains on one session and
# tests on another (MOABB's standard leave-one-session-out protocol) — exactly
# the setting where between-session covariance drift is present for
# recentering to correct.

dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:4]  # keep the example fast
paradigm = LeftRightImagery(fmin=8, fmax=32)

cross_session_eval = CrossSessionEvaluation(
    paradigm=paradigm, datasets=[dataset], overwrite=True, random_state=42
)
cross_results = cross_session_eval.process(pipelines)

###############################################################################
# Within-session evaluation: the control condition
# ---------------------------------------------------
#
# :class:`~moabb.evaluations.WithinSessionEvaluation` trains and tests within
# the *same* recording session, so there is no between-session shift for
# recentering to correct. Per [1]_, recentering should therefore make little
# to no difference here — the mechanism-isolating control for the
# cross-session result above.

within_session_eval = WithinSessionEvaluation(
    paradigm=paradigm, datasets=[dataset], overwrite=True, random_state=42
)
within_results = within_session_eval.process(pipelines)

###############################################################################
# The double dissociation
# -------------------------
#
# Plot the mean score per pipeline in both protocols side by side. Per [1]_,
# the expected pattern is: a clear Geometry-Aware advantage cross-session,
# and near-parity within-session.

cross_means = cross_results.groupby("pipeline")["score"].mean()
within_means = within_results.groupby("pipeline")["score"].mean()

fig, ax = plt.subplots(figsize=(6, 4.5))
x = np.arange(2)
width = 0.35
names = list(pipelines.keys())
for i, name in enumerate(names):
    vals = [within_means.get(name, np.nan), cross_means.get(name, np.nan)]
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(["Within-session\n(no drift)", "Cross-session\n(drift)"])
ax.set_ylabel("Mean score")
ax.set_title("Recentering helps specifically where there is drift to correct")
ax.legend()
fig.tight_layout()
plt.show()

print("Within-session means:\n", within_means)
print("\nCross-session means:\n", cross_results.groupby("pipeline")["score"].mean())

###############################################################################
# Using it inside your own evaluation
# --------------------------------------
#
# The recentering step is a single argument, no new dependency, and drops
# into any tangent-space pipeline::
#
#     from pyriemann.estimation import Covariances
#     from pyriemann.tangentspace import TangentSpace
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.pipeline import make_pipeline
#
#     pipelines = {
#         "Geometry-Aware": make_pipeline(
#             Covariances(estimator="oas"),
#             TangentSpace(metric="riemann", tsupdate=True),
#             LogisticRegression(max_iter=1000),
#         )
#     }
#
# As in the note above, use ``tsupdate=True`` under
# :class:`~moabb.evaluations.CrossSessionEvaluation` /
# :class:`~moabb.evaluations.CrossSubjectEvaluation` (there is a shift to
# correct), and ``tsupdate=False`` under
# :class:`~moabb.evaluations.WithinSessionEvaluation` (there is not) — mixing
# these up is exactly the confound the double-dissociation design in [1]_ is
# built to rule out.
#
# For the full eight-dataset benchmark, deep-model comparison (bidirectional
# Mamba mixture-of-experts, SPDNet), and complete statistical validation
# (Friedman omnibus, FDR/Holm-corrected Wilcoxon, Cohen's d, bootstrap CIs,
# Critical-Difference analysis), see [1]_.
#
# References
# ----------
# .. [1] Rahimipour, M., Yang, L., & Van Hulle, M. Simple Geometric
#        Recentering Rivals Deep Sequence Models for Cross-Session EEG
#        Motor-Imagery Decoding. In preparation, 2026.
