r"""
==========================================================
Comparing fitted and test-batch tangent-space references
==========================================================
Riemannian tangent-space projection maps symmetric positive-definite covariance
matrices to vectors around a reference matrix [1]_, [2]_. With
:class:`pyriemann.tangentspace.TangentSpace`, ``tsupdate=False`` estimates that
reference during ``fit`` and reuses it during ``transform``. By contrast,
``tsupdate=True`` estimates a reference from all matrices passed to each
``transform`` call.

The updated setting does not use evaluation labels, but it is transductive:
each prediction depends on the other samples in the same evaluation batch.
Its output can therefore depend on batch size and composition, and pyRiemann
documents it as incompatible with online use. This example reports both
settings for four subjects from one public motor-imagery dataset under MOABB's
within-session and cross-session protocols. It is an illustration, not a
benchmark or evidence that either reference policy is generally better.

Note
----
This example changes only the reference used by the tangent-space projection.
It is distinct from :class:`moabb.datasets.preprocessing.EuclideanAlignment`,
which whitens raw trials before covariance estimation, and from domain-aware
transfer estimators such as :class:`pyriemann.transfer.TLCenter`.
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
# Both pipelines are identical except for whether the tangent-space reference
# comes from the evaluation batch or the fitted training data. Holding the
# covariance estimator and classifier fixed makes that policy the only pipeline
# difference in this illustrative run.


def make_ts_pipeline(tsupdate):
    """Build a tangent-space pipeline with a fitted or updated reference."""
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann", tsupdate=tsupdate),
        LogisticRegression(max_iter=1000, C=1.0),
    )


pipelines = {
    "Updated test-batch reference": make_ts_pipeline(tsupdate=True),
    "Frozen training reference": make_ts_pipeline(tsupdate=False),
}

###############################################################################
# Cross-session evaluation
# ------------------------
#
# :class:`~moabb.evaluations.CrossSessionEvaluation` trains on one session and
# tests on another using MOABB's leave-one-session-out protocol. We record the
# observed scores without prespecifying which reference policy should win.

dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:4]  # keep the example fast
paradigm = LeftRightImagery(fmin=8, fmax=32)

cross_session_eval = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    suffix="geometry_aware",
    overwrite=True,
    random_state=42,
)
cross_results = cross_session_eval.process(pipelines)

###############################################################################
# Within-session evaluation
# -------------------------
#
# :class:`~moabb.evaluations.WithinSessionEvaluation` trains and tests within
# each recording session. This supplies a second evaluation context for the
# same two pipeline settings; it is not a no-shift control condition.

within_session_eval = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    suffix="geometry_aware",
    overwrite=True,
    random_state=42,
)
within_results = within_session_eval.process(pipelines)

###############################################################################
# Observed scores
# ---------------
#
# Plot the mean score per pipeline in both protocols side by side. These means
# summarize this four-subject run only; no inferential statistic is computed.

cross_means = cross_results.groupby("pipeline")["score"].mean()
within_means = within_results.groupby("pipeline")["score"].mean()

fig, ax = plt.subplots(figsize=(6, 4.5))
x = np.arange(2)
width = 0.35
names = list(pipelines.keys())
for i, name in enumerate(names):
    vals = [within_means[name], cross_means[name]]
    ax.bar(x + i * width, vals, width, label=name)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(["Within-session", "Cross-session"])
ax.set_ylabel("Mean score")
ax.set_title("Fitted vs test-batch tangent-space reference")
ax.legend()
fig.tight_layout()
plt.show()

print("Within-session means:\n", within_means)
print("\nCross-session means:\n", cross_means)

###############################################################################
# Choosing a reference policy
# ---------------------------
#
# The policy is controlled by one argument::
#
#     from pyriemann.estimation import Covariances
#     from pyriemann.tangentspace import TangentSpace
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.pipeline import make_pipeline
#
#     pipelines = {
#         "Updated test-batch reference": make_pipeline(
#             Covariances(estimator="oas"),
#             TangentSpace(metric="riemann", tsupdate=True),
#             LogisticRegression(max_iter=1000),
#         ),
#         "Frozen training reference": make_pipeline(
#             Covariances(estimator="oas"),
#             TangentSpace(metric="riemann", tsupdate=False),
#             LogisticRegression(max_iter=1000),
#         ),
#     }
#
# Use ``tsupdate=True`` only when transductive access to the complete prediction
# batch matches the intended deployment and evaluation contract. Use
# ``tsupdate=False`` when predictions must depend only on fitted training state,
# including online or independently processed samples. The evaluation class
# alone does not determine that choice.
#
# References
# ----------
# .. [1] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. Multiclass
#        Brain-Computer Interface Classification by Riemannian Geometry.
#        IEEE Transactions on Biomedical Engineering, 59(4):920-928, 2012.
#        doi:10.1109/TBME.2011.2172210.
# .. [2] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. Classification of
#        covariance matrices using a Riemannian-based kernel for BCI
#        applications. Neurocomputing, 112:172-178, 2013.
#        doi:10.1016/j.neucom.2012.12.039.
