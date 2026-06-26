"""
=========================================================================
Cross-subject transfer learning through the standard CrossSubjectEvaluation
=========================================================================

Target-aware transfer methods (e.g. Riemannian Alignment / RPA) need two things
the ordinary cross-subject split does not give them:

1. which source subject each training trial belongs to, and
2. controlled access to an (unlabeled) slice of the held-out target subject.

Both fit into the **standard** :class:`~moabb.evaluations.CrossSubjectEvaluation`
with three scikit-learn-native pieces -- no dedicated evaluation engine:

* ``cv_kwargs={"calibration_size": ...}`` makes each fold yield
  ``(train, calibration, test)``
  (see :class:`~moabb.evaluations.splitters.CrossSubjectSplitter`);
* **metadata routing** (``set_fit_request``) delivers ``subjects`` and the raw
  calibration slice to the steps that request them;
* ``Pipeline(transform_input=...)`` transforms that raw slice through the earlier
  steps (here ``Covariances``), so a mid-pipeline alignment step receives the
  target in the *same representation* as ``X`` (covariances, not raw epochs).
"""

# License: BSD (3-clause)

import numpy as np
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
from moabb.paradigms import LeftRightImagery


# ---------------------------------------------------------------------------
# pyriemann's ``Covariances`` is stateless and does not mark itself fitted,
# which ``Pipeline(transform_input=...)`` requires (it calls ``transform`` on
# the sub-pipeline, which runs ``check_is_fitted``). One line fixes it; this
# ideally lands upstream in pyriemann.
# ---------------------------------------------------------------------------
class Covariances_(Covariances):
    def __sklearn_is_fitted__(self):
        return True


# ---------------------------------------------------------------------------
# A minimal target-aware step: Riemannian Alignment (recenter each domain to the
# identity). It consumes ``subjects`` (per-source-subject reference) and the
# unlabeled target covariances (target reference). Both come from the evaluation
# via metadata routing.
# ---------------------------------------------------------------------------
class RiemannianAlignment(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, subjects=None, X_target_unlabeled=None):
        X = np.asarray(X)
        self.fit_subjects_ = np.asarray(subjects)
        # one whitening reference per source subject
        self.source_refs_ = {
            s: invsqrtm(mean_riemann(X[self.fit_subjects_ == s]))
            for s in np.unique(self.fit_subjects_)
        }
        # X_target_unlabeled arrives as covariances thanks to transform_input
        self.target_ref_ = (
            invsqrtm(mean_riemann(np.asarray(X_target_unlabeled)))
            if X_target_unlabeled is not None and len(X_target_unlabeled)
            else None
        )
        return self

    def transform(self, X):
        X = np.asarray(X)
        if len(X) == len(self.fit_subjects_):  # training data: per-source-subject
            out = np.empty_like(X)
            for s, ref in self.source_refs_.items():
                m = self.fit_subjects_ == s
                out[m] = ref @ X[m] @ ref
            return out
        # held-out target at predict time: recenter by the target reference
        ref = self.target_ref_
        if ref is None:
            ref = next(iter(self.source_refs_.values()))
        return ref @ X @ ref


def make_transfer_pipeline():
    with config_context(enable_metadata_routing=True):
        align = RiemannianAlignment().set_fit_request(
            subjects=True, X_target_unlabeled=True
        )
    # transform_input -> the raw target slice is passed through Covariances_()
    # before RiemannianAlignment.fit receives it (so it is covariances, not raw).
    return make_pipeline(
        Covariances_("oas"),
        align,
        TangentSpace("riemann"),
        LogisticRegression(max_iter=500),
        transform_input=["X_target_unlabeled"],
    )


if __name__ == "__main__":
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=4, n_sessions=2, seed=42
    )
    paradigm = LeftRightImagery()

    # calibration_size=0.2: the first 20% of each held-out subject is the
    # unlabeled target slice used by RiemannianAlignment; the remaining 80% is
    # scored. calibration_size=0.0 reproduces the ordinary cross-subject split.
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        cv_kwargs={"calibration_size": 0.2},
        overwrite=True,
    )
    results = evaluation.process({"RA+TS+LR": make_transfer_pipeline()})
    print(results[["subject", "session", "pipeline", "score"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Trialwise / one-shot prediction with pure scikit-learn: freeze the
    # source-trained model and predict each held-out trial in isolation via
    # LeaveOneOut (test fold size == 1). No custom wrapper.
    # ------------------------------------------------------------------
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    X, y, meta = paradigm.get_data(dataset=dataset)
    target = meta["subject"].to_numpy() == np.unique(meta["subject"])[-1]
    source_model = make_pipeline(
        Covariances_("oas"), TangentSpace("riemann"), LogisticRegression(max_iter=500)
    ).fit(X[~target], y[~target])
    trialwise = cross_val_predict(
        FrozenEstimator(source_model),
        X[target],
        y[target],
        cv=LeaveOneOut(),
        method="predict",
    )
    print(
        "trialwise == blockwise (inductive):",
        np.array_equal(trialwise, source_model.predict(X[target])),
    )
