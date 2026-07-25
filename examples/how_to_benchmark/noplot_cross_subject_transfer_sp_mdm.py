"""
=============================================================================
Subject-prototype distance features using CrossSubject TRAIN_TRIALWISE mode
=============================================================================

Some cross-subject methods need to know which source subject each training trial
belongs to. This example implements a simple subject-aware feature extractor:

1. for each source subject, compute one Riemannian prototype per class;
2. represent each trial by its distances to all subject-specific prototypes;
3. train a linear SVM on these subject-prototype distance features.

The available cross-subject modes are defined and documented in
:mod:`moabb.evaluations.protocols`.

This example demonstrates how the standard
:class:`~moabb.evaluations.CrossSubjectEvaluation` can support subject-aware
methods through metadata routing. The feature extractor requests two fit
metadata fields:

* ``subjects``: the source-subject label of each training trial;
* ``cs_mode``: the selected cross-subject protocol preset.

The subject-prototype feature extractor is intentionally restricted here to
TRAIN_TRIALWISE. The example also runs a standard source-only MDM baseline with
TRAIN to compare against the subject-prototype method. Finally, it demonstrates
how the subject-prototype feature extractor can reject a mode that is not
appropriate for this example.

``TRAIN_TRIALWISE`` means: train on source subjects only, use no target
calibration data, and score the held-out target subject one trial at a time.

"""

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
    # 1. Standard source-only MDM baseline.
    #
    # This is the ordinary CrossSubjectEvaluation behavior:
    # train on source subjects, test on the held-out subject, no target
    # calibration data, and score the target block normally.
    # ------------------------------------------------------------------
    baseline_results = run_mode(
        dataset=dataset,
        paradigm=paradigm,
        mode=CrossSubjectMode.TRAIN,
        pipeline_name="SourceOnlyMDM",
        pipeline=make_pipeline(Covariances("oas"), MDM(metric="riemann")),
    )
    all_results.append(baseline_results)

    # ------------------------------------------------------------------
    # 2. Subject-prototype distance features in TRAIN_TRIALWISE mode.
    #
    # The feature extractor receives the source-subject metadata during fit.
    # The evaluation then scores the held-out target subject one trial at a
    # time. No target calibration data is used.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. One intentionally incompatible mode.
    #
    # This transformer is restricted to TRAIN_TRIALWISE in this example.
    # It is not a target-adaptation method, so it rejects modes that provide
    # unlabeled target calibration data.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Compact comparison for the successful runs.
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
