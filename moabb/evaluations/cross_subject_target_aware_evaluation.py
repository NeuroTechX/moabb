"""
A cross-subject implementation specifically targeting transfer learning.

It provides a leave-one-subject-out (LOSO) evaluation with controlled access
to the held-out subject. The held-out subject is the target subject being
evaluated. The code allows users to compare results in the domain of transfer
learning BCI.

Currently, it provides 6 modes:

HOS_SOURCE_ONLY_TRIALWISE - Source-only training; each held-out target
                            trial is predicted independently. This mode is
                            the closest to real-world BCI when we do not
                            have enough data.

HOS_SOURCE_ONLY_BLOCKWISE - Source-only training; held-out target trials
                            are predicted as a block, matching standard
                            MOABB behavior. It is a compatibility mode
                            with CrossSubjectEvaluation. Do not use it;
                            use one of the other modes, as they are more
                            clearly defined.

HOS_UNLABELED_20P - First 20% of held-out target trials are used unlabeled
                    for adaptation; the remaining 80% are evaluated.

HOS_UNLABELED_50P - First 50% of held-out target trials are used unlabeled
                    for adaptation; the remaining 50% are evaluated.

HOS_UNLABELED_100P - All held-out target trials are used unlabeled for
                     transductive adaptation and are also evaluated.

HOS_LABELED_20P - First 20% of held-out target trials are used with labels
                  for supervised calibration; the remaining 80% are
                  evaluated.

Important notes:

1) HOS_SOURCE_ONLY_BLOCKWISE vs HOS_UNLABELED_100P

   HOS_SOURCE_ONLY_BLOCKWISE predicts the target-subject samples as a
   block after source-only training. The held-out subject is not provided
   during training.

   By contrast, HOS_UNLABELED_100P explicitly provides the entire
   unlabeled target block during fit/adaptation before predicting it.
   Therefore, HOS_SOURCE_ONLY_BLOCKWISE should be used with care, as it
   can be misused if a pipeline delays training and uses the held-out
   subject as training data, as in HOS_UNLABELED_100P.

2) Labeled target data is not provided to old pipelines

   In HOS_LABELED_20P, target labeled data is only passed if the estimator
   accepts X_target_labeled / y_target_labeled. This means that old/regular
   pipelines continue not to receive any data from the held-out subject.

3) Unlabeled target data also only works for special estimators

   HOS_UNLABEnbLED_20P/50P/100P modes work only if a pipeline step explicitly
   accepts X_target_unlabeled. Regular pipelines will silently ignore the
   target adaptation data, as they are unaware of it.

4) Split order defines adaptation data
   The “first 20%” depends on trial order.

"""

from __future__ import annotations

import inspect
import time
import warnings
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations import CrossSubjectEvaluation, CrossSubjectSplitter
from moabb.evaluations.utils import _create_scorer, _ensure_fitted


@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm on batch completion.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# ---------------------------------------------------------------------
# Target-aware cross-subject modes
# ---------------------------------------------------------------------


class CsMode(Enum):
    HOS_SOURCE_ONLY_TRIALWISE = (
        auto()
    )  # Source-only training; each held-out target trial is predicted independently.
    HOS_SOURCE_ONLY_BLOCKWISE = auto()  # Source-only training; held-out target trials are predicted as a block, matching standard MOABB behavior.
    HOS_UNLABELED_20P = auto()  # First 20% of held-out target trials are used unlabeled for adaptation; remaining 80% are evaluated.
    HOS_UNLABELED_50P = auto()  # First 50% of held-out target trials are used unlabeled for adaptation; remaining 50% are evaluated.
    HOS_UNLABELED_100P = auto()  # All held-out target trials are used unlabeled for transductive adaptation and also evaluated.
    HOS_LABELED_20P = auto()  # First 20% of held-out target trials are used with labels for supervised calibration; remaining 80% are evaluated.


def cs_mode_uses_unlabeled_target(mode: CsMode) -> bool:
    return mode in {
        CsMode.HOS_UNLABELED_20P,
        CsMode.HOS_UNLABELED_50P,
        CsMode.HOS_UNLABELED_100P,
    }


def cs_mode_uses_labeled_target(mode: CsMode) -> bool:
    return mode == CsMode.HOS_LABELED_20P


def cs_mode_target_fraction(mode: CsMode) -> float:
    if mode in {CsMode.HOS_SOURCE_ONLY_TRIALWISE, CsMode.HOS_SOURCE_ONLY_BLOCKWISE}:
        return 0.0

    if mode == CsMode.HOS_UNLABELED_20P:
        return 0.20

    if mode == CsMode.HOS_UNLABELED_50P:
        return 0.50

    if mode == CsMode.HOS_UNLABELED_100P:
        return 1.00

    if mode == CsMode.HOS_LABELED_20P:
        return 0.20

    raise ValueError(f"Unknown CsMode: {mode!r}")


def split_target_for_cs_mode(
    test_idx: np.ndarray, cs_mode: CsMode
) -> tuple[np.ndarray, np.ndarray]:
    test_idx = np.asarray(test_idx, dtype=int)

    if len(test_idx) == 0:
        raise ValueError("Empty held-out test index.")

    fraction = cs_mode_target_fraction(cs_mode)

    if fraction == 0.0:
        return np.array([], dtype=int), test_idx

    if fraction == 1.0:
        return test_idx, test_idx

    n_adapt = int(round(fraction * len(test_idx)))
    n_adapt = max(1, n_adapt)
    n_adapt = min(n_adapt, len(test_idx) - 1)

    target_adapt_idx = test_idx[:n_adapt]
    eval_idx = test_idx[n_adapt:]

    return target_adapt_idx, eval_idx


# ---------------------------------------------------------------------
# Metadata-aware fitting utilities
# ---------------------------------------------------------------------


def estimator_accepts_argument(estimator: Any, method_name: str, arg_name: str) -> bool:
    """
    Return True if an estimator method explicitly accepts a metadata argument.

    This is the only compatibility layer kept here: metadata-aware estimators
    may accept subjects, cs_mode, X_target_unlabeled, X_target_labeled, or
    y_target_labeled, while normal sklearn estimators usually do not.
    """
    method = getattr(estimator, method_name, None)

    if method is None:
        return False

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False

    return arg_name in signature.parameters


def _safe_transform(step: Any, X: Any):
    if X is None:
        return None

    if not hasattr(step, "transform"):
        return X

    return step.transform(X)


def _fit_transform_step_with_metadata(
    step: Any,
    X: Any,
    y: np.ndarray,
    subjects: Optional[np.ndarray] = None,
    cs_mode: Optional[str] = None,
    X_target_unlabeled: Optional[Any] = None,
    X_target_labeled: Optional[Any] = None,
    y_target_labeled: Optional[np.ndarray] = None,
):
    fit_kwargs = {}

    if subjects is not None and estimator_accepts_argument(step, "fit", "subjects"):
        fit_kwargs["subjects"] = subjects

    if cs_mode is not None and estimator_accepts_argument(step, "fit", "cs_mode"):
        fit_kwargs["cs_mode"] = cs_mode

    if X_target_unlabeled is not None and estimator_accepts_argument(
        step, "fit", "X_target_unlabeled"
    ):
        fit_kwargs["X_target_unlabeled"] = X_target_unlabeled

    if X_target_labeled is not None and estimator_accepts_argument(
        step, "fit", "X_target_labeled"
    ):
        fit_kwargs["X_target_labeled"] = X_target_labeled

    if y_target_labeled is not None and estimator_accepts_argument(
        step, "fit", "y_target_labeled"
    ):
        fit_kwargs["y_target_labeled"] = y_target_labeled

    # Use fit_transform only for ordinary sklearn-style steps without metadata.
    # When metadata is passed, call fit() and transform() explicitly because
    # fit_transform() may be implemented separately and may not accept the same
    # metadata arguments as fit().
    if hasattr(step, "fit_transform") and not fit_kwargs:
        Xt = step.fit_transform(X, y)
    else:
        step.fit(X, y, **fit_kwargs)

        if not hasattr(step, "transform"):
            raise TypeError(
                f"Intermediate pipeline step {step!r} has no transform method."
            )

        Xt = step.transform(X)

    Xt_target_unlabeled = _safe_transform(step, X_target_unlabeled)
    Xt_target_labeled = _safe_transform(step, X_target_labeled)

    return Xt, Xt_target_unlabeled, Xt_target_labeled


def fit_pipeline_with_subject_metadata(
    estimator: Any,
    X_train: Any,
    y_train: np.ndarray,
    subjects_train: Optional[np.ndarray] = None,
    cs_mode: Optional[str] = None,
    X_target_unlabeled: Optional[Any] = None,
    X_target_labeled: Optional[Any] = None,
    y_target_labeled: Optional[np.ndarray] = None,
) -> Any:
    """
    Fit an estimator or sklearn Pipeline while passing metadata only to steps
    that explicitly support it.

    Normal sklearn estimators remain supported because unsupported metadata
    arguments are not passed to them.
    """
    if not isinstance(estimator, Pipeline):
        fit_kwargs = {}

        if subjects_train is not None and estimator_accepts_argument(
            estimator, "fit", "subjects"
        ):
            fit_kwargs["subjects"] = subjects_train

        if cs_mode is not None and estimator_accepts_argument(
            estimator, "fit", "cs_mode"
        ):
            fit_kwargs["cs_mode"] = cs_mode

        if X_target_unlabeled is not None and estimator_accepts_argument(
            estimator, "fit", "X_target_unlabeled"
        ):
            fit_kwargs["X_target_unlabeled"] = X_target_unlabeled

        if X_target_labeled is not None and estimator_accepts_argument(
            estimator, "fit", "X_target_labeled"
        ):
            fit_kwargs["X_target_labeled"] = X_target_labeled

        if y_target_labeled is not None and estimator_accepts_argument(
            estimator, "fit", "y_target_labeled"
        ):
            fit_kwargs["y_target_labeled"] = y_target_labeled

        estimator.fit(X_train, y_train, **fit_kwargs)
        return estimator

    Xt_train = X_train
    Xt_target_unlabeled = X_target_unlabeled
    Xt_target_labeled = X_target_labeled

    for _step_name, step in estimator.steps[:-1]:
        (Xt_train, Xt_target_unlabeled, Xt_target_labeled) = (
            _fit_transform_step_with_metadata(
                step=step,
                X=Xt_train,
                y=y_train,
                subjects=subjects_train,
                cs_mode=cs_mode,
                X_target_unlabeled=Xt_target_unlabeled,
                X_target_labeled=Xt_target_labeled,
                y_target_labeled=y_target_labeled,
            )
        )

    _final_name, final_step = estimator.steps[-1]

    final_fit_kwargs = {}

    if subjects_train is not None and estimator_accepts_argument(
        final_step, "fit", "subjects"
    ):
        final_fit_kwargs["subjects"] = subjects_train

    if cs_mode is not None and estimator_accepts_argument(final_step, "fit", "cs_mode"):
        final_fit_kwargs["cs_mode"] = cs_mode

    if Xt_target_unlabeled is not None and estimator_accepts_argument(
        final_step, "fit", "X_target_unlabeled"
    ):
        final_fit_kwargs["X_target_unlabeled"] = Xt_target_unlabeled

    if Xt_target_labeled is not None and estimator_accepts_argument(
        final_step, "fit", "X_target_labeled"
    ):
        final_fit_kwargs["X_target_labeled"] = Xt_target_labeled

    if y_target_labeled is not None and estimator_accepts_argument(
        final_step, "fit", "y_target_labeled"
    ):
        final_fit_kwargs["y_target_labeled"] = y_target_labeled

    final_step.fit(Xt_train, y_train, **final_fit_kwargs)

    return estimator


class TrialwisePredictWrapper(ClassifierMixin, BaseEstimator):
    """
    Wrap an already-fitted estimator and force one-trial-at-a-time prediction.

    This keeps MOABB/sklearn scorer compatibility while preventing the wrapped
    estimator from receiving the full target test block during prediction.

    Has multi-class support.
    """

    _estimator_type = "classifier"

    def __init__(self, fitted_estimator):
        self.fitted_estimator = fitted_estimator
        _ensure_fitted(fitted_estimator)
        self.classes_ = self._get_classes(fitted_estimator)

    def fit(self, X, y=None):
        raise RuntimeError("TrialwisePredictWrapper wraps an already fitted estimator.")

    def predict(self, X):
        return np.asarray(
            [
                self.fitted_estimator.predict(self._slice_one(X, i))[0]
                for i in range(len(X))
            ]
        )

    def predict_proba(self, X):
        if not hasattr(self.fitted_estimator, "predict_proba"):
            raise AttributeError("Wrapped estimator does not provide predict_proba.")

        rows = [
            self._first_row(self.fitted_estimator.predict_proba(self._slice_one(X, i)))
            for i in range(len(X))
        ]

        return np.vstack(rows)

    def decision_function(self, X):
        if not hasattr(self.fitted_estimator, "decision_function"):
            raise AttributeError("Wrapped estimator does not provide decision_function.")

        rows = [
            self._first_row(
                self.fitted_estimator.decision_function(self._slice_one(X, i))
            )
            for i in range(len(X))
        ]

        out = np.asarray(rows)

        # sklearn convention:
        # binary decision_function -> shape (n_samples,)
        # multiclass decision_function -> shape (n_samples, n_classes)
        if out.ndim == 2 and out.shape[1] == 1:
            return out.ravel()

        return out

    @staticmethod
    def _slice_one(X, i):
        return X[i : i + 1]

    @staticmethod
    def _first_row(output):
        arr = np.asarray(output)

        if arr.ndim == 0:
            return arr.item()

        if arr.shape[0] == 1:
            arr = arr[0]

        return arr

    @staticmethod
    def _get_classes(estimator):
        if hasattr(estimator, "classes_"):
            return estimator.classes_

        if hasattr(estimator, "steps"):
            final_estimator = estimator.steps[-1][1]
            if hasattr(final_estimator, "classes_"):
                return final_estimator.classes_

        raise AttributeError("Wrapped estimator does not expose classes_.")


# ---------------------------------------------------------------------
# Main evaluation class
# ---------------------------------------------------------------------


class CrossSubjectTargetAwareEvaluation(CrossSubjectEvaluation):
    _eval_type = "CrossSubjectTargetAware"
    _score_per_session = True
    _needs_all_subjects = True

    def __init__(
        self, *args, cs_mode: CsMode = CsMode.HOS_SOURCE_ONLY_BLOCKWISE, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(cs_mode, CsMode):
            raise ValueError(f"cs_mode must be an instance of CsMode. Got {cs_mode!r}.")

        self.cs_mode = cs_mode

    def _build_task_list(self, dataset, y, metadata, splitter, work_plan, param_grid):
        """
        Build lightweight flattened MOABB 1.6 tasks with target-aware split
        metadata.

        The task dictionary intentionally does not store X, y, metadata, groups,
        or sessions. These are passed separately to the worker to avoid duplicating
        large objects in every joblib task.
        """
        groups = metadata["subject"].values

        tasks = []

        for cv_ind, (train_idx, test_idx) in enumerate(splitter.split(y, metadata)):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)

            subject = groups[test_idx[0]]

            if subject in work_plan:
                subject_key = subject
            elif str(subject) in work_plan:
                subject_key = str(subject)
            else:
                continue

            target_adapt_idx, eval_idx = split_target_for_cs_mode(test_idx, self.cs_mode)

            if len(eval_idx) == 0:
                warnings.warn(
                    f"{dataset.code} | subject={subject}: empty evaluation set "
                    f"for cs_mode={self.cs_mode.name}. Skipping fold.",
                    RuntimeWarning,
                )
                continue

            for pipeline_name, pipeline in work_plan[subject_key].items():
                tasks.append(
                    {
                        "dataset": dataset,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "target_adapt_idx": target_adapt_idx,
                        "eval_idx": eval_idx,
                        "subject": subject,
                        "pipeline_name": pipeline_name,
                        "pipeline": pipeline,
                        "param_grid": param_grid,
                        "cv_ind": cv_ind,
                    }
                )

        return tasks

    def _create_splitter(self):
        """
        Create the MOABB 1.6 cross-subject splitter.

        This delegates subject-level CV handling to MOABB's CrossSubjectSplitter,
        so cv_class, random_state, and cv_kwargs keep the same meaning as in
        CrossSubjectEvaluation.
        """
        cv_kwargs = getattr(self, "cv_kwargs", {}) or {}
        cv_class = getattr(self, "cv_class", None)

        if cv_class is None:
            return CrossSubjectSplitter(random_state=self.random_state, **cv_kwargs)

        return CrossSubjectSplitter(
            cv_class=cv_class, random_state=self.random_state, **cv_kwargs
        )

    def _evaluate_task(self, task, X, y, groups, sessions):
        """
        Evaluate one flattened target-aware task.

        Large shared objects are passed as worker arguments instead of being stored
        inside each task dictionary.
        """
        dataset = task["dataset"]

        train_idx = task["train_idx"]
        test_idx = task["test_idx"]
        target_adapt_idx = task["target_adapt_idx"]
        eval_idx = task["eval_idx"]

        subject = task["subject"]
        name = task["pipeline_name"]
        clf = task["pipeline"]
        param_grid = task["param_grid"]
        cv_ind = task["cv_ind"]

        if param_grid is not None:
            raise NotImplementedError(
                "param_grid/grid search is not supported by "
                "CrossSubjectTargetAwareEvaluation yet. Inner GridSearchCV does "
                "not pass subjects, X_target_unlabeled, X_target_labeled, or "
                "y_target_labeled to inner fits. Please set param_grid=None."
            )

        nchan = self._get_nchan(X)

        cvclf = clone(clf)

        X_train = X[train_idx]

        if self.mne_labels:
            y_train = y[train_idx]
            y_eval_all = y
        else:
            fold_label_idx = np.unique(np.concatenate([train_idx, test_idx]))

            le = LabelEncoder()
            le.fit(y[fold_label_idx])

            y_train = le.transform(y[train_idx])

            y_eval_all = np.empty_like(y, dtype=int)
            y_eval_all[fold_label_idx] = le.transform(y[fold_label_idx])

        subjects_train = groups[train_idx]

        X_target_unlabeled = None
        X_target_labeled = None
        y_target_labeled = None

        n_target_unlabeled = 0
        n_target_labeled = 0

        if cs_mode_uses_unlabeled_target(self.cs_mode):
            X_target_unlabeled = X[target_adapt_idx]
            n_target_unlabeled = int(len(target_adapt_idx))

        elif cs_mode_uses_labeled_target(self.cs_mode):
            X_target_labeled = X[target_adapt_idx]
            y_target_labeled = y_eval_all[target_adapt_idx]
            n_target_labeled = int(len(target_adapt_idx))

        duration = self._fit_estimator_with_target_metadata(
            estimator=cvclf,
            X_train=X_train,
            y_train=y_train,
            subjects_train=subjects_train,
            cs_mode=self.cs_mode.name,
            X_target_unlabeled=X_target_unlabeled,
            X_target_labeled=X_target_labeled,
            y_target_labeled=y_target_labeled,
        )

        self._maybe_save_model_cv(
            cvclf, dataset, subject, "", name, cv_ind, eval_type=self._eval_type
        )

        if self.cs_mode == CsMode.HOS_SOURCE_ONLY_TRIALWISE:
            scoring_estimator = TrialwisePredictWrapper(cvclf)
        else:
            scoring_estimator = cvclf

        scorer = _create_scorer(scoring_estimator, self.paradigm.scoring)

        results = []

        for session in np.unique(sessions[eval_idx]):
            session_eval_idx = eval_idx[sessions[eval_idx] == session]

            if len(session_eval_idx) == 0:
                continue

            res = self._build_scored_result(
                dataset=dataset,
                subject=subject,
                session=session,
                pipeline=name,
                n_samples=len(X_train),
                n_channels=nchan,
                duration=duration,
                scorer=scorer,
                model=scoring_estimator,
                X_test=X[session_eval_idx],
                y_test=y_eval_all[session_eval_idx],
            )

            res["cs_mode"] = self.cs_mode.name
            res["n_source_train"] = int(len(train_idx))
            res["n_source_fit"] = int(len(X_train))
            res["n_train_total"] = int(len(X_train) + n_target_labeled)
            res["n_heldout_total"] = int(len(test_idx))
            res["n_target_adapt"] = int(len(target_adapt_idx))
            res["n_target_eval"] = int(len(eval_idx))
            res["n_target_unlabeled"] = int(n_target_unlabeled)
            res["n_target_labeled"] = int(n_target_labeled)
            res["target_subject"] = subject

            if self.cs_mode == CsMode.HOS_SOURCE_ONLY_TRIALWISE:
                res["predict_mode"] = "trialwise"
            else:
                res["predict_mode"] = "blockwise"

            results.append(res)

        return results

    def _evaluate_parallel_dataset(
        self,
        dataset,
        pipelines,
        param_grid,
        process_pipeline,
        postprocess_pipeline,
        work_plan,
    ):
        """
        MOABB > 1.6-native flattened parallel evaluation.

        One task = one held-out subject fold x one pipeline.

        The task dictionaries are kept lightweight. Large shared objects are passed
        separately to the worker.
        """
        from joblib import Parallel, delayed

        if param_grid is not None:
            raise NotImplementedError(
                "param_grid/grid search is not supported by "
                "CrossSubjectTargetAwareEvaluation yet. Inner GridSearchCV does "
                "not pass subjects, X_target_unlabeled, X_target_labeled, or "
                "y_target_labeled to inner fits. Please set param_grid=None."
            )

        subjects_to_load = (
            dataset.subject_list
            if getattr(self, "_needs_all_subjects", False)
            else list(work_plan.keys())
        )

        run_pipes = {
            name: pipe
            for subject_pipelines in work_plan.values()
            for name, pipe in subject_pipelines.items()
        }

        X, y_raw, metadata = self._load_data(
            dataset=dataset,
            run_pipes=run_pipes,
            process_pipeline=process_pipeline,
            postprocess_pipeline=postprocess_pipeline,
            subjects=subjects_to_load,
        )

        y = np.asarray(y_raw)

        groups = metadata["subject"].values
        sessions = metadata["session"].values

        splitter = self._create_splitter()

        tasks = self._build_task_list(
            dataset=dataset,
            y=y,
            metadata=metadata,
            splitter=splitter,
            work_plan=work_plan,
            param_grid=param_grid,
        )

        if not tasks:
            return []

        desc = f"{dataset.code}-{self._eval_type}"

        if self.n_jobs == 1:
            nested_results = []

            for task in tqdm(
                tasks, total=len(tasks), desc=desc, unit="task", dynamic_ncols=True
            ):
                nested_results.append(
                    self._evaluate_task(
                        task=task, X=X, y=y, groups=groups, sessions=sessions
                    )
                )

        else:
            with tqdm_joblib(
                tqdm(total=len(tasks), desc=desc, unit="task", dynamic_ncols=True)
            ):
                nested_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(self._evaluate_task)(
                        task=task, X=X, y=y, groups=groups, sessions=sessions
                    )
                    for task in tasks
                )

        all_results = []

        for rows in nested_results:
            all_results.extend(rows)

        return all_results

    def _fit_estimator_with_target_metadata(
        self,
        estimator: Any,
        X_train: Any,
        y_train: np.ndarray,
        subjects_train: Optional[np.ndarray] = None,
        cs_mode: Optional[str] = None,
        X_target_unlabeled: Optional[Any] = None,
        X_target_labeled: Optional[Any] = None,
        y_target_labeled: Optional[np.ndarray] = None,
    ):
        """
        Fit one estimator with optional target-aware metadata.
        """
        start_time = time.time()

        fit_pipeline_with_subject_metadata(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            subjects_train=subjects_train,
            cs_mode=cs_mode,
            X_target_unlabeled=X_target_unlabeled,
            X_target_labeled=X_target_labeled,
            y_target_labeled=y_target_labeled,
        )

        _ensure_fitted(estimator)

        duration = time.time() - start_time
        return duration
