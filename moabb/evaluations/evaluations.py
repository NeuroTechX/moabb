import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.protocols import CrossSubjectMode, validate_transfer_protocol
from moabb.evaluations.splitters import (
    _RESOLVED_CV_KEY,
    CrossSessionSplitter,
    CrossSubjectSplitter,
    WithinSessionSplitter,
    WithinSubjectSplitter,
    _ResolvedCV,
    epoch_interval_groups,
)


if TYPE_CHECKING:
    from moabb.datasets.base import BaseDataset

from moabb.evaluations.utils import (
    _average_scores,
    _carbonfootprint,
    _create_scorer,
    _update_result_with_scores,
)


log = logging.getLogger(__name__)


class WithinSessionEvaluation(BaseEvaluation):
    """Performance evaluation within session (k-fold cross-validation)

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject.

    For learning curve evaluation, use ``cv_class=LearningCurveSplitter`` with
    appropriate ``cv_kwargs`` containing ``data_size`` and ``n_perms`` parameters.

    Parameters
    ----------
    paradigm : :class:`~moabb.paradigms.base.BaseParadigm`
        The paradigm to use.
    datasets : list of :class:`~moabb.datasets.base.BaseDataset`
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state : int or None
        If not None, can guarantee same seed for shuffling examples.
        Defaults to ``None``.
    n_jobs : int
        Number of jobs for fitting of pipeline. Defaults to ``1``.
    overwrite : bool
        If true, overwrite the results. Defaults to ``False``.
    error_score : str or float
        Value to assign to the score if an error occurs in estimator fitting. If set to
        ``'raise'``, the error is raised. Defaults to ``"raise"``.
    suffix : str
        Suffix for the results file.
    hdf5_path : str
        Specific path for storing the results and models.
    additional_columns : None
        Adding information to results.
    return_epochs : bool
        Use MNE epoch to train pipelines. Defaults to ``False``.
    return_raws : bool
        Use MNE raw to train pipelines. Defaults to ``False``.
    mne_labels : bool
        If returning MNE epoch, use original dataset label if True.
        Defaults to ``False``.
    cv_class : type or None
        Optional cross-validation class (e.g., LearningCurveSplitter for learning curves).
        Defaults to ``None``.
    cv_kwargs : dict or None
        Keyword arguments for cv_class. Defaults to ``None``.

    """

    _eval_type = "WithinSession"
    _aggregate_folds = True

    def _create_splitter(self):
        """Create the WithinSessionSplitter for parallel evaluation."""
        cv_class, resolved_cv = self._resolve_cv(StratifiedKFold)
        explicit_keys = resolved_cv.explicit_keys
        splitter_kwargs = {
            "n_folds": self.n_splits or 5,
            "shuffle": resolved_cv.inner_kwargs.get("shuffle", True),
            "random_state": resolved_cv.inner_kwargs.get(
                "random_state", self.random_state
            ),
            _RESOLVED_CV_KEY: resolved_cv,
        }
        if not splitter_kwargs["shuffle"] and "random_state" not in explicit_keys:
            splitter_kwargs["random_state"] = None
        groups = self.groups
        if groups is None and getattr(cv_class, "requires_epoch_timing", False):
            groups = epoch_interval_groups
        if groups is not None:
            splitter_kwargs["groups"] = groups
        return WithinSessionSplitter(cv_class=cv_class, **splitter_kwargs)

    # flake8: noqa: C901
    def _evaluate(
        self,
        dataset: "BaseDataset",
        pipelines: dict,
        param_grid: Optional[dict],
        process_pipeline,
        postprocess_pipeline,
    ):
        # Progress Bar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-WithinSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                continue

            X, y, metadata = self._load_data(
                dataset,
                run_pipes,
                process_pipeline,
                postprocess_pipeline,
                subjects=[subject],
            )

            self.cv = self._create_splitter()

            # iterate over sessions
            for session in np.unique(metadata.session):
                ix = metadata.session == session

                for name, clf in run_pipes.items():
                    inner_cv = StratifiedKFold(
                        3, shuffle=True, random_state=self.random_state
                    )

                    # Implement Grid Search
                    grid_clf = clone(clf)
                    grid_clf = self._grid_search(
                        param_grid=param_grid,
                        name=name,
                        grid_clf=grid_clf,
                        inner_cv=inner_cv,
                    )

                    le = LabelEncoder()
                    y_cv = le.fit_transform(y[ix])
                    X_ = X[ix]
                    y_ = y[ix] if self.mne_labels else y_cv
                    meta_ = metadata[ix].reset_index(drop=True)
                    acc = []
                    durations = []
                    test_sizes = []
                    nchan = self._get_nchan(X)

                    if _carbonfootprint:
                        # Initialise CodeCarbon per cross-validation
                        tracker = self.emissions.create_tracker()
                        tracker.start()

                    # Create scorer once before CV loop
                    scorer = _create_scorer(grid_clf, self.paradigm.scoring)

                    per_split = hasattr(self.cv.cv_class, "get_metadata")
                    # Initialize variables for edge case where CV split returns zero iterations
                    duration = 0
                    emissions = np.nan
                    task_name = None
                    for cv_ind, (train, test) in enumerate(self.cv.split(y_, meta_)):
                        cvclf = clone(grid_clf)

                        duration, emissions, task_name = self._fit_cv(
                            cvclf,
                            X_[train],
                            y_[train],
                            tracker if _carbonfootprint else None,
                        )
                        durations.append(duration)
                        self._maybe_save_model_cv(
                            cvclf,
                            dataset,
                            subject,
                            session,
                            name,
                            cv_ind,
                            eval_type="WithinSession",
                        )
                        if per_split:
                            res = self._build_scored_result(
                                dataset,
                                subject,
                                session,
                                name,
                                len(train),
                                nchan,
                                duration,
                                scorer,
                                cvclf,
                                X_[test],
                                y_[test],
                            )
                            if _carbonfootprint:
                                self._attach_emissions(res, emissions, task_name)
                            yield res
                        else:
                            score = scorer(cvclf, X_[test], y_[test])
                            acc.append(score)
                            test_sizes.append(len(test))

                    if _carbonfootprint:
                        tracker.stop()

                    if not per_split:
                        avg_duration = float(np.mean(durations)) if durations else 0.0
                        res = self._build_result(
                            dataset,
                            subject,
                            session,
                            name,
                            len(y_cv),
                            nchan,
                            avg_duration,
                        )
                        res["n_samples_test"] = (
                            int(np.mean(test_sizes)) if test_sizes else 0
                        )
                        res["n_classes"] = len(np.unique(y_cv))
                        _update_result_with_scores(res, _average_scores(acc))
                        if _carbonfootprint:
                            self._attach_emissions(res, emissions, task_name)
                        yield res

    def evaluate(
        self,
        dataset: "BaseDataset",
        pipelines: dict,
        param_grid: Optional[dict],
        process_pipeline,
        postprocess_pipeline=None,
    ):
        yield from self._evaluate(
            dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline
        )

    def is_valid(self, dataset: "BaseDataset") -> bool:
        return True


class CrossSessionEvaluation(BaseEvaluation):
    """Cross-session performance evaluation.

    Evaluate performance of the pipeline across sessions but for a single
    subject. Verifies that there is at least two sessions before starting
    the evaluation.

    Parameters
    ----------
    paradigm : :class:`~moabb.paradigms.base.BaseParadigm`
        The paradigm to use.
    datasets : list of :class:`~moabb.datasets.base.BaseDataset`
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state : int or None
        If not None, can guarantee same seed for shuffling examples.
        Defaults to ``None``.
    n_jobs : int
        Number of jobs for fitting of pipeline. Defaults to ``1``.
    overwrite : bool
        If true, overwrite the results. Defaults to ``False``.
    error_score : str or float
        Value to assign to the score if an error occurs in estimator fitting. If set to
        ``'raise'``, the error is raised. Defaults to ``"raise"``.
    suffix : str
        Suffix for the results file.
    hdf5_path : str
        Specific path for storing the results and models.
    additional_columns : None
        Adding information to results.
    return_epochs : bool
        Use MNE epoch to train pipelines. Defaults to ``False``.
    return_raws : bool
        Use MNE raw to train pipelines. Defaults to ``False``.
    mne_labels : bool
        If returning MNE epoch, use original dataset label if True.
        Defaults to ``False``.
    save_model : bool
        Save model after training, for each fold of cross-validation if needed.
        Defaults to ``False``.
    cache_config : :class:`~moabb.datasets.base.CacheConfig` or None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.
        Defaults to ``None``.

    Notes
    -----
    .. versionadded:: 1.1.0
       Add save_model and cache_config parameters.
    """

    _eval_type = "CrossSession"
    _score_per_session = True

    def _create_splitter(self):
        """Create the CrossSessionSplitter for parallel evaluation."""
        cv_class, resolved_cv = self._resolve_cv(LeaveOneGroupOut)
        splitter_kwargs = {
            "shuffle": resolved_cv.inner_kwargs.get("shuffle", False),
            "random_state": resolved_cv.inner_kwargs.get(
                "random_state", self.random_state
            ),
            _RESOLVED_CV_KEY: resolved_cv,
        }
        if self.groups is not None:
            splitter_kwargs["groups"] = self.groups
        return CrossSessionSplitter(cv_class=cv_class, **splitter_kwargs)

    # flake8: noqa: C901
    def evaluate(
        self,
        dataset: "BaseDataset",
        pipelines: dict,
        param_grid: Optional[dict],
        process_pipeline,
        postprocess_pipeline=None,
    ):
        if not self.is_valid(dataset):
            reason = self._get_incompatibility_reason(dataset)
            raise AssertionError(
                f"Dataset '{dataset.code}' is not appropriate for {self.__class__.__name__}: {reason}"
            )
            # Progressbar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CrossSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                log.info(f"Subject {subject} already processed")
                continue

            X, y, metadata = self._load_data(
                dataset,
                run_pipes,
                process_pipeline,
                postprocess_pipeline,
                subjects=[subject],
            )
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)
            groups = metadata.session.values
            nchan = self._get_nchan(X)

            for name, clf in run_pipes.items():
                # we want to store a results per session
                self.cv = self._create_splitter()
                inner_cv = StratifiedKFold(
                    3, shuffle=True, random_state=self.random_state
                )

                # Implement Grid Search
                grid_clf = clone(clf)
                grid_clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=grid_clf, inner_cv=inner_cv
                )

                if _carbonfootprint:
                    # Initialise CodeCarbon per cross-validation
                    tracker = self.emissions.create_tracker()
                    tracker.start()

                # Create scorer once before CV loop
                scorer = _create_scorer(grid_clf, self.paradigm.scoring)

                for cv_ind, (train, test) in enumerate(self.cv.split(y, metadata)):
                    cvclf = clone(grid_clf)

                    duration, emissions, task_name = self._fit_cv(
                        cvclf, X[train], y[train], tracker if _carbonfootprint else None
                    )
                    self._maybe_save_model_cv(
                        cvclf,
                        dataset,
                        subject,
                        "",
                        name,
                        cv_ind,
                        eval_type="CrossSession",
                    )

                    test_sessions = groups[test]
                    for session in np.unique(test_sessions):
                        session_test = test[test_sessions == session]
                        res = self._build_scored_result(
                            dataset,
                            subject,
                            session,
                            name,
                            len(train),
                            nchan,
                            duration,
                            scorer,
                            cvclf,
                            X[session_test],
                            y[session_test],
                        )

                        if _carbonfootprint:
                            self._attach_emissions(res, emissions, task_name)

                        yield res

                if _carbonfootprint:
                    tracker.stop()

    def is_valid(self, dataset: "BaseDataset") -> bool:
        return dataset.n_sessions > 1

    def _get_incompatibility_reason(self, dataset):
        """Get specific reason for dataset incompatibility."""
        n_sessions = dataset.n_sessions
        if n_sessions <= 1:
            return (
                f"dataset has only {n_sessions} session(s), "
                f"but {self.__class__.__name__} requires at least 2 sessions"
            )
        return "requirements not met"


class CrossSubjectEvaluation(BaseEvaluation):
    """Cross-subject evaluation performance.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    Parameters
    ----------
    paradigm : :class:`~moabb.paradigms.base.BaseParadigm`
        The paradigm to use.
    datasets : list of :class:`~moabb.datasets.base.BaseDataset`
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state : int or None
        If not None, can guarantee same seed for shuffling examples.
        Defaults to ``None``.
    n_jobs : int
        Number of jobs for fitting of pipeline. Defaults to ``1``.
    overwrite : bool
        If true, overwrite the results. Defaults to ``False``.
    error_score : str or float
        Value to assign to the score if an error occurs in estimator fitting. If set to
        ``'raise'``, the error is raised. Defaults to ``"raise"``.
    suffix : str
        Suffix for the results file.
    hdf5_path : str
        Specific path for storing the results and models.
    additional_columns : None
        Adding information to results.
    return_epochs : bool
        Use MNE epoch to train pipelines. Defaults to ``False``.
    return_raws : bool
        Use MNE raw to train pipelines. Defaults to ``False``.
    mne_labels : bool
        If returning MNE epoch, use original dataset label if True.
        Defaults to ``False``.
    save_model : bool
        Save model after training, for each fold of cross-validation if needed.
        Defaults to ``False``.
    cache_config : :class:`~moabb.datasets.base.CacheConfig` or None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.
        Defaults to ``None``.
    n_splits : int or None
        Number of splits for cross-validation. If None, the number of splits
        is equal to the number of subjects. Defaults to ``None``.
    cv_class : type or None
        Cross-validation strategy used to hold out subjects (e.g.
        ``LeaveOneGroupOut``, ``GroupShuffleSplit``, ``GroupKFold``). Defaults to
        ``None`` (``LeaveOneGroupOut``, or ``GroupKFold`` when ``n_splits`` is set).
    cv_kwargs : dict
        Keyword arguments for ``cv_class``. ``calibration_size`` (float in
        ``[0, 1]``, default ``0.0``) enables transfer learning: when ``> 0`` each
        fold becomes ``(train, calibration, test)``. The fraction is taken
        within every held-out subject/session pair so each remains scorable,
        and the calibration slice is routed (raw) to pipeline steps via
        ``set_fit_request``. With ``calibration_labeled=False``, only
        ``X_target_unlabeled`` may be routed. With ``calibration_labeled=True``,
        ``X_target_labeled`` and ``y_target_labeled`` may be routed.
        Labeled calibration is only allowed with ``calibration_size <= 0.5``.
    cs_mode : CrossSubjectMode or str, default=CrossSubjectMode.TRAIN
        Named cross-subject protocol preset. By default, this is the standard
        train-only cross-subject evaluation with no target calibration. The
        ``TRAIN_TRIALWISE`` mode additionally enforces one-trial-at-a-time
        prediction during scoring and supports the built-in ``"accuracy"`` and
        ``"roc_auc"`` metrics. Cannot be combined with manual
        ``calibration_size`` or ``calibration_labeled`` in ``cv_kwargs``, except
        for the default ``TRAIN`` mode.
    splitter : BaseCrossValidator or None
        Optional top-level cross-subject splitter. It must follow MOABB's
        ``split(y, metadata)`` contract and yield unique, in-range,
        one-dimensional positional integer indices into ``y`` and
        ``metadata`` for either train/test or train/calibration/test slices.
        The slices must be pairwise disjoint. Each test fold must contain exactly
        one subject, matching
        MOABB's per-subject result-row semantics. When provided, it replaces
        ``CrossSubjectSplitter`` and cannot
        be combined with ``cv_class``, ``cv_kwargs``, ``n_splits``,
        ``groups``, or a non-default ``cs_mode``. Defaults to ``None``.

    Notes
    -----
    .. versionadded:: 1.1.0
         Add save_model, cache_config and n_splits parameters
    """

    _eval_type = "CrossSubject"
    _score_per_session = True
    _score_per_subject = True
    _needs_all_subjects = True

    def __init__(
        self,
        *args,
        cs_mode=CrossSubjectMode.TRAIN,
        splitter: Optional[BaseCrossValidator] = None,
        **kwargs,
    ):
        if cs_mode is None:
            cs_mode = CrossSubjectMode.TRAIN
        cs_mode = CrossSubjectMode(cs_mode)

        if splitter is not None:
            if not isinstance(splitter, BaseCrossValidator):
                raise TypeError("splitter must be a sklearn BaseCrossValidator instance.")

            conflicts = []
            if kwargs.get("cv_class") is not None:
                conflicts.append("cv_class")
            if kwargs.get("cv_kwargs"):
                conflicts.append("cv_kwargs")
            if kwargs.get("n_splits") is not None:
                conflicts.append("n_splits")
            if kwargs.get("groups") is not None:
                conflicts.append("groups")
            if cs_mode != CrossSubjectMode.TRAIN:
                conflicts.append("cs_mode")
            if conflicts:
                names = ", ".join(conflicts)
                raise ValueError(
                    f"splitter cannot be combined with protocol options: {names}."
                )

            self.splitter = splitter
            self.cs_mode = cs_mode
            self.trialwise = False
            additional_columns = list(kwargs.get("additional_columns") or ())
            for column in getattr(splitter, "metadata_columns", ()):
                if column not in additional_columns:
                    additional_columns.append(column)
            kwargs["additional_columns"] = additional_columns
            super().__init__(*args, **kwargs)
            self._cv_internal_keys = frozenset()
            self._cv_explicit_keys = frozenset()
            return

        self.splitter = None
        cv_kwargs = dict(kwargs.get("cv_kwargs") or {})
        internal_cv_keys = frozenset()
        self.cs_mode = cs_mode

        # Manual cv_kwargs still work when the default train-only blockwise
        # mode is used.
        has_manual_calibration = (
            "calibration_size" in cv_kwargs or "calibration_labeled" in cv_kwargs
        )

        if has_manual_calibration and cs_mode != CrossSubjectMode.TRAIN:
            raise ValueError(
                "Pass either cs_mode or calibration_size/calibration_labeled, not both."
            )

        if not has_manual_calibration:
            cv_kwargs["calibration_size"] = cs_mode.calibration_size
            cv_kwargs["calibration_labeled"] = cs_mode.calibration_labeled
            internal_cv_keys = frozenset({"calibration_size", "calibration_labeled"})

        self.trialwise = cs_mode.trialwise

        validate_transfer_protocol(
            cv_kwargs.get("calibration_size", 0.0),
            cv_kwargs.get("calibration_labeled", False),
        )

        kwargs["cv_kwargs"] = cv_kwargs
        super().__init__(*args, **kwargs)
        self._cv_internal_keys = internal_cv_keys
        self._cv_explicit_keys = frozenset(self.cv_kwargs) - internal_cv_keys

    def _validate_fold_indices(
        self, train_idx, calib_idx, test_idx, *, n_samples, cv_ind
    ):
        if self.splitter is None:
            return

        named_indices = {
            "train": np.asarray(train_idx),
            "calibration": np.asarray(calib_idx),
            "test": np.asarray(test_idx),
        }
        for name, indices in named_indices.items():
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise TypeError(
                    "A top-level CrossSubjectEvaluation splitter must return "
                    f"one-dimensional integer positional indices; fold {cv_ind} "
                    f"{name} indices have shape {indices.shape} and dtype "
                    f"{indices.dtype}."
                )
            if indices.size and (indices.min() < 0 or indices.max() >= n_samples):
                raise ValueError(
                    "A top-level CrossSubjectEvaluation splitter returned "
                    f"out-of-range {name} indices in fold {cv_ind} for "
                    f"{n_samples} samples."
                )
            if np.unique(indices).size != indices.size:
                raise ValueError(
                    "A top-level CrossSubjectEvaluation splitter returned "
                    f"duplicate {name} indices in fold {cv_ind}."
                )

        for left, right in (
            ("train", "calibration"),
            ("train", "test"),
            ("calibration", "test"),
        ):
            if np.intersect1d(named_indices[left], named_indices[right]).size:
                raise ValueError(
                    "A top-level CrossSubjectEvaluation splitter must return "
                    "disjoint train/calibration/test slices; "
                    f"fold {cv_ind} has overlapping {left} and {right} indices."
                )

    def _validate_test_fold_metadata(self, test_metadata):
        super()._validate_test_fold_metadata(test_metadata)
        if self.splitter is None:
            return
        test_subjects = test_metadata["subject"].unique()
        if len(test_subjects) != 1:
            raise ValueError(
                "A top-level CrossSubjectEvaluation splitter must hold out "
                "exactly one subject per test fold because MOABB records one "
                "subject identity per result row; got test subjects "
                f"{test_subjects.tolist()}."
            )

    def _create_splitter(self):
        """Create the top-level splitter for parallel evaluation.

        An explicit ``splitter`` is used directly. Otherwise,
        ``calibration_size`` and ``calibration_labeled`` passed via
        ``cv_kwargs`` configure the default ``CrossSubjectSplitter``.
        """
        if self.splitter is not None:
            return self.splitter

        if self.n_splits is None:
            default_class = LeaveOneGroupOut
            default_kwargs = {}
        else:
            default_class = GroupKFold
            default_kwargs = {"n_splits": self.n_splits}

        cv_class, resolved_cv = self._resolve_cv(default_class, default_kwargs)
        calibration_size = resolved_cv.inner_kwargs.get("calibration_size", 0.0)
        calibration_labeled = resolved_cv.inner_kwargs.get("calibration_labeled", False)
        inner_kwargs = {
            name: value
            for name, value in resolved_cv.inner_kwargs.items()
            if name not in {"calibration_size", "calibration_labeled"}
        }
        resolved_cv = _ResolvedCV(
            inner_kwargs,
            resolved_cv.explicit_keys - {"calibration_size", "calibration_labeled"},
        )
        splitter_kwargs = {
            "random_state": inner_kwargs.get("random_state", self.random_state),
            "calibration_size": calibration_size,
            "calibration_labeled": calibration_labeled,
            _RESOLVED_CV_KEY: resolved_cv,
        }
        if self.groups is not None:
            splitter_kwargs["groups"] = self.groups
        return CrossSubjectSplitter(cv_class=cv_class, **splitter_kwargs)

    def evaluate(
        self,
        dataset: "BaseDataset",
        pipelines: dict,
        param_grid: Optional[dict],
        process_pipeline,
        postprocess_pipeline=None,
    ):
        if not self.is_valid(dataset):
            reason = self._get_incompatibility_reason(dataset)
            raise AssertionError(
                f"Dataset '{dataset.code}' is not appropriate for "
                f"{self.__class__.__name__}: {reason}"
            )
        yield from self._evaluate_parallel_dataset(
            dataset=dataset,
            pipelines=pipelines,
            param_grid=param_grid,
            process_pipeline=process_pipeline,
            postprocess_pipeline=postprocess_pipeline,
        )

    def is_valid(self, dataset: "BaseDataset") -> bool:
        return len(dataset.subject_list) > 1

    def _get_incompatibility_reason(self, dataset):
        """Get specific reason for dataset incompatibility."""
        n_subjects = len(dataset.subject_list)

        if n_subjects <= 1:
            return (
                f"dataset has only {n_subjects} subject(s), "
                f"but {self.__class__.__name__} requires at least 2 subjects"
            )

        return "requirements not met"


class WithinSubjectEvaluation(BaseEvaluation):
    """Within-subject k-fold cross-validation pooling all sessions.

    Pools all sessions of each subject and performs k-fold cross-validation
    on the combined data. Scores are reported per session within each subject,
    averaged across folds.

    This differs from WithinSessionEvaluation (k-fold within each session
    separately) and CrossSessionEvaluation (leave-one-session-out).

    Parameters
    ----------
    paradigm : :class:`~moabb.paradigms.base.BaseParadigm`
        The paradigm to use.
    datasets : list of :class:`~moabb.datasets.base.BaseDataset`
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state : int or None
        If not None, can guarantee same seed for shuffling examples.
        Defaults to ``None``.
    n_jobs : int
        Number of jobs for fitting of pipeline. Defaults to ``1``.
    overwrite : bool
        If true, overwrite the results. Defaults to ``False``.
    error_score : str or float
        Value to assign to the score if an error occurs in estimator fitting. If set to
        ``'raise'``, the error is raised. Defaults to ``"raise"``.
    suffix : str
        Suffix for the results file.
    hdf5_path : str
        Specific path for storing the results and models.
    additional_columns : None
        Adding information to results.
    return_epochs : bool
        Use MNE epoch to train pipelines. Defaults to ``False``.
    return_raws : bool
        Use MNE raw to train pipelines. Defaults to ``False``.
    mne_labels : bool
        If returning MNE epoch, use original dataset label if True.
        Defaults to ``False``.
    save_model : bool
        Save model after training, for each fold of cross-validation if needed.
        Defaults to ``False``.
    cache_config : :class:`~moabb.datasets.base.CacheConfig` or None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig`
        for details. Defaults to ``None``.
    """

    _eval_type = "WithinSubject"
    _aggregate_folds = True
    _score_per_session = True

    def _create_splitter(self):
        """Create the WithinSubjectSplitter for parallel evaluation."""
        cv_class, resolved_cv = self._resolve_cv(StratifiedKFold)
        explicit_keys = resolved_cv.explicit_keys
        splitter_kwargs = {
            "n_folds": self.n_splits or 5,
            "shuffle": resolved_cv.inner_kwargs.get("shuffle", True),
            "random_state": resolved_cv.inner_kwargs.get(
                "random_state", self.random_state
            ),
            _RESOLVED_CV_KEY: resolved_cv,
        }
        if not splitter_kwargs["shuffle"] and "random_state" not in explicit_keys:
            splitter_kwargs["random_state"] = None
        if self.groups is not None:
            splitter_kwargs["groups"] = self.groups
        return WithinSubjectSplitter(cv_class=cv_class, **splitter_kwargs)

    def evaluate(
        self,
        dataset: "BaseDataset",
        pipelines: dict,
        param_grid: Optional[dict],
        process_pipeline,
        postprocess_pipeline=None,
    ):
        if not self.is_valid(dataset):
            reason = self._get_incompatibility_reason(dataset)
            raise AssertionError(
                f"Dataset '{dataset.code}' is not appropriate for "
                f"{self.__class__.__name__}: {reason}"
            )
        yield from self._evaluate_parallel_dataset(
            dataset=dataset,
            pipelines=pipelines,
            param_grid=param_grid,
            process_pipeline=process_pipeline,
            postprocess_pipeline=postprocess_pipeline,
        )

    def is_valid(self, dataset: "BaseDataset") -> bool:
        return True
