import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import (
    BaseEvaluation,
    _one_shot_estimator,
    _route_transfer_metadata,
)
from moabb.evaluations.protocols import (
    CrossSubjectMode,
    is_trialwise_mode,
    resolve_cross_subject_mode,
    validate_transfer_protocol,
)
from moabb.evaluations.splitters import (
    CrossSessionSplitter,
    CrossSubjectSplitter,
    WithinSessionSplitter,
    WithinSubjectSplitter,
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
        cv_class, cv_kwargs = self._resolve_cv(StratifiedKFold)
        if self.groups is not None:
            cv_kwargs = {**cv_kwargs, "groups": self.groups}
        return WithinSessionSplitter(
            n_folds=self.n_splits or 5,
            shuffle=True,
            random_state=self.random_state,
            cv_class=cv_class,
            **cv_kwargs,
        )

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

    def _create_splitter(self):
        """Create the CrossSessionSplitter for parallel evaluation."""
        cv_class, cv_kwargs = self._resolve_cv(LeaveOneGroupOut)
        if self.groups is not None:
            cv_kwargs = {**cv_kwargs, "groups": self.groups}
        return CrossSessionSplitter(
            cv_class=cv_class, random_state=self.random_state, **cv_kwargs
        )

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

                    res = self._build_scored_result(
                        dataset,
                        subject,
                        groups[test][0],
                        name,
                        len(train),
                        nchan,
                        duration,
                        scorer,
                        cvclf,
                        X[test],
                        y[test],
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
        fold becomes ``(train, calibration, test)`` and the held-out calibration
        slice is routed (raw) to the pipeline steps that request it via
        ``set_fit_request``. With ``calibration_labeled=False``, only
        ``X_target_unlabeled`` may be routed. With ``calibration_labeled=True``,
        ``X_target_labeled`` and ``y_target_labeled`` may be routed.
        Labeled calibration is only allowed with ``calibration_size <= 0.5``.
    cs_mode : CrossSubjectMode or str, default=CrossSubjectMode.TRAIN
        Named cross-subject protocol preset. By default, this is the standard
        train-only cross-subject evaluation with no target calibration. The
        ``TRAIN_TRIALWISE`` mode additionally enforces one-trial-at-a-time
        prediction during scoring. Cannot be combined with manual
        ``calibration_size`` or ``calibration_labeled`` in ``cv_kwargs``, except
        for the default ``TRAIN`` mode.

    Notes
    -----
    .. versionadded:: 1.1.0
         Add save_model, cache_config and n_splits parameters
    """

    _eval_type = "CrossSubject"
    _score_per_session = True
    _needs_all_subjects = True

    def __init__(self, *args, cs_mode=CrossSubjectMode.TRAIN, **kwargs):
        cv_kwargs = dict(kwargs.get("cv_kwargs") or {})
        self.one_shot_predict = False

        if cs_mode is None:
            cs_mode = CrossSubjectMode.TRAIN

        cs_mode = CrossSubjectMode(cs_mode)
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
            params = resolve_cross_subject_mode(cs_mode)
            cv_kwargs["calibration_size"] = params["calibration_size"]
            cv_kwargs["calibration_labeled"] = params["calibration_labeled"]

        self.one_shot_predict = is_trialwise_mode(cs_mode)

        validate_transfer_protocol(
            cv_kwargs.get("calibration_size", 0.0),
            cv_kwargs.get("calibration_labeled", False),
        )

        kwargs["cv_kwargs"] = cv_kwargs
        super().__init__(*args, **kwargs)

    def _create_splitter(self):
        """Create the CrossSubjectSplitter for parallel evaluation.

        ``calibration_size`` and ``calibration_labeled`` passed via
        ``cv_kwargs`` turn each fold into a transfer split:

        ``(train, calibration, test)``.
        """
        if self.n_splits is None:
            default_class = LeaveOneGroupOut
            default_kwargs = {}
        else:
            default_class = GroupKFold
            default_kwargs = {"n_splits": self.n_splits}

        cv_class, cv_kwargs = self._resolve_cv(default_class, default_kwargs)
        if self.groups is not None:
            cv_kwargs = {**cv_kwargs, "groups": self.groups}
        return CrossSubjectSplitter(
            cv_class=cv_class, random_state=self.random_state, **cv_kwargs
        )

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
                f"Dataset '{dataset.code}' is not appropriate for "
                f"{self.__class__.__name__}: {reason}"
            )

        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(
                self.results.not_yet_computed(
                    pipelines, dataset, subject, process_pipeline
                )
            )

        if len(run_pipes) == 0:
            return

        X, y, metadata = self._load_data(
            dataset, run_pipes, process_pipeline, postprocess_pipeline
        )

        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)
        nchan = self._get_nchan(X)

        self.cv = self._create_splitter()

        if self.n_splits is not None and self.cv_class is None:
            n_subjects = self.n_splits

        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

        if _carbonfootprint:
            tracker = self.emissions.create_tracker()
            tracker.start()

        for cv_ind, (train, *cal, test) in enumerate(
            tqdm(
                self.cv.split(y, metadata),
                total=n_subjects,
                desc=f"{dataset.code}-CrossSubject",
            )
        ):
            calib = cal[0] if cal else train[:0]
            subject = groups[test[0]]

            split_metadata = None
            if hasattr(self.cv, "get_metadata"):
                split_metadata = self.cv.get_metadata()
                if split_metadata is not None:
                    split_metadata = dict(split_metadata)

            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )

            for name, clf in run_pipes.items():
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
                )
                cvclf = clone(clf)

                calib_md = None
                if len(calib):
                    calib_md = {"X": X[calib], "y": y[calib]}

                calibration_labeled = False
                if split_metadata is not None:
                    calibration_labeled = bool(
                        split_metadata.get("calibration_labeled", False)
                    )

                fit_params = _route_transfer_metadata(
                    cvclf,
                    groups[train],
                    calib=calib_md,
                    calibration_labeled=calibration_labeled,
                    cs_mode=self.cs_mode,
                )

                duration, emissions, task_name = self._fit_cv(
                    cvclf,
                    X[train],
                    y[train],
                    tracker if _carbonfootprint else None,
                    fit_params=fit_params,
                )

                self._maybe_save_model_cv(
                    cvclf, dataset, subject, "", name, cv_ind, eval_type="CrossSubject"
                )

                score_estimator = (
                    _one_shot_estimator(cvclf) if self.one_shot_predict else cvclf
                )
                scorer = _create_scorer(score_estimator, self.paradigm.scoring)

                for session in np.unique(sessions[test]):
                    ix = sessions[test] == session

                    res = self._build_scored_result(
                        dataset,
                        subject,
                        session,
                        name,
                        len(train),
                        nchan,
                        duration,
                        scorer,
                        score_estimator,
                        X[test[ix]],
                        y[test[ix]],
                        split_metadata=split_metadata,
                    )

                    if _carbonfootprint:
                        self._attach_emissions(res, emissions, task_name)

                    yield res

        if _carbonfootprint:
            tracker.stop()

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
        cv_class, cv_kwargs = self._resolve_cv(StratifiedKFold)
        if self.groups is not None:
            cv_kwargs = {**cv_kwargs, "groups": self.groups}
        return WithinSubjectSplitter(
            n_folds=self.n_splits or 5,
            shuffle=True,
            random_state=self.random_state,
            cv_class=cv_class,
            **cv_kwargs,
        )

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
