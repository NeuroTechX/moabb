import logging
from copy import deepcopy
from typing import Union

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.splitters import (
    CrossSessionSplitter,
    CrossSubjectSplitter,
    WithinSessionSplitter,
)
from moabb.evaluations.utils import (
    _average_scores,
    _create_scorer,
    _update_result_with_scores,
)


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]


class WithinSessionEvaluation(BaseEvaluation):
    """Performance evaluation within session (k-fold cross-validation)

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject.

    For learning curve evaluation, use ``cv_class=LearningCurveSplitter`` with
    appropriate ``cv_kwargs`` containing ``data_size`` and ``n_perms`` parameters.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results and models.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    cv_class: type, default=None
        Optional cross-validation class (e.g., LearningCurveSplitter for learning curves).
    cv_kwargs: dict, default=None
        Keyword arguments for cv_class.

    """

    # flake8: noqa: C901
    def _evaluate(
        self,
        dataset,
        pipelines,
        param_grid,
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

            cv_class, cv_kwargs = self._resolve_cv(StratifiedKFold)
            self.cv = WithinSessionSplitter(
                n_folds=5,
                shuffle=True,
                random_state=self.random_state,
                cv_class=cv_class,
                **cv_kwargs,
            )

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
                    acc = list()
                    durations = []
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
                        _update_result_with_scores(res, _average_scores(acc))
                        if _carbonfootprint:
                            self._attach_emissions(res, emissions, task_name)
                        yield res

    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        yield from self._evaluate(
            dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline
        )

    def is_valid(self, dataset):
        return True


class CrossSessionEvaluation(BaseEvaluation):
    """Cross-session performance evaluation.

    Evaluate performance of the pipeline across sessions but for a single
    subject. Verifies that there is at least two sessions before starting
    the evaluation.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results and models.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    save_model: bool, default=False
        Save model after training, for each fold of cross-validation if needed
    cache_config: bool, default=None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.

    Notes
    -----
    .. versionadded:: 1.1.0
       Add save_model and cache_config parameters.
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
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
                cv_class, cv_kwargs = self._resolve_cv(LeaveOneGroupOut)
                self.cv = CrossSessionSplitter(
                    cv_class=cv_class, random_state=self.random_state, **cv_kwargs
                )
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
                        cvclf,
                        X[train],
                        y[train],
                        tracker if _carbonfootprint else None,
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

    def is_valid(self, dataset):
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
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results and models.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    save_model: bool, default=False
        Save model after training, for each fold of cross-validation if needed
    cache_config: bool, default=None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.
    n_splits: int, default=None
        Number of splits for cross-validation. If None, the number of splits
        is equal to the number of subjects.

    Notes
    -----
    .. versionadded:: 1.1.0
         Add save_model, cache_config and n_splits parameters
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if not self.is_valid(dataset):
            reason = self._get_incompatibility_reason(dataset)
            raise AssertionError(
                f"Dataset '{dataset.code}' is not appropriate for {self.__class__.__name__}: {reason}"
            )
        # this is a bit awkward, but we need to check if at least one pipe
        # have to be run before loading the data. If at least one pipeline
        # need to be run, we have to load all the data.
        # we might need a better granularity, if we query the DB
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
            dataset,
            run_pipes,
            process_pipeline,
            postprocess_pipeline,
        )
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        # extract metadata
        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)
        nchan = self._get_nchan(X)

        # perform leave one subject out CV
        if self.n_splits is None:
            default_class = LeaveOneGroupOut
            default_kwargs = {}
            adjust_subjects = False
        else:
            default_class = GroupKFold
            default_kwargs = {"n_splits": self.n_splits}
            adjust_subjects = True

        cv_class, cv_kwargs = self._resolve_cv(default_class, default_kwargs)
        if self.cv_class is None and adjust_subjects:
            n_subjects = self.n_splits

        self.cv = CrossSubjectSplitter(
            cv_class=cv_class, random_state=self.random_state, **cv_kwargs
        )

        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

        if _carbonfootprint:
            # Initialise CodeCarbon per cross-validation
            tracker = self.emissions.create_tracker()
            tracker.start()

        # Progressbar at subject level
        for cv_ind, (train, test) in enumerate(
            tqdm(
                self.cv.split(y, metadata),
                total=n_subjects,
                desc=f"{dataset.code}-CrossSubject",
            )
        ):
            subject = groups[test[0]]
            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            # iterate over pipelines
            for name, clf in run_pipes.items():
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
                )
                cvclf = deepcopy(clf)

                duration, emissions, task_name = self._fit_cv(
                    cvclf,
                    X[train],
                    y[train],
                    tracker if _carbonfootprint else None,
                )
                self._maybe_save_model_cv(
                    cvclf,
                    dataset,
                    subject,
                    "",
                    name,
                    cv_ind,
                    eval_type="CrossSubject",
                )

                # Create scorer once per pipeline
                scorer = _create_scorer(cvclf, self.paradigm.scoring)

                # Evaluate on each session
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
                        cvclf,
                        X[test[ix]],
                        y[test[ix]],
                    )

                    if _carbonfootprint:
                        self._attach_emissions(res, emissions, task_name)

                    yield res

        if _carbonfootprint:
            tracker.stop()

    def is_valid(self, dataset):
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
