import logging
from copy import deepcopy
from time import time
from typing import Optional, Union

import numpy as np
from mne.epochs import BaseEpochs
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list


try:
    from codecarbon import EmissionsTracker

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]


class WithinSessionEvaluation(BaseEvaluation):
    """Performance evaluation within session (k-fold cross-validation)

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject, it is possible to
    estimate the performance on a subset of training examples to obtain
    learning curves.

    Parameters
    ----------
    n_perms :
        Number of permutations to perform. If an array
        is passed it has to be equal in size to the data_size array.
        Values in this array must be monotonically decreasing (performing
        more permutations for more data is not useful to reduce standard
        error of the mean).
        Default: None
    data_size :
        If None is passed, it performs conventional WithinSession evaluation.
        Contains the policy to pick the datasizes to
        evaluate, as well as the actual values. The dict has the
        key 'policy' with either 'ratio' or 'per_class', and the key
        'value' with the actual values as an numpy array. This array should be
        sorted, such that values in data_size are strictly monotonically increasing.
        Default: None
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    n_jobs_evaluation: int, default=1
        Number of jobs for evaluation, processing in parallel the within session,
        cross-session or cross-subject.
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
    """

    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        **kwargs,
    ):
        self.data_size = data_size
        self.n_perms = n_perms
        self.calculate_learning_curve = self.data_size is not None
        if self.calculate_learning_curve:
            # Check correct n_perms parameter
            if self.n_perms is None:
                raise ValueError(
                    "When passing data_size, please also indicate number of permutations"
                )
            if isinstance(n_perms, int):
                self.n_perms = np.full_like(self.data_size["value"], n_perms, dtype=int)
            elif len(self.n_perms) != len(self.data_size["value"]):
                raise ValueError(
                    "Number of elements in n_perms must be equal "
                    "to number of elements in data_size['value']"
                )
            elif not np.all(np.diff(n_perms) <= 0):
                raise ValueError(
                    "If n_perms is passed as an array, it has to be monotonically decreasing"
                )
            # Check correct data size parameter
            if not np.all(np.diff(self.data_size["value"]) > 0):
                raise ValueError(
                    "data_size['value'] must be sorted in strictly monotonically increasing order."
                )
            if data_size["policy"] not in WithinSessionEvaluation.VALID_POLICIES:
                raise ValueError(
                    f"{data_size['policy']} is not valid. Please use one of"
                    f"{WithinSessionEvaluation.VALID_POLICIES}"
                )
            self.test_size = 0.2  # Roughly similar to 5-fold CV
            add_cols = ["data_size", "permutation"]
            super().__init__(additional_columns=add_cols, **kwargs)
        else:
            # Perform default within session evaluation
            super().__init__(**kwargs)

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
                return []

            # get the data
            X, y, metadata = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                postprocess_pipeline=postprocess_pipeline,
            )
            # iterate over sessions
            for session in np.unique(metadata.session):
                ix = metadata.session == session

                for name, clf in run_pipes.items():
                    if _carbonfootprint:
                        # Initialize CodeCarbon
                        tracker = EmissionsTracker(save_to_file=False, log_level="error")
                        tracker.start()
                    t_start = time()
                    cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)
                    inner_cv = StratifiedKFold(
                        3, shuffle=True, random_state=self.random_state
                    )
                    scorer = get_scorer(self.paradigm.scoring)
                    le = LabelEncoder()
                    y_cv = le.fit_transform(y[ix])
                    X_ = X[ix]
                    y_ = y[ix] if self.mne_labels else y_cv

                    grid_clf = clone(clf)

                    # Create folder for grid search results
                    create_save_path(
                        self.hdf5_path,
                        dataset.code,
                        subject,
                        session,
                        name,
                        grid=True,
                        eval_type="WithinSession",
                    )

                    # Implement Grid Search
                    grid_clf = self._grid_search(
                        param_grid=param_grid,
                        name=name,
                        grid_clf=grid_clf,
                        inner_cv=inner_cv,
                    )
                    if self.hdf5_path is not None and self.save_model:
                        model_save_path = create_save_path(
                            self.hdf5_path,
                            dataset.code,
                            subject,
                            session,
                            name,
                            grid=False,
                            eval_type="WithinSession",
                        )

                    if isinstance(X, BaseEpochs):
                        scorer = get_scorer(self.paradigm.scoring)
                        acc = list()
                        X_ = X[ix]
                        y_ = y[ix] if self.mne_labels else y_cv
                        for cv_ind, (train, test) in enumerate(cv.split(X_, y_)):
                            cvclf = clone(grid_clf)
                            cvclf.fit(X_[train], y_[train])
                            acc.append(scorer(cvclf, X_[test], y_[test]))

                            if self.hdf5_path is not None and self.save_model:
                                save_model_cv(
                                    model=cvclf,
                                    save_path=model_save_path,
                                    cv_index=cv_ind,
                                )

                        acc = np.array(acc)
                        score = acc.mean()
                    else:
                        results = cross_validate(
                            grid_clf,
                            X[ix],
                            y_cv,
                            cv=cv,
                            scoring=self.paradigm.scoring,
                            n_jobs=self.n_jobs,
                            error_score=self.error_score,
                            return_estimator=True,
                        )
                        score = results["test_score"].mean()
                        if self.hdf5_path is not None and self.save_model:
                            save_model_list(
                                results["estimator"],
                                score_list=results["test_score"],
                                save_path=model_save_path,
                            )

                    if _carbonfootprint:
                        emissions = tracker.stop()
                        if emissions is None:
                            emissions = np.NaN
                    duration = time() - t_start

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration / 5.0,  # 5 fold CV
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "score": score,
                        "n_samples": len(y_cv),  # not training sample
                        "n_channels": nchan,
                        "pipeline": name,
                    }
                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)

                    yield res

    def get_data_size_subsets(self, y):
        if self.data_size is None:
            raise ValueError(
                "Cannot create data subsets without valid policy for data_size."
            )
        if self.data_size["policy"] == "ratio":
            vals = np.array(self.data_size["value"])
            if np.any(vals < 0) or np.any(vals > 1):
                raise ValueError("Data subset ratios must be in range [0, 1]")
            upto = np.ceil(vals * len(y)).astype(int)
            indices = [np.array(range(i)) for i in upto]
        elif self.data_size["policy"] == "per_class":
            classwise_indices = dict()
            n_smallest_class = np.inf
            for cl in np.unique(y):
                cl_i = np.where(cl == y)[0]
                classwise_indices[cl] = cl_i
                n_smallest_class = (
                    len(cl_i) if len(cl_i) < n_smallest_class else n_smallest_class
                )
            indices = []
            for ds in self.data_size["value"]:
                if ds > n_smallest_class:
                    raise ValueError(
                        f"Smallest class has {n_smallest_class} samples. "
                        f"Desired samples per class {ds} is too large."
                    )
                indices.append(
                    np.concatenate(
                        [classwise_indices[cl][:ds] for cl in classwise_indices]
                    )
                )
        else:
            raise ValueError(f"Unknown policy {self.data_size['policy']}")
        return indices

    def score_explicit(self, clf, X_train, y_train, X_test, y_test):
        if not self.mne_labels:
            # convert labels if array, keep them if epochs and mne_labels is set
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        scorer = get_scorer(self.paradigm.scoring)
        t_start = time()
        try:
            model = clf.fit(X_train, y_train)
            score = _score(model, X_test, y_test, scorer)
        except ValueError as e:
            if self.error_score == "raise":
                raise e
            score = self.error_score
        duration = time() - t_start
        return score, duration

    def _evaluate_learning_curve(
        self, dataset, pipelines, process_pipeline, postprocess_pipeline
    ):
        # Progressbar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-WithinSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                continue

            # get the data
            X_all, y_all, metadata_all = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                postprocess_pipeline=postprocess_pipeline,
            )
            # shuffle_data = True if self.n_perms > 1 else False
            for session in np.unique(metadata_all.session):
                sess_idx = metadata_all.session == session
                X_sess = X_all[sess_idx]
                y_sess = y_all[sess_idx]
                # metadata_sess = metadata_all[sess_idx]
                sss = StratifiedShuffleSplit(
                    n_splits=self.n_perms[0], test_size=self.test_size
                )
                for perm_i, (train_idx, test_idx) in enumerate(sss.split(X_sess, y_sess)):
                    X_train_all = X_sess[train_idx]
                    y_train_all = y_sess[train_idx]
                    X_test = X_sess[test_idx]
                    y_test = y_sess[test_idx]
                    data_size_steps = self.get_data_size_subsets(y_train_all)
                    for di, subset_indices in enumerate(data_size_steps):
                        if perm_i >= self.n_perms[di]:
                            continue
                        not_enough_data = False
                        log.info(
                            f"Permutation: {perm_i + 1},"
                            f" Training samples: {len(subset_indices)}"
                        )

                        if self.return_epochs:
                            X_train = X_train_all[subset_indices]
                        else:
                            X_train = X_train_all[subset_indices, :]
                        y_train = y_train_all[subset_indices]
                        # metadata = metadata_perm[:subset_indices]

                        if len(np.unique(y_train)) < 2:
                            log.warning(
                                "For current data size, only one class" "would remain."
                            )
                            not_enough_data = True
                        nchan = (
                            X_train.info["nchan"]
                            if isinstance(X_train, BaseEpochs)
                            else X_train.shape[1]
                        )
                        for name, clf in run_pipes.items():
                            res = {
                                "dataset": dataset,
                                "subject": subject,
                                "session": session,
                                "n_samples": len(y_train),
                                "n_channels": nchan,
                                "pipeline": name,
                                # Additional columns
                                "data_size": len(subset_indices),
                                "permutation": perm_i + 1,
                            }
                            if not_enough_data:
                                res["time"] = 0
                                res["score"] = np.nan
                            else:
                                res["score"], res["time"] = self.score_explicit(
                                    deepcopy(clf), X_train, y_train, X_test, y_test
                                )
                            yield res

    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if self.calculate_learning_curve:
            yield from self._evaluate_learning_curve(
                dataset, pipelines, process_pipeline, postprocess_pipeline
            )
        else:
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
    n_jobs_evaluation: int, default=1
        Number of jobs for evaluation, processing in parallel the within session,
        cross-session or cross-subject.
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
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
            # Progressbar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CrossSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                print(f"Subject {subject} already processed")
                return []

            # get the data
            X, y, metadata = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                postprocess_pipeline=postprocess_pipeline,
            )
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)
            groups = metadata.session.values
            scorer = get_scorer(self.paradigm.scoring)

            for name, clf in run_pipes.items():
                if _carbonfootprint:
                    # Initialise CodeCarbon
                    tracker = EmissionsTracker(save_to_file=False, log_level="error")
                    tracker.start()

                # we want to store a results per session
                cv = LeaveOneGroupOut()
                inner_cv = StratifiedKFold(
                    3, shuffle=True, random_state=self.random_state
                )

                grid_clf = clone(clf)

                # Implement Grid Search
                grid_clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=grid_clf, inner_cv=inner_cv
                )

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=False,
                        eval_type="CrossSession",
                    )

                for cv_ind, (train, test) in enumerate(cv.split(X, y, groups)):
                    model_list = []
                    if _carbonfootprint:
                        tracker.start()
                    t_start = time()
                    if isinstance(X, BaseEpochs):
                        cvclf = clone(grid_clf)
                        cvclf.fit(X[train], y[train])
                        model_list.append(cvclf)
                        score = scorer(cvclf, X[test], y[test])

                        if self.hdf5_path is not None and self.save_model:
                            save_model_cv(
                                model=cvclf,
                                save_path=model_save_path,
                                cv_index=str(cv_ind),
                            )
                    else:
                        result = _fit_and_score(
                            clone(grid_clf),
                            X,
                            y,
                            scorer,
                            train,
                            test,
                            verbose=False,
                            parameters=None,
                            fit_params=None,
                            error_score=self.error_score,
                            return_estimator=True,
                        )
                        score = result["test_scores"]
                        model_list = result["estimator"]
                    if _carbonfootprint:
                        emissions = tracker.stop()
                        if emissions is None:
                            emissions = 0

                    duration = time() - t_start
                    if self.hdf5_path is not None and self.save_model:
                        save_model_list(
                            model_list=model_list,
                            score_list=score,
                            save_path=model_save_path,
                        )

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": groups[test][0],
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }
                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)

                    yield res

    def is_valid(self, dataset):
        return dataset.n_sessions > 1


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
    n_jobs_evaluation: int, default=1
        Number of jobs for evaluation, processing in parallel the within session,
        cross-session or cross-subject.
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
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        # this is a bit akward, but we need to check if at least one pipe
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

        # get the data
        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            postprocess_pipeline=postprocess_pipeline,
        )

        # encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        # extract metadata
        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)

        scorer = get_scorer(self.paradigm.scoring)

        # perform leave one subject out CV
        cv = LeaveOneGroupOut()
        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

        # Implement Grid Search

        if _carbonfootprint:
            # Initialise CodeCarbon
            tracker = EmissionsTracker(save_to_file=False, log_level="error")

        # Progressbar at subject level
        for cv_ind, (train, test) in enumerate(
            tqdm(
                cv.split(X, y, groups),
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
                if _carbonfootprint:
                    tracker.start()
                t_start = time()
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
                )
                model = deepcopy(clf).fit(X[train], y[train])
                if _carbonfootprint:
                    emissions = tracker.stop()
                    if emissions is None:
                        emissions = 0
                duration = time() - t_start

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=False,
                        eval_type="CrossSubject",
                    )

                    save_model_cv(
                        model=model, save_path=model_save_path, cv_index=str(cv_ind)
                    )
                # we eval on each session
                for session in np.unique(sessions[test]):
                    ix = sessions[test] == session
                    score = _score(model, X[test[ix]], y[test[ix]], scorer)

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)
                    yield res

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1
