from copy import deepcopy
from time import time

import numpy as np

from typing import Optional, Union

from mne import BaseEpochs
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations import create_save_path, save_model_cv
from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.splitters import CrossSubjectSplitter

try:
    from codecarbon import EmissionsTracker

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

Vector = Union[list, tuple, np.ndarray]


class GroupEvaluation(BaseEvaluation):
    """Perform specific evaluation based on a given data splitter.

    Possible modes:
    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject, it is possible to
    estimate the performance on a subset of training examples to obtain
    learning curves.

    Cross-session evaluation uses a Leave-One-Group-Out cross-validation to
    evaluate performance across sessions, but for a single subject.

    Cross-subject evaluation also uses Leave-One-Subject-Out to evaluate performance
    on a pipeline trained in all subjects but one.

    It's also possible to determine how test data is being used
    .... meta splitters

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
    SPLITTERS = {'CrossSubject': CrossSubjectSplitter,
                 }
    META_SPLITTERS = {}

    def __init__(
            self,
            split_method,
            meta_split_method,
            n_folds=None,
            n_perms: Optional[Union[int, Vector]] = None,
            data_size: Optional[dict] = None,
            calib_size: Optional[int] = None,
            **kwargs,
    ):
        self.data_size = data_size
        self.n_perms = n_perms
        self.split_method = split_method
        self.meta_split_method = meta_split_method
        self.n_folds = n_folds
        self.calib_size = calib_size

        # Check if splitters are valid
        if self.split_method not in self.SPLITTERS:
            raise ValueError(f"{self.split_method} does not correspond to a valid data splitter."
                             f"Please use one of {self.SPLITTERS.keys()}")

        if self.meta_split not in self.META_SPLITTERS:
            raise ValueError(f"{self.meta_split} does not correspond to a valid evaluation split."
                             f"Please use one of {self.META_SPLITTERS.keys()}")

        # Initialize splitter
        self.data_splitter = self.SPLITTERS[self.split_method](self.n_folds)

        # If SamplerSplit
        if self.meta_split_method == 'sampler':

            # Check if data_size is defined
            if self.data_size is None:
                raise ValueError(
                    "Please pass data_size parameter with the policy and values for the evaluation"
                    "split."
                )

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

            if data_size["policy"] not in self.VALID_POLICIES:
                raise ValueError(
                    f"{data_size['policy']} is not valid. Please use one of"
                    f"{self.VALID_POLICIES}"
                )

            self.test_size = 0.2  # Roughly similar to 5-fold CV

            # TODO: Initialize Meta Splitter
            # self.meta_splitter ......

            add_cols = ["data_size", "permutation"]
            super().__init__(additional_columns=add_cols, **kwargs)

        else:

            # Initialize Meta Splitter
            # self.meta_splitter ......

            super().__init__(**kwargs)

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):

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

        # Get data spliter type
        splitter = self.data_splitter

        # Like this, I will need to lead all data before looping through subjects
        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            cache_config=self.cache_config,
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

        # Define inner cv
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        if _carbonfootprint:
            # Initialise CodeCarbon
            tracker = EmissionsTracker(save_to_file=False, log_level="error")

        # Progressbar at subject level
        for cv_ind, (train, test) in enumerate(
            tqdm(
                splitter.split(X, y, groups),
                total=n_subjects,
                desc=f"{dataset.code}-{self.split_method}",
            )
        ):

            subject = groups[test[0]]

            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                print(f"Subject {subject} already processed")
                return []

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
                        eval_type=f"{self.split_method}",
                    )

                    save_model_cv(
                        model=model, save_path=model_save_path, cv_index=str(cv_ind)
                    )

                # Remove and use the meta splitters
                # Now, for evaluation, we will need to use the new metasplitters Offline or TimeSeries
                X_test, y_test, meta_test = X[test], y[test], metadata[test]

                meta_splitter = self.meta_split_method
                for test_split in meta_splitter.split(X_test, y_test, meta_test):
                    score = _score(model, X_test[test_split], y_test[test_split], self.scorer)

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": meta_test[test_split].session_name,
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)
                    yield res


    def is_valid(self, dataset):

        if self.split_method == 'within_subject':
            return True
        elif self.split_method == 'cross_session':
            return dataset.n_sessions > 1
        elif self.split_method == 'cross_subject':
            return len(dataset.subject_list) > 1


class LazyEvaluation(BaseEvaluation):
    """Perform specific evaluation based on a given data splitter.

    Possible modes:
    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject, it is possible to
    estimate the performance on a subset of training examples to obtain
    learning curves.

    Cross-session evaluation uses a Leave-One-Group-Out cross-validation to
    evaluate performance across sessions, but for a single subject.

    Cross-subject evaluation also uses Leave-One-Subject-Out to evaluate performance
    on a pipeline trained in all subjects but one.

    It's also possible to determine how test data is being used
    .... meta splitters

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
    SPLITTERS = {'CrossSubject': CrossSubjectSplitter,
                 }
    META_SPLITTERS = {}

    def __init__(
            self,
            split_method,
            eval_split_method,
            n_folds=None,
            meta_split_method=None,
            n_perms: Optional[Union[int, Vector]] = None,
            data_size: Optional[dict] = None,
            calib_size: Optional[int] = None,
            **kwargs,
    ):
        self.data_size = data_size
        self.n_perms = n_perms
        self.split_method = split_method
        self.meta_split_method = meta_split_method
        self.n_folds = n_folds
        self.calib_size = calib_size

        # Check if splitters are valid
        if self.split_method not in self.SPLITTERS:
            raise ValueError(f"{self.split_method} does not correspond to a valid data splitter."
                             f"Please use one of {self.SPLITTERS.keys()}")

        if self.meta_split_method not in self.META_SPLITTERS:
            raise ValueError(f"{self.meta_split_method} does not correspond to a valid evaluation split."
                             f"Please use one of {self.META_SPLITTERS.keys()}")

        # Initialize splitter
        self.data_splitter = self.SPLITTERS[self.split_method](self.n_folds)
        self.meta_splitter = self.META_SPLITTERS[self.split_method](self.n_folds)

        # If SamplerSplit
        if self.meta_split_method == 'sampler':

            # Check if data_size is defined
            if self.data_size is None:
                raise ValueError(
                    "Please pass data_size parameter with the policy and values for the evaluation"
                    "split."
                )

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

            if data_size["policy"] not in self.VALID_POLICIES:
                raise ValueError(
                    f"{data_size['policy']} is not valid. Please use one of"
                    f"{self.VALID_POLICIES}"
                )

            self.test_size = 0.2  # Roughly similar to 5-fold CV

            # self.meta_splitter ......

            add_cols = ["data_size", "permutation"]
            super().__init__(additional_columns=add_cols, **kwargs)

        else:

            # Initialize Meta Splitter
            # self.meta_splitter ......

            super().__init__(**kwargs)

    def evaluate(self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None):

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

        if _carbonfootprint:
            # Initialise CodeCarbon
            tracker = EmissionsTracker(save_to_file=False, log_level="error")

        self.scorer = get_scorer(self.paradigm.scoring)

        # Define inner cv
        self.inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        subjects_list = dataset.subject_list
        # Avoid downloading everything at once if possible
        if self.split_method == 'CrossSubject':
            subjects = subjects_list
            self._evaluate(dataset, pipelines, param_grid, process_pipeline,
                           subjects=subjects, tracker=None, postprocess_pipeline=None)
        else:

            for subject in tqdm(subjects_list, desc=f"{dataset.code}-{self.split_method}"):
                subjects = [subject]
                run_pipes = self.results.not_yet_computed(
                    pipelines, dataset, subject, process_pipeline
                )
                self._evaluate(dataset, pipelines, param_grid, process_pipeline,
                               subjects=subjects, tracker=None, postprocess_pipeline=None)

    def _evaluate(self, dataset, pipelines, param_grid, process_pipeline, subjects, tracker=None, postprocess_pipeline=None):

        # Get data spliter type
        splitter = self.data_splitter

        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            subjects=subjects,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            cache_config=self.cache_config,
            postprocess_pipeline=postprocess_pipeline,
        )

        # encode labels
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        # extract metadata
        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(groups)

        for cv_ind, (train, test) in enumerate(
                tqdm(
                    splitter.split(X, y, metadata),
                    total=self.split_method.get_n_splits(metadata),
                    desc=f"{dataset.code}-{self.split_method}",
                )
        ):

            if self.split_method == 'CrossSubject':
                subject = groups[test[0]]
            else:
                subject = subjects[0]

            # now we can check if this subject has results
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                print(f"Subject {subject} already processed")
                return []

            # iterate over pipelines
            for name, clf in run_pipes.items():

                # Start tracker
                if _carbonfootprint:
                    tracker.start()

                t_start = time()
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=self.inner_cv
                )
                model = deepcopy(clf).fit(X[train], y[train])

                # Check carbon emissions
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
                        eval_type=f"{self.split_method}",
                    )

                    save_model_cv(
                        model=model, save_path=model_save_path, cv_index=str(cv_ind)
                    )

                # Remove and use the meta splitters
                # Now, for evaluation, we will need to use the new metasplitters Offline or TimeSeries
                X_test, y_test, meta_test = X[test], y[test], metadata[test]

                meta_splitter = self.meta_split_method
                for test_split in meta_splitter.split(X_test, y_test, meta_test):
                    score = _score(model, X_test[test_split], y_test[test_split], self.scorer)

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": meta_test[test_split].session_name,
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }

                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)
                    yield res

    def is_valid(self, dataset):

        if self.split_method == 'within_subject':
            return True
        elif self.split_method == 'cross_session':
            return dataset.n_sessions > 1
        elif self.split_method == 'cross_subject':
            return len(dataset.subject_list) > 1

