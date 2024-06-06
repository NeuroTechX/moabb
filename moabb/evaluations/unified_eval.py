import numpy as np

from typing import Optional, Union

from moabb.evaluations.base import BaseEvaluation
from moabb.evaluations.splitters import CrossSubjectSplitter

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
    SPLITTERS = {'cross_subject': CrossSubjectSplitter,
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


        # Take into account the split
        # for ....

        # Inside, for inference, take into account the meta split

        return

    def is_valid(self, dataset):

        if self.split_method == 'within_subject':
            return True
        elif self.split_method == 'cross_session':
            return dataset.n_sessions > 1
        elif self.split_method == 'cross_subject':
            return len(dataset.subject_list) > 1

