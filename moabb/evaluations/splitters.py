import inspect
import logging
from typing import Optional, Union
from warnings import warn

import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.utils import check_random_state


log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]


class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets for each subject in each session. This splitter
    assumes that all data from all subjects is already known and loaded.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/withinsess.png
        :alt: The schematic diagram of the WithinSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses StratifiedKFold.

    Parameters
    ----------
    n_folds : int, default=5
        Number of folds. Must be at least 2. If
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of splits. Only used when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    cv_class: cross-validation class, default=StratifiedKFold
        Inner cross-validation strategy for splitting the sessions.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
        cv_class: type[BaseCrossValidator] = StratifiedKFold,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.random_state = random_state
        self._rng = check_random_state(random_state) if shuffle else None

        if not shuffle and random_state is not None:
            raise ValueError("random_state should be None when shuffle is False")

        # Create a dictionary of parameters by adding arguments only if they
        # are part of the inner cross-validation strategy's signature
        params = inspect.signature(self.cv_class).parameters
        for p, v in [
            ("n_splits", n_folds),
            ("shuffle", shuffle),
            ("random_state", self._rng),
        ]:
            if p in params:
                self._cv_kwargs[p] = v

    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        splitter = self.cv_class(**self._cv_kwargs)
        inner_splits = None
        try:
            inner_splits = splitter.get_n_splits()
        except TypeError:
            try:
                inner_splits = splitter.get_n_splits(None, None, None)
            except (TypeError, ValueError):
                inner_splits = None
        except ValueError:
            inner_splits = None
        if inner_splits is None:
            inner_splits = self.n_folds
        return inner_splits * num_sessions_subjects

    def split(self, y, metadata):
        all_index = metadata.index.values

        # Shuffle subjects if required
        # Convert to numpy array to avoid ArrowStringArray shuffle warning
        subjects = np.array(metadata["subject"].unique())
        if self.shuffle:
            self._rng.shuffle(subjects)

        for subject in subjects:
            subject_mask = metadata["subject"] == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            y_subject = y[subject_mask]

            # Shuffle sessions if required
            # Convert to numpy array to avoid ArrowStringArray shuffle warning
            sessions = np.array(subject_metadata["session"].unique())

            if self.shuffle:
                self._rng.shuffle(sessions)

            for session in sessions:
                session_mask = subject_metadata["session"] == session
                indices = subject_indices[session_mask]
                y_session = y_subject[session_mask]

                # Instantiate a new internal splitter for each session
                splitter = self.cv_class(**self._cv_kwargs)

                # Store reference to the current inner splitter for metadata access
                self._current_splitter = splitter

                # Split using the current instance of StratifiedKFold by default
                for train_ix, test_ix in splitter.split(indices, y_session):

                    yield indices[train_ix], indices[test_ix]


class WithinSubjectSplitter(BaseCrossValidator):
    """Data splitter for within subject evaluation.

    Within-subject evaluation uses k-fold cross-validation to determine train
    and test sets for each subject across all their sessions. This splitter
    assumes that all data from all subjects is already known and loaded.

    Unlike WithinSessionSplitter which performs cross-validation within each session,
    this splitter performs cross-validation across all sessions within each subject.

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses StratifiedKFold.

    Parameters
    ----------
    n_folds : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of splits. Only used when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    cv_class: cross-validation class, default=StratifiedKFold
        Inner cross-validation strategy for splitting within each subject.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
        cv_class: type[BaseCrossValidator] = StratifiedKFold,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.random_state = random_state
        self._rng = check_random_state(random_state) if shuffle else None

        if not shuffle and random_state is not None:
            raise ValueError("random_state should be None when shuffle is False")

        # Create a dictionary of parameters by adding arguments only if they
        # are part of the inner cross-validation strategy's signature
        params = inspect.signature(self.cv_class).parameters
        for p, v in [
            ("n_splits", n_folds),
            ("shuffle", shuffle),
            ("random_state", self._rng),
        ]:
            if p in params:
                self._cv_kwargs[p] = v

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of folds.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """
        num_subjects = metadata["subject"].nunique()
        splitter = self.cv_class(**self._cv_kwargs)
        inner_splits = None
        try:
            inner_splits = splitter.get_n_splits()
        except TypeError:
            try:
                inner_splits = splitter.get_n_splits(None, None, None)
            except (TypeError, ValueError):
                inner_splits = None
        except ValueError:
            inner_splits = None
        if inner_splits is None:
            inner_splits = self.n_folds
        return inner_splits * num_subjects

    def split(self, y, metadata):
        all_index = metadata.index.values

        # Shuffle subjects if required
        # Convert to numpy array to avoid ArrowStringArray shuffle warning
        subjects = np.array(metadata["subject"].unique())
        if self.shuffle:
            self._rng.shuffle(subjects)

        for subject in subjects:
            subject_mask = metadata["subject"] == subject
            subject_indices = all_index[subject_mask]
            y_subject = y[subject_mask]

            # Instantiate a new internal splitter for each subject
            splitter = self.cv_class(**self._cv_kwargs)

            # Store reference to the current inner splitter for metadata access
            self._current_splitter = splitter

            # Split using the cross-validation strategy across all sessions of the subject
            for train_ix, test_ix in splitter.split(subject_indices, y_subject):
                yield subject_indices[train_ix], subject_indices[test_ix]


class CrossSessionSplitter(BaseCrossValidator):
    """Data splitter for cross session evaluation.

    This splitter enables cross-session evaluation by performing a Leave-One-Session-Out (LOSO)
    cross-validation on data from each subject.

    It assumes that the entire metainformation across all subjects is already loaded.

    Unlike the `CrossSessionEvaluation` class from `moabb.evaluation`, which manages
    the complete evaluation process end-to-end, this splitter is solely responsible
    for dividing the data into training and testing sets based on sessions.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/crosssess.jpg
        :alt: The schematic diagram of the CrossSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which performs Leave-One-Session-Out cross-validation.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Inner cross-validation strategy for splitting the sessions of one subject.
        LeaveOneGroupOut is the most common default.
    shuffle: bool, default=False
        Whether to shuffle the session order for each subject. It can only be
        used when changing the `cv_class` to a class compatible with `shuffle`.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the inner cross-validation when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
        For `cv_class` accepting `random_state`, they are provided with a shared rng.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """

    def __init__(
        self,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        shuffle: bool = False,
        random_state: int = None,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.shuffle = shuffle
        self.random_state = random_state

        params = inspect.signature(self.cv_class).parameters
        # When shuffle=True, only allow cv_classes that explicitly support shuffling
        # (i.e., have a 'shuffle' parameter)
        # changing here
        params_key = params.keys()

        if shuffle and ("shuffle" not in params_key and "random_state" not in params_key):
            raise ValueError(
                f"Shuffling is not supported for {cv_class.__name__}. "
                "Choose a different `cv_class` or use `shuffle=False`. "
                "Example of `cv_class`: `GroupShuffleSplit`: "
                "CrossSessionSplitter(shuffle=True, random_state=42, cv_class=GroupShuffleSplit)"
            )

        if not shuffle and "shuffle" in params and random_state is not None:
            raise ValueError(
                "`random_state` should be None when `shuffle` is False for {cv_class.__name__}"
            )

        self._need_rng = "random_state" in params and (shuffle or "shuffle" not in params)

        if "shuffle" in params:
            self._cv_kwargs["shuffle"] = shuffle

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of splits
        of the inner cross-validation strategy.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """
        subjects = metadata["subject"].unique()
        n_splits = 0
        for subject in subjects:
            subject_metadata = metadata.query("subject == @subject")
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                continue  # Skip subjects with only one session

            splitter = self.cv_class(**self._cv_kwargs)
            n_splits += splitter.get_n_splits(
                subject_metadata, groups=subject_metadata["session"]
            )
        return n_splits

    def split(self, y, metadata):
        # here, I am getting the index across all the subject
        all_index = metadata.index.values
        # I check how many subjects are here:
        subjects = metadata["subject"].unique()

        # To make sure that when I shuffle the subject, I shuffle the same way
        # the session when the object is created
        cv_kwargs = {**self._cv_kwargs}  # Copy the original kwargs
        if self._need_rng:
            cv_kwargs["random_state"] = check_random_state(self.random_state)

        # For each subject I am creating the mask to select the subject metainformation.
        for subject in subjects:
            # Creating the subject_mask
            subject_mask = metadata["subject"] == subject
            # from all the index, I am getting the trial index
            subject_indices = all_index[subject_mask]
            # Here, I am getting the metainformation to use the column session
            subject_metadata = metadata[subject_mask]
            # getting the label at subject level
            y_subject = y[subject_mask]
            # check the number of sessions and check how many sessions we
            # have!
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                log.info(
                    f"Skipping subject {subject}: Only one session available. "
                    f"Cross-session evaluation requires at least two sessions."
                )
                continue  # Skip subjects with only one session

            # by default, I am using LeaveOneGroupOut
            splitter = self.cv_class(**cv_kwargs)
            self._current_splitter = splitter

            # Yield the splits for a given subject
            for train_session_idx, test_session_idx in splitter.split(
                X=subject_indices, y=y_subject, groups=subject_metadata["session"]
            ):
                # returning the index
                yield subject_indices[train_session_idx], subject_indices[
                    test_session_idx
                ]


class CrossSubjectSplitter(BaseCrossValidator):
    """Data splitter for cross subject evaluation.

    This splitter enables cross-subject evaluation by performing a Leave-One-Session-Out (LOSO)
    cross-validation on the dataset.

    It assumes that the entire metainformation across all subjects is already loaded.

    Unlike the `CrossSubjectEvaluation` class from `moabb.evaluation`, which manages
    the complete evaluation process end-to-end, this splitter is solely responsible
    for dividing the data into training and testing sets based on subjects.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/crosssubj.png
        :alt: The schematic diagram of the CrossSubject split
        :align: center

    The splitting strategy for the subjects can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which performs Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Cross-validation strategy for splitting the subjects between train and test sets.
        By default, use LeaveOneGroupOut, which keeps one subject as a test.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the cross-validation.
        Pass an int for reproducible output across multiple calls.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """

    def __init__(
        self,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        random_state: int = None,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        params = inspect.signature(self.cv_class).parameters
        if "random_state" in params:
            self._cv_kwargs["random_state"] = random_state

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of splits
        of the inner cross-validation strategy.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """

        splitter = self.cv_class(**self._cv_kwargs)
        n_splits = splitter.get_n_splits(metadata.index, groups=metadata["subject"])
        return n_splits

    def split(self, y, metadata):
        # here, I am getting the index across all the subject
        all_index = metadata.index.values

        splitter = self.cv_class(**self._cv_kwargs)

        # Store reference to the current inner splitter for metadata access
        self._current_splitter = splitter

        # Yield the splits for the entire dataset
        for train_session_idx, test_session_idx in splitter.split(
            X=all_index, y=y, groups=metadata["subject"]
        ):
            # returning the index
            yield all_index[train_session_idx], all_index[test_session_idx]


class LearningCurveSplitter(BaseCrossValidator):
    """Learning curve splitter following sklearn CV interface.

    This splitter creates train/test splits for learning curve evaluation,
    where the training set is progressively subsampled to different sizes
    while the test set remains fixed within each permutation.

    Can be used as cv_class for WithinSessionSplitter, CrossSessionSplitter, etc.

    Parameters
    ----------
    data_size : dict
        Dictionary with 'policy' (either 'ratio' or 'per_class') and 'value'
        (array of sizes). For 'ratio', values should be floats between 0 and 1
        representing the fraction of training data to use. For 'per_class',
        values should be integers representing the number of samples per class.
    n_perms : int or array-like
        Number of permutations per data_size value. If an int, the same number
        of permutations is used for all data sizes. If an array, it should have
        the same length as data_size['value'] and values should be monotonically
        decreasing (more permutations for smaller data sizes).
    test_size : float, default=0.2
        Fraction of data to use for testing.
    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the permutations. Pass an int for
        reproducible output across multiple function calls.

    Attributes
    ----------
    n_splits_ : int
        Total number of splits (permutations x data_size steps).
    _current_perm : int
        Current permutation index (set during split iteration).
    _current_data_size : int
        Current data size (set during split iteration).

    Examples
    --------
    Using LearningCurveSplitter with WithinSessionSplitter:

    >>> from moabb.evaluations.splitters import (
    ...     WithinSessionSplitter, LearningCurveSplitter
    ... )
    >>> import numpy as np
    >>> splitter = WithinSessionSplitter(
    ...     n_folds=1,  # LearningCurveSplitter handles its own permutations
    ...     cv_class=LearningCurveSplitter,
    ...     data_size={'policy': 'ratio', 'value': np.array([0.1, 0.5, 1.0])},
    ...     n_perms=10,
    ...     test_size=0.2,
    ... )

    Notes
    -----
    The splitter yields (train_indices, test_indices) pairs where:
    - The test set is fixed for each permutation
    - The training set is subsampled according to data_size policy

    This replicates the old _evaluate_learning_curve behavior from
    WithinSessionEvaluation.
    """

    VALID_POLICIES = ["per_class", "ratio"]
    metadata_columns = ("data_size", "permutation")

    def __init__(
        self,
        data_size: dict,
        n_perms: Union[int, Vector],
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        # The following are accepted but not used (for compatibility with parent splitters)
        n_splits: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.data_size = data_size
        self.test_size = test_size
        self.random_state = random_state

        # Validate and process data_size
        if data_size is None:
            raise ValueError("data_size must be provided")

        if "policy" not in data_size or "value" not in data_size:
            raise ValueError("data_size must have 'policy' and 'value' keys")

        if data_size["policy"] not in self.VALID_POLICIES:
            raise ValueError(
                f"{data_size['policy']} is not valid. "
                f"Please use one of {self.VALID_POLICIES}"
            )

        data_size_values = np.asarray(data_size["value"])
        if not np.all(np.diff(data_size_values) > 0):
            raise ValueError(
                "data_size['value'] must be sorted in strictly "
                "monotonically increasing order."
            )

        # Validate and process n_perms
        if n_perms is None:
            raise ValueError("n_perms must be provided")

        if isinstance(n_perms, int):
            self.n_perms = np.full_like(data_size_values, n_perms, dtype=int)
        else:
            self.n_perms = np.asarray(n_perms, dtype=int)
            if len(self.n_perms) != len(data_size_values):
                raise ValueError(
                    "Number of elements in n_perms must be equal "
                    "to number of elements in data_size['value']"
                )
            if not np.all(np.diff(self.n_perms) <= 0):
                raise ValueError(
                    "If n_perms is passed as an array, it has to be "
                    "monotonically decreasing"
                )

        # Calculate total number of splits
        self.n_splits_ = int(np.sum(self.n_perms))

        # Store metadata about current split (updated during iteration)
        self._current_perm = None
        self._current_data_size = None

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return total number of splits.

        Parameters
        ----------
        X : array-like, optional
            Ignored.
        y : array-like, optional
            Ignored.
        groups : array-like, optional
            Ignored.

        Returns
        -------
        n_splits : int
            Total number of splits (upper bound; can be lower if some splits
            are skipped due to single-class training sets).
        """
        return self.n_splits_

    def _get_data_size_subsets(self, y):
        """Get indices for each data size step.

        Parameters
        ----------
        y : array-like
            Target labels for the training set.

        Returns
        -------
        indices : list of arrays
            List of index arrays, one for each data size step.
        """
        if self.data_size["policy"] == "ratio":
            vals = np.array(self.data_size["value"])
            if np.any(vals < 0) or np.any(vals > 1):
                raise ValueError("Data subset ratios must be in range [0, 1]")
            upto = np.ceil(vals * len(y)).astype(int)
            indices = [np.array(range(i)) for i in upto]
        elif self.data_size["policy"] == "per_class":
            classwise_indices = {}
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

    def split(self, X, y, groups=None):
        """Generate train/test indices for learning curve evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        groups : array-like of shape (n_samples,), optional
            Group labels. If provided, splits are made on groups (no group
            appears in both train and test).

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        y = np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)

        # Create base train/test splits
        # Use StratifiedShuffleSplit when no groups are provided, otherwise
        # split by groups to avoid session/subject leakage.
        max_perms = int(np.max(self.n_perms))
        if groups is None:
            splitter = StratifiedShuffleSplit(
                n_splits=max_perms,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            base_splits = splitter.split(X, y)
        else:
            splitter = GroupShuffleSplit(
                n_splits=max_perms,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            base_splits = splitter.split(X, y, groups=groups)

        # Generate all permutations
        for perm_i, (train_idx_full, test_idx) in enumerate(base_splits):
            # For this permutation, get the training data labels
            y_train_full = y[train_idx_full]

            # Get subset indices for each data size step
            data_size_steps = self._get_data_size_subsets(y_train_full)

            # Iterate over data size steps
            for di, subset_indices in enumerate(data_size_steps):
                # Skip if we've exceeded n_perms for this data size
                if perm_i >= self.n_perms[di]:
                    continue

                # Get the actual training indices (subset of the full training set)
                train_idx = train_idx_full[subset_indices]

                # Skip splits where training set collapses to single class
                # (can happen with small data_size values or imbalanced datasets)
                if len(np.unique(y[train_idx])) < 2:
                    warn(
                        "Skipping split: training set has only one class "
                        f"(data_size={len(subset_indices)}, perm={perm_i + 1})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                # Update current split metadata
                self._current_perm = perm_i + 1  # 1-indexed for user display
                self._current_data_size = len(subset_indices)

                yield train_idx, test_idx

    def get_metadata(self):
        """Get metadata about the current split.

        Returns
        -------
        metadata : dict
            Dictionary with 'permutation' and 'data_size' keys.
            Only valid during or after iteration.
        """
        return {
            "permutation": self._current_perm,
            "data_size": self._current_data_size,
        }
