import inspect

import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.utils import check_random_state


class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets for each subject in each session. This splitter
    assumes that all data from all subjects is already known and loaded.

    .. image:: docs/source/images/withinsess.png
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
    cv_class: cros-validation class, default=StratifiedKFold
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
        **cv_kwargs: dict,
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
        return self.n_folds * num_sessions_subjects

    def split(self, y, metadata):
        all_index = metadata.index.values

        # Shuffle subjects if required
        subjects = metadata["subject"].unique()
        if self.shuffle:
            self._rng.shuffle(subjects)

        for subject in subjects:
            subject_mask = metadata["subject"] == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            y_subject = y[subject_mask]

            # Shuffle sessions if required
            sessions = subject_metadata["session"].unique()
            if self.shuffle:
                self._rng.shuffle(sessions)

            for session in sessions:
                session_mask = subject_metadata["session"] == session
                indices = subject_indices[session_mask]
                y_session = y_subject[session_mask]

                # Instantiate a new internal splitter for each session
                splitter = self.cv_class(**self._cv_kwargs)

                # Split using the current instance of StratifiedKFold by default
                for train_ix, test_ix in splitter.split(indices, y_session):

                    yield indices[train_ix], indices[test_ix]


class CrossSubjectSplitter(BaseCrossValidator):
    """Data splitter for cross-subject evaluation.

    Cross-subject evaluation uses Leave-One-Subject-Out cross-validation
    to evaluate performance across different subjects. This splitter
    assumes that data from all subjects is already loaded.

    .. image:: images/crosssubject.png
        :alt: Schematic diagram of the CrossSubject split
        :align: center

    Parameters
    ----------
    shuffle : bool, default=False
        Whether to shuffle the order of subjects.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    cv_class : cross-validation class, default=LeaveOneGroupOut
        Inner cross-validation strategy for splitting subjects.
        By default, LeaveOneGroupOut is appropriate for cross-subject splitting.
    cv_kwargs : dict
        Additional arguments to pass to the inner cross-validation strategy.
    """

    def __init__(
        self,
        shuffle: bool = True,
        random_state: int = None,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        **cv_kwargs: dict,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self.shuffle = shuffle
        self.random_state = random_state

        self._rng = check_random_state(random_state) if shuffle else None
        self._cv_kwargs = dict(cv_kwargs)

        if not shuffle and random_state is not None:
            raise ValueError("`random_state` should be None when `shuffle` is False")

        if self.cv_class == LeaveOneGroupOut and len(self.cv_kwargs) == 0 and shuffle:
            self.cv_class = GroupShuffleSplit

        params = inspect.signature(self.cv_class).parameters
        for p, v in [("shuffle", shuffle), ("random_state", self._rng), ("n_splits", 5)]:
            if p in params:
                self._cv_kwargs[p] = v

    def get_n_splits(self, metadata):
        """Return the number of splits for the cross-validation."""
        # Number of subjects in the dataset
        if self.cv_class == LeaveOneGroupOut:
            return metadata["subject"].nunique()
        else:
            # Number of splits in the inner cross-validation strategy
            return self.cv_kwargs["n_splits"]

    def split(self, y, metadata):
        """Generate indices to split data into training and test set."""

        subjects_index = metadata["subject"].unique()

        if self.shuffle:
            self._rng.shuffle(subjects_index)
            if self._cv_kwargs.get("n_splits") is None:
                self._cv_kwargs["n_splits"] = len(subjects_index)

        splitter = self.cv_class(**self._cv_kwargs)

        for train_subject_idx, test_subject_idx in splitter.split(
            X=np.zeros(len(subjects_index)), y=None, groups=subjects_index
        ):
            train_mask = metadata["subject"].isin(subjects_index[train_subject_idx])
            test_mask = metadata["subject"].isin(subjects_index[test_subject_idx])

            train_idx = metadata.index[train_mask].values
            test_idx = metadata.index[test_mask].values

            yield train_idx, test_idx
