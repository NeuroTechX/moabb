import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
)


class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject. This splitter assumes that
    all data from all subjects is already known and loaded.

     . image:: images/withinsess.pdf
    :alt: The schematic diagram of the WithinSession split
    :align: center

    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2.
    random_state: int, RandomState instance or None, default=None
        Important when `shuffle` is True. Controls the randomness of splits.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    Examples
    -----------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from moabb.evaluations.splitters import WithinSessionSplitter
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [1,4], [7, 4], [5, 8], [0,3], [2,4]])
    >>> y = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    >>> subjects = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    >>> sessions = np.array(['T', 'T', 'E', 'E', 'T', 'T', 'E', 'E'])
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions})
    >>> csess = WithinSessionSplitter(n_folds=2)
    >>> csess.get_n_splits(metadata)
    >>> for i, (train_index, test_index) in enumerate(csess.split(X, y, metadata)):
    ...    print(f"Fold {i}:")
    ...    print(f"  Train: index={train_index}, group={subjects[train_index]}, session={sessions[train_index]}")
    ...    print(f"  Test:  index={test_index}, group={subjects[test_index]}, sessions={sessions[test_index]}")
    Fold 0:
      Train: index=[2 7], group=[1 1], session=['E' 'E']
      Test:  index=[3 6], group=[1 1], sessions=['E' 'E']
    Fold 1:
      Train: index=[3 6], group=[1 1], session=['E' 'E']
      Test:  index=[2 7], group=[1 1], sessions=['E' 'E']
    Fold 2:
      Train: index=[4 5], group=[1 1], session=['T' 'T']
      Test:  index=[0 1], group=[1 1], sessions=['T' 'T']
    Fold 3:
      Train: index=[0 1], group=[1 1], session=['T' 'T']
      Test:  index=[4 5], group=[1 1], sessions=['T' 'T']


    """

    def __init__(self, n_folds=5, random_state=42, shuffle=True):

        # Check type
        assert isinstance(n_folds, int)

        self.n_folds = n_folds
        # Setting random state
        self.random_state = random_state
        self.shuffle = shuffle

    def get_n_splits(self, metadata):
        sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * sessions_subjects

    def split(self, X, y, metadata, **kwargs):

        assert isinstance(self.n_folds, int)

        subjects = metadata.subject.values
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle,
                             random_state=self.random_state)

        for subject in np.unique(subjects):
            mask = subjects == subject
            X_, y_, meta_ = (
                X[mask],
                y[mask],
                metadata[mask],
            )

            sessions = meta_.session.values

            for session in np.unique(sessions):
                mask_s = sessions == session
                X_s, y_s, _ = (
                    X_[mask_s],
                    y_[mask_s],
                    meta_[mask_s],
                )

                for ix_train, ix_test in cv.split(X_s, y_s):

                    ix_train_global = np.where(mask)[0][np.where(mask_s)[0][ix_train]]
                    ix_test_global = np.where(mask)[0][np.where(mask_s)[0][ix_test]]
                    yield ix_train_global, ix_test_global


class IndividualWithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject. This splitter does not assume
    that all data and metadata from all subjects is already loaded. If X, y and metadata
    are from a single subject, it returns data split for this subject only.

    It can be used as basis for WithinSessionSplitter or to avoid downloading all data at
    once when it is not needed,

    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2.
    random_state: int, RandomState instance or None, default=None
        Important when `shuffle` is True. Controls the randomness of splits.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    """

    def __init__(self, n_folds=5, random_state=42, shuffle=True):
        # Check type
        assert isinstance(n_folds, int)

        self.n_folds = n_folds
        # Setting random state
        self.random_state = random_state
        self.shuffle = shuffle

    def get_n_splits(self, metadata):
        return self.n_folds

    def split(self, X, y, metadata, **kwargs):

        assert len(np.unique(metadata.subject)) == 1
        assert isinstance(self.n_folds, int)

        sessions = metadata.subject.values
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle,
                             random_state=self.random_state)

        for session in np.unique(sessions):
            mask = sessions == session
            X_, y_, _ = (
                X[mask],
                y[mask],
                metadata[mask],
            )

            for ix_train, ix_test in cv.split(X_, y_):
                yield ix_train, ix_test
