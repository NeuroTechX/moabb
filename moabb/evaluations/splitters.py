import numpy as np
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.utils import check_random_state


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
    4
    >>> for i, (train_index, test_index) in enumerate(csess.split(y, metadata)):
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

    def __init__(self, n_folds: int = 5, random_state: int = 42, shuffle: bool = True):
        # Check type
        assert isinstance(n_folds, int)

        self.n_folds = n_folds
        # Setting random state
        self.random_state = check_random_state(random_state) if shuffle else None
        self.shuffle = shuffle

    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * num_sessions_subjects

    def split(self, y, metadata, **kwargs):

        assert isinstance(self.n_folds, int)

        all_index = metadata.index.values
        subjects = metadata.subject.values
        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state
        )

        for subject in np.unique(subjects):
            subject_mask = subjects == subject
            subject_indices, subject_y, subject_metadata = (
                all_index[subject_mask],
                y[subject_mask],
                metadata[subject_mask],
            )

            sessions = subject_metadata.session.values

            for session in np.unique(sessions):
                session_mask = sessions == session
                session_indices, session_y = (
                    subject_indices[session_mask],
                    subject_y[session_mask],
                )

                for ix_train, ix_test in cv.split(session_indices, session_y):
                    yield session_indices[ix_train], session_indices[ix_test]
