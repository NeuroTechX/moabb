from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.utils import check_random_state


class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject. This splitter assumes that
    all data from all subjects is already known and loaded.

    .. image:: images/withinsess.png
        :alt: The schematic diagram of the WithinSession split
        :align: center


    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2.
    random_state: int, RandomState instance or None, default=None
        Important when `shuffle` is True. Controls the randomness of splits.
        Pass an int for reproducible output across multiple function calls.
    shuffle_session : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    shuffle_subjects : bool, default=False
        Apply shuffle in mixing subjects and sessions, this parameter allows
        sample iterations of the sppliter.

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

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        shuffle_subjects: bool = False,
        shuffle_session: bool = True,
    ):
        self.n_folds = n_folds
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_session = shuffle_session
        self.random_state = check_random_state(random_state)

    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * num_sessions_subjects

    def split(self, y, metadata, **kwargs):
        all_index = metadata.index.values
        subjects = metadata.subject.unique()

        # Shuffle subjects if required
        if self.shuffle_subjects:
            self.random_state.shuffle(subjects)

        for subject in subjects:
            subject_mask = metadata.subject == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            sessions = subject_metadata.session.unique()

            # Shuffle sessions if required
            if self.shuffle_session:
                self.random_state.shuffle(sessions)

            for session in sessions:
                session_mask = subject_metadata.session == session
                indices = subject_indices[session_mask]
                group_y = y[indices]

                # Use StratifiedKFold with the group-specific random state
                cv = StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=self.shuffle_session,
                    random_state=self.random_state,
                )
                for ix_train, ix_test in cv.split(indices, group_y):
                    yield indices[ix_train], indices[ix_test]
