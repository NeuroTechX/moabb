import copy

from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.utils import check_random_state

from moabb.evaluations.metasplitters import PseudoOnlineSplit

class WrapperStratifiedKFold:
    def __init__(self, n_folds=5, shuffle=True, global_seed=42):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.global_seed = global_seed

    def initialize(self, number_subjects, number_sessions):

        for i in range(number_subjects):
            for j in range(number_sessions):

                yield StratifiedKFold(n_splits=self.n_folds,
                                       shuffle=self.shuffle,
                                       random_state= self.global_seed + 10000*i + j)



class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets for each subject in each session. This splitter
    assumes that all data from all subjects is already known and loaded.

    .. image:: docs/source/images/withinsess.png
        :alt: The schematic diagram of the WithinSession split
        :align: center


    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2. If
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of splits. Only used when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    custom_cv: bool, default=False
        Indicates if you are using PseudoOnlineSplit as cv strategy
    calib_size: int, default=None
        Size of calibration set if custom_cv==True
    cv: cros-validation object, default=StratifiedKFold
        Inner cross-validation strategy for splitting the sessions. Be careful, if
        PseudoOnlineSplit is used, it will return calibration and test indexes.


    Examples
    -----------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from moabb.evaluations.splitters import WithinSessionSplitter
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [1,4], [7, 4], [5, 8], [0,3], [2,4]])
    >>> y = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    >>> subjects = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    >>> sessions = np.array(['T', 'T', 'T', 'T', 'E', 'E', 'E', 'E'])
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions})
    >>> csess = WithinSessionSplitter(n_folds=2)
    >>> csess.get_n_splits(metadata)
    4
    >>> for i, (train_index, test_index) in enumerate(csess.split(y, metadata)):
    ...    print(f"Fold {i}:")
    ...    print(f"  Train: index={train_index}, group={subjects[train_index]}, session={sessions[train_index]}")
    ...    print(f"  Test:  index={test_index}, group={subjects[test_index]}, sessions={sessions[test_index]}")
    Fold 0:
      Train: index=[4 5], group=[1 1], session=['E' 'E']
      Test:  index=[6 7], group=[1 1], sessions=['E' 'E']
    Fold 1:
      Train: index=[6 7], group=[1 1], session=['E' 'E']
      Test:  index=[4 5], group=[1 1], sessions=['E' 'E']
    Fold 2:
      Train: index=[0 1], group=[1 1], session=['T' 'T']
      Test:  index=[2 3], group=[1 1], sessions=['T' 'T']
    Fold 3:
      Train: index=[2 3], group=[1 1], session=['T' 'T']
      Test:  index=[0 1], group=[1 1], sessions=['T' 'T']

    """

    def __init__(
        self,
        cv = None,
        custom_cv = False,
        n_folds: int = 5,
        random_state: int = 42,
        global_seed = 66,
        shuffle: bool = True,
        calib_size: int = None
    ):
        self.cv = cv
        self.custom_cv = custom_cv
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = check_random_state(random_state) if shuffle else None
        self.global_seed = global_seed
        self.calib_size = calib_size


    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.cv.get_n_splits(metadata) if self.custom_cv else self.n_folds * num_sessions_subjects

    def split(self, y, metadata, **kwargs):
        all_index = metadata.index.values
        subjects = metadata["subject"].unique()

        # Shuffle subjects if required
        if self.shuffle:
            self.random_state.shuffle(subjects)

        for i, subject in enumerate(subjects):
            subject_mask = metadata.subject == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            sessions = subject_metadata.session.unique()

            # Shuffle sessions if required
            if self.shuffle:
                self.random_state.shuffle(sessions)

            for j, session in enumerate(sessions):
                session_mask = subject_metadata.session == session
                indices = subject_indices[session_mask]
                group_y = y[indices]

                # Handle custom splitter
                if self.custom_cv:
                    if self.cv is None:
                        raise ValueError("Need to pass a custom cv strategy.")
                    splitter = self.cv(calib_size=self.calib_size)
                    for calib_ix, test_ix in splitter.split(indices, group_y, subject_metadata[session_mask]):
                        yield indices[calib_ix], indices[test_ix]
                # If we want to use normal Kfold
                else:
                    # Defining a different cv object per (subject,session)
                    wrapper_cv = WrapperStratifiedKFold(
                        n_folds=self.n_folds,
                        shuffle=self.shuffle,
                        global_seed=self.global_seed,
                    )
                    wrapper_iterator = wrapper_cv.initialize(len(subjects), len(sessions))
                    splitter = next(wrapper_iterator)

                    # Split using the current instance of StratifiedKFold
                    for train_ix, test_ix in splitter.split(indices, group_y):
                        yield indices[train_ix], indices[test_ix]
