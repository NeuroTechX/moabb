from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.utils import check_random_state


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
    cross_val: cros-validation class, default=StratifiedKFold
        Inner cross-validation strategy for splitting the sessions.
    n_folds : int, default=5
        Number of folds. Must be at least 2. If
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of splits. Only used when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    """

    def __init__(
        self,
        cross_val: type[BaseCrossValidator] = StratifiedKFold,
        n_folds: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        self.cross_val = cross_val
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.rng = check_random_state(random_state) if shuffle else None

    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * num_sessions_subjects

    def split(self, y, metadata):
        all_index = metadata.index.values
        subjects = metadata["subject"].unique()

        for subject in subjects:
            subject_mask = metadata["subject"] == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            sessions = subject_metadata["session"].unique()
            y_subject = y[subject_mask]

            # Shuffle sessions if required
            if self.shuffle:
                self.rng.shuffle(sessions)

            for session in sessions:
                session_mask = subject_metadata["session"] == session
                indices = subject_indices[session_mask]
                y_session = y_subject[session_mask]

                splitter = self.cross_val(
                    n_splits=self.n_folds,
                    shuffle=self.shuffle,
                    random_state=self.rng,
                )

                # Split using the current instance of StratifiedKFold by default
                for train_ix, test_ix in splitter.split(indices, y_session):

                    yield indices[train_ix], indices[test_ix]
