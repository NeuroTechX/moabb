from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut, StratifiedKFold


class WithinSessionValidator(BaseCrossValidator):
    """
    CrossValidator iterator for inside the session validation.

    Within-session validation is an inter-subject validation where
    for each person and each session, we perform Stratified KFold
    cross-validation.

    Here, we implement the train and test generator for the data with
    training and testing sets.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 5.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None, *args, **kwargs):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        super(WithinSessionValidator, self).__init__(*args, **kwargs)

    def _iter_test_masks(self, X=None, y=None, groups=None):
        subject_list = groups["subject"].unique()

        for subject in subject_list:
            groups_subject = groups[groups["subject"] == subject]

            sessions_list = groups_subject["session"].unique()

            for session in sessions_list:
                groups_within = groups_subject[groups_subject["session"] == session]
                subject_indices = groups_within.index

                X_subject_session = X[subject_indices]
                y_subject_session = y[subject_indices]

                cv = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )

                for train_idx, test_idx in cv.split(X_subject_session, y_subject_session):
                    yield train_idx, test_idx

    def split(self, X, y=None, groups=None):
        return self._iter_test_masks(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return (
            len(groups["session"].unique())
            * len(groups["subject"].unique())
            * self.n_splits
        )


class CrossSessionValidator(BaseCrossValidator):
    """
    CrossValidator iterator for cross-session validation.

    Cross-session validator is an inter-subject validation where
    we perform Leave One GroupOut for each person in the sessions.
    So, we have as many folds as the number of sessions for each person.

    Here, we implement the train and test generator for the data with
    training and testing sets.
    Suppose we have two sessions for each subject. The training set is composed
    of all the sessions, the testing set is the other remaining session.
    """

    def __init__(self, *args, **kwargs):
        super(CrossSessionValidator, self).__init__(*args, **kwargs)

    def _iter_test_masks(self, X=None, y=None, groups=None):
        subject_list = groups["subject"].unique()

        for subject in subject_list:
            groups_subject = groups[groups["subject"] == subject]
            subject_indices = groups_subject.index

            X_subject = X[subject_indices]
            y_subject = y[subject_indices]
            session_subject = groups_subject["session"]

            cv = LeaveOneGroupOut()

            for train_idx, test_idx in cv.split(
                X=X_subject, y=y_subject, groups=session_subject
            ):
                yield train_idx, test_idx

    def split(self, X, y=None, groups=None):
        return self._iter_test_masks(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(groups["session"].unique()) * len(groups["subject"].unique())
