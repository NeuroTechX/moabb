import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
)


class WithinSubjectSplitter(BaseCrossValidator):
    """ Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets on separate session for each subject. This splitter assumes that
    all data from all subjects is already known and loaded.

    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2.

    """

    def __init__(self, n_folds: int):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * sessions_subjects

    def split(self, X, y, metadata, **kwargs):

        subjects = metadata.subject.values

        split = IndividualWithinSubjectSplitter(self.n_folds)

        for subject in np.unique(subjects):

            X_, y_, meta_ = (
                X[subjects == subject],
                y[subjects == subject],
                metadata[subjects == subject],
            )

            yield split.split(X_, y_, meta_, **kwargs)


class IndividualWithinSubjectSplitter(BaseCrossValidator):
    """ Data splitter for within session evaluation.

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

    """
    def __init__(self, n_folds: int):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        return self.n_folds

    def split(self, X, y, metadata, **kwargs):

        sessions = metadata.subject.values

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, **kwargs)

        for session in np.unique(sessions):
            X_, y_, meta_ = (
                X[sessions == session],
                y[sessions == session],
                metadata[sessions == session],
            )

            for ix_train, ix_test in cv.split(X_, y_):

                yield ix_train, ix_test


class CrossSessionSplitter(BaseCrossValidator):
    """ Data splitter for cross session evaluation.

    Cross-session evaluation uses a Leave-One-Group-Out cross-validation to
    evaluate performance across sessions, but for a single subject. This splitter
    assumes that all data from all subjects is already known and loaded.

    Parameters
    ----------
    n_folds :
        Not used. For compatibility with other cross-validation splitters.
        Default:None

    """

    def __init__(self, n_folds=None):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        sessions_subjects = len(metadata.groupby(["subject", "session"]).first())
        return sessions_subjects

    def split(self, X, y, metadata):

        subjects = metadata.subject.values
        split = IndividualCrossSessionSplitter()

        for subject in np.unique(subjects):
            X_, y_, meta_ = (
                X[subjects == subject],
                y[subjects == subject],
                metadata[subjects == subject],
            )

            yield split.split(X_, y_, meta_)


class IndividualCrossSessionSplitter(BaseCrossValidator):
    """ Data splitter for cross session evaluation.

    Cross-session evaluation uses a Leave-One-Group-Out cross-validation to
    evaluate performance across sessions, but for a single subject. This splitter does
    not assumethat all data and metadata from all subjects is already loaded. If X, y
    and metadata are from a single subject, it returns data split for this subject only.

    It can be used as basis for CrossSessionSplitter or to avoid downloading all data at
    once when it is not needed,

    Parameters
    ----------
    n_folds :
        Not used. For compatibility with other cross-validation splitters.
        Default:None

    """
    def __init__(self, n_folds=None):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        sessions = metadata.session.values
        return np.unique(sessions)

    def split(self, X, y, metadata):

        cv = LeaveOneGroupOut()

        sessions = metadata.session.values

        for ix_train, ix_test in cv.split(X, y, groups=sessions):

            yield ix_train, ix_test


class CrossSubjectSplitter(BaseCrossValidator):
    """ Data splitter for cross session evaluation.

    Cross-session evaluation uses a Leave-One-Group-Out cross-validation to
    evaluate performance across sessions, but for a single subject. This splitter
    assumes that all data from all subjects is already known and loaded.

    Parameters
    ----------
    n_groups : int or None
        If None, Leave-One-Subject-Out is performed.
        If int, Leave-k-Subjects-Out is performed.


    """
    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_n_splits(self, dataset=None):
        return self.n_groups

    def split(self, X, y, metadata):

        groups = metadata.subject.values

        # Define split
        if self.n_groups is None:
            cv = LeaveOneGroupOut()
        else:
            cv = GroupKFold(n_splits=self.n_groups)

        for ix_train, ix_test in cv.split(metadata, groups=groups):

            yield ix_train, ix_test
