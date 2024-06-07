import numpy as np
from sklearn.model_selection import BaseCrossValidator, GroupKFold, LeaveOneGroupOut, StratifiedKFold


class WithinSubjectSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        sessions_subjects = len(metadata.groupby(['subject', 'session']).first())
        return self.n_folds * sessions_subjects

    def split(self, X, y, metadata, **kwargs):

        subjects = metadata.subject.values

        split = IndividualWithinSubjectSplitter(self.n_folds)

        for subject in np.unique(subjects):

            X_, y_, meta_ = X[subjects == subject], y[subjects == subject], metadata[subjects == subject]

            yield split.split(X_, y_, meta_)


class IndividualWithinSubjectSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        return self.n_folds

    def split(self, X, y, metadata, **kwargs):

        sessions = metadata.subject.values

        cv = StratifiedKFold(self.n_folds, **kwargs)

        for session in np.unique(sessions):
            X_, y_, meta_ = X[sessions == session], y[sessions == session], metadata[sessions == session]

            for ix_train, ix_test in cv.split(X_, y_):

                yield ix_train, ix_test


class CrossSessionSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        sessions_subjects = len(metadata.groupby(['subject', 'session']).first())
        return sessions_subjects

    def split(self, X, y, metadata, **kwargs):

        subjects = metadata.subject.values
        split = IndividualCrossSessionSplitter(self.n_folds)

        for subject in np.unique(subjects):
            X_, y_, meta_ = X[subjects == subject], y[subjects == subject], metadata[subjects == subject]

            yield split.split(X_, y_, meta_)


class IndividualCrossSessionSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        sessions = metadata.session.values
        return np.unique(sessions)

    def split(self, X, y, metadata, **kwargs):

        cv = LeaveOneGroupOut()

        sessions = metadata.session.values

        for ix_train, ix_test in cv.split(X, y, groups=sessions):

            yield ix_train, ix_test


class CrossSubjectSplitter(BaseCrossValidator):

    def __init__(self, n_groups=None):
        self.n_groups = n_groups

    def get_n_splits(self, dataset=None):
        return self.n_groups

    def split(self, X, y, metadata, **kwargs):

        groups = metadata.subject.values

        # Define split
        if self.n_groups is None:
            cv = LeaveOneGroupOut()
        else:
            cv = GroupKFold(n_splits=self.n_groups)

        for ix_train, ix_test in cv.split(metadata, groups=groups):

            yield ix_train, ix_test
