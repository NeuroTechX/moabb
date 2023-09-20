from sklearn.model_selection import BaseCrossValidator, GroupKFold, LeaveOneGroupOut


class CrossSubjectValidator(BaseCrossValidator):
    """
    CrossValidator iterator for leave one-subject out or leave multiple
    subjects out.

    The Cross-Validator validator is an intra-subject validation where
    for the whole dataset, we perform Leave One GroupOut for each person, or
    Leave Multiple Groups Out for a list of people.

    Here, if you don't pass the n_splits, we assume that you want to perform
    Leave One Subject Out, so we have as many folds as the number of subjects.

    Parameters
    ----------
    n_splits : None or int, default=None
        Number of folds.
    """

    def __init__(self, n_splits=None, *args, **kwargs):
        self.n_splits = n_splits

        if self.n_splits is None:
            self.cv = LeaveOneGroupOut()
        else:
            self.cv = GroupKFold(n_splits=n_splits)

        super(CrossSubjectValidator, self).__init__(*args, **kwargs)

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for train_idx, test_idx in self.cv.split(X=X, y=y, groups=groups):
            yield train_idx, test_idx

    def split(self, X, y=None, groups=None):
        return self._iter_test_masks(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        if self.n_splits is None:
            return len(groups["subject"].unique())
        else:
            return self.n_splits
