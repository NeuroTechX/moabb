import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedShuffleSplit,
)


class OfflineSplit(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        subjects = len(metadata.subject.unique())
        sessions = len(metadata.session.unique())
        return subjects * sessions

    def split(self, X, y, metadata, **kwargs):

        subjects = metadata.subject

        for subject in subjects.unique():
            X_, y_, meta_ = (
                X[subjects == subject],
                y[subjects == subject],
                metadata[subjects == subject],
            )
            sessions = meta_.session.values

            for session in sessions:
                ix_test = np.nonzero(sessions == session)[0]

                yield ix_test


class TimeSeriesSplit(BaseCrossValidator):

    def __init__(self, calib_size):
        self.calib_size = calib_size

    def get_n_splits(self, metadata):
        sessions = metadata.session.unique()
        subjects = metadata.subject.unique()

        splits = len(sessions) * len(subjects)
        return splits

    def split(self, X, y, metadata, **kwargs):

        runs = metadata.run.unique()
        sessions = metadata.session.unique()
        subjects = metadata.subject.unique()

        if len(runs) > 1:
            for subject in subjects:
                for session in sessions:
                    # Index of specific session of this subejct
                    session_indices = metadata[
                        (metadata.subject == subject) & (metadata.session == session)
                    ].index

                    for run in runs:
                        test_ix = session_indices[
                            metadata.loc[session_indices].run != run
                        ]
                        calib_ix = session_indices[
                            metadata.loc[session_indices].run == run
                        ]
                        yield test_ix, calib_ix
                        break  # Take the fist run as calibration
        else:
            for subject in subjects:
                for session in sessions:
                    session_indices = metadata[
                        (metadata.subject == subject) & (metadata.session == session)
                    ].index
                    calib_size = self.calib_size

                    indices = session_indices.to_numpy()
                    calib_ix = indices[:calib_size]
                    test_ix = indices[calib_size:]

                    yield test_ix, calib_ix  # Take first #calib_size samples as calibration


class SamplerSplit(BaseCrossValidator):

    def __init__(self, test_size, n_perms, data_size=None):
        self.data_size = data_size
        self.test_size = test_size
        self.n_perms = n_perms

        self.split = IndividualSamplerSplit(
            self.test_size, self.n_perms, data_size=self.data_size
        )

    def get_n_splits(self, y=None):
        return self.n_perms[0] * len(self.split.get_data_size_subsets(y))

    def split(self, X, y, metadata, **kwargs):
        subjects = metadata.subject.values
        split = self.split

        for subject in np.unique(subjects):
            X_, y_, meta_ = (
                X[subjects == subject],
                y[subjects == subject],
                metadata[subjects == subject],
            )

            yield split.split(X_, y_, meta_)


class IndividualSamplerSplit(BaseCrossValidator):

    def __init__(self, test_size, n_perms, data_size=None):
        self.data_size = data_size
        self.test_size = test_size
        self.n_perms = n_perms

    def get_n_splits(self, y=None):
        return self.n_perms[0] * len(self.get_data_size_subsets(y))

    def get_data_size_subsets(self, y):
        if self.data_size is None:
            raise ValueError(
                "Cannot create data subsets without valid policy for data_size."
            )
        if self.data_size["policy"] == "ratio":
            vals = np.array(self.data_size["value"])
            if np.any(vals < 0) or np.any(vals > 1):
                raise ValueError("Data subset ratios must be in range [0, 1]")
            upto = np.ceil(vals * len(y)).astype(int)
            indices = [np.array(range(i)) for i in upto]
        elif self.data_size["policy"] == "per_class":
            classwise_indices = dict()
            n_smallest_class = np.inf
            for cl in np.unique(y):
                cl_i = np.where(cl == y)[0]
                classwise_indices[cl] = cl_i
                n_smallest_class = (
                    len(cl_i) if len(cl_i) < n_smallest_class else n_smallest_class
                )
            indices = []
            for ds in self.data_size["value"]:
                if ds > n_smallest_class:
                    raise ValueError(
                        f"Smallest class has {n_smallest_class} samples. "
                        f"Desired samples per class {ds} is too large."
                    )
                indices.append(
                    np.concatenate(
                        [classwise_indices[cl][:ds] for cl in classwise_indices]
                    )
                )
        else:
            raise ValueError(f"Unknown policy {self.data_size['policy']}")
        return indices

    def split(self, X, y, metadata, **kwargs):

        sessions = metadata.session.unique()

        cv = StratifiedShuffleSplit(n_splits=self.n_perms[0], test_size=self.test_size)

        for session in np.unique(sessions):
            X_, y_, meta_ = (
                X[sessions == session],
                y[sessions == session],
                metadata[sessions == session],
            )

            for ix_train, ix_test in cv.split(X_, y_):

                y_split = y_[ix_train]
                data_size_steps = self.get_data_size_subsets(y_split)
                for subset_indices in data_size_steps:
                    ix_train = ix_train[subset_indices]
                    yield ix_train, ix_test
