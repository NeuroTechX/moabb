import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedShuffleSplit,
)


class OfflineSplit(BaseCrossValidator):
    """ Offline split for evaluation test data.

    Assumes that, per session, all test trials are available for inference. It can be used when
    no filtering or data alignment is needed.

    Parameters
    ----------
    n_folds: int
    Not used in this case, just so it can be initialized in the same way as
    TimeSeriesSplit.

    """

    def __init__(self, n_folds: int):
        self.n_folds = n_folds

    def get_n_splits(self, metadata):
        subjects = len(metadata.subject.unique())
        sessions = len(metadata.session.unique())
        return subjects * sessions

    def split(self, X, y, metadata):

        subjects = metadata.subject.unique()

        for subject in subjects:
            X_, y_, meta_ = X[subjects == subject], y[subjects == subject], metadata[subjects == subject]
            sessions = meta_.session.unique()

            for session in sessions:
                ix_test = meta_[meta_['session'] == session].index

                yield ix_test


class TimeSeriesSplit(BaseCrossValidator):
    """ Pseudo-online split for evaluation test data.

    It takes into account the time sequence for obtaining the test data, and uses first run,
    or first #calib_size trial as calibration data, and the rest as evaluation data.
    Calibration data is important in the context where data alignment or filtering is used on
    training data.

    OBS: Be careful! Since this inference split is based on time disposition of obtained data,
    if your data is not organized by time, but by other parameter, such as class, you may want to
    be extra careful when using this split.

    Parameters
    ----------
    calib_size: int
    Size of calibration set, used if there is just one run.

    """

    def __init__(self, calib_size:int):
        self.calib_size = calib_size

    def get_n_splits(self, metadata):
        sessions = metadata.session.unique()
        subjects = metadata.subject.unique()

        splits = len(sessions) * len(subjects)
        return splits

    def split(self, X, y, metadata):

        runs = metadata.run.unique()
        sessions = metadata.session.unique()
        subjects = metadata.subject.unique()

        if len(runs) > 1:
            for subject in subjects:
                for session in sessions:
                    # Index of specific session of this subject
                    session_indices = metadata[(metadata['subject'] == subject) &
                                               (metadata['session'] == session)].index

                    for run in runs:
                        test_ix = session_indices[metadata.loc[session_indices]['run'] != run]
                        calib_ix = session_indices[metadata.loc[session_indices]['run'] == run]
                        yield test_ix, calib_ix
                        break  # Take the fist run as calibration
        else:
            for subject in subjects:
                for session in sessions:
                    session_indices = metadata[(metadata['subject'] == subject) &
                                               (metadata['session'] == session)].index
                    calib_size = self.calib_size

                    calib_ix = session_indices[:calib_size]
                    test_ix = session_indices[calib_size:]

                    yield test_ix, calib_ix  # Take first #calib_size samples as calibration


class SamplerSplit(BaseCrossValidator):
    """ Return subsets of the training data with different number of samples.

    Util for estimating the performance of a model when using different number of
    training samples and plotting the learning curve. You must define the data
    evaluation type (WithinSubject, CrossSession, CrossSubject) so the training set
    can be sampled.

    Parameters
    ----------
    data_eval: BaseCrossValidator object
        Evaluation splitter already initialized. It can be WithinSubject, CrossSession,
        or CrossSubject Splitters.
    data_size : dict
        Contains the policy to pick the datasizes to
        evaluate, as well as the actual values. The dict has the
        key 'policy' with either 'ratio' or 'per_class', and the key
        'value' with the actual values as a numpy array. This array should be
        sorted, such that values in data_size are strictly monotonically increasing.

    """

    def __init__(self, data_eval, data_size):
        self.data_eval = data_eval
        self.data_size = data_size

        self.sampler = IndividualSamplerSplit(self.data_size)

    def get_n_splits(self, y, metadata):
        return self.data_eval.get_n_splits(metadata) * len(self.sampler.get_data_size_subsets(y))

    def split(self, X, y, metadata, **kwargs):
        cv = self.data_eval
        sampler = self.sampler

        for ix_train, _ in cv.split(X, y, metadata, **kwargs):
            X_train, y_train, meta_train = X[ix_train], y[ix_train], metadata[ix_train]
            yield sampler.split(X_train, y_train, meta_train)


class IndividualSamplerSplit(BaseCrossValidator):
    """ Return subsets of the training data with different number of samples.

    Util for estimating the performance of a model when using different number of
    training samples and plotting the learning curve. It must be used after already splitting
    data using one of the other evaluation data splitters (WithinSubject, CrossSession, CrossSubject)
    since it corresponds to a subsampling of the training data.

    This 'Individual' Sampler Split assumes that data and metadata being passed is training, and was
    already split by WithinSubject, CrossSession, or CrossSubject splitters.

    Parameters
    ----------
    data_size : dict
        Contains the policy to pick the datasizes to
        evaluate, as well as the actual values. The dict has the
        key 'policy' with either 'ratio' or 'per_class', and the key
        'value' with the actual values as a numpy array. This array should be
        sorted, such that values in data_size are strictly monotonically increasing.

    """

    def __init__(self, data_size):
        self.data_size = data_size

    def get_n_splits(self, y=None):
        return len(self.get_data_size_subsets(y))

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

    def split(self, X, y, metadata):

        data_size_steps = self.get_data_size_subsets(y)
        for subset_indices in data_size_steps:
            ix_train = subset_indices
            yield ix_train
