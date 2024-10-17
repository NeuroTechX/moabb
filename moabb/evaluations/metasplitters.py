"""
The data splitters defined in this file are not directly related to a evaluation method
the way that WithinSession, CrossSession and CrossSubject splitters are.

OfflineSplit and TimeSeriesSplit split the test data, indicating weather model inference
will be computed using a Offline or a Pseudo-Online validation. Pseudo-online evaluation
is important when training data is pre-processed with some data-dependent transformation.
One part of the test data is separated as a calibration to compute the transformation.

SamplerSplit is an optional subsplit done on the training set to generate subsets with
different numbers of samples. It can be used to estimate the performance of the model
on different training sizes and plot learning curves.

"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class SamplerSplit(BaseCrossValidator):
    """Return subsets of the training data with different number of samples.

    This splitter can be used for estimating a model's performance when using different
    numbers of training samples, and for plotting the learning curve per number of training
    samples. You must define the data evaluation type (WithinSubject, CrossSession, CrossSubject)
    so the training set can be sampled. It is also needed to pass a dictionary indicating the
    policy used for sampling the training set and the number of examples (or the percentage) that
    each sample must contain.

    Parameters
    ----------
    data_eval: BaseCrossValidator object
        Evaluation splitter already initialized. It can be WithinSubject, CrossSession,
        or CrossSubject Splitters.
    data_size : dict
        Contains the policy to pick the datasizes to evaluate, as well as the actual values.
        The dict has the key 'policy' with either 'ratio' or 'per_class', and the key
        'value' with the actual values as a numpy array. This array should be
        sorted, such that values in data_size are strictly monotonically increasing.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.array([[[5, 6]]*12])[0]
    >>> y = np.array([[1, 2]*6])[0]
    >>> subjects = np.array([1]*12)
    >>> sessions = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> runs = np.array(['0', '0', '1', '1', '2', '2', '0', '0', '1', '1', '2', '2',])
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions, 'run':runs})
    >>> from moabb.evaluations.metasplitters import SamplerSplit
    >>> from moabb.evaluations.splitters import WithinSessionSplitter
    >>> data_size = dict(policy="per_class", value=np.array([1,3]))
    >>> data_eval = WithinSessionSplitter(n_folds=3)
    >>> sampler = SamplerSplit(data_eval, data_size)
    >>> for i, (train_index, test_index) in enumerate(sampler.split(X, y, metadata)):
    >>>    print(f"Fold {i}:")
    >>>    print(f"  Train: index={train_index}, sessions={sessions[train_index]}")
    >>>    print(f"  Test:  index={test_index}, sessions={sessions[test_index]}")

    Fold 0:
      Train: index=[2 1], sessions=[0 0]
      Test:  index=[0 5], sessions=[0 0]
    Fold 1:
      Train: index=[2 4 1 3], sessions=[0 0 0 0]
      Test:  index=[0 5], sessions=[0 0]
    Fold 2:
      Train: index=[0 3], sessions=[0 0]
      Test:  index=[1 2], sessions=[0 0]
    Fold 3:
      Train: index=[0 4 3 5], sessions=[0 0 0 0]
      Test:  index=[1 2], sessions=[0 0]
    Fold 4:
      Train: index=[0 1], sessions=[0 0]
      Test:  index=[3 4], sessions=[0 0]
    Fold 5:
      Train: index=[0 2 1 5], sessions=[0 0 0 0]
      Test:  index=[3 4], sessions=[0 0]
    Fold 6:
      Train: index=[8 7], sessions=[1 1]
      Test:  index=[ 6 11], sessions=[1 1]
    Fold 7:
      Train: index=[ 8 10  7  9], sessions=[1 1 1 1]
      Test:  index=[ 6 11], sessions=[1 1]
    Fold 8:
      Train: index=[6 9], sessions=[1 1]
      Test:  index=[7 8], sessions=[1 1]
    Fold 9:
      Train: index=[ 6 10  9 11], sessions=[1 1 1 1]
      Test:  index=[7 8], sessions=[1 1]
    Fold 10:
      Train: index=[6 7], sessions=[1 1]
      Test:  index=[ 9 10], sessions=[1 1]
    Fold 11:
      Train: index=[ 6  8  7 11], sessions=[1 1 1 1]
      Test:  index=[ 9 10], sessions=[1 1]

    """

    def __init__(self, data_eval, data_size):
        self.data_eval = data_eval
        self.data_size = data_size

        self.sampler = IndividualSamplerSplit(self.data_size)

    def get_n_splits(self, y, metadata):
        return self.data_eval.get_n_splits(metadata) * len(
            self.sampler.get_data_size_subsets(y)
        )

    def split(self, X, y, metadata, **kwargs):
        cv = self.data_eval
        sampler = self.sampler

        for ix_train, ix_test in cv.split(X, y, metadata, **kwargs):
            X_train, y_train, meta_train = (
                X[ix_train],
                y[ix_train],
                metadata.iloc[ix_train],
            )
            for ix_train_sample in sampler.split(X_train, y_train, meta_train):
                ix_train_sample = ix_train[ix_train_sample]
                yield ix_train_sample, ix_test


class IndividualSamplerSplit(BaseCrossValidator):
    """Return subsets of the training data with different number of samples.

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
