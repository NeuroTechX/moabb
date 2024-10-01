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

from moabb.evaluations.utils import sort_group


class OfflineSplit(BaseCrossValidator):
    """Offline split for evaluation test data.

    It can be used for further splitting of the test data based on sessions or runs as needed.

    Assumes that, per session, all test trials are available for inference. It can be used when
    no filtering or data alignment is needed.

    Parameters
    ----------
    n_folds: int
    Not used in this case, just so it can be initialized in the same way as
    PseudoOnlineSplit.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from moabb.evaluations.splitters import CrossSubjectSplitter
    >>> X = np.array([[[5, 6]]*12])[0]
    >>> y = np.array([[1, 2]*12])[0]
    >>> subjects = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    >>> sessions = np.array([[0, 0, 1, 1]*3])[0]
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions})
    >>> csubj = CrossSubjectSplitter()
    >>> off = OfflineSplit()
    >>> csubj.get_n_splits(metadata)
    3
    >>> for i, (train_index, test_index) in enumerate(csubj.split(X, y, metadata)):
    >>>     print(f"Fold {i}:")
    >>>     print(f"  Train: index={train_index}, group={subjects[train_index]}, sessions={sessions[train_index]}")
    >>>     print(f"  Test:  index={test_index}, group={subjects[test_index]}, sessions={sessions[test_index]}")
    >>>     X_test, y_test, meta_test = X[test_index], y[test_index], metadata.loc[test_index]
    >>>     for j, test_session in enumerate(off.split(X_test, y_test, meta_test)):
    >>>         print(f"  By session - Test:  index={test_session}, group={subjects[test_session]}, sessions={sessions[test_session]}")

    Fold 0:
      Train: index=[ 4  5  6  7  8  9 10 11], group=[2 2 2 2 3 3 3 3], sessions=[0 0 1 1 0 0 1 1]
      Test:  index=[0 1 2 3], group=[1 1 1 1], sessions=[0 0 1 1]
      By session - Test:  index=[0, 1], group=[1 1], sessions=[0 0]
      By session - Test:  index=[2, 3], group=[1 1], sessions=[1 1]
    Fold 1:
      Train: index=[ 0  1  2  3  8  9 10 11], group=[1 1 1 1 3 3 3 3], sessions=[0 0 1 1 0 0 1 1]
      Test:  index=[4 5 6 7], group=[2 2 2 2], sessions=[0 0 1 1]
      By session - Test:  index=[4, 5], group=[2 2], sessions=[0 0]
      By session - Test:  index=[6, 7], group=[2 2], sessions=[1 1]
    Fold 2:
      Train: index=[0 1 2 3 4 5 6 7], group=[1 1 1 1 2 2 2 2], sessions=[0 0 1 1 0 0 1 1]
      Test:  index=[ 8  9 10 11], group=[3 3 3 3], sessions=[0 0 1 1]
      By session - Test:  index=[8, 9], group=[3 3], sessions=[0 0]
      By session - Test:  index=[10, 11], group=[3 3], sessions=[1 1]

    """

    def __init__(self, n_folds=None, run=False):
        self.n_folds = n_folds
        self.run = run

    def get_n_splits(self, metadata):
        return metadata.groupby(["subject", "session"]).ngroups

    def split(self, X, y, metadata):

        subjects = metadata["subject"]

        for subject in subjects.unique():
            mask = subjects == subject
            X_, y_, meta_ = X[mask], y[mask], metadata[mask]
            sessions = meta_.session.unique()

            for session in sessions:
                session_mask = meta_["session"] == session
                _, _, meta_session = X_[session_mask], y_[session_mask], meta_[session_mask]

                # If you can (amd want) to split by run also
                if self.run and "run" in meta_session.columns:
                    runs = meta_session["run"].unique()

                    for run in runs:
                        run_mask = meta_session["run"] == run
                        ix_test = meta_session[run_mask].index
                        yield list(ix_test)

                else:
                    ix_test = meta_session.index
                    yield list(ix_test)


class PseudoOnlineSplit(BaseCrossValidator):
    """Pseudo-online split for evaluation test data.

    It takes into account the time sequence for obtaining the test data, and uses first run,
    or first #calib_size trials as calibration data, and the rest as evaluation data.
    Calibration data is important in the context where data alignment or filtering is used on
    training data.

    OBS: Be careful! Since this inference split is based on time disposition of obtained data,
    if your data is not organized by time, but by other parameter, such as class, you may want to
    be extra careful when using this split.

    Parameters
    ----------
    calib_size: int
    Size of calibration set, used if there is just one run.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from moabb.evaluations.splitters import CrossSubjectSplitter
    >>> from moabb.evaluations.metasplitters import PseudoOnlineSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [8, 9], [5, 4], [2, 5], [1, 7]])
    >>> y = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    >>> subjects = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    >>> sessions = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> runs = np.array(['0', '1', '0', '1', '0', '1', '0', '1'])
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions, 'run':runs})
    >>> csubj = CrossSubjectSplitter()
    >>> posplit = PseudoOnlineSplit()
    >>> posplit.get_n_splits(metadata)
    4
    >>> for i, (train_index, test_index) in enumerate(csubj.split(X, y, metadata)):
    >>>     print(f"Fold {i}:")
    >>>     print(f"  Train: index={train_index}, group={subjects[train_index]}, sessions={sessions[train_index]}, runs={runs[train_index]}")
    >>>     print(f"  Test:  index={test_index}, group={subjects[test_index]}, sessions={sessions[test_index]}, runs={runs[test_index]}")
    >>>     X_test, y_test, meta_test = X[test_index], y[test_index], metadata.loc[test_index]
    >>>     for j, (test_ix, calib_ix) in enumerate(posplit.split(X_test, y_test, meta_test)):
    >>>         print(f"     Evaluation:  index={test_ix}, group={subjects[test_ix]}, sessions={sessions[test_ix]}, runs={runs[test_ix]}")
    >>>         print(f"     Calibration:  index={calib_ix}, group={subjects[calib_ix]}, sessions={sessions[calib_ix]}, runs={runs[calib_ix]}")

    Fold 0:
      Train: index=[4 5 6 7], group=[2 2 2 2], sessions=[0 0 1 1], runs=['0' '1' '0' '1']
      Test:  index=[0 1 2 3], group=[1 1 1 1], sessions=[0 0 1 1], runs=['0' '1' '0' '1']
         Evaluation:  index=[1], group=[1], sessions=[0], runs=['1']
         Calibration:  index=[0], group=[1], sessions=[0], runs=['0']
         Evaluation:  index=[3], group=[1], sessions=[1], runs=['1']
         Calibration:  index=[2], group=[1], sessions=[1], runs=['0']
    Fold 1:
      Train: index=[0 1 2 3], group=[1 1 1 1], sessions=[0 0 1 1], runs=['0' '1' '0' '1']
      Test:  index=[4 5 6 7], group=[2 2 2 2], sessions=[0 0 1 1], runs=['0' '1' '0' '1']
         Evaluation:  index=[5], group=[2], sessions=[0], runs=['1']
         Calibration:  index=[4], group=[2], sessions=[0], runs=['0']
         Evaluation:  index=[7], group=[2], sessions=[1], runs=['1']
         Calibration:  index=[6], group=[2], sessions=[1], runs=['0']

    """

    def __init__(self, calib_size: int = None):
        self.calib_size = calib_size

    def get_n_splits(self, metadata):
        return len(metadata.groupby(["subject", "session"]))

    def split(self, X, y, metadata):

        for _, group in metadata.groupby(["subject", "session"]):
            runs = group.run.unique()
            if len(runs) > 1:
                # To guarantee that the runs are on the right order
                runs = sort_group(runs)
                for run in runs:
                    test_ix = group[group["run"] != run].index
                    calib_ix = group[group["run"] == run].index
                    yield list(test_ix), list(calib_ix)
                    break  # Take the fist run as calibration
            else:
                calib_size = self.calib_size
                calib_ix = group[:calib_size].index
                test_ix = group[calib_size:].index

                yield list(test_ix), list(
                    calib_ix
                )  # Take first #calib_size samples as calibration


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
    >>> y = np.array([[1, 2]*12])[0]
    >>> subjects = np.array([1]*12)
    >>> sessions = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> runs = np.array(['0', '0', '1', '1', '2', '2', '0', '0', '1', '1', '2', '2',])
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions, 'run':runs})
    >>> from moabb.evaluations.metasplitters import SamplerSplit
    >>> from moabb.evaluations.splitters import CrossSessionSplitter
    >>> data_size = dict(policy="per_class", value=np.array([2,3]))
    >>> data_eval = CrossSessionSplitter()
    >>> sampler = SamplerSplit(data_eval, data_size)
    >>> for i, (train_index, test_index) in enumerate(sampler.split(X, y, metadata)):
    >>>    print(f"Fold {i}:")
    >>>    print(f"  Train: index={train_index}, sessions={sessions[train_index]}")
    >>>    print(f"  Test:  index={test_index}, sessions={sessions[test_index]}")

    Fold 0:
      Train: index=[6 8 7 9], sessions=[1 1 1 1]
      Test:  index=[0 1 2 3 4 5], sessions=[0 0 0 0 0 0]
    Fold 1:
      Train: index=[ 6  8 10  7  9 11], sessions=[1 1 1 1 1 1]
      Test:  index=[0 1 2 3 4 5], sessions=[0 0 0 0 0 0]
    Fold 2:
      Train: index=[0 2 1 3], sessions=[0 0 0 0]
      Test:  index=[ 6  7  8  9 10 11], sessions=[1 1 1 1 1 1]
    Fold 3:
      Train: index=[0 2 4 1 3 5], sessions=[0 0 0 0 0 0]
      Test:  index=[ 6  7  8  9 10 11], sessions=[1 1 1 1 1 1]

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
