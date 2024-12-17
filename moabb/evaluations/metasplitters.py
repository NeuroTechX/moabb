from sklearn.model_selection import BaseCrossValidator

from moabb.evaluations.utils import sort_group

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
    >>> from moabb.evaluations.splitters import WithinSessionSplitter
    >>> from moabb.evaluations.metasplitters import PseudoOnlineSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [8, 9], [5, 4], [2, 5], [1, 7]])
    >>> y = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    >>> subjects = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    >>> sessions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> runs = np.array(['0', '0', '1', '1', '0', '0', '1', '1'])
    >>> metadata = pd.DataFrame(data={'subject': subjects, 'session': sessions, 'run':runs})
    >>> posplit = PseudoOnlineSplit
    >>> csubj = WithinSessionSplitter(cv=posplit, calib_size=1, custom_cv=True)
    >>> posplit.get_n_splits(metadata)
    2
    >>> for i, (train_index, test_index) in enumerate(csubj.split(y, metadata)):
    >>>     print(f"Fold {i}:")
    >>>     print(f"  Calibration: index={train_index}, group={subjects[train_index]}, sessions={sessions[train_index]}, runs={runs[train_index]}")
    >>>     print(f"  Test:  index={test_index}, group={subjects[test_index]}, sessions={sessions[test_index]}, runs={runs[test_index]}")

    Fold 0:
      Calibration: index=[4], group=[1], sessions=[1], runs=['0']
      Test:  index=[5], group=[1], sessions=[1], runs=['0']
    Fold 1:
      Calibration: index=[6], group=[1], sessions=[1], runs=['1']
      Test:  index=[7], group=[1], sessions=[1], runs=['1']
    Fold 2:
      Calibration: index=[0], group=[1], sessions=[0], runs=['0']
      Test:  index=[1], group=[1], sessions=[0], runs=['0']
    Fold 3:
      Calibration: index=[2], group=[1], sessions=[0], runs=['1']
      Test:  index=[3], group=[1], sessions=[0], runs=['1']

    """

    def __init__(self, calib_size: int = None):
        self.calib_size = calib_size

    def get_n_splits(self, metadata):
        return len(metadata.groupby(["subject", "session"]))

    def split(self, indices, y, metadata=None):

        if metadata is not None:
            for _, group in metadata.groupby(["subject", "session"]):
                group = group.reset_index()
                runs = group.run.unique()
                if len(runs) > 1:
                    # To guarantee that the runs are on the right order
                    runs = sort_group(runs)
                    for run in runs:
                        calib_mask = group["run"] == run
                        calib_ix = group[calib_mask].index

                        if self.calib_size is None:
                            test_ix = group[~calib_mask].index
                            yield calib_ix, test_ix
                            break  # Take the fist run as calibration
                        else:
                            mask_run = group["run"] == run
                            if self.calib_size > len(mask_run):
                                raise ValueError('Calibration size must be less than number of runs.')
                            yield calib_ix[:self.calib_size], calib_ix[self.calib_size:]


                # Else, get the first #calib_size trials
                else:
                    if self.calib_size is None:
                        raise ValueError('Data contains just one run. Need to provide calibration size.')
                    # Take first #calib_size samples as calibration
                    calib_size = self.calib_size
                    if calib_size < len(group):
                        raise ValueError('Data contains just one run. Need to provide calibration size.')

                    # Get indexes of respective groups
                    calib_ix = group[:calib_size].index
                    test_ix = group[calib_size:].index

                    yield calib_ix, test_ix
        # If not
        else:
            if self.calib_size is None:
                raise ValueError('No metadata information. Need to provide calibration size.')
            calib_size = self.calib_size
            yield list(range(calib_size)), list(range(calib_size,len(indices)))
