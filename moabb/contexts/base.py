import pandas as pd
from time import time
import numpy as np
import sys

from sklearn.base import BaseEstimator

from ..datasets.base import BaseDataset
from .. import utils

import mne

class BaseContext():
    """Base Context.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    """

    def __init__(self, pipelines, datasets=None):
        """init"""
        if datasets is None:
            datasets = utils.dataset_list
        # check dataset
        if not isinstance(datasets, list):
            if isinstance(datasets, BaseDataset):
                datasets = [datasets]
            else:
                raise(ValueError("datasets must be a list or a dataset instance"))

        for dataset in datasets:
            if not(isinstance(dataset, BaseDataset)):
                raise(ValueError("datasets must only contains dataset instance"))

        self.datasets = datasets

        # check pipelines
        if not isinstance(pipelines, dict):
            raise(ValueError("pipelines must be a dict or a Pipeline instance"))

        for name, pipeline in pipelines.items():
            if not(isinstance(pipeline, BaseEstimator)):
                raise(ValueError("pipelines must only contains Pipelines instance"))
        self.pipelines = pipelines

    def _epochs(self, raws, event_dict, time, bp_low=None, bp_high=None, channels=None):
        if type(raws) is not list:
            raws = [raws]
        ep=[]
        for raw in raws:
            print(np.unique(raw[-1,:][0]))
            events = mne.find_events(raw, shortest_event=0, verbose=False)
            if channels is None:
                raw.pick_types(eeg=True, stim=True)
            else:
                raw.pick_types(include=channels, stim=True)  # TODO: letter case test
            raw.filter(bp_low, bp_high, method='iir')
            if len(events) > 0:
                keep_events = dict([(key,val) for key, val in event_dict.items() if
                                    val in np.unique(events[:,2])])

                epochs = mne.Epochs(raw, events, keep_events, time[0], time[1],
                                proj=False, baseline=None, preload=True,
                                verbose=False)
                ep.append(epochs)
        return ep

    def prepare_data(self, dataset, subjects):
        pass


class WithinSubjectContext(BaseContext):
    """Within Subject evaluation Context.

    Evaluate performance of the pipeline on each subject independently,
    concatenating sessions.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.

    See Also
    --------
    BaseContext
    """

    def evaluate(self, verbose=False):
        """Evaluate performances

        Parameters
        ----------
        verbose: bool (defaul False)
            if true, print results durint the evaluation

        Returns
        -------
        results: Dict of panda DataFrame
            Return a dict of pandas dataframe, one for each pipeline

        """
        columns = ['Score', 'Dataset', 'Subject', 'Pipeline', 'Time']
        results = dict()
        for pipeline in self.pipelines:
            results[pipeline] = pd.DataFrame(columns=columns)

        for dataset in self.datasets:
            print('\nProcessing dataset: {:s}'.format(dataset.code))
            dataset_name = dataset.code
            subjects = dataset.subject_list

            for subject in subjects:
                X, y = self.prepare_data(dataset, [subject], stack_sessions=True)[0]

                for pipeline in self.pipelines:
                    clf = self.pipelines[pipeline]
                    t_start = time()
                    score = self.score(clf, X=X, y=y)
                    duration = time() - t_start
                    row = [score, dataset_name, subject, pipeline, duration]
                    results[pipeline].loc[len(results[pipeline])] = row
                    if verbose:
                        print(row)
        return results
