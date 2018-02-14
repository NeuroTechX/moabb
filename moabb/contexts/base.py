from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np
import sys

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

import mne

from ..datasets.base import BaseDataset
from .. import utils


class Results(ABC):

    def __init__(self, evaluation, pipelines):
        """
        class that will abstract result storage
        """
        self.evaluation = evaluation
        self.data_columns = ['id', 'time', 'score', 'dataset', 'n_samples']
        dfs = [[] for p in pipelines.keys()]
        self.data = dict(zip(pipelines.keys(), dfs))

    def add(self, data_dict, pipeline):
        if type(data_dict) is dict:
            data_dict = [data_dict]
        elif type(data_dict) is not list:
            raise ValueError('Results are given as neither dict nor list but {}'.format(
                type(data_dict).__name__))
        self.data[pipeline].extend(data_dict)

    def to_dataframe(self):
        for k in self.data.keys():
            df = pd.DataFrame.from_records(
                self.data[k], columns=self.data_columns)
            self.data[k] = df


class BaseImageryParadigm(ABC):
    """Base Context.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    """

    def __init__(self, pipelines, evaluator, datasets=None):
        """init"""
        self.evaluator = evaluator
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
            raise(ValueError("pipelines must be a dict"))

        for name, pipeline in pipelines.items():
            if not(isinstance(pipeline, BaseEstimator)):
                raise(ValueError("pipelines must only contains Pipelines instance"))
        self.pipelines = pipelines
        self.results = Results(type(evaluator).__name__, pipelines)

    def verify(self, dataset):
        assert(dataset.paradigm == 'imagery')

    def process(self):
        # Verify that datasets are valid for given paradigm first
        for d in self.datasets:
            self.verify(d)
        for d in self.datasets:
            print('\n\nProcessing dataset: {}'.format(d.code))
            self.evaluator.preprocess_data(d)
            for s in d.subject_list:
                for name, clf in self.pipelines.items():
                    self.results.add(self.process_subject(d, s, clf), name)
        self.results.to_dataframe()

    def process_subject(self, dataset, subj, clf):
        return self.evaluator.evaluate(dataset, subj, clf, self)

    def _epochs(self, raws, event_dict, time, bp_low=1, bp_high=40, channels=None):
        '''Take list of raws and returns a list of epoch objects. Implements 
        imagery-specific processing as well

        '''
        if type(raws) is not list:
            raws = [raws]
        ep = []
        for raw in raws:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
            if channels is None:
                raw.pick_types(eeg=True, stim=True)
            else:
                # TODO: letter case test
                raw.pick_types(include=channels, stim=True)
            raw.filter(bp_low, bp_high, method='iir')
            # ensure events are desired:
            if len(events) > 0:
                keep_events = dict([(key, val) for key, val in event_dict.items() if
                                    val in np.unique(events[:, 2])])
                if len(keep_events) > 0:
                    epochs = mne.Epochs(raw, events, keep_events, time[0], time[1],
                                        proj=False, baseline=None, preload=True,
                                        verbose=False)
                    ep.append(epochs)
                    
        return ep

    @abstractproperty
    def scoring(self):
        pass


class BaseEvaluation(ABC):

    def __init__(self, random_state=None, n_jobs=1):
        """

        """
        self.random_state = random_state
        self.n_jobs = n_jobs

    @abstractmethod
    def evaluate(self, dataset, subject, clf, paradigm):
        '''
        Return results in a dict
        '''
        pass

    def preprocess_data(self, dataset):
        '''
        optional if you want to optimize data loading for a given dataset/do augmentation/etc
        '''
        pass
