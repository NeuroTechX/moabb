from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import sys

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

import mne

from ..datasets.base import BaseDataset
from ..viz import Results
from .. import utils



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
            self.evaluator.preprocess_data(d, self)
            for s in d.subject_list:
                for name, clf in self.pipelines.items():
                    self.results.add(self.process_subject(d, s, clf), name)
        self.results.to_dataframe()

    def process_subject(self, dataset, subj, clf):
        return self.evaluator.evaluate(dataset, subj, clf, self)

    def _epochs(self, raws, event_dict, time, channels=None):
        '''Take list of raws and returns a list of epoch objects. Implements 
        imagery-specific processing as well

        '''
        bp_low = self.fmin
        bp_high = self.fmax
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

    def preprocess_data(self, dataset, paradigm):
        '''
        optional if you want to optimize data loading for a given dataset/do augmentation/etc
        '''
        pass
