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
    datasets : List of Dataset instances, or None
        List of dataset instances on which the pipelines will be evaluated.
        If None, uses all datasets (and should break...)
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    evaluator: Evaluator instance
        Instance that defines evaluation scheme
    """

    def __init__(self, pipelines, evaluator, datasets=None, fmin=1, fmax=45, channels=None):
        """init"""
        self.fmin=fmin
        self.fmax=fmax
        self.channels=channels
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

    def verify(self, dataset):
        '''
        Method that verifies dataset is correct for given parameters
        '''
        assert dataset.paradigm == 'imagery'

    def process(self, results=None):
        '''
        Runs tasks on all given datasets. 
        '''
        # Verify that datasets are valid for given paradigm first
        if results is None:
            self.results = Results(self.evaluator)
        elif type(results) is str:
            self.results = Results(path=results, evaluation=self.evaluator)
        elif type(results) is Results:
            self.results = results
        for d in self.datasets:
            self.verify(d)
        for d in self.datasets:
            print('\n\nProcessing dataset: {}'.format(d.code))
            self.evaluator.preprocess_data(d, self)
            for s in d.subject_list:
                run_pipes = self.results.not_yet_computed(self.pipelines, d, s)
                if len(run_pipes)>0:
                    try:
                        self.results.add(self.process_subject(d, s, run_pipes))
                    except Exception as e:
                        print(e)
                        print('Skipping subject {}'.format(s))
        return self.results

    def process_subject(self, dataset, subj, pplines):
        return self.evaluator.evaluate(dataset, subj, pplines, self)

    def _epochs(self, raws, event_dict, time):
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
            if self.channels is None:
                # TODO: generalize to other sorts of channels
                raw.pick_types(eeg=True, stim=False)
            else:
                raw.pick_types(include=self.channels, stim=False)
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
        '''Property that defines scoring metric (e.g. ROC-AUC or accuracy or f-score),
        given as a sklearn-compatible string

        '''
        pass


class BaseEvaluation(ABC):
    '''Base class that defines necessary operations for an evaluation. Evaluations
    determine what the train and test sets are and can implement additional data
    preprocessing steps for more complicated algorithms.

    random_state: if not None, can guarantee same seed
    n_jobs: 1; number of jobs for fitting of pipeline

    '''
    

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
        Optional paramter if any sort of dataset-wide computation is needed per subject
        '''
        pass
