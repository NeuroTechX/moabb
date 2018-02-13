import pandas as pd
from time import time
import numpy as np
import sys

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

import mne
from mne.epochs import concatenate_epochs, equalize_epoch_counts

from ..datasets.base import BaseDataset
from .. import utils


class Results:

    def __init__(self, evaluation, pipelines):
        """
        class that will abstract result storage
        """
        self.evaluation = evaluation
        self.data_columns = ['id','time','score','dataset']
        dfs = [[] for p in pipelines.keys()]
        self.data = dict(zip(pipelines.keys(), dfs))

    def add(self, data_dict, pipeline):
        print(data_dict)
        self.data[pipeline].append(data_dict)

    def to_dataframe(self):
        for k in self.data.keys():
            print(self.data[k])
            df = pd.DataFrame(columns=self.data_columns)
            new = df.append(self.data[k])
            self.data[k] = new

class BaseImageryParadigm():
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
        self.evaluator=evaluator
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
        assert(dataset.paradigm=='imagery')
        
    def process(self):
        for d in self.datasets:
            self.verify(d)
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
            print(np.unique(raw[-1, :][0]))
            events = mne.find_events(raw, shortest_event=0, verbose=False)
            if channels is None:
                raw.pick_types(eeg=True, stim=True)
            else:
                # TODO: letter case test
                raw.pick_types(include=channels, stim=True)
            raw.filter(bp_low, bp_high, method='iir')
            if len(events) > 0:
                keep_events = dict([(key, val) for key, val in event_dict.items() if
                                    val in np.unique(events[:, 2])])

                epochs = mne.Epochs(raw, events, keep_events, time[0], time[1],
                                    proj=False, baseline=None, preload=True,
                                    verbose=False)
                ep.append(epochs)
        return ep

class BaseEvaluation:

    def __init__(self, random_state=None, n_jobs=1):
        """
        
        """
        self.random_state = random_state
        self.n_jobs = n_jobs

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




class WithinSubjectEvaluation(BaseEvaluation):
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

    def evaluate(self, dataset, subject, clf, paradigm):
        """Prepare data for classification."""
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        sub= dataset.get_data([subject], stack_sessions=True)[0]
        # get all epochs for individual files in given subject
        epochs = paradigm._epochs(sub, event_id, dataset.interval)
        # equalize events from different classes
        event_epochs = dict(zip(event_id.keys(), [[]]*len(event_id)))
        for epoch in epochs:
            for key in event_id.keys():
                if key in epoch.event_id.keys():
                    event_epochs[key].append(epoch[key])
        for key in event_id.keys():
            event_epochs[key] = concatenate_epochs(event_epochs[key])

        # equalize for accuracy
        equalize_epoch_counts(list(event_epochs.values()))
        ep = concatenate_epochs(list( event_epochs.values() ))
        X, y = (ep.get_data(), ep.events[:,-1]) #previously multipled data by 1e6
        t_start = time()
        score = self.score(clf, X, y)
        duration = time() - t_start
        return {'time':duration, 'dataset':dataset.code,'id':subject, 'score':score}
    def score(self, clf, X, y):
        cv = KFold(5, shuffle=True, random_state=self.random_state)

        acc = cross_val_score(clf, X, y, cv=cv,
                              scoring='accuracy', n_jobs=self.n_jobs)
        return acc.mean()

# class WithinSessionContext(BaseContext):
#     """Within Subject evaluation Context.

#     Evaluate performance of the pipeline on each session independently,
#     artificially expanding datasets

#     Parameters
#     ----------
#     datasets : List of Dataset instances.
#         List of dataset instances on which the pipelines will be evaluated.
#     pipelines : Dict of pipelines instances.
#         Dictionary of pipelines. Keys identifies pipeline names, and values
#         are scikit-learn pipelines instances.

#     See Also
#     --------
#     BaseContext
#     """

#     def evaluate(self, verbose=False):
#         """Evaluate performances

#         Parameters
#         ----------
#         verbose: bool (defaul False)
#             if true, print results durint the evaluation

#         Returns
#         -------
#         results: Dict of panda DataFrame
#             Return a dict of pandas dataframe, one for each pipeline

#         """
#         columns = ['Score', 'Dataset', 'Subject', 'Pipeline', 'Time']
#         results = dict()
#         for pipeline in self.pipelines:
#             results[pipeline] = pd.DataFrame(columns=columns)

#         for dataset in self.datasets:
#             print('\nProcessing dataset: {:s}'.format(dataset.code))
#             dataset_name = dataset.code
#             subjects = dataset.subject_list

#             for subject in subjects:
#                 sessions = self.prepare_data(
#                     dataset, [subject], stack_sessions=False)

#                 for ind, (X, y) in enumerate(sessions):
#                     for pipeline in self.pipelines:
#                         clf = self.pipelines[pipeline]
#                         t_start = time()
#                         score = self.score(clf, X=X, y=y)
#                         duration = time() - t_start
#                         row = [score, dataset_name, '{:s}_{:d}'.format(
#                             subject, ind), pipeline, duration]
#                         results[pipeline].loc[len(results[pipeline])] = row
#                         if verbose:
#                             print(row)
#         return results

# class CrossSessionContext(BaseContext):
#     """Cross session Context.

#     Evaluate performance of the pipeline across sessions,

#     Parameters
#     ----------
#     datasets : List of Dataset instances.
#         List of dataset instances on which the pipelines will be evaluated.
#     pipelines : Dict of pipelines instances.
#         Dictionary of pipelines. Keys identifies pipeline names, and values
#         are scikit-learn pipelines instances.

#     See Also
#     --------
#     BaseContext
#     """

#     def evaluate(self, verbose=False):
#         """Evaluate performances

#         Parameters
#         ----------
#         verbose: bool (defaul False)
#             if true, print results durint the evaluation

#         Returns
#         -------
#         results: Dict of panda DataFrame
#             Return a dict of pandas dataframe, one for each pipeline

#         """
#         columns = ['Score', 'Dataset', 'Subject', 'Pipeline', 'Time']
#         results = dict()
#         for pipeline in self.pipelines:
#             results[pipeline] = pd.DataFrame(columns=columns)

#         for dataset in self.datasets:
#             print('\nProcessing dataset: {:s}'.format(dataset.code))
#             if dataset.n_sessions == 1:
#                 print('Skipping dataset since there is only one recording session')
#             dataset_name = dataset.code
#             subjects = dataset.subject_list

#             for subject in subjects:
#                 sessions = self.prepare_data(
#                     dataset, [subject], stack_sessions=False)

#                 allX = []
#                 ally = []
#                 groups = []
#                 for ind, (X, y) in enumerate(sessions):
#                     allX.append(X)
#                     ally.append(y)
#                     groups.append(np.ones((X.shape[0],1))*ind)
#                 allX = np.concatenate(allX, axis=0)
#                 ally = np.concatenate(ally, axis=0)
#                 groups = np.concatenate(groups, axis=0)

#                 for pipeline in self.pipelines:
#                     clf = self.pipelines[pipeline]
#                     t_start = time()
#                     score = self.score(clf, X=X, y=y, groups=groups)
#                     duration = time() - t_start
#                     row = [score, dataset_name, subject, pipeline, duration]
#                     results[pipeline].loc[len(results[pipeline])] = row
#                     if verbose:
#                         print(row)
#         return results
