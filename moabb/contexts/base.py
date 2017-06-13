import pandas as pd
from time import time
import numpy as np
import sys
from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

from sklearn.base import BaseEstimator

from ..datasets.base import BaseDataset


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

    def __init__(self, datasets, pipelines):
        """init"""
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

    @abstractmethod
    def score(self, clf, X, y, groups, n_jobs=1):
        '''
        Return score 
        '''
        pass

    @abstractmethod
    def prepare_data(self, dataset, subjectlist):
        '''
        Given dataset, fetch data from subjects

        Parameters:
            dataset:       Dataset instance
            subjectlist:   List of ids **(strings? numbers?)** for subjects

        Output:
            X:       ndarray (trials, channels, timepoints) of data
            y:       ndarray (trials,) **1 or 2d** of labels
            groups:  ndarray (trials,) specifying which subject each trial belongs to
        '''
        pass
    
class WithinSubjectContext(BaseContext):
    """Within Subject evaluation Context.

    Evaluate performance of the pipeline on each subject independently.

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
            dataset_name = dataset.get_name()
            subjects = dataset.get_subject_list()

            for subject in subjects:
                X, y, groups = self.prepare_data(dataset, [subject])

                for pipeline in self.pipelines:
                    clf = self.pipelines[pipeline]
                    t_start = time()
                    score = self.score(clf, X=X, y=y, groups=groups)

                    duration = time() - t_start
                    row = [score, dataset_name, subject, pipeline, duration]
                    results[pipeline].loc[len(results[pipeline])] = row
                    if verbose:
                        print(row)
        return results

    def score(self, clf, X, y, groups, scoring, n_jobs=1, k=5):
        """get the score"""
        cv = KFold(k, shuffle=True, random_state=45)
        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring=scoring, n_jobs=n_jobs)
        return auc.mean()    

class CrossSubjectContext(BaseContext):
    '''
    Cross-subject evaluation Context

    Evaluate performance of the pipeline by training on (n-1) subjects and testing on the last

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
    '''
    def evaluate(self, verbose=False):
        """Evaluate performances

        Parameters
        ----------
        verbose: bool (defaul False)
            if true, print results during the evaluation

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
            dataset_name = dataset.get_name()
            subjects = dataset.get_subject_list()
            X, y, groups = self.prepare_data(dataset, subjects)

            for pipeline in self.pipelines:
                clf = self.pipelines[pipeline]
                t_start = time()
                score = self.score(clf, X=X, y=y, groups=groups)

                duration = (time() - t_start) / len(subjects)
                for subject, accuracy in zip(subjects, score):
                    row = [accuracy, dataset_name, subject, pipeline, duration]
                    results[pipeline].loc[len(results[pipeline])] = row
                    if verbose:
                        print(row)
        return results

    def score(self, clf, X, y, groups, scoring, n_jobs=1):
        """get the score"""
        cv = LeaveOneGroupOut()
        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                      scoring=scoring, n_jobs=n_jobs)
        return auc
