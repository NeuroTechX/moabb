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
    def score(self, clf, X, y, info, n_jobs=1):
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
         info:       DataFrame specifying session, subject, and possibly other information
        '''
        pass
    
    def evaluate(self, verbose=False, **kwargs):
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
        for dataset in self.datasets:
            dataset_name = dataset.get_name()
            subjects = dataset.get_subject_list()
            X, y, info = self.prepare_data(dataset, subjects)
            for pipeline in self.pipelines:
                clf = self.pipelines[pipeline]
                score = self.score(clf, X=X, y=y, info=info, **kwargs)
                score = score.assign(Dataset=dataset_name,Pipeline=pipeline)
                results[pipeline] = score
                if verbose:
                    print(score)
        return results

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
    def score(self, clf, X, y, info, scoring, n_jobs=1, k=5):
        """ for each subject cross-validate"""
        # how can we enforce the fields of the dataframe?
        subj_ind = np.unique(info['Subject'])
        cv = KFold(k, shuffle=True, random_state=45)
        out = pd.DataFrame(np.zeros((len(subj_ind),3)),
                           columns=['Score', 'Subject', 'Time'])
        for ind, sub in enumerate(subj_ind):
            t_start = time()
            # extract x and y corresponding to subject
            X_sub = X[info['Subject']==sub,...]
            y_sub = y[info['Subject']==sub]
            auc = cross_val_score(clf, X_sub, y_sub, cv=cv,
                                  scoring=scoring, n_jobs=n_jobs)
            t_end = time()
            # do this properly
            out.loc[ind] = [auc.mean(), sub, t_end-t_start]
        return out

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
    def score(self, clf, X, y, info, scoring, n_jobs=1):
        """get the score"""
        cv = LeaveOneGroupOut()
        # extract subject data regardless of session
        X_sub = []
        y_sub = []
        group = []
        subj_ind = np.unique(info['Subject'])
        for ind,sub in enumerate(subj_ind):
            X_sub.append(X[info['Subject']==sub,...])
            y_sub.append(y[info['Subject']==sub])
            group.extend([ind]*len(y_sub[-1]))
        X_sub = np.vstack(X_sub)
        y_sub = np.concatenate(y_sub)
        # this method doesn't allow us to get a time per subject...
        t_start = time()
        auc = cross_val_score(clf, X_sub, y_sub, groups=np.asarray(group), cv=cv,
                              scoring=scoring, n_jobs=n_jobs)
        t_end = time()
        
        return pd.DataFrame({'Subject': subj_ind,'Score':auc,
                             'Time':[((t_end-t_start)/len(subj_ind))]*len(subj_ind)})

class SubjectUpdateContext(BaseContext):

    def score(self, pipe, X, y, info, scoring, n_jobs=None, k=5):
        '''
        Return score when pre-training on all subjects and then doing cross-validation within a given subject. 
        '''
        
        cv = KFold(k, shuffle=True, random_state=45)
        # extract subject data regardless of session
        X_sub = []
        y_sub = []
        subj_ind = np.unique(info['Subject'])
        nsubj = len(subj_ind)
        for ind,sub in enumerate(subj_ind):
            X_sub.append(X[info['Subject']==sub,...])
            y_sub.append(y[info['Subject']==sub])

        out = pd.DataFrame(np.empty((len(X_sub),4)),columns=['Score','Subject','Subject time','Pretrain time'])
        jobsarg = {'{}__n_jobs'.format(type(pipe._final_estimator).__name__.lower()):n_jobs} 
        for ind in range(nsubj):
            # this step will probably have to change...
            trainsubj = [i for i,v in enumerate(subj_ind) if i != ind]
            ttrain_st = time()
            clf = pipe.pre_fit([X_sub[i] for i in trainsubj],
                               [y_sub[i] for i in trainsubj],**jobsarg)
            ttrain_end = time()
            ttest_st = time()
            auc = cross_val_score(clf, X_sub[ind], y_sub[ind], cv=cv,
                                  scoring=scoring, n_jobs=n_jobs)
            ttest_end = time()
            out.loc[ind] = [auc.mean(), subj_ind[ind], ttest_end-ttest_st, ttrain_end-ttrain_st]
        return out
