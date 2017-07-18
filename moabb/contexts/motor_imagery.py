"""Motor Imagery contexts"""

import numpy as np
from .base import *
from mne import Epochs, find_events
from mne.epochs import concatenate_epochs, equalize_epoch_counts
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold
import pandas as pd

class MotorImageryContext(BaseContext):
    """Base motor imagery context

    Defines motor imagery-specific epoching and preprocessing so that, combined
    with a child of BaseContext, it is possible to create contexts of interest e.g.
    MotorImageryMultiClassWithinSubject

    """
    def __init__(self, datasets, pipelines, fmin=7., fmax=35.):
        self.fmin = fmin
        self.fmax = fmax
        super().__init__(datasets, pipelines)

    def _epochs(self, dataset, subjects, event_id):
        """epoch data

        NOTE: Is this necessary to have defined within the 'motor imagery'
        context, or should we simply put it in BaseContext?


        Returns a 2-d list, level 1 is subject and level 2 is sessions/runs within a subject
        """
        raws = dataset.get_data(subjects=subjects)
        ep = []
        # we process each subject independently
        for subj in raws:
            ep.append([])
            # ***we assume that all files are from single sessions here***
            for raw in subj:
                # find events
                events = find_events(raw, shortest_event=0, verbose=False)

                # pick some channels
                raw.pick_types(meg=True, eeg=True, stim=False,
                               eog=False, exclude='bads')

                # filter data
                raw.filter(self.fmin, self.fmax, method='iir')

                # epoch data
                epochs = Epochs(raw, events, event_id, dataset.tmin, dataset.tmax,
                                proj=False, baseline=None, preload=True,
                                verbose=False)
                ep[-1].append(epochs)
        return ep #shall we have it yield, since we don't need to load everything at the start?

    def prepare_data(self, dataset, subjects, equalize=True):
        """Prepare data for classification. Also (optionally) equalize classes.

        Returns:
            X:    data as ndarray (trials, channels, timepoints)
            y:    labels as ndarray (trials, )
            info: DataFrame with trial index, session, and subject information"""
        
        event_id = dataset.event_id
        subject_list = self._epochs(dataset, subjects, event_id)
        # because I don't know Pandas that well:
        session_list = []
        subjectid_list = []
        X_list = []
        y_list = []
        for i, subj in enumerate(subject_list):
            # Each Epochs object corresponds to one session
            session = []
            full_epochs = []
            for ii, epochs in enumerate(subj):
                epochs_list = [epochs[k] for k in event_id]
                # equalize for accuracy within sessions
                if equalize:
                    equalize_epoch_counts(epochs_list)
                ep = concatenate_epochs(epochs_list)
                session.extend([ii]*len(ep))
                full_epochs.append(ep)
            epochs = concatenate_epochs(full_epochs)
            X = epochs.get_data()*1e6
            y = epochs.events[:, -1]
            # replace events with values from 0-len(event_id)
            ynew = np.zeros((len(y),))
            for ind, v in enumerate(np.unique(y)):
                ynew[y==v]=ind
            # add to the list
            X_list.append(X)
            y_list.append(ynew)
            subjectid_list.extend([i]*X.shape[0])
            session_list.extend(session)
        info = pd.DataFrame({'Session':session_list,
                             'Subject':subjectid_list})
        return np.vstack(X_list,), np.concatenate(y_list), info


class MotorImageryMultiClassWithinSubject(MotorImageryContext, WithinSubjectContext):
    """Motor Imagery for multi class classification

    Multiclass motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with accuracy as a metric. Epochs count are equalized.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """
    def __init__(self, *args, **kwargs):
        MotorImageryContext.__init__(self, *args, **kwargs)

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""
        if len(dataset.event_id) < 3:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s only contains two classes" %
                             dataset.name))
        return MotorImageryContext.prepare_data(self, dataset, subjects, equalize=True)

    def score(self, clf, X, y, info, scoring='accuracy', n_jobs=1):
        """get the score"""
        return WithinSubjectContext.score(self, clf, X, y, info, scoring, n_jobs)


class MotorImageryTwoClassWithinSubject(MotorImageryContext, WithinSubjectContext):
    """Motor Imagery for binary classification

    Binary motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with AUC as a metric.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """
    def __init__(self,datasets, pipelines, fmin=7., fmax=35.):
        MotorImageryContext.__init__(self, datasets, pipelines, fmin, fmax)

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""
        if len(dataset.event_id) > 2:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s contains more than 2 classes" %
                             dataset.name))
        return MotorImageryContext.prepare_data(self, dataset, subjects, equalize=False)

    def score(self, clf, X, y, info, scoring='roc_auc',n_jobs=1):
        """get the score"""
        return WithinSubjectContext.score(self, clf, X, y, info, scoring, n_jobs)

class MotorImageryMultiClassCrossSubject(MotorImageryContext, CrossSubjectContext):
    """Motor Imagery for multi class classification

    Multiclass motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with accuracy as a metric. Epochs count are equalized.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """
    def __init__(self, *args, **kwargs):
        MotorImageryContext.__init__(self, *args, **kwargs)

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""
        if len(dataset.event_id) < 3:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s only contains two classes" %
                             dataset.name))
        return MotorImageryContext.prepare_data(self, dataset, subjects, equalize=True)

    def score(self, clf, X, y, info, scoring='accuracy', n_jobs=1):
        """get the score"""
        return CrossSubjectContext.score(self, clf, X, y, info, scoring, n_jobs)


class MotorImageryTwoClassCrossSubject(MotorImageryContext, CrossSubjectContext):
    """Motor Imagery for binary classification

    Binary motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with AUC as a metric.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """
    def __init__(self,datasets, pipelines, fmin=7., fmax=35.):
        MotorImageryContext.__init__(self, datasets, pipelines, fmin, fmax)

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""
        if len(dataset.event_id) > 2:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s contains more than 2 classes" %
                             dataset.name))
        return MotorImageryContext.prepare_data(self, dataset, subjects, equalize=False)

    def score(self, clf, X, y, info, scoring='roc_auc',n_jobs=1):
        """get the score"""
        return CrossSubjectContext.score(self, clf, X, y, info, scoring, n_jobs)

class MotorImageryTwoClassSubjectUpdate(MotorImageryContext, SubjectUpdateContext):
    """Motor Imagery for binary classification

    Binary motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with AUC as a metric.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """
    def __init__(self,datasets, pipelines, fmin=7., fmax=35.):
        MotorImageryContext.__init__(self, datasets, pipelines, fmin, fmax)

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""
        if len(dataset.event_id) > 2:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s contains more than 2 classes" %
                             dataset.name))
        return MotorImageryContext.prepare_data(self, dataset, subjects, equalize=False)

    def score(self, clf, X, y, info, scoring='accuracy',n_jobs=1):
        """get the score"""
        return SubjectUpdateContext.score(self, clf, X, y, info, scoring, n_jobs)
