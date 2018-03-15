"""Motor Imagery contexts"""

import mne
import numpy as np
import logging

from moabb.paradigms.base import BaseParadigm
from moabb.datasets import utils

log = logging.getLogger()


class BaseMotorImagery(BaseParadigm):
    """Base Imagery paradigm  Context.

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
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.
    interval: list | None, (default None)
        Interval to cut trial. If None, defaults to the dataset-defined interval

    """
    def __repr__(self):
        return '{}(fmin={},fmax={},channels={},interval={})'.format(
            type(self).__name__,
            self.fmin,
            self.fmax,
            self.channels,
            self.interval)

    def __init__(self, fmin=7, fmax=35, channels=None, interval=None, **kwargs):
        """init"""
        super().__init__(**kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.channels = channels
        self.interval = interval
        if self.interval is not None:
            assert self.interval[0] >= 0

    def verify(self, dataset):
        '''
        Method that verifies dataset is correct for given parameters
        '''
        assert dataset.paradigm == 'imagery'
        if self.interval is not None:
            assert dataset.interval[1] > self.interval[1]

    def raw_to_trials(self, raw, events, picks, event_dict, time):
        '''
        Given a single raw file turn it into a features matrix and labels
        '''
        raw_f = raw.filter(self.fmin, self.fmax, method='iir', picks=picks,
                           verbose=False)
        epochs = mne.Epochs(raw_f, events, event_id=event_dict,
                            tmin=time[0], tmax=time[1], proj=False,
                            baseline=None, preload=True,
                            verbose=False, picks=picks)
        return epochs.get_data(), epochs.events[:, -1]

    def cont_to_trials(self, raws, event_dict, time):
        '''Take list of raws and returns trial data and labels. Implements
        imagery-specific processing as well

        '''
        if type(raws) is not list:
            raws = [raws]
        Xall = []
        yall = []
        for raw in raws:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
            channels = () if self.channels is None else self.channels
            # picks channels
            picks = mne.pick_types(raw.info, eeg=True, stim=False,
                                   include=channels)
            log.debug('Picks: {}'.format(picks))
            # ensure events are desired:
            if len(events) > 0:
                keep_events = {key: val for key, val in event_dict.items() if
                               val in np.unique(events[:, 2])}
                if len(keep_events) > 0:
                    # copy before filtering, so we let raws intact.
                    X, y = self.raw_to_trials(
                        raw, events, picks, keep_events, time)
                    Xall.append(X)
                    yall.append(y)
        return np.concatenate(Xall, axis=0), np.concatenate(yall, axis=0)

    @property
    def datasets(self):
        return utils.dataset_search(paradigm='imagery')

    @property
    def scoring(self):
        return 'accuracy'


class MotorImageryMultiPass(BaseMotorImagery):

    def __repr__(self):
        return '{}(fbands={}, channels={},interval={})'.format(
            type(self).__name__,
            self.fbands,
            self.channels,
            self.interval)

    def __init__(self, fbands=np.array([[8, 14], [20, 30]]),
                 channels=None, **kwargs):
        """init"""
        super().__init__(**kwargs)
        self.fbands = fbands
        self.channels = channels

    def raw_to_trials(self, raw, events, picks, event_dict, time):
        y = None
        X = []
        for fmin, fmax in self.fbands:
            raw_f = raw.filter(fmin, fmax, method='iir', picks=picks,
                               verbose=False)
            epochs = mne.Epochs(raw_f, events, event_id=event_dict,
                                tmin=time[0], tmax=time[1], proj=False,
                                baseline=None, preload=True,
                                verbose=False, picks=picks)
            xband, y = epochs.get_data(), epochs.events[:, -1]
            X.append(xband)
        return np.stack(X, axis=3), y


def ImageryNClassFactory(parent):

    class ImageryNClass(parent):
        """Imagery for multi class classification

        Returns n-class imagery results, visualization agnostic but forces all
        datasets to have exactly n classes. Uses 'accuracy' as metric

        """

        def __init__(self, n_classes, **kwargs):
            self.n_classes = n_classes
            self.human_paradigm = 'ImageryNClass'
            super().__init__(**kwargs)

        def verify(self, d):
            log.warning(
                'Assumes events have already been selected per dataset')
            super().verify(d)
            assert len(d.selected_events) == self.n_classes

        @property
        def datasets(self):
            return utils.dataset_search(paradigm='imagery',
                                        total_classes=self.n_classes)

    return ImageryNClass


def LeftRightImageryFactory(parent):
    class LeftRightImagery(parent):
        """Motor Imagery for left hand/right hand classification

        Metric is 'roc_auc'

        """
        def __init__(self, **kwargs):
            """
        
            """
            super().__init__(**kwargs)
            self.human_paradigm = 'LeftRightImagery'
            
        def verify(self, d):
            events = ['left_hand', 'right_hand']
            super().verify(d)
            assert set(events) <= set(d.event_id.keys())
            d.selected_events = dict(
                zip(events, [d.event_id[s] for s in events]))

        @property
        def scoring(self):
            return 'roc_auc'

        @property
        def datasets(self):
            return utils.dataset_search(paradigm='imagery',
                                        events=['right_hand', 'left_hand'],
                                        has_all_events=True)
    return LeftRightImagery


globals()['LeftRightImagerySinglePass'] = LeftRightImageryFactory(
    BaseMotorImagery)
globals()['ImageryNClassSinglePass'] = ImageryNClassFactory(BaseMotorImagery)
globals()['LeftRightImageryMultiPass'] = LeftRightImageryFactory(
    MotorImageryMultiPass)
globals()['ImageryNClassMultiPass'] = ImageryNClassFactory(
    MotorImageryMultiPass)
