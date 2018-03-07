"""Motor Imagery contexts"""

import mne
import numpy as np

from moabb.analysis import Results
from moabb.contexts.base import BaseParadigm


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

    """

    def __init__(self, pipelines, evaluator, datasets=None, fmin=7, fmax=35,
                 channels=None, **kwargs):
        """init"""
        super().__init__(pipelines=pipelines, evaluator=evaluator,
                         datasets=datasets, **kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.channels = channels

    def verify(self, dataset):
        '''
        Method that verifies dataset is correct for given parameters
        '''
        assert dataset.paradigm == 'imagery'

    def process(self, overwrite=False, suffix=''):
        '''
        Runs tasks on all given datasets.
        '''
        # Verify that datasets are valid for given paradigm first
        self.results = Results(type(self.evaluator),
                               type(self), overwrite=overwrite, suffix=suffix)
        for d in self.datasets:
            self.verify(d)
        for d in self.datasets:
            print('\n\nProcessing dataset: {}'.format(d.code))
            self.evaluator.preprocess_data(d, self)
            for s in d.subject_list:
                run_pipes = self.results.not_yet_computed(self.pipelines, d, s)
                if len(run_pipes) > 0:
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
            channels = () if self.channels is None else self.channels
            # picks channels
            picks = mne.pick_types(raw.info, eeg=True, stim=False,
                                   include=channels)

            # ensure events are desired:
            if len(events) > 0:
                keep_events = {key: val for key, val in event_dict.items() if
                               val in np.unique(events[:, 2])}
                if len(keep_events) > 0:
                    # copy before filtering, so we let raws intact.
                    raw_f = raw.copy().filter(bp_low, bp_high, method='iir',
                                              picks=picks, verbose=False)
                    epochs = mne.Epochs(raw_f, events, event_id=keep_events,
                                        tmin=time[0], tmax=time[1], proj=False,
                                        baseline=None, preload=True,
                                        verbose=False, picks=picks)
                    ep.append(epochs)

        return ep


class ImageryNClass(BaseMotorImagery):
    """Imagery for multi class classification

    Returns n-class imagery results, visualization agnostic but forces all
    datasets to have exactly n classes. Uses 'accuracy' as metric

    """

    def __init__(self, pipelines, evaluator, n_classes, **kwargs):
        self.n_classes = n_classes
        super().__init__(pipelines, evaluator, **kwargs)

    def verify(self, d):
        print('Warning: Assumes events have already been selected per dataset')
        super().verify(d)
        assert len(d.selected_events) == self.n_classes

    @property
    def scoring(self):
        return 'accuracy'


class LeftRightImagery(BaseMotorImagery):
    """Motor Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def verify(self, d):
        events = ['left_hand', 'right_hand']
        super().verify(d)
        assert set(events) <= set(d.event_id.keys())
        d.selected_events = dict(zip(events, [d.event_id[s] for s in events]))

    @property
    def scoring(self):
        return 'roc_auc'
