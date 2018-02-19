"""Motor Imagery contexts"""

import numpy as np
from .base import BaseImageryParadigm
from mne import Epochs, find_events
from mne.epochs import concatenate_epochs, equalize_epoch_counts
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

class BaseMotorImagery(BaseImageryParadigm):
    """Base Motor imagery context


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

    def __init__(self, pipelines, evaluator, datasets=None, fmin=7, fmax=35):
        super().__init__(pipelines, evaluator, datasets, fmin=fmin, fmax=fmax)


class ImageryNClass(BaseMotorImagery):
    """Imagery for multi class classification
    
    Returns n-class imagery results, visualization agnostic but forces all
    datasets to have exactly n classes. Uses 'accuracy' as metric

    """

    def __init__(self, pipelines, evaluator, n_classes , **kwargs):
        self.n_classes = n_classes
        super().__init__(pipelines, evaluator, **kwargs)

    def verify(self, d):
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
        super().verify(d)
        assert set(d.selected_events.keys()) == set(('left_hand','right_hand'))

    @property
    def scoring(self):
        return 'roc_auc'
