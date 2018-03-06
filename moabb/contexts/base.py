from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from sklearn.base import BaseEstimator

from moabb.datasets.base import BaseDataset
from moabb import utils


class BaseParadigm(ABC):
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
                raise(ValueError("datasets must be a list or a dataset "
                                 "instance"))

        for dataset in datasets:
            if not(isinstance(dataset, BaseDataset)):
                raise(ValueError("datasets must only contains dataset "
                                 "instance"))

        self.datasets = datasets

        # check pipelines
        if not isinstance(pipelines, dict):
            raise(ValueError("pipelines must be a dict"))

        for name, pipeline in pipelines.items():
            if not(isinstance(pipeline, BaseEstimator)):
                raise(ValueError("pipelines must only contains Pipelines "
                                 "instance"))
        self.pipelines = pipelines

    @abstractproperty
    def scoring(self):
        '''Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        '''
        pass
