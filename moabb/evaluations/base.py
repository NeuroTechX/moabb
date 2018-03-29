import logging
import traceback
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from moabb.analysis import Results
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm

log = logging.getLogger()


class BaseEvaluation(ABC):
    '''Base class that defines necessary operations for an evaluation.
    Evaluations determine what the train and test sets are and can implement
    additional data preprocessing steps for more complicated algorithms.

    Parameters
    ----------
    paradigm : Paradigm instance
        the paradigm to use.
    datasets : List of Dataset Instance.
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state:
        if not None, can guarantee same seed
    n_jobs: 1;
        number of jobs for fitting of pipeline

    '''

    def __init__(self, paradigm, datasets=None, random_state=None, n_jobs=1):
        """
        Init.
        """

        self.random_state = random_state
        self.n_jobs = n_jobs

        # check paradigm
        if not isinstance(paradigm, BaseParadigm):
            raise(ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm

        # if no dataset provided, then we get the list from the paradigm
        if datasets is None:
            datasets = self.paradigm.datasets

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

        for dataset in datasets:
            self.paradigm.verify(dataset)

        self.datasets = datasets

    def process(self, pipelines, overwrite=False, suffix=''):
        '''
        Runs tasks on all given datasets.
        '''

        # check pipelines
        if not isinstance(pipelines, dict):
            raise(ValueError("pipelines must be a dict"))

        for _, pipeline in pipelines.items():
            if not(isinstance(pipeline, BaseEstimator)):
                raise(ValueError("pipelines must only contains Pipelines "
                                 "instance"))

        self.results = Results(type(self),
                               type(self.paradigm),
                               overwrite=overwrite,
                               suffix=suffix)

        for dataset in self.datasets:
            log.info('Processing dataset: {}'.format(dataset.code))
            self.evaluate(dataset, pipelines)

        return self.results

    def push_result(self, res, pipelines):
        for pipe, r in res.items():
            message = '{} | '.format(pipe)
            message += '{} | {} | {}'.format(r['dataset'].code,
                                             r['id'], r['session'])
            message += ': Score %.3f' % r['score']
            log.info(message)
        self.results.add(res, pipelines=pipelines)

    @abstractmethod
    def evaluate(self, dataset, pipelines):
        '''

        '''
        pass
