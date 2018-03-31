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

    def __init__(self, paradigm, datasets=None, random_state=None, n_jobs=1,
                 overwrite=False, suffix=''):
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

        self.results = Results(type(self),
                               type(self.paradigm),
                               overwrite=overwrite,
                               suffix=suffix)

    def process(self, pipelines):
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

        for dataset in self.datasets:
            log.info('Processing dataset: {}'.format(dataset.code))
            results = self.evaluate(dataset, pipelines)
            for res in results:
                self.push_result(res, pipelines)

        return self.results.to_dataframe()

    def push_result(self, res, pipelines):
        message = '{} | '.format(res['pipeline'])
        message += '{} | {} | {}'.format(res['dataset'].code,
                                         res['id'], res['session'])
        message += ': Score %.3f' % res['score']
        log.info(message)
        self.results.add({res['pipeline']: res}, pipelines=pipelines)

    @abstractmethod
    def evaluate(self, dataset, pipelines):
        '''Evaluate results on a single dataset.

        This method return a generator. each results item is a dict with
        the following convension :

            res = {'time': Duration of the training ,
                   'dataset': dataset id,
                   'id': subject id,
                   'session': session id,
                   'score': score,
                   'n_samples': number of training examples,
                   'n_channels': number of channel,
                   'pipeline': pipeline name}
        '''
        pass
