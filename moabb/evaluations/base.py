from abc import ABC, abstractmethod
import numpy as np


class BaseEvaluation(ABC):
    '''Base class that defines necessary operations for an evaluation.
    Evaluations determine what the train and test sets are and can implement
    additional data preprocessing steps for more complicated algorithms.

    random_state: if not None, can guarantee same seed
    n_jobs: 1; number of jobs for fitting of pipeline

    '''

    def __init__(self, random_state=None, n_jobs=1):
        """
        Init.
        """
        if random_state is None:
            self.random_state = np.random.randint(0, 1000, 1)[0]
        self.n_jobs = n_jobs

    @abstractmethod
    def evaluate(self, dataset, subject, clf, paradigm):
        '''
        Return results in a dict
        '''
        pass

    def preprocess_data(self, dataset, paradigm):
        '''
        Optional paramter if any sort of dataset-wide computation is needed
        per subject
        '''
        pass
