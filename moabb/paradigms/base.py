from abc import ABC, abstractproperty, abstractmethod


class BaseParadigm(ABC):
    """Base Context.
    """

    def __init__(self):
        """init"""
        pass

    @abstractproperty
    def scoring(self):
        '''Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        '''
        pass

    @abstractproperty
    def datasets(self):
        '''Property that define the list of compatible datasets

        '''
        pass

    @abstractmethod
    def verify(self, dataset):
        '''
        Method that verifies dataset is correct for given parameters
        '''
